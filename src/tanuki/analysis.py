"""Stage 5 - post-concat analysis: filters, (optional) batch correction, DensMAP / UMAP / t-SNE / PCA / NMF,
clusterings, mean spectra, QC tables, Bonsai. run_post_concat_analysis() is the inline stage 5 code of
v5's __main__ block, unchanged."""
from __future__ import annotations

import os
import re

import igraph as ig
import leidenalg
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import sklearn.manifold
import umap
from plotly.subplots import make_subplots
from sklearn.cluster import KMeans
from sklearn.decomposition import NMF, PCA
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import MinMaxScaler, RobustScaler

from .config import (ALLOW_SINGLETON_BATCHES, ANALYSIS_AREA_BOUNDS_BY_TYPE, ANALYSIS_AREA_MAX_PX,
    ANALYSIS_AREA_MIN_PX, BONSAI_USE_RAW_INTENSITIES, COMBAT_BATCH_COL, COMBAT_ENABLED, COMBAT_LABEL,
    FEATURE_LABEL, FEATURE_TAG, MASTER_DIR, MEAN_SPECTRUM_ANNOTATE_TOP, MEAN_SPECTRUM_ENABLED,
    MEAN_SPECTRUM_GROUPS, MEAN_SPECTRUM_HIGHLIGHT_FLOOR, MEAN_SPECTRUM_HIGHLIGHT_FOLD,
    MEAN_SPECTRUM_HIGHLIGHT_TOP_N, MEAN_SPECTRUM_LOG_Y, MEAN_SPECTRUM_SOURCE, MZ_COL_PRECISION, N_CLUSTERS,
    RANDOM_SEED, SPLIT_POS_NEG, USE_MORPHOLOGY_FEATURES, ZERO_RATE_PER_CELL_TYPE, ZERO_RATE_THRESHOLD)
from .bonsai import run_bonsai_stage
from .utils import (cell_type_of, get_masses_of_interest, group_color_map, group_sort_key, load_df_parquet,
    mz_col_name, slide_id_of)


# ============================================================
# v5: BATCH CORRECTION AND COLOUR HELPERS (shared with the neutrophil scripts)
# ============================================================

def combat_design_report(df, batch_col, group_cols, covariate_col="Individual Sample Type"):
    """
    Print how the groups sit on the batches before ComBat runs, and name every group that
    occurs in only ONE batch. For a label that is NOT the covariate (e.g. the IPSC cell line)
    the correction cannot separate that batch's effect from the group's own difference and
    removes both. The covariate's own groups are protected; one that sits on a single batch
    just has that batch's shift estimated from the other groups on it.
    """
    if batch_col not in df.columns:
        return
    print(f"📋 ComBat design: cells per {batch_col} x group")
    for col in group_cols:
        if col not in df.columns or col == batch_col:
            continue
        table = pd.crosstab(df[batch_col].astype(str), df[col].astype(str))
        print(table.to_string())
        if table.shape[0] > 1:
            nested = [g for g in table.columns if int((table[g] > 0).sum()) == 1]
            if nested and col == covariate_col:
                print(f"ℹ️ {col}: {nested} occur on a single {batch_col}; their {batch_col} shift is "
                      f"estimated from the other groups on it (the covariate keeps their own difference).")
            elif nested:
                print(f"⚠️ {col}: {len(nested)} group(s) occur on a single {batch_col}: {nested}. "
                      f"Correcting by {batch_col} also removes what distinguishes them.")


def combat_correct(df, feature_cols, batch_col, covariate_col="Individual Sample Type"):
    """
    Batch correction for single-cell MSI intensities: ComBat's location/scale model, fitted
    per m/z on the DETECTED (non-zero) values only. Returns (values, applied): the corrected
    matrix aligned with df's rows, and which rows were corrected.

    Why not pycombat_norm: the intensities are TIC fractions (~1e-5 .. 1e-2), so log1p did
    nothing, and about half of them are zero. ComBat then shifted every zero to a slide-
    specific constant (0% zeros, a third negative, cells separated MORE by slide), and even
    with the zeros put back its batch estimates were driven by each slide's zero FRACTION,
    so the detected values were shifted by the wrong amount. inmoose refuses missing values,
    so the zeros cannot simply be masked out of it. Here, for each m/z:
      * log of the detected values ~ intercept + covariate (cell type) + batch, least squares;
      * the batch terms are removed (relative to their cell-weighted mean) and the residual
        spread of each batch is rescaled to the pooled spread - ComBat's location/scale step
        without its empirical-Bayes shrinkage, which barely moves estimates from hundreds of
        cells per batch;
      * zeros stay zero and detections stay detections: no shift can create or erase one;
      * only batches with more than one SAMPLE take part; the rest keep their raw values.
    """
    X = df[feature_cols].to_numpy(dtype=float)
    out = X.copy()
    applied = np.zeros(len(df), dtype=bool)
    n_samples = df.groupby(batch_col)['sample name'].nunique()
    valid = n_samples[n_samples > 1].index
    mask = df[batch_col].isin(valid).to_numpy()
    if len(valid) < len(n_samples):
        print(f"⚠️ Excluding {len(n_samples) - len(valid)} {batch_col} batch(es) with a single sample "
              f"from correction: {sorted(set(n_samples.index) - set(valid))}")
    if len(valid) < 2:
        print(f"⚠️ Fewer than two {batch_col} batches with more than one sample - nothing to correct.")
        return out, applied

    Xv = X[mask]
    codes, levels = pd.factorize(df.loc[mask, batch_col].astype(str), sort=True)
    n_batches = len(levels)
    cov = pd.get_dummies(df.loc[mask, covariate_col].astype(str), drop_first=True).to_numpy(dtype=float)
    Xc = Xv.copy()
    n_done = n_skipped = 0
    for j in range(Xv.shape[1]):
        det = Xv[:, j] > 0
        cj = codes[det]
        counts = np.bincount(cj, minlength=n_batches).astype(float)
        present = np.flatnonzero(counts > 0)
        if present.size < 2:
            continue
        cov_j = cov[det]
        cov_j = cov_j[:, cov_j.any(axis=0)] if cov_j.size else cov_j
        design = np.hstack([np.ones((cj.size, 1)), cov_j,
                            (cj[:, None] == present[None, 1:]).astype(float)])
        if cj.size <= design.shape[1] + 1:
            n_skipped += 1
            continue
        L = np.log(Xv[det, j])
        beta, _, rank, _ = np.linalg.lstsq(design, L, rcond=None)
        if rank < design.shape[1]:
            # batch and covariate confounded for this m/z: the shift cannot be told apart
            n_skipped += 1
            continue
        gamma = np.zeros(n_batches)
        gamma[present[1:]] = beta[-(present.size - 1):]
        gamma_bar = float((counts[present] * gamma[present]).sum() / counts[present].sum())
        fitted = design @ beta
        resid = L - fitted
        biology = fitted - gamma[cj] + gamma_bar
        dof = np.maximum(counts - 1.0, 0.0)
        ss = np.bincount(cj, weights=resid * resid, minlength=n_batches)
        with np.errstate(invalid='ignore', divide='ignore'):
            s_b = np.sqrt(ss / dof)
            s_pool = float(np.sqrt(ss[dof > 0].sum() / dof[dof > 0].sum())) if dof.sum() > 0 else np.nan
            scale = np.where((counts > 2) & (s_b > 0) & np.isfinite(s_pool), s_pool / s_b, 1.0)
        Xc[det, j] = np.exp(biology + resid * scale[cj])
        n_done += 1
    out[mask] = Xc
    applied[mask] = True
    print(f"✅ Batch correction on {int(mask.sum())} cells over {n_batches} {batch_col} batches: "
          f"{n_done} m/z corrected on their detected values, {n_skipped} left as measured "
          f"(too few detections or confounded); zeros kept ({100 * float((Xv == 0).mean()):.0f}% of entries).")
    return out, applied



# Suffix stripped when comparing groups for the mean-spectrum sidebar: a peak's
# "distinctiveness" is judged per BASE sample type, not per stim status.
_POS_NEG_SUFFIX = re.compile(r'(_POS|_NEG)$', re.IGNORECASE)


def _pos_neg_base_name(name) -> str:
    """'GBA_ASTRO_POS' -> 'GBA_ASTRO'. Leaves a name with no POS/NEG suffix unchanged."""
    return _POS_NEG_SUFFIX.sub('', str(name))


def compute_group_highlight_peaks(names, mean_by_group, mz, fold=10.0, floor=0.0, top_n=5):
    """
    For each ORIGINAL group in `names` (e.g. 'GBA_ASTRO_POS'), return up to `top_n`
    (mz, kind, value) tuples describing the peaks that make its BASE sample type (POS/NEG
    suffix stripped) stand out from every other base type in this same figure:

        'unique'    only this base type has any signal there (mean > floor); ranked by
                    mean intensity, highest first.
        'dominant'  this base type's mean is at least `fold` times every other base
                    type's mean, but more than one base type has some signal; ranked by
                    that fold-over-next-highest ratio, highest first. Used to fill out
                    the list only once the unique peaks run out, since a unique peak is
                    always the more informative kind (infinite fold, by definition).

    POS and NEG variants of the same base type get the IDENTICAL list back, since the
    comparison happens at the base-type level - see MEAN_SPECTRUM_HIGHLIGHT_* above.
    Returns {original_group_name: [(mz, kind, value), ...]}.
    """
    base_names = sorted({_pos_neg_base_name(g) for g in names})
    base_mean = {}
    for base in base_names:
        member_means = [mean_by_group[g] for g in names if _pos_neg_base_name(g) == base]
        base_mean[base] = np.mean(member_means, axis=0)

    vals = np.array([base_mean[b] for b in base_names])          # (n_base, n_mass)
    detected = vals > floor
    n_detected = detected.sum(axis=0)

    order = np.argsort(vals, axis=0)                             # ascending, per mass
    top_idx = order[-1, :]
    col_range = np.arange(vals.shape[1])
    top_val = vals[top_idx, col_range]
    if vals.shape[0] > 1:
        second_idx = order[-2, :]
        second_val = vals[second_idx, col_range]
    else:
        second_val = np.zeros_like(top_val)
    with np.errstate(divide='ignore', invalid='ignore'):
        fold_ratio = np.where(second_val > 0, top_val / second_val,
                              np.where(top_val > 0, np.inf, np.nan))

    highlights_by_base = {}
    for bi, base in enumerate(base_names):
        unique_mask = (n_detected == 1) & detected[bi]
        unique_idx = np.where(unique_mask)[0]
        unique_idx = unique_idx[np.argsort(-vals[bi, unique_idx])]

        dominant_mask = (~unique_mask) & (top_idx == bi) & np.isfinite(fold_ratio) & (fold_ratio >= fold)
        dominant_idx = np.where(dominant_mask)[0]
        dominant_idx = dominant_idx[np.argsort(-fold_ratio[dominant_idx])]

        picks = [(float(mz[i]), 'unique', float(vals[bi, i])) for i in unique_idx[:top_n]]
        if len(picks) < top_n:
            for i in dominant_idx[:top_n - len(picks)]:
                picks.append((float(mz[i]), 'dominant', float(fold_ratio[i])))
        highlights_by_base[base] = picks

    return {g: highlights_by_base[_pos_neg_base_name(g)] for g in names}


def _format_highlight_text(picks) -> str:
    """Sidebar text for one panel: one line per peak, unique peaks marked with a star."""
    if not picks:
        return '<i>no distinctive peaks</i>'
    lines = []
    for mz_val, kind, value in picks:
        if kind == 'unique':
            lines.append(f'★ {mz_val:.4f}')
        else:
            fold_str = f'{value:.0f}x' if np.isfinite(value) else 'inf'
            lines.append(f'{mz_val:.4f} ({fold_str})')
    return '<br>'.join(lines)


def write_group_mean_spectra(df, feature_cols, values, group_col, tag='raw',
                             color_map=None, order_list=None):
    """
    Mean spectrum per label over the full mass list, from the single-cell matrix.
    One stacked panel per group, plus a tidy CSV of the same numbers.

    `values` is the n_cells x n_features block to average, passed separately from `df`
    so the caller decides whether that is the raw extraction or the ComBat output.

    Error bars are the standard error on the group mean ACROSS CELLS, so they say how
    well the group mean is pinned down, not how variable single cells are.
    """
    if values is None or len(feature_cols) == 0 or group_col not in df.columns:
        return None

    mz = np.array([float(str(c).replace('mz_', '')) for c in feature_cols], dtype=float)
    order = np.argsort(mz)
    mz = mz[order]
    mat = np.asarray(values, dtype=float)[:, order]

    groups = df[group_col].astype(str)
    if order_list:
        names = [g for g in order_list if g in set(groups)]
        names += sorted(set(groups) - set(names))
    else:
        # numeric-aware so cluster panels read 0, 1, 2, ... 10 rather than 0, 1, 10, 2
        def _panel_key(v):
            try:
                return (0, float(v), '')
            except (TypeError, ValueError):
                return (1, 0.0, str(v))
        names = sorted(groups.unique(), key=_panel_key)

    out = pd.DataFrame({'mz': mz})
    panels = []
    for g in names:
        mask = (groups == g).values
        n = int(mask.sum())
        if n == 0:
            continue
        block = mat[mask]
        mean = np.nanmean(block, axis=0)
        sem = (np.nanstd(block, axis=0, ddof=1) / np.sqrt(n)) if n > 1 else np.zeros_like(mean)
        out['mean_' + str(g)] = mean
        out['sem_' + str(g)] = sem
        panels.append((g, n, mean))

    if not panels:
        return None

    mean_by_group = {g: mean for g, n, mean in panels}
    highlights_by_group = compute_group_highlight_peaks(
        [g for g, n, mean in panels], mean_by_group, mz,
        fold=MEAN_SPECTRUM_HIGHLIGHT_FOLD, floor=MEAN_SPECTRUM_HIGHLIGHT_FLOOR,
        top_n=MEAN_SPECTRUM_HIGHLIGHT_TOP_N)

    safe = str(group_col).lower().replace(' ', '_')
    csv_path = f'mean_spectrum_{safe}_{tag}.csv'
    out.to_csv(csv_path, index=False)

    rows = len(panels)
    # A FIXED fraction of vertical_spacing gives a shrinking absolute pixel gap as the
    # figure gets shorter (few rows -> few pixels -> the subplot title collides with the
    # panel above it). Fix the GAP in pixels instead and derive vertical_spacing/height
    # from it, so every panel gets the same real spacing whether there are 2 rows or 90.
    _ROW_HEIGHT_PX, _GAP_PX = 260, 40
    _MARGIN_T, _MARGIN_B = 90, 60
    if rows > 1:
        _plot_area_px = rows * _ROW_HEIGHT_PX + (rows - 1) * _GAP_PX
        vertical_spacing = _GAP_PX / _plot_area_px
    else:
        _plot_area_px = _ROW_HEIGHT_PX
        vertical_spacing = 0.02
    fig_height = _MARGIN_T + _MARGIN_B + _plot_area_px

    fig = make_subplots(rows=rows, cols=1, shared_xaxes=True, vertical_spacing=vertical_spacing,
                        subplot_titles=[f'{g}   (n = {n} cells)' for g, n, _ in panels])
    span = float(np.ptp(mz)) if mz.size > 1 else 1.0
    bar_width = max(0.05, span / max(mz.size, 1) * 0.8)

    for i, (g, n, mean) in enumerate(panels, start=1):
        colour = color_map.get(g) if color_map else None
        fig.add_trace(go.Bar(x=mz, y=mean, width=bar_width, marker_color=colour,
                             marker_line_width=0, name=str(g), showlegend=False,
                             hovertemplate='m/z %{x:.4f}<br>%{y:.4g}<extra></extra>'),
                      row=i, col=1)
        if MEAN_SPECTRUM_ANNOTATE_TOP:
            for j in np.argsort(-mean)[:int(MEAN_SPECTRUM_ANNOTATE_TOP)]:
                if mean[j] <= 0:
                    continue
                fig.add_annotation(x=float(mz[j]), y=float(mean[j]), text=f'{mz[j]:.3f}',
                                   showarrow=False, yshift=9, font=dict(size=9), row=i, col=1)
        fig.update_yaxes(title_text='mean intensity', row=i, col=1,
                         type='log' if MEAN_SPECTRUM_LOG_Y else 'linear')

        # Distinctive-peaks sidebar: printed OUTSIDE the plot area (paper coordinates),
        # vertically centred on this row's own y-domain, so it reads as this panel's
        # sidebar regardless of how many rows the figure has. ★ marks a peak unique to
        # this group (see MEAN_SPECTRUM_HIGHLIGHT_* above); an (Nx) entry is a peak
        # shared with other groups but at least MEAN_SPECTRUM_HIGHLIGHT_FOLD higher here.
        y0, y1 = fig.get_subplot(row=i, col=1).yaxis.domain
        fig.add_annotation(
            xref='paper', yref='paper', x=1.015, y=(y0 + y1) / 2,
            xanchor='left', yanchor='middle', align='left', showarrow=False,
            text=_format_highlight_text(highlights_by_group.get(g, [])),
            font=dict(size=9),
            bgcolor='rgba(255,255,255,0.75)', bordercolor='rgba(0,0,0,0.15)',
            borderwidth=1, borderpad=4,
        )

    fig.update_xaxes(title_text='m/z', row=rows, col=1)
    src = 'raw, before batch correction' if tag == 'raw' else COMBAT_LABEL
    fig.update_layout(title=f'Mean single-cell spectrum by {group_col} '
                            f'({len(feature_cols)} masses, {src})',
                      template='plotly_white', bargap=0,
                      height=fig_height, width=1780,
                      margin=dict(l=70, r=260, t=_MARGIN_T, b=_MARGIN_B))

    base = f'mean_spectrum_{safe}_{tag}'
    fig.write_html(base + '.html', include_plotlyjs='cdn')
    try:
        fig.write_image(base + '.png', scale=2)
    except Exception as e:
        print(f'PNG export skipped for {base} ({e}) - HTML still written.')
    print(f'📈 Mean spectra by {group_col} ({tag}): {rows} group(s) -> {base}.html, {csv_path}')
    return out


# ============================================================
# STAGE 5 - POST-CONCAT ANALYSIS (v5: inline in the __main__ block)
# ============================================================

def run_post_concat_analysis():
    """Everything between the concat and the benchmark file in v5's __main__ block, unchanged: reads the
    concatenated tables in MASTER_DIR, writes cleaned_feature_matrix.csv and every plot / table, runs the
    Bonsai stage. Returns what the benchmark file and the Dash viewer need."""
    # set from what actually happened; write_group_mean_spectra and the Dash viewer read it
    global COMBAT_LABEL
    masses_of_interest = get_masses_of_interest()
    print("\n========== POST-CONCAT ANALYSIS ==========\n")

    master_directory = MASTER_DIR
    os.chdir(master_directory)
    basename = os.path.basename(os.path.abspath(master_directory))

    # FIX: renamed from masses_of_interest. Rebinding the module-level name here shadowed
    # the float list that stage 1 and stage 3 use, which is confusing at best.
    mass_col_names = [mz_col_name(x) for x in masses_of_interest]


    def detect_mass_columns(df_cols, masses_list):
        df_cols = list(df_cols)
        direct = [m for m in masses_list if m in df_cols]
        # Legacy fallback: files written before mz_col_name existed carry bare float labels.
        legacy = [c for c in df_cols
                  if not isinstance(c, str) and f"mz_{float(c):.{MZ_COL_PRECISION}f}" in masses_list]
        candidate = direct + legacy
        if not candidate:
            numeric_like = []
            ignore = {
                "mask_id", "area_pixels", "area_pixels_x", "area_pixels_y",
                "eccentricity", "major_axis_length", "minor_axis_length",
                "solidity", "branch_points", "centroid_row", "centroid_col",
                "number_trunks", "number_non_trunk_branches", "number_branch_ends", "total_object_skeleton_length",
                "well_cells_segmented", "well_cells_after_area_gate",
                "well_cells_final", "well_pct_retained"
            }
            for c in df_cols:
                if c in ignore:
                    continue
                try:
                    # FIX: this was `float(c.replace(...))` under a BARE `except:`. On a
                    # non-string column label .replace raises AttributeError, which the bare
                    # except swallowed, so the fallback silently returned nothing and you got
                    # "Could not detect any mass columns" with no clue why.
                    float(str(c).replace("mz_", ""))
                    numeric_like.append(c)
                except (TypeError, ValueError):
                    continue
            candidate = numeric_like

        out = []
        seen = set()
        for c in candidate:
            if c not in seen:
                out.append(c)
                seen.add(c)
        return out


    final_df_avg = load_df_parquet(str(MASTER_DIR / f"{basename}_concatenated_mask_id_averages.parquet"))
    feature_cols = detect_mass_columns(final_df_avg.columns, mass_col_names)

    if not feature_cols:
        raise ValueError(
            "Could not detect any mass columns in averages. "
            f"Expected names like {mass_col_names[:3]}; "
            f"columns present: {list(final_df_avg.columns)[:20]}"
        )


    print(f"MASS COLUMNS USED (AVG): {len(feature_cols)} of {len(mass_col_names)} requested")

    # per-cell standard errors, keyed by (sample, mask) - Bonsai's noise input
    sem_lookup = pd.DataFrame()
    _sem_path = MASTER_DIR / f"{basename}_concatenated_mask_id_sems.parquet"
    if _sem_path.exists():
        _sem_raw = load_df_parquet(str(_sem_path))
        if not _sem_raw.empty and {'mask_id', 'sample name'} <= set(_sem_raw.columns):
            _sem_raw = _sem_raw.drop_duplicates(subset=['sample name', 'mask_id'], keep='first')
            sem_lookup = _sem_raw.copy()
            sem_lookup.index = pd.MultiIndex.from_arrays(
                [sem_lookup['sample name'].astype(str), sem_lookup['mask_id'].astype(int)])
            print(f'Loaded per-cell standard errors for {len(sem_lookup)} cells.')
    else:
        print('⚠️ No per-cell standard errors on disk. Re-run stage 3 to generate them; '
              'without them Bonsai gets a noise estimate unrelated to the intensity '
              'scale and will exit with "No gene made the zscore-cutoff".')

    area_col = "area_pixels"
    if area_col not in final_df_avg.columns:
        raise ValueError(f"No valid area column found. Columns present: {list(final_df_avg.columns)}")

    # =========================================================================
    # v3: AREA FILTER, PER CELL TYPE
    # =========================================================================
    # Bounds come from ANALYSIS_AREA_BOUNDS_BY_TYPE (unlisted types fall back to
    # ANALYSIS_AREA_MIN_PX / ANALYSIS_AREA_MAX_PX); None on either side = no bound there.
    # A missing area (NaN) is kept, as before. The area distribution and the loss on each
    # side are printed per cell type, so a bound that bites one type far harder than the
    # others is visible in the log instead of only in the per-well retention table.
    print("Before area filter:", len(final_df_avg))
    _area_ct = final_df_avg["sample name"].map(cell_type_of)
    _area_default = (ANALYSIS_AREA_MIN_PX, ANALYSIS_AREA_MAX_PX)
    _area_lo = pd.to_numeric(
        _area_ct.map(lambda t: ANALYSIS_AREA_BOUNDS_BY_TYPE.get(t, _area_default)[0]), errors='coerce')
    _area_hi = pd.to_numeric(
        _area_ct.map(lambda t: ANALYSIS_AREA_BOUNDS_BY_TYPE.get(t, _area_default)[1]), errors='coerce')
    area_vals = pd.to_numeric(final_df_avg[area_col], errors='coerce')
    _too_small = _area_lo.notna() & area_vals.notna() & (area_vals < _area_lo)
    _too_big = _area_hi.notna() & area_vals.notna() & (area_vals > _area_hi)

    print("Area distribution by cell type (segmentation px, ~1.3 um/px):")
    print(area_vals.groupby(_area_ct.rename('CellType')).describe(percentiles=[.05, .5, .95, .99]).round(0).to_string())
    _area_report = pd.DataFrame({"CellType": _area_ct, "below_min": _too_small, "above_max": _too_big}) \
        .groupby("CellType").agg(cells=("below_min", "size"),
                                 below_min=("below_min", "sum"),
                                 above_max=("above_max", "sum"))
    _area_report["min_px"] = [ANALYSIS_AREA_BOUNDS_BY_TYPE.get(t, _area_default)[0] for t in _area_report.index]
    _area_report["max_px"] = [ANALYSIS_AREA_BOUNDS_BY_TYPE.get(t, _area_default)[1] for t in _area_report.index]
    _area_report["kept_pct"] = (100 * (1 - (_area_report["below_min"] + _area_report["above_max"])
                                       / _area_report["cells"])).round(1)
    print("Area filter by cell type:")
    print(_area_report.to_string())

    mask_valid_area = ~(_too_small | _too_big)
    final_df_avg = final_df_avg[mask_valid_area].reset_index(drop=True)
    final_df_avg.to_csv(r'postfilteravg.csv', index=False, na_rep='0')
    print("After area filter:", len(final_df_avg))

    print("Cleaning feature matrix (remove NaNs / empty rows)...")
    feature_cols = list(dict.fromkeys(feature_cols))

    final_df_avg[feature_cols] = final_df_avg[feature_cols].fillna(0)
    final_df_avg[feature_cols] = final_df_avg[feature_cols].astype(float)
    mask_nonzero = ~(final_df_avg[feature_cols] == 0).all(axis=1)
    final_df_avg = final_df_avg.loc[mask_nonzero].reset_index(drop=True)

    # =========================================================================
    # v3: ZERO-RATE FEATURE FILTER, PER CELL TYPE
    # =========================================================================
    _is_zero = (final_df_avg[feature_cols] == 0)
    _pooled_pass = _is_zero.mean().le(ZERO_RATE_THRESHOLD)
    if ZERO_RATE_PER_CELL_TYPE:
        _zero_by_type = _is_zero.groupby(final_df_avg["sample name"].map(cell_type_of)).mean()
        _type_pass = _zero_by_type.le(ZERO_RATE_THRESHOLD)
        _keep_feature = _type_pass.any(axis=0)
        print(f"Zero-rate filter per cell type (<={ZERO_RATE_THRESHOLD:.0%} zeros in at least one type): "
              f"{int(_keep_feature.sum())} kept, {int((_keep_feature & ~_pooled_pass).sum())} of them "
              f"would have been dropped by the pooled rule")
        print(f"   features passing, by cell type: {_type_pass.sum(axis=1).to_dict()}")
        print("   NOTE: a feature that is mostly zero across ALL cells has an interquartile range of 0,")
        print("   which RobustScaler leaves unscaled, so it carries little weight in UMAP/DensMAP/t-SNE.")
        print("   NMF (MinMax), Bonsai and the mean spectra use it fully.")
    else:
        _keep_feature = _pooled_pass
    cols_to_drop = _keep_feature.index[~_keep_feature].unique().tolist()
    final_df_avg = final_df_avg.drop(columns=cols_to_drop, errors='ignore')
    feature_cols = [c for c in feature_cols if c not in cols_to_drop]

    print(f"Keeping {len(feature_cols)} features with <={ZERO_RATE_THRESHOLD:.0%} zeros"
          + (" in at least one cell type" if ZERO_RATE_PER_CELL_TYPE else ""))
    if len(final_df_avg) == 0:
        raise ValueError("After cleaning, no valid rows remain for UMAP.")

    raw_intensities_df = final_df_avg[feature_cols].copy()

    unique_df_cols = list(dict.fromkeys(final_df_avg.columns))
    metadata_cols = [col for col in unique_df_cols if col not in feature_cols]
    final_df_avg = final_df_avg[metadata_cols + feature_cols]

    if USE_MORPHOLOGY_FEATURES:
        morph_cols = [
                "area_pixels", "perimeter", "extent", "eccentricity",
                "major_axis_length", "minor_axis_length", "solidity",
                "branch_points", "number_trunks", "number_non_trunk_branches",
                "number_branch_ends", "total_object_skeleton_length"
        ]
        morph_cols = [c for c in morph_cols if c in final_df_avg.columns]
    else:
        morph_cols = []

    # =========================================================================
    # EXTRACT METADATA EARLY FOR BATCH CORRECTION
    # =========================================================================
    # Built from the base <group>_<cell> combinations so SPLIT_POS_NEG has exactly one
    # place to act on: with it on, each base gets a _POS and a _NEG entry; with it off,
    # the base name itself is the group.
    _BASE_SAMPLE_TYPES = [
        'GBA_ASTRO', 'GBA_MICRO', 'GBA_NEURO',
        'CONTROL_ASTRO', 'CONTROL_MICRO', 'CONTROL_NEURO',
    ]
    if SPLIT_POS_NEG:
        LEGEND_ORDER = [f'{b}_{st}' for b in _BASE_SAMPLE_TYPES for st in ('POS', 'NEG')]
    else:
        LEGEND_ORDER = list(_BASE_SAMPLE_TYPES)
    LEGEND_ORDER.append('Other_Sample')
    print(f"🏷️ SPLIT_POS_NEG = {SPLIT_POS_NEG}: "
          f"{len(LEGEND_ORDER) - 1} sample-type group(s) "
          f"({'stim status kept separate' if SPLIT_POS_NEG else 'POS/NEG merged'}).")


    def get_gba_category(name):
        name_str = str(name)
        match_id = re.search(r'(?<!\d)(11291|11328|11302|11320|11574|11450|11507)(?!\d)', name_str)
        if match_id:
            id_val = match_id.group(0)
        else:
            fallback_match = re.search(r'(?<!\d)(\d{5})(?!\d)', name_str)
            id_val = fallback_match.group(1) if fallback_match else "Unknown"

        id_mapping = {
            '11291': 'CONTROL',
            '11328': 'GBA',
            '11302': 'CONTROL',
            '11320': 'CONTROL',
            '11574': 'GBA',
            '11450': 'CONTROL',
            '11507': 'GBA'
        }
        group = id_mapping.get(id_val, "Unknown")
        status = 'POS' if ('stim' in name_str.lower() or 'pos' in name_str.lower()) else 'NEG'

        cell = cell_type_of(name_str)

        # `status` is computed above either way, but only becomes part of the label
        # when SPLIT_POS_NEG is on - see the setting at the top of the file.
        raw_cat = f"{group}_{cell}_{status}" if SPLIT_POS_NEG else f"{group}_{cell}"
        if raw_cat not in LEGEND_ORDER:
            return "Other_Sample"
        return raw_cat


    final_df_avg["Individual Sample Type"] = final_df_avg["sample name"].apply(get_gba_category)

    # Stim status as its OWN column, independent of SPLIT_POS_NEG. "Individual Sample
    # Type" stays whatever that setting makes it (merged by default); these two columns
    # carry the stim / no-stim split alongside it for colouring, mean spectra and QC
    # without touching the ComBat covariate or the Bonsai stratification.
    def get_stim_status(name):
        s = str(name).lower()
        # "pos" only as a whole token, so "post" in a scan name can never count as stim
        if "stim" in s or re.search(r"(?<![a-z])pos(?![a-z])", s):
            return "STIM"
        return "NO_STIM"

    final_df_avg["Stim"] = final_df_avg["sample name"].apply(get_stim_status)
    final_df_avg["Sample Type + Stim"] = (final_df_avg["Individual Sample Type"].astype(str)
                                          + "_" + final_df_avg["Stim"])
    print(f"🏷️ Stim column: {final_df_avg['Stim'].value_counts().to_dict()}")


    # =========================================================================
    # DYNAMIC BATCH AND WELL IDENTIFICATION
    # =========================================================================
    def get_cell_type(sample_str):
        # v3: one definition for the whole script, see cell_type_of at the top
        return cell_type_of(sample_str)


    def get_slide_batch(sample_name):
        # notes: Batch = cell type + cell line number. Every batch should therefore contain
        # notes: multiple replicate samples; a batch of size 1 means the folder name failed to
        # notes: match the expected cell-type / cell-line pattern (naming error) and is checked below.
        # v3: this label is for colouring and the naming check. ComBat uses COMBAT_BATCH_COL.
        sample_str = str(sample_name)

        match_id = re.search(r'(?<!\d)(11291|11328|11302|11320|11574|11450|11507)(?!\d)', sample_str)
        if match_id:
            cell_id = match_id.group(0)
        else:
            fallback_match = re.search(r'(?<!\d)(\d{5})(?!\d)', sample_str)
            cell_id = fallback_match.group(1) if fallback_match else "UnknownID"

        cell_type = get_cell_type(sample_str)

        return f"{cell_type}_{cell_id}"


    def get_well_id(sample_name):
        # FIX: \b never fires between 'A1' and the following '_' (underscore is a word
        # character), so this returned 'Unknown' for every folder. Use explicit
        # non-alphanumeric boundaries instead; 'S2B1' still cannot match.
        match = re.search(r'(?<![A-Za-z0-9])([A-Ba-b][1-5])(?![A-Za-z0-9])', str(sample_name))
        if match:
            return match.group(1).upper()
        return "Unknown"


    print("🔍 Extracting Cell-Type + Cell-Line Batch IDs, Slides and Well Numbers...")
    final_df_avg["Batch"] = final_df_avg["sample name"].apply(get_slide_batch)
    final_df_avg["Well"] = final_df_avg["sample name"].apply(get_well_id)
    final_df_avg["Slide"] = final_df_avg["sample name"].apply(slide_id_of)
    final_df_avg["CellType"] = final_df_avg["sample name"].apply(cell_type_of)

    detected_batches = final_df_avg['Batch'].unique().tolist()
    print(f"📊 Detected {len(detected_batches)} Unique Batches (Cell Type + Cell Line): {detected_batches}")
    _slide_table = (final_df_avg.groupby('Slide')
                    .agg(samples=('sample name', 'nunique'), cells=('sample name', 'size'),
                         cell_types=('CellType', lambda s: ','.join(sorted(s.unique())))))
    print("📊 Slides (S#B# token in the folder name):")
    print(_slide_table.to_string())
    if 'UnknownSlide' in _slide_table.index:
        print("⚠️ Some folder names carry no S#B# slide token; their cells form one 'UnknownSlide' "
              "batch for ComBat. Rename them, or set COMBAT_BATCH_COL = 'Batch'.")

    # notes: Every batch (cell type + cell line) is expected to have multiple replicate samples.
    # notes: A singleton batch means a folder name didn't match the expected pattern - treat as a naming error.
    # FIX: this used value_counts() on a dataframe that is one row per CELL, so a batch had
    # to contain exactly one CELL to trip the guard. A genuinely mis-named folder still
    # contributes thousands of cells and sailed straight through, which is the opposite of
    # what the comment above promises. Count distinct samples instead.
    samples_per_batch = final_df_avg.groupby('Batch')['sample name'].nunique()
    singleton_batches = samples_per_batch[samples_per_batch == 1].index.tolist()
    if singleton_batches:
        offending = final_df_avg.loc[
            final_df_avg['Batch'].isin(singleton_batches), ['sample name', 'Batch']].drop_duplicates()
        print('🚨 SINGLETON BATCH(ES) DETECTED. Two possible causes:')
        print('   (a) a folder name is missing/misspelling its cell type or cell line ID, or')
        print('   (b) the batch HAD replicates but they failed earlier in the pipeline -')
        print('       check the stage 1 log for "Final Detection Failed" / registration errors')
        print('       in other folders of the same cell type + cell line.')
        for _, row in offending.iterrows():
            print(f"    '{row['sample name']}'  ->  Batch = '{row['Batch']}'")

        if not ALLOW_SINGLETON_BATCHES:
            raise ValueError(
                f"{len(singleton_batches)} batch(es) contain only a single sample: "
                f"{singleton_batches}. Every batch should be (cell type + cell line) with "
                "multiple replicates. Either fix the folder name(s) above, or - if the "
                "replicates failed in stage 1 - recover those folders first. Set "
                "ALLOW_SINGLETON_BATCHES = True to proceed anyway.")

        # v3: this is now a naming / replicate check only. ComBat batches on
        # COMBAT_BATCH_COL, and only a ComBat batch with a single sample is excluded from
        # correction (and tagged _FAILED_CORRECTION in Batch below).
        n_affected = int(final_df_avg['Batch'].isin(singleton_batches).sum())
        print(f'⚠️ ALLOW_SINGLETON_BATCHES is on: continuing with {len(singleton_batches)} '
              f'singleton batch(es), {n_affected} cell(s). Colour by Batch before reading '
              f'anything into a cluster made only of these cells.')

    # =========================================================================
    # PREPARE MORPHOLOGY AND MASS MATRICES
    # =========================================================================
    from sklearn.preprocessing import normalize

    # 1. Scale Morphology (This doesn't change before/after batch correction)
    if morph_cols:
        df_morph_temp = final_df_avg[morph_cols].copy()
        if "area_pixels" in df_morph_temp.columns:
            df_morph_temp["area_pixels"] = np.log1p(df_morph_temp["area_pixels"])
        X_morph = MinMaxScaler().fit_transform(df_morph_temp.values)
        print(f'🔬 Embedding on {len(feature_cols)} mass + {len(morph_cols)} morphology features.')
    else:
        # an (n, 0) block so every np.hstack below stays unchanged
        X_morph = np.zeros((len(final_df_avg), 0), dtype=float)
        print(f'🔬 Embedding on {len(feature_cols)} mass features; morphology excluded '
              f'(USE_MORPHOLOGY_FEATURES = False).')

    clusterz = N_CLUSTERS

    # =========================================================================
    # PRE-CORRECTION: SAVE UNCORRECTED SCALED DATA AND RUN DENSMAP
    # =========================================================================
    print("🧬 Running DensMAP on UNCORRECTED data for comparison...")

    # Scale the raw uncorrected mass data
    X_mass_uncorrected = RobustScaler().fit_transform(final_df_avg[feature_cols].values)
    X_mass_uncorrected = normalize(X_mass_uncorrected, norm='l2')

    # Combine Mass + Morph for uncorrected baseline
    X_combined_uncorrected = np.hstack([X_mass_uncorrected, X_morph])

    # FIX: passing random_state together with n_jobs=-1 makes UMAP fall back to a single
    # thread AND emit a warning on every call. random_state is the one that matters here.
    dens_uncorrected = umap.UMAP(
        densmap=True, n_neighbors=20, min_dist=0.1, metric='euclidean', random_state=RANDOM_SEED
    )
    x_dens_uncorrected = dens_uncorrected.fit_transform(X_combined_uncorrected)

    final_df_avg["DensMAP_Uncorrected1"] = x_dens_uncorrected[:, 0]
    final_df_avg["DensMAP_Uncorrected2"] = x_dens_uncorrected[:, 1]

    # Third panel: UNCORRECTED and MASS ONLY. With the other two (uncorrected
    # mass+morph, ComBat-corrected mass+morph) this separates the two effects:
    # panel 1 -> 2 is what morphology adds, panel 2 -> 3 is what batch correction adds.
    # X_mass_uncorrected is already the scaled, L2-normalised mass block, so nothing is
    # re-scaled here and the only difference from panel 2 is the absence of X_morph.
    if morph_cols:
        print('🧬 Running DensMAP on UNCORRECTED, MASS-ONLY data (third panel)...')
        dens_uncorr_mass = umap.UMAP(
            densmap=True, n_neighbors=20, min_dist=0.1, metric='euclidean',
            random_state=RANDOM_SEED)
        x_dens_uncorr_mass = dens_uncorr_mass.fit_transform(X_mass_uncorrected)
        final_df_avg['DensMAP_UncorrMassOnly1'] = x_dens_uncorr_mass[:, 0]
        final_df_avg['DensMAP_UncorrMassOnly2'] = x_dens_uncorr_mass[:, 1]
    else:
        print('🧬 Third DensMAP skipped: morphology is off, so uncorrected mass-only is '
              'the same embedding as DensMAP_Uncorrected.')

    # Fourth panel: MORPHOLOGY ONLY. No mass channel enters this embedding, so it shows
    # what cell shape alone separates; comparing it with the mass-only panel says whether
    # the mass+morph structure is driven by shape, by chemistry, or by both. X_morph is
    # the same MinMax-scaled block that goes into the combined panels.
    if morph_cols:
        print(f'🧬 Running DensMAP on MORPHOLOGY-ONLY features ({len(morph_cols)} features, fourth panel)...')
        dens_morph = umap.UMAP(
            densmap=True, n_neighbors=20, min_dist=0.1, metric='euclidean',
            random_state=RANDOM_SEED)
        x_dens_morph = dens_morph.fit_transform(X_morph)
        final_df_avg['DensMAP_MorphOnly1'] = x_dens_morph[:, 0]
        final_df_avg['DensMAP_MorphOnly2'] = x_dens_morph[:, 1]
        final_df_avg['ClusterMorphOnly'] = KMeans(
            n_clusters=clusterz, n_init=10, random_state=RANDOM_SEED).fit_predict(x_dens_morph)

    # =========================================================================
    # BATCH CORRECTION VIA INMOOSE (PYCOMBAT)
    # =========================================================================
    if COMBAT_BATCH_COL not in final_df_avg.columns:
        raise ValueError(f"COMBAT_BATCH_COL = {COMBAT_BATCH_COL!r} is not a column of the matrix "
                         f"(expected 'Slide' or 'Batch').")
    # v5: show what the batch correction would be entangled with BEFORE it runs
    combat_design_report(final_df_avg, COMBAT_BATCH_COL, ['Batch', 'Individual Sample Type'])
    if COMBAT_ENABLED:
        print(f"🧬 Applying ComBat batch correction via inmoose (batch = '{COMBAT_BATCH_COL}', "
              f"covariate = 'Individual Sample Type')...")
        _corrected, _applied = combat_correct(final_df_avg, feature_cols, COMBAT_BATCH_COL)
        final_df_avg[feature_cols] = _corrected
        # per ROW: cells of a single-sample batch keep raw values and are marked as such
        final_df_avg["ComBatApplied"] = _applied
        if _applied.any() and not _applied.all():
            final_df_avg.loc[~_applied, "Batch"] = final_df_avg.loc[~_applied, "Batch"] + "_FAILED_CORRECTION"
    else:
        print("🧬 ComBat is OFF (COMBAT_ENABLED = False): every 'corrected' embedding below is "
              "computed on the raw values.")
        final_df_avg["ComBatApplied"] = False
    COMBAT_LABEL = 'PyComBat corrected' if final_df_avg["ComBatApplied"].any() else 'not batch-corrected'

    # =========================================================================
    # POST-CORRECTION: SCALE CORRECTED DATA AND RUN DENSMAP
    # =========================================================================
    print("🧬 Running DensMAP on CORRECTED data...")

    # Scale the new PyComBat corrected mass data
    X_scaled = RobustScaler().fit_transform(final_df_avg[feature_cols].values)
    X_mass_corrected = normalize(X_scaled, norm='l2')

    # Combine Corrected Mass + Morph
    X_combined_corrected = np.hstack([X_mass_corrected, X_morph])

    dens_corrected = umap.UMAP(
        densmap=True, n_neighbors=20, min_dist=0.1, metric='euclidean', random_state=RANDOM_SEED
    )
    x_dens_corrected = dens_corrected.fit_transform(X_combined_corrected)

    final_df_avg["DensMAP_Corrected1"] = x_dens_corrected[:, 0]
    final_df_avg["DensMAP_Corrected2"] = x_dens_corrected[:, 1]

    final_df_avg["ClusterUnifiedDens"] = KMeans(
        n_clusters=clusterz, n_init=10, random_state=RANDOM_SEED).fit_predict(x_dens_corrected)

    # =========================================================================
    # ORIGINAL DIMENSIONALITY REDUCTIONS (Using Corrected Mass Data)
    # =========================================================================
    # UMAP / t-SNE / PCA / NMF / Leiden stay MASS ONLY by design. Morphology enters only
    # the DensMAP panels, so the three DensMAPs below isolate what it contributes.
    nmf_scaler = MinMaxScaler()
    X_nmf = nmf_scaler.fit_transform(final_df_avg[feature_cols].values)

    nmf = NMF(n_components=8, init="nndsvda", max_iter=500, random_state=RANDOM_SEED)
    x_nmf = nmf.fit_transform(X_nmf)

    # FIX: neither of these was seeded, so the UMAP and t-SNE panels moved between runs while
    # the DensMAP panels stayed put. Seed both. (random_state supersedes n_jobs in UMAP.)
    fit_umap = umap.UMAP(n_neighbors=20, min_dist=0.1, init='pca', n_components=2, metric='cosine',
                         random_state=RANDOM_SEED)
    x_umap = fit_umap.fit_transform(X_scaled)

    fit_tsne = sklearn.manifold.TSNE(n_components=2, init='pca', metric='cosine', n_jobs=-1,
                                     random_state=RANDOM_SEED)
    x_tsne = fit_tsne.fit_transform(X_scaled)

    pca = PCA(n_components=3, random_state=RANDOM_SEED)
    x_pca = pca.fit_transform(X_scaled)

    projections = {"UMAP": x_umap, "TSNE": x_tsne, "PCA": x_pca}

    for prefix, data in projections.items():
        for i in range(data.shape[1]):
            final_df_avg[f"{prefix}{i + 1}"] = data[:, i]

    for i in range(nmf.n_components):
        final_df_avg[f"NMF{i + 1}"] = x_nmf[:, i]

    # FIX: all four KMeans calls were unseeded, so cluster IDs (and therefore every legend
    # and every cluster heatmap row) were reshuffled on each run.
    final_df_avg["KmeansClusterUMAP"] = KMeans(
        n_clusters=clusterz, n_init=10, random_state=RANDOM_SEED).fit_predict(x_umap)
    final_df_avg["KmeansClusterTSNE"] = KMeans(
        n_clusters=clusterz, n_init=10, random_state=RANDOM_SEED).fit_predict(x_tsne)
    final_df_avg["KmeansClusterPCA"] = KMeans(
        n_clusters=clusterz, n_init=10, random_state=RANDOM_SEED).fit_predict(x_pca[:, :3])
    final_df_avg["ClusterNMF"] = KMeans(
        n_clusters=clusterz, n_init=10, random_state=RANDOM_SEED).fit_predict(x_nmf)

    nmf_loadings = pd.DataFrame(nmf.components_, columns=feature_cols,
                                index=[f"NMF{i + 1}" for i in range(nmf.n_components)])

    knn_adj = kneighbors_graph(X_scaled, n_neighbors=20, mode='connectivity', include_self=False)
    sources, targets = knn_adj.nonzero()
    edgelist = list(zip(sources.tolist(), targets.tolist()))
    # FIX: ig.Graph(edgelist) infers the vertex count from the largest index present. That
    # happens to be safe for a kNN graph, but state the vertex count instead of relying on it.
    g = ig.Graph(n=X_scaled.shape[0], edges=edgelist)

    partition = leidenalg.find_partition(g, leidenalg.ModularityVertexPartition, weights=None,
                                         n_iterations=-1, seed=RANDOM_SEED)
    final_df_avg["ClusterLeiden"] = np.array(partition.membership)

    if final_df_avg.columns.duplicated().any():
        print("Deduplicating dataframe columns...")
        new_cols = []
        used = {}
        for c in final_df_avg.columns:
            if c not in used:
                used[c] = 0
                new_cols.append(c)
            else:
                used[c] += 1
                new_cols.append(f"{c}_{used[c]}")
        final_df_avg.columns = new_cols


    # =========================================================================
    # PLOTTING FUNCTIONS
    # =========================================================================
    def create_scatter_plot(df, x_col, y_col, group_col, intensity_col=None, title="Scatter Plot",
                            color_map=None, symbol_map=None, order_list=None, hover_data=None,
                            save_html=True, save_png=True, png_show_legend=False, png_show_colorbar=False,
                            font_size=28, marker_size=10, feature_type="unknown_features"):

        if hover_data is None:
            hover_data = ["area_pixels"]

        required_cols = [x_col, y_col, group_col]
        if intensity_col:
            required_cols.append(intensity_col)

        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Column(s) not found: {missing}")

        df_plot = df.copy()
        df_plot[group_col] = df_plot[group_col].astype(str)

        if order_list is None:
            # FIX: sort numerically when the group values are numbers, so cluster legends
            # read 1,2,...,10 rather than '1','10','2'.
            def _group_sort_key(v):
                try:
                    return (0, float(v), "")
                except (TypeError, ValueError):
                    return (1, 0.0, str(v))

            order_list = sorted(df_plot[group_col].unique(), key=_group_sort_key)

        cat_orders = {group_col: order_list}
        # v5: fixed colour per group, and enough colours for >10 groups (Leiden had 17)
        if not intensity_col:
            color_map = group_color_map(order_list, color_map)

        hover_name = "sample name" if "sample name" in df_plot.columns else None

        if intensity_col:
            color_arg = intensity_col
            symbol_arg = group_col
            final_color_map = None
        else:
            color_arg = group_col
            symbol_arg = None
            final_color_map = color_map

        fig = px.scatter(
            df_plot, x=x_col, y=y_col,
            color=color_arg,
            symbol=symbol_arg,
            color_discrete_map=final_color_map,
            symbol_map=symbol_map,
            category_orders=cat_orders,
            hover_name=hover_name,
            hover_data=[c for c in hover_data if c in df_plot.columns],
            title=f"{title} (N={len(df_plot)})"
        )

        fig.update_layout(
            font=dict(size=font_size),
            title=dict(font=dict(size=font_size + 4)),
            legend=dict(
                title_text='',
                itemsizing='constant',
                yanchor="top", y=1,
                xanchor="left", x=1.02,
                traceorder="normal"
            ),
            margin=dict(r=250 if intensity_col else 100),
            showlegend=True
        )

        fig.update_traces(marker=dict(size=marker_size, sizemode='diameter'))

        if intensity_col:
            fig.update_layout(coloraxis_colorbar=dict(
                title=intensity_col,
                tickfont=dict(size=font_size - 4),
                titlefont=dict(size=font_size - 4),
                x=1.15
            ))

        clean_filename_base = f"{title.lower().replace(' ', '_').replace('+', 'plus')}_{feature_type}"

        if save_png:
            # v5: a kaleido failure must not stop the rest of stage 5
            try:
                fig.write_image(f"{clean_filename_base}.png", scale=3)
                print(f"Saved PNG: {clean_filename_base}.png")
            except Exception as e:
                print(f"⚠️ PNG export skipped for '{title}' ({e}) - HTML still written.")

        if save_html:
            fig.write_html(f"{clean_filename_base}.html", include_plotlyjs='cdn')
            print(f"Saved HTML: {clean_filename_base}.html")


    def create_densmap_triptych_html(df, group_col, color_map=None, order_list=None,
                                     filename='densmap_triptych.html'):
        """
        The three DensMAPs side by side, coloured identically:
            1. uncorrected, mass only
            2. uncorrected, mass + morphology
            3. PyComBat corrected, mass + morphology
        Left to right, 1 -> 2 is what morphology contributes and 2 -> 3 is what batch
        correction contributes. Missing panels are dropped, so this degrades to two
        panels when USE_MORPHOLOGY_FEATURES is False.
        """
        panels = [('DensMAP_UncorrMassOnly1', 'DensMAP_UncorrMassOnly2',
                   'Uncorrected, mass only'),
                  ('DensMAP_Uncorrected1', 'DensMAP_Uncorrected2',
                   f'Uncorrected, {FEATURE_LABEL}'),
                  ('DensMAP_Corrected1', 'DensMAP_Corrected2',
                   f'{COMBAT_LABEL}, {FEATURE_LABEL}'),
                  ('DensMAP_MorphOnly1', 'DensMAP_MorphOnly2',
                   'Morphology only')]
        panels = [pn for pn in panels if pn[0] in df.columns and pn[1] in df.columns]
        if len(panels) < 2:
            return
        fig = make_subplots(rows=1, cols=len(panels),
                            subplot_titles=[pn[2] for pn in panels])
        groups = df[group_col].astype(str)
        if order_list is None:
            order_list = sorted(groups.unique(), key=group_sort_key)
        # v5: one colour per group in EVERY panel (plotly otherwise colours by trace position)
        color_map = group_color_map(order_list, color_map)
        for idx, (x_col, y_col, _) in enumerate(panels):
            for group in order_list:
                subset = df[groups == group]
                if subset.empty:
                    continue
                fig.add_trace(go.Scatter(
                    x=subset[x_col], y=subset[y_col], mode='markers',
                    marker=dict(size=5, color=color_map.get(group) if color_map else None),
                    name=str(group), legendgroup=str(group), showlegend=(idx == 0),
                    hovertext=subset['sample name'] if 'sample name' in subset.columns else None),
                    row=1, col=idx + 1)
            fig.update_xaxes(title_text='DensMAP 1', row=1, col=idx + 1)
            fig.update_yaxes(title_text='DensMAP 2', row=1, col=idx + 1)
        fig.update_layout(
            title_text=f'DensMAP: morphology and batch correction, coloured by {group_col}',
            height=750, width=560 * len(panels), template='plotly_white',
            legend=dict(itemsizing='constant', traceorder='normal'))
        fig.write_html(filename, include_plotlyjs='cdn')
        print(f'Saved DensMAP triptych: {filename}')

    def create_side_by_side_html(df, group_col, color_map=None, order_list=None, filename_prefix="comparative",
                                 feature_type="unknown"):
        df_plot = df.copy()
        df_plot[group_col] = df_plot[group_col].astype(str)

        if order_list is None:
            order_list = sorted(df_plot[group_col].unique(), key=group_sort_key)
        # v5: one colour per group in BOTH panels (plotly otherwise colours by trace position,
        # so each batch changed colour between the uncorrected and corrected panel)
        color_map = group_color_map(order_list, color_map)

        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=(f"UNCORRECTED DensMAP", f"{COMBAT_LABEL.upper()} DensMAP")
        )

        for group in order_list:
            df_group = df_plot[df_plot[group_col] == group]
            if df_group.empty:
                continue

            color = color_map.get(group) if color_map else None

            fig.add_trace(go.Scatter(
                x=df_group["DensMAP_Uncorrected1"],
                y=df_group["DensMAP_Uncorrected2"],
                mode='markers',
                marker=dict(color=color, size=6),
                name=group,
                legendgroup=group,
                hovertext=df_group["sample name"]
            ), row=1, col=1)

            fig.add_trace(go.Scatter(
                x=df_group["DensMAP_Corrected1"],
                y=df_group["DensMAP_Corrected2"],
                mode='markers',
                marker=dict(color=color, size=6),
                name=group,
                legendgroup=group,
                showlegend=False,
                hovertext=df_group["sample name"]
            ), row=1, col=2)

        fig.update_layout(
            title_text=f"Batch Correction Evaluation: Colored by {group_col}",
            height=800,
            width=1600,
            template="plotly_white",
            legend=dict(itemsizing='constant', traceorder="normal")
        )

        fig.update_xaxes(title_text="DensMAP 1", row=1, col=1)
        fig.update_yaxes(title_text="DensMAP 2", row=1, col=1)
        fig.update_xaxes(title_text="DensMAP 1", row=1, col=2)
        fig.update_yaxes(title_text="DensMAP 2", row=1, col=2)

        safe_name = f"{filename_prefix}_{group_col.lower().replace(' ', '_')}_{feature_type}.html"
        fig.write_html(safe_name)
        print(f"Saved Comparative HTML: {safe_name}")


    # FIX: cluster columns used to be cast to str here, which made
    # groupby("ClusterUnifiedDens") order lexicographically ('10' before '2') and gave
    # legends the same wrong order. They stay numeric now; create_scatter_plot and the
    # Dash callback each stringify on their own copy for discrete colouring.
    cluster_cols = [c for c in final_df_avg.columns if "Cluster" in c]

    # Both the split (_POS/_NEG) and the merged base names are listed, so the colour map
    # is valid either way SPLIT_POS_NEG is set and no group silently loses its colour.
    # When merged, a base type takes the colour its _POS variant had.
    custom_colors = {
        # split: one colour per stim status
        'GBA_ASTRO_POS': '#E69F00', 'GBA_ASTRO_NEG': '#D55E00',
        'GBA_MICRO_POS': '#56B4E9', 'GBA_MICRO_NEG': '#0072B2',
        'GBA_NEURO_POS': '#009E73', 'GBA_NEURO_NEG': '#000000',
        'CONTROL_ASTRO_POS': '#F0E442', 'CONTROL_ASTRO_NEG': '#CC79A7',
        'CONTROL_MICRO_POS': '#332288', 'CONTROL_MICRO_NEG': '#88CCEE',
        'CONTROL_NEURO_POS': '#117733', 'CONTROL_NEURO_NEG': '#AA4499',
        # merged (SPLIT_POS_NEG = False)
        'GBA_ASTRO': '#E69F00', 'GBA_MICRO': '#56B4E9', 'GBA_NEURO': '#009E73',
        'CONTROL_ASTRO': '#F0E442', 'CONTROL_MICRO': '#332288', 'CONTROL_NEURO': '#117733',
        'Other_Sample': '#999999'
    }
    # 'Sample Type + Stim' reuses the split-mode colours: STIM takes the _POS colour,
    # NO_STIM the _NEG colour, so the two labelling schemes never disagree on a hue.
    for _b in _BASE_SAMPLE_TYPES + ['Other_Sample']:
        custom_colors.setdefault(f'{_b}_STIM', custom_colors.get(f'{_b}_POS', custom_colors.get(_b, '#999999')))
        custom_colors.setdefault(f'{_b}_NO_STIM', custom_colors.get(f'{_b}_NEG', custom_colors.get(_b, '#999999')))
    STIM_LEGEND_ORDER = [f'{_b}_{_s}' for _b in _BASE_SAMPLE_TYPES + ['Other_Sample'] for _s in ('STIM', 'NO_STIM')]
    custom_colors.setdefault('STIM', '#D55E00')
    custom_colors.setdefault('NO_STIM', '#0072B2')

    # =========================================================================
    # PLOT GENERATION
    # =========================================================================
    # v5: Bonsai runs LAST (after the CSV, QC table and plots below). It takes ~1 h single-
    # process on Windows; before, a hang or Ctrl+C there lost every stage-5 output.
    bonsai_segments = None

    # ---- mean spectrum per label, from the final single-cell matrix ----
    if MEAN_SPECTRUM_ENABLED:
        print('\n📈 ===== MEAN SPECTRA BY LABEL =====')
        _spec_sources = []
        if MEAN_SPECTRUM_SOURCE in ('raw', 'both'):
            _spec_sources.append(('raw', raw_intensities_df[feature_cols].astype(float).values))
        if MEAN_SPECTRUM_SOURCE in ('corrected', 'both'):
            _spec_sources.append(('corrected', final_df_avg[feature_cols].astype(float).values))
        for _gcol in MEAN_SPECTRUM_GROUPS:
            if _gcol not in final_df_avg.columns:
                print(f"   skipping '{_gcol}': not a column in the matrix.")
                continue
            for _tag, _vals in _spec_sources:
                try:
                    write_group_mean_spectra(
                        final_df_avg, feature_cols, _vals, _gcol, tag=_tag,
                        color_map=custom_colors if _gcol in ('Individual Sample Type', 'Sample Type + Stim', 'Stim') else None,
                        order_list=(LEGEND_ORDER if _gcol == 'Individual Sample Type'
                                    else STIM_LEGEND_ORDER if _gcol == 'Sample Type + Stim' else None))
                except Exception as e:
                    print(f'⚠️ Mean spectrum failed for {_gcol} ({_tag}): {e}')

    print("Generating side-by-side comparison plots...")
    create_side_by_side_html(final_df_avg, group_col="Batch", filename_prefix="side_by_side_technical",
                             feature_type=FEATURE_TAG)
    # v3: the column ComBat actually corrected on, so its effect can be judged directly
    if COMBAT_BATCH_COL != "Batch" and COMBAT_BATCH_COL in final_df_avg.columns:
        create_side_by_side_html(final_df_avg, group_col=COMBAT_BATCH_COL,
                                 filename_prefix="side_by_side_technical",
                                 feature_type=FEATURE_TAG)
    create_side_by_side_html(final_df_avg, group_col="Individual Sample Type", color_map=custom_colors,
                             order_list=LEGEND_ORDER, filename_prefix="side_by_side_biological",
                             feature_type=FEATURE_TAG)

    create_densmap_triptych_html(final_df_avg, 'Individual Sample Type',
                                 color_map=custom_colors, order_list=LEGEND_ORDER)
    create_densmap_triptych_html(final_df_avg, 'Batch',
                                 filename='densmap_triptych_batch.html')
    if COMBAT_BATCH_COL != "Batch" and COMBAT_BATCH_COL in final_df_avg.columns:
        create_densmap_triptych_html(final_df_avg, COMBAT_BATCH_COL,
                                     filename=f'densmap_triptych_{COMBAT_BATCH_COL.lower()}.html')

    if 'DensMAP_UncorrMassOnly1' in final_df_avg.columns:
        create_scatter_plot(final_df_avg, 'DensMAP_UncorrMassOnly1', 'DensMAP_UncorrMassOnly2',
                            'Individual Sample Type',
                            title='DensMAP Uncorrected Individual Samples',
                            color_map=custom_colors, order_list=LEGEND_ORDER,
                            feature_type='mass_only')

    if 'DensMAP_MorphOnly1' in final_df_avg.columns:
        create_scatter_plot(final_df_avg, 'DensMAP_MorphOnly1', 'DensMAP_MorphOnly2',
                            'Individual Sample Type',
                            title='DensMAP Morphology Only Individual Samples',
                            color_map=custom_colors, order_list=LEGEND_ORDER,
                            feature_type='morph_only')
        create_scatter_plot(final_df_avg, 'DensMAP_MorphOnly1', 'DensMAP_MorphOnly2',
                            'ClusterMorphOnly', title='DensMAP Morphology Only Clusters',
                            feature_type='morph_only')
        create_scatter_plot(final_df_avg, 'DensMAP_MorphOnly1', 'DensMAP_MorphOnly2',
                            'Stim', title='DensMAP Morphology Only Stim vs No Stim',
                            color_map=custom_colors, order_list=['STIM', 'NO_STIM'],
                            feature_type='morph_only')

    create_scatter_plot(final_df_avg, "DensMAP_Corrected1", "DensMAP_Corrected2", "ClusterUnifiedDens",
                        title="DensMAP Clusters", feature_type=FEATURE_TAG)
    create_scatter_plot(final_df_avg, "DensMAP_Corrected1", "DensMAP_Corrected2", "Individual Sample Type",
                        title="DensMAP Individual Samples", color_map=custom_colors, order_list=LEGEND_ORDER,
                        feature_type=FEATURE_TAG)

    # stim / no-stim, kept separate from the default grouping
    create_scatter_plot(final_df_avg, "DensMAP_Corrected1", "DensMAP_Corrected2", "Stim",
                        title="DensMAP Stim vs No Stim", color_map=custom_colors,
                        order_list=["STIM", "NO_STIM"], feature_type=FEATURE_TAG)
    create_scatter_plot(final_df_avg, "DensMAP_Corrected1", "DensMAP_Corrected2", "Sample Type + Stim",
                        title="DensMAP Individual Samples by Stim", color_map=custom_colors,
                        order_list=STIM_LEGEND_ORDER, feature_type=FEATURE_TAG)
    create_scatter_plot(final_df_avg, "UMAP1", "UMAP2", "Sample Type + Stim",
                        title="UMAP Individual Samples by Stim", color_map=custom_colors,
                        order_list=STIM_LEGEND_ORDER, feature_type="mass_only")

    create_scatter_plot(final_df_avg, "UMAP1", "UMAP2", "KmeansClusterUMAP", title="UMAP Clusters",
                        feature_type="mass_only")
    create_scatter_plot(final_df_avg, "UMAP1", "UMAP2", "Individual Sample Type", title="UMAP Individual Samples",
                        color_map=custom_colors, order_list=LEGEND_ORDER, feature_type="mass_only")

    create_scatter_plot(final_df_avg, "TSNE1", "TSNE2", "KmeansClusterTSNE", title="TSNE Clusters",
                        feature_type="mass_only")
    create_scatter_plot(final_df_avg, "TSNE1", "TSNE2", "Individual Sample Type", title="TSNE Individual Samples",
                        color_map=custom_colors, order_list=LEGEND_ORDER, feature_type="mass_only")

    create_scatter_plot(final_df_avg, x_col="TSNE1", y_col="TSNE2", group_col='ClusterNMF',
                        title="tSNE colored by NMF clusters", feature_type="mass_only")
    create_scatter_plot(final_df_avg, x_col="TSNE1", y_col="TSNE2", group_col='ClusterLeiden',
                        title="tSNE colored by Leiden clusters", feature_type="mass_only")

    nmf_cols = [c for c in final_df_avg.columns if re.match(r'^NMF\d+$', c)]
    final_df_avg["NMF_dominant"] = final_df_avg[nmf_cols].idxmax(axis=1)

    unique_factors = sorted(final_df_avg["NMF_dominant"].unique())
    auto_color_map = {factor: px.colors.qualitative.Safe[i % len(px.colors.qualitative.Safe)] for i, factor in
                      enumerate(unique_factors)}

    create_scatter_plot(final_df_avg, x_col="TSNE1", y_col="TSNE2", group_col='NMF_dominant',
                        title="tSNE colored by dominant NMF factor", color_map=auto_color_map,
                        hover_data=nmf_cols + ["area_pixels"], feature_type="mass_only")

    # FIX: a set gave a different column order on every run. Collect in a stable order and
    # sort numerically so the heatmap axis is reproducible and reads as an m/z axis.
    def _feature_sort_key(c):
        try:
            return (0, float(str(c).replace("mz_", "")), "")
        except ValueError:
            return (1, 0.0, str(c))

    top_features = []
    seen_features = set()
    for i in range(nmf_loadings.shape[0]):
        for c in nmf_loadings.iloc[i].nlargest(15).index:
            if c not in seen_features:
                seen_features.add(c)
                top_features.append(c)
    top_features = sorted(top_features, key=_feature_sort_key)

    loadings_to_plot = nmf_loadings[top_features]

    fig_loadings = px.imshow(
        loadings_to_plot, labels=dict(x="Mass Features (m/z)", y="NMF Factor", color="Weighting"),
        x=loadings_to_plot.columns, y=loadings_to_plot.index, color_continuous_scale='Viridis',
        aspect="auto", title="NMF Loadings: Top Features Defining Each Factor"
    )

    fig_loadings.update_xaxes(side="bottom", tickangle=45)
    fig_loadings.update_layout(height=600, width=1200, margin=dict(l=50, r=50, b=100, t=100))
    fig_loadings.write_html("nmf_loadings_heatmap_mass_only.html")

    # =========================================================================
    # PSEUDO-LOADINGS FOR CORRECTED DENSMAP (NON-LINEAR FEATURE IMPORTANCE)
    # =========================================================================
    print("Calculating Pseudo-Loadings for Corrected DensMAP...")

    all_unified_features = feature_cols + morph_cols

    dens1_corr = final_df_avg[all_unified_features].apply(
        lambda x: x.corr(final_df_avg["DensMAP_Corrected1"])).dropna().sort_values()
    dens2_corr = final_df_avg[all_unified_features].apply(
        lambda x: x.corr(final_df_avg["DensMAP_Corrected2"])).dropna().sort_values()

    fig_height = max(600, len(dens1_corr) * 20)

    fig_dens1 = px.bar(
        x=dens1_corr.values, y=dens1_corr.index, orientation='h',
        title="Corrected DensMAP Axis 1 - Full Pseudo-Loadings",
        labels={'x': "Pearson Correlation (Pull Strength)", 'y': "Feature"},
        color=dens1_corr.values, color_continuous_scale="RdBu"
    )
    fig_dens1.update_layout(height=fig_height, width=1000, margin=dict(l=200))
    fig_dens1.write_html("densmap1_pseudo_loadings_" + FEATURE_TAG + ".html")

    fig_dens2 = px.bar(
        x=dens2_corr.values, y=dens2_corr.index, orientation='h',
        title="Corrected DensMAP Axis 2 - Full Pseudo-Loadings",
        labels={'x': "Pearson Correlation (Pull Strength)", 'y': "Feature"},
        color=dens2_corr.values, color_continuous_scale="RdBu"
    )
    fig_dens2.update_layout(height=fig_height, width=1000, margin=dict(l=200))
    fig_dens2.write_html("densmap2_pseudo_loadings_" + FEATURE_TAG + ".html")

    # =========================================================================
    # CLUSTER ENRICHMENT HEATMAP (DENSMAP MARKER FEATURES)
    # =========================================================================
    print("Extracting defining features for Corrected DensMAP Clusters...")

    # FIX: groupby now runs on numeric cluster IDs, so sort_index() is a true numeric sort.
    cluster_means = final_df_avg.groupby("ClusterUnifiedDens")[all_unified_features].mean().sort_index()
    cluster_zscores = cluster_means.apply(lambda x: (x - x.mean()) / (x.std() + 1e-8), axis=0)

    # FIX: same non-determinism as the NMF heatmap. Stable collection, then sorted so
    # mass features come first in m/z order and morphology features follow alphabetically.
    top_cluster_features = []
    seen_cluster_features = set()
    for i in range(len(cluster_zscores)):
        for c in cluster_zscores.iloc[i].nlargest(5).index:
            if c not in seen_cluster_features:
                seen_cluster_features.add(c)
                top_cluster_features.append(c)
    top_cluster_features = sorted(top_cluster_features, key=_feature_sort_key)

    fig_dens_heatmap = px.imshow(
        cluster_zscores[top_cluster_features],
        labels=dict(x=f"Features ({FEATURE_LABEL})", y="DensMAP Cluster ID", color="Z-Score"),
        x=top_cluster_features,
        y=cluster_zscores.index.astype(str),
        color_continuous_scale='Viridis',
        aspect="auto",
        title="Corrected DensMAP Clusters: Top Enriched Features"
    )
    fig_dens_heatmap.update_xaxes(side="bottom", tickangle=45)
    fig_dens_heatmap.update_layout(height=600, width=1200, margin=dict(l=50, r=50, b=150, t=100))
    fig_dens_heatmap.write_html("corrected_densmap_cluster_heatmap_" + FEATURE_TAG + ".html")

    # ---- per-well cell survivorship, carried on every row of the final matrix ----
    if 'sample name' in final_df_avg.columns:
        final_counts = final_df_avg.groupby('sample name').size()
        final_df_avg['well_cells_final'] = final_df_avg['sample name'].map(final_counts).astype(int)
        if 'well_cells_segmented' in final_df_avg.columns:
            seg = final_df_avg['well_cells_segmented'].replace(0, np.nan)
            final_df_avg['well_pct_retained'] = (100 * final_df_avg['well_cells_final'] / seg).round(1)

        qc_cols = [c for c in ['sample name', 'Individual Sample Type', 'CellType', 'Stim', 'Sample Type + Stim',
                               'Batch', 'Slide', 'Well',
                               'well_cells_segmented', 'well_cells_after_area_gate',
                               'well_cells_final', 'well_pct_retained']
                   if c in final_df_avg.columns]
        qc = final_df_avg[qc_cols].drop_duplicates('sample name').sort_values('sample name')
        qc.to_csv('qc_cells_per_well.csv', index=False)
        print(f'🔍 Per-well cell counts -> qc_cells_per_well.csv ({len(qc)} wells)')
        if 'well_cells_segmented' in qc.columns:
            print(f"   segmented total {int(qc['well_cells_segmented'].sum())}, "
                  f"in final matrix {int(qc['well_cells_final'].sum())}")
            worst = qc.nsmallest(5, 'well_pct_retained')[
                ['sample name', 'well_cells_segmented', 'well_cells_final', 'well_pct_retained']]
            print('   lowest retention:')
            print(worst.to_string(index=False))
            # v3: the same survivorship per cell type, from the stage 3 gate onwards, so a
            # filter that bites one type harder than the others is visible in one table
            if 'CellType' in qc.columns and 'well_cells_after_area_gate' in qc.columns:
                _by_type = qc.groupby('CellType')[['well_cells_segmented', 'well_cells_after_area_gate',
                                                   'well_cells_final']].sum()
                _by_type['pct_of_stage3'] = (100 * _by_type['well_cells_final']
                                             / _by_type['well_cells_after_area_gate'].replace(0, np.nan)).round(1)
                print('   by cell type (segmented -> past stage 3 gate -> final matrix):')
                print(_by_type.to_string())

    print("Saving fully enriched feature matrix (with TSNE/UMAP coords) for Cloud Run...")
    final_df_avg.to_csv("cleaned_feature_matrix.csv", index=False)

    # ---- Bonsai tree, now that everything else is on disk ----
    _n_cols_before = final_df_avg.shape[1]
    final_df_avg, bonsai_segments = run_bonsai_stage(
        final_df_avg, feature_cols, sem_lookup, custom_colors, LEGEND_ORDER,
        raw_values=(raw_intensities_df[feature_cols].astype(float).values
                    if BONSAI_USE_RAW_INTENSITIES else None))
    if final_df_avg.shape[1] != _n_cols_before:
        if MEAN_SPECTRUM_ENABLED and 'ClusterBonsai' in final_df_avg.columns:
            # cells outside the exported subsample carry -1: not a cluster, so not a panel
            _in_tree = (final_df_avg['ClusterBonsai'] >= 0).to_numpy()
            try:
                write_group_mean_spectra(
                    final_df_avg[_in_tree], feature_cols,
                    raw_intensities_df[feature_cols].astype(float).values[_in_tree], 'ClusterBonsai', tag='raw')
            except Exception as e:
                print(f'⚠️ Mean spectrum failed for ClusterBonsai (raw): {e}')
        print("Re-saving the feature matrix with the Bonsai coordinates...")
        final_df_avg.to_csv("cleaned_feature_matrix.csv", index=False)


    return {"final_df_avg": final_df_avg, "feature_cols": feature_cols, "raw_intensities_df": raw_intensities_df,
            "custom_colors": custom_colors, "LEGEND_ORDER": LEGEND_ORDER, "basename": basename,
            "bonsai_segments": bonsai_segments}
