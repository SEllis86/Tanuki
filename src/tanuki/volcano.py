"""Volcano plots from a finished Tanuki run (called by the pipeline at the end of stage 5; VOLCANO_ENABLED).

Two sets:
  1. stimulation within each cell type     STIM vs NO_STIM per cell type (cell types without stim wells skipped)
  2. cell type vs cell type, pooled        MICRO vs ASTRO, MICRO vs NEURO, ASTRO vs NEURO

Statistics. Cells are never the replicates. Two units are available per preset (cfg "unit"):
  "donor" (the ipsc preset) - the biological replicate is the donor line: each well's cells are averaged
           (repeat acquisitions of one well pooled first), a donor's wells are averaged, and the two groups
           are compared PAIRED by donor (every line was differentiated into every cell type, and each line's
           stimulated and unstimulated wells share a slide). The per-donor log2 differences are tested with a
           moderated one-sample t-test (limma-style empirical Bayes: each m/z's variance is shrunk towards a
           prior fitted across all m/z, Smyth 2004, doi:10.2202/1544-6115.1027), which keeps a 3-donor design
           usable. log2 fold change = mean of the per-donor log2 ratios; the tables also give each donor's
           log2 ratio and how many donors agree in direction.
  "well"   - each well's cells are averaged and the wells of the two groups are compared with Welch's t-test
           on log2(well mean + pseudocount 1% of the feature's pooled mean).
p-values are Benjamini-Hochberg adjusted; significant = FDR < FDR_MAX and |log2FC| >= LOG2FC_MIN. A cell-level
Mann-Whitney test (also BH-adjusted) is reported in the tables for reference only - with thousands of cells it
calls nearly everything significant. Intensities are the pipeline's RAW (TIC-normalised, not batch-corrected)
per-cell means, for exactly the cells kept in its cleaned_feature_matrix.csv. An m/z is tested if it is
non-zero in >= MIN_DETECTION of the cells of either group.

Outputs in <run folder>/volcano/: one PNG + interactive HTML + CSV per comparison, an overview PNG per set and
volcano_summary.csv.

    python -m tanuki.volcano "<run folder>"
"""
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy import stats

FDR_MAX = 0.05
LOG2FC_MIN = 1.0
MIN_DETECTION = 0.20
N_LABELS = 6
NS_COLOUR, OTHER_COLOUR = "#c8c8c6", "#3d3d3a"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e6e6"

# Comparisons and label columns per pipeline. folder / raw are filled in by preset() from the run folder.
DATASETS = {
    "ipsc": dict(group_col="CellType", stim_col="Stim", stim=("STIM", "NO_STIM"),
                 stim_names=("stimulated", "unstimulated"),
                 groups=["MICRO", "ASTRO", "NEURO"],
                 names={"MICRO": "Microglia", "ASTRO": "Astrocytes", "NEURO": "Neurons"},
                 colours={"MICRO": "#2a78d6", "ASTRO": "#eb6834", "NEURO": "#1baf7a"}, exclude=None,
                 # A1_1_neuro_... and A1_2_neuro_... are two MSI acquisitions of the same well: one replicate
                 replicate_merge=r"^([AB]\d)_\d+_", names_csv="BIGmatched_peaks_raw.csv",
                 # the biological replicate is the donor line (3 per cell type), paired across groups
                 unit="donor", donor_col="Batch", donor_re=r"_(\d+)$"),
}
MZ_RE = re.compile(r"(mz_)?(\d+\.\d+)")


def load(cfg):
    """raw per-cell intensities for the cells kept in cleaned_feature_matrix.csv, + their labels"""
    keep_cols = ["sample name", "mask_id", cfg["group_col"], cfg["stim_col"]]
    if cfg.get("donor_col") and cfg["donor_col"] not in keep_cols:
        keep_cols.append(cfg["donor_col"])
    cells = pd.read_csv(cfg["folder"] / "cleaned_feature_matrix.csv", usecols=keep_cols, low_memory=False,
                        dtype={c: str for c in keep_cols[2:]})
    names = pq.read_schema(cfg["raw"]).names
    mz_cols = [c for c in names if MZ_RE.fullmatch(c)]
    raw = pq.read_table(cfg["raw"], columns=["sample name", "mask_id"] + mz_cols).to_pandas()
    df = cells.merge(raw, on=["sample name", "mask_id"], how="inner", validate="one_to_one")
    if len(df) != len(cells):
        print(f"   WARNING: {len(cells) - len(df)} cleaned cells had no raw row")
    mz = np.array([float(MZ_RE.fullmatch(c).group(2)) for c in mz_cols])
    # the unit of replication: the well. Repeat acquisitions of one well are pooled (cfg replicate_merge).
    df.insert(0, "replicate", df["sample name"])
    if cfg.get("replicate_merge"):
        df["replicate"] = df["sample name"].str.replace(cfg["replicate_merge"], r'\1_', regex=True)
    if cfg.get("donor_col"):
        df.insert(0, "donor", df[cfg["donor_col"]].str.extract(cfg.get("donor_re", r"(.*)"))[0])
        if df["donor"].isna().any():
            raise ValueError(f"donor not found in {cfg['donor_col']} for {int(df['donor'].isna().sum())} cells")
    return df, mz_cols, mz


def peak_names(cfg):
    """{m/z rounded to 4 dp: name} from the pipeline's mass list, if it has a 'name' column"""
    cands = cfg.get("names_csv") or ()
    cands = [cands] if isinstance(cands, str) else list(cands)
    found = [Path(cfg["folder"]) / f for f in cands if (Path(cfg["folder"]) / f).exists()]
    if not found:
        return {}
    t = pd.read_csv(found[0])
    cols = {c.lower(): c for c in t.columns}
    if "name" not in cols:
        return {}
    mzc = cols.get("mz", t.columns[0])
    return {round(float(m), 4): str(n) for m, n in zip(t[mzc], t[cols["name"]])}


def bh(p):
    p = np.asarray(p, float)
    out = np.full_like(p, np.nan)
    ok = np.isfinite(p)
    n = ok.sum()
    if n:
        order = np.argsort(p[ok])
        ranked = p[ok][order] * n / np.arange(1, n + 1)
        ranked = np.minimum.accumulate(ranked[::-1])[::-1]
        tmp = np.empty(n)
        tmp[order] = np.clip(ranked, 0, 1)
        out[ok] = tmp
    return out


def compare(df, mz_cols, mz, mask_a, mask_b):
    A, B = df.loc[mask_a], df.loc[mask_b]
    det = np.maximum((A[mz_cols] > 0).mean().to_numpy(), (B[mz_cols] > 0).mean().to_numpy())
    use = det >= MIN_DETECTION
    cols = [c for c, u in zip(mz_cols, use) if u]
    sa = A.groupby("replicate")[cols].mean()
    sb = B.groupby("replicate")[cols].mean()
    ma, mb = sa.mean().to_numpy(), sb.mean().to_numpy()
    pc = 0.01 * (ma + mb) / 2 + 1e-12
    log2fc = np.log2((ma + pc) / (mb + pc))
    la, lb = np.log2(sa.to_numpy() + pc), np.log2(sb.to_numpy() + pc)
    with np.errstate(all="ignore"):
        p_s = stats.ttest_ind(la, lb, axis=0, equal_var=False).pvalue
        p_c = stats.mannwhitneyu(A[cols].to_numpy(), B[cols].to_numpy(), axis=0).pvalue
    p_s = np.where(np.isfinite(p_s), p_s, 1.0)
    res = pd.DataFrame({"mz": mz[use], "feature": cols, "log2FC": log2fc, "mean_A": ma, "mean_B": mb,
                        "detected_A": (A[cols] > 0).mean().to_numpy(), "detected_B": (B[cols] > 0).mean().to_numpy(),
                        "p_sample": p_s, "FDR_sample": bh(p_s), "p_cell": p_c, "FDR_cell": bh(p_c)})
    res["n_samples_A"], res["n_samples_B"] = len(sa), len(sb)
    res["n_cells_A"], res["n_cells_B"] = len(A), len(B)
    res["significant"] = np.where((res.FDR_sample < FDR_MAX) & (res.log2FC >= LOG2FC_MIN), "A",
                                  np.where((res.FDR_sample < FDR_MAX) & (res.log2FC <= -LOG2FC_MIN), "B", "ns"))
    return res.sort_values("FDR_sample").reset_index(drop=True)


def _trigamma_inverse(x):
    """Inverse of the trigamma function (Newton iteration as in limma::trigammaInverse)."""
    from scipy.special import polygamma
    x = float(x)
    if x > 1e7:
        return 1.0 / np.sqrt(x)
    if x < 1e-6:
        return 1.0 / x
    y = 0.5 + 1.0 / x
    for _ in range(50):
        tri = float(polygamma(1, y))
        dif = tri * (1 - tri / x) / float(polygamma(2, y))
        y += dif
        if -dif / y < 1e-8:
            break
    return y


def fit_f_dist(s2, df1):
    """Empirical-Bayes prior (d0, s0^2) for a set of sample variances with df1 degrees of freedom
    (moment estimator of limma::fitFDist). d0 = inf means no extra variation between features."""
    from scipy.special import digamma, polygamma
    s2 = np.asarray(s2, float)
    s2 = s2[np.isfinite(s2) & (s2 > 0)]
    if s2.size < 3:
        return np.inf, float(np.mean(s2)) if s2.size else 1.0
    e = np.log(s2) - digamma(df1 / 2) + np.log(df1 / 2)
    evar = e.var(ddof=1) - polygamma(1, df1 / 2)
    if evar <= 0:
        return np.inf, float(np.exp(e.mean()))
    d0 = 2 * _trigamma_inverse(evar)
    return d0, float(np.exp(e.mean() + digamma(d0 / 2) - np.log(d0 / 2)))


def moderated_one_sample(D):
    """D: replicates x features (paired log2 differences). Moderated t-test of mean = 0 per feature.
    Returns mean, moderated t, p, total df, prior df."""
    D = np.asarray(D, float)
    n = D.shape[0]
    df1 = n - 1
    m = D.mean(0)
    s2 = D.var(0, ddof=1)
    d0, s02 = fit_f_dist(s2, df1)
    if np.isinf(d0):
        s2_post, df_tot = np.full_like(s2, s02), np.inf
    else:
        s2_post, df_tot = (d0 * s02 + df1 * s2) / (d0 + df1), d0 + df1
    t = m / np.sqrt(s2_post / n)
    p = 2 * (stats.norm.sf(np.abs(t)) if np.isinf(df_tot) else stats.t.sf(np.abs(t), df_tot))
    return m, t, p, df_tot, d0


def compare_donor(df, mz_cols, mz, mask_a, mask_b):
    """Paired donor-level comparison: wells -> donors, moderated t on the per-donor log2 ratios."""
    A, B = df.loc[mask_a], df.loc[mask_b]
    det = np.maximum((A[mz_cols] > 0).mean().to_numpy(), (B[mz_cols] > 0).mean().to_numpy())
    use = det >= MIN_DETECTION
    cols = [c for c, u in zip(mz_cols, use) if u]
    wa = A.groupby(["donor", "replicate"])[cols].mean()            # wells
    wb = B.groupby(["donor", "replicate"])[cols].mean()
    da, db = wa.groupby(level="donor").mean(), wb.groupby(level="donor").mean()   # donors
    donors = sorted(set(da.index) & set(db.index))
    if len(donors) < 2:
        return None
    da, db = da.loc[donors], db.loc[donors]
    ma, mb = da.mean().to_numpy(), db.mean().to_numpy()
    pc = 0.01 * (ma + mb) / 2 + 1e-12
    D = np.log2(da.to_numpy() + pc) - np.log2(db.to_numpy() + pc)
    lfc, t, p, df_tot, d0 = moderated_one_sample(D)
    p = np.where(np.isfinite(p), p, 1.0)
    with np.errstate(all="ignore"):
        p_c = stats.mannwhitneyu(A[cols].to_numpy(), B[cols].to_numpy(), axis=0).pvalue
    res = pd.DataFrame({"mz": mz[use], "feature": cols, "log2FC": lfc, "t_moderated": t,
                        "mean_A": ma, "mean_B": mb,
                        "detected_A": (A[cols] > 0).mean().to_numpy(), "detected_B": (B[cols] > 0).mean().to_numpy(),
                        "p_sample": p, "FDR_sample": bh(p), "p_cell": p_c, "FDR_cell": bh(p_c)})
    for i, d in enumerate(donors):
        res[f"log2FC_donor_{d}"] = D[i]
    res["donors_same_direction"] = (np.sign(D) == np.sign(lfc)).sum(0)
    res["unit"] = "donor"
    res["n_samples_A"] = res["n_samples_B"] = len(donors)
    res["n_wells_A"], res["n_wells_B"] = len(wa), len(wb)
    res["n_cells_A"], res["n_cells_B"] = len(A), len(B)
    res["df_moderated"], res["df_prior"] = df_tot, d0
    res["significant"] = np.where((res.FDR_sample < FDR_MAX) & (res.log2FC >= LOG2FC_MIN), "A",
                                  np.where((res.FDR_sample < FDR_MAX) & (res.log2FC <= -LOG2FC_MIN), "B", "ns"))
    return res.sort_values("FDR_sample").reset_index(drop=True)


def _unit(res):
    return res["unit"].iloc[0] if "unit" in res.columns else "well"


def draw(ax, res, name_a, name_b, col_a, col_b, title, small=False):
    y = -np.log10(np.clip(res.FDR_sample.to_numpy(), 1e-300, 1))
    x = res.log2FC.to_numpy()
    sig = res.significant.to_numpy()
    s = 9 if small else 14
    ax.scatter(x[sig == "ns"], y[sig == "ns"], s=s * 0.6, color=NS_COLOUR, linewidths=0, zorder=1)
    for lab, c in (("A", col_a), ("B", col_b)):
        ax.scatter(x[sig == lab], y[sig == lab], s=s, color=c, edgecolor="white", linewidth=0.4, zorder=3)
    ax.axhline(-np.log10(FDR_MAX), color="#888888", lw=0.8, ls=(0, (3, 2)), zorder=0)
    for v in (-LOG2FC_MIN, LOG2FC_MIN):
        ax.axvline(v, color="#888888", lw=0.8, ls=(0, (3, 2)), zorder=0)
    lim = max(2.0, np.nanpercentile(np.abs(x), 99.5) * 1.05)
    ytop = max(2.0, np.nanmax(y) * 1.38)        # headroom: the legend sits above the highest point
    ax.set_xlim(-lim * 1.5, lim * 1.5)          # outer 1/3 on each side holds the label columns
    ax.set_ylim(0, ytop)
    fs = 5.5 if small else 6.5
    renderer = ax.figure.canvas.get_renderer()
    for lab, side in (("A", 1), ("B", -1)):
        top = res[res.significant == lab].head(N_LABELS)
        top = top.assign(_y=-np.log10(np.clip(top.FDR_sample, 1e-300, 1))).sort_values("_y", ascending=False)
        for k, (_, r) in enumerate(top.iterrows()):
            ty = ytop * (0.70 - 0.07 * k)
            # both label columns sit INSIDE the outer third, anchored at the panel edge, so they never
            # run into the tick labels; the leader line ends at the measured inner edge of the text
            tx = side * lim * 1.47
            t = ax.text(tx, ty, f"{r.mz:.3f}", fontsize=fs, color=INK2, va="center",
                        ha="right" if side > 0 else "left")
            bb = t.get_window_extent(renderer).transformed(ax.transData.inverted())
            inner = bb.x0 if side > 0 else bb.x1
            ax.plot([r.log2FC, inner - side * lim * 0.02], [r._y, ty], color="#b5b5b2", lw=0.5, zorder=2)
    ax.grid(color=GRID, lw=0.5)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlabel(f"log$_2$ fold change ({name_a} / {name_b})", fontsize=7 if small else 8, color=INK2)
    unit = _unit(res)
    ax.set_ylabel(f"−log$_{{10}}$ FDR ({unit} level)", fontsize=7 if small else 8, color=INK2)
    n_a, n_b = int((sig == "A").sum()), int((sig == "B").sum())
    ax.set_title(title, fontsize=8 if small else 9, loc="left", color=INK, pad=13)
    if unit == "donor":
        sub = f"{int(res.n_samples_A.iloc[0])} donors, paired"
        if not small:
            sub += (f" ({int(res.n_wells_A.iloc[0])} vs {int(res.n_wells_B.iloc[0])} wells, "
                    f"{int(res.n_cells_A.iloc[0]):,} vs {int(res.n_cells_B.iloc[0]):,} cells)")
    else:
        sub = f"{int(res.n_samples_A.iloc[0])} vs {int(res.n_samples_B.iloc[0])} wells"
        if not small:      # small panels: cell numbers go in the caption, the line would be clipped
            sub += f" ({int(res.n_cells_A.iloc[0]):,} vs {int(res.n_cells_B.iloc[0]):,} cells)"
    sub += f"; {len(res)} m/z tested"
    ax.text(0.0, 1.012, sub, transform=ax.transAxes, ha="left", va="bottom", fontsize=5.8 if small else 6.5,
            color=INK2)
    handles = [plt.Line2D([], [], marker="o", ls="", color=col_a, markersize=4.5,
                          label=f"Higher in {name_a} ({n_a})"),
               plt.Line2D([], [], marker="o", ls="", color=col_b, markersize=4.5,
                          label=f"Higher in {name_b} ({n_b})"),
               plt.Line2D([], [], marker="o", ls="", color=NS_COLOUR, markersize=4, label="Not significant")]
    ax.legend(handles=handles, fontsize=5.6 if small else 6.5, frameon=True, framealpha=0.9, edgecolor="none",
              loc="upper center", handletextpad=0.2, borderaxespad=0.3)


def html(res, name_a, name_b, col_a, col_b, title, path):
    import plotly.graph_objects as go
    fig = go.Figure()
    y = -np.log10(np.clip(res.FDR_sample, 1e-300, 1))
    for lab, colr, nm in (("ns", NS_COLOUR, "Not significant"), ("A", col_a, f"Higher in {name_a}"),
                          ("B", col_b, f"Higher in {name_b}")):
        r = res[res.significant == lab]
        fig.add_trace(go.Scattergl(
            x=r.log2FC, y=y[r.index], mode="markers", name=f"{nm} ({len(r)})",
            marker=dict(color=colr, size=6 if lab != "ns" else 5, line=dict(width=0.5, color="white")),
            customdata=np.c_[r.mz, r.FDR_sample, r.mean_A, r.mean_B, r.detected_A, r.detected_B, r.FDR_cell],
            hovertemplate=("m/z %{customdata[0]:.4f}<br>log2FC %{x:.2f}<br>" + f"FDR ({_unit(res)} level) "
                           + "%{customdata[1]:.2e}"
                           f"<br>mean {name_a} " "%{customdata[2]:.3e}" f"<br>mean {name_b} " "%{customdata[3]:.3e}"
                           f"<br>detected {name_a} " "%{customdata[4]:.0%}" f" / {name_b} " "%{customdata[5]:.0%}"
                           "<br>FDR (cell level) %{customdata[6]:.2e}<extra></extra>")))
    for v in (-LOG2FC_MIN, LOG2FC_MIN):
        fig.add_vline(x=v, line=dict(color="#888888", dash="dash", width=1))
    fig.add_hline(y=-np.log10(FDR_MAX), line=dict(color="#888888", dash="dash", width=1))
    fig.update_layout(title=title, template="plotly_white", width=900, height=700,
                      xaxis_title=f"log2 fold change ({name_a} / {name_b})",
                      yaxis_title=f"-log10 FDR ({_unit(res)} level)")
    fig.write_html(str(path), include_plotlyjs="cdn")


def preset(key, folder, script=None):
    """The DATASETS entry for key, pointed at a run folder (a pipeline's MASTER_DIR); the raw table is
    <folder>/<folder name>_concatenated_mask_id_averages.parquet."""
    cfg = dict(DATASETS[key])
    folder = Path(folder)
    cfg["folder"] = folder
    cfg["raw"] = folder / f"{folder.name}_concatenated_mask_id_averages.parquet"
    if script:
        cfg["script"] = script
    return cfg


def run(key, folder, script=None):
    cfg = preset(key, folder, script)
    out = cfg["folder"] / "volcano"
    out.mkdir(exist_ok=True)
    print(f"\n=== {key}: {cfg['folder']}")
    df, mz_cols, mz = load(cfg)
    names = peak_names(cfg)
    if cfg["exclude"]:
        df = df[df[cfg["exclude"][0]] != cfg["exclude"][1]]
    g, s = cfg["group_col"], cfg["stim_col"]
    print(f"   {len(df):,} cells, {len(mz_cols)} m/z; groups {df[g].value_counts().to_dict()}; "
          f"{s} {df[s].value_counts().to_dict()}")
    plt.rcParams.update({"font.family": "Arial", "font.size": 7.5, "axes.edgecolor": "#888888",
                         "axes.linewidth": 0.8, "xtick.color": "#444444", "ytick.color": "#444444"})
    sets = {"1_stim_within_group": [], "2_group_vs_group": []}
    st_a, st_b = cfg["stim"]
    sn_a, sn_b = cfg["stim_names"]
    donor_unit = cfg.get("unit") == "donor"
    cmp = compare_donor if donor_unit else compare
    rep = "donor" if donor_unit else "replicate"
    for grp in cfg["groups"]:
        a = (df[g] == grp) & (df[s] == st_a)
        b = (df[g] == grp) & (df[s] == st_b)
        paired = set(df.loc[a, rep]) & set(df.loc[b, rep]) if donor_unit else None
        if (donor_unit and len(paired) < 2) or (not donor_unit and (
                df.loc[a, rep].nunique() < 2 or df.loc[b, rep].nunique() < 2)):
            print(f"   skip {grp} {st_a} vs {st_b}: fewer than 2 {'paired donors' if donor_unit else 'wells'} "
                  f"in a group")
            continue
        nm = cfg["names"][grp]
        sets["1_stim_within_group"].append((f"{grp}_{st_a}_vs_{st_b}", f"{nm}: {sn_a} vs {sn_b}",
                                            sn_a, sn_b, cfg["colours"][grp], OTHER_COLOUR, a, b))
    groups = cfg["groups"]
    for i in range(len(groups)):
        for j in range(i + 1, len(groups)):
            ga, gb = groups[i], groups[j]
            sets["2_group_vs_group"].append((f"{ga}_vs_{gb}", f"{cfg['names'][ga]} vs {cfg['names'][gb]}",
                                             cfg["names"][ga], cfg["names"][gb], cfg["colours"][ga],
                                             cfg["colours"][gb], df[g] == ga, df[g] == gb))
    summary = []
    for set_name, comps in sets.items():
        if not comps:
            continue
        results = []
        for tag, title, na, nb, ca, cb, a, b in comps:
            res = cmp(df, mz_cols, mz, a, b)
            if res is None:
                print(f"   skip {title}: fewer than 2 paired donors")
                continue
            if names:
                res.insert(1, "name", res["mz"].round(4).map(names))
            base = out / f"volcano_{set_name}_{tag}"
            res.to_csv(base.with_suffix(".csv"), index=False)
            fig, ax = plt.subplots(figsize=(4.2, 3.8), dpi=300)
            draw(ax, res, na, nb, ca, cb, title)
            fig.tight_layout()
            fig.savefig(base.with_suffix(".png"), facecolor="white")
            plt.close(fig)
            html(res, na, nb, ca, cb, title, base.with_suffix(".html"))
            results.append((title, na, nb, ca, cb, res))
            summary.append({"set": set_name, "comparison": title, "unit": _unit(res),
                            "n A": int(res.n_samples_A.iloc[0]), "n B": int(res.n_samples_B.iloc[0]),
                            "m/z tested": len(res),
                            f"up in A": int((res.significant == "A").sum()), "up in B": int((res.significant == "B").sum()),
                            "cell-level FDR<0.05 & |log2FC|>=1": int(((res.FDR_cell < FDR_MAX) & (res.log2FC.abs() >= LOG2FC_MIN)).sum())})
        n = len(results)
        if not n:
            continue
        fig, axes = plt.subplots(1, n, figsize=(3.3 * n, 3.3), dpi=300, squeeze=False)
        for ax, (title, na, nb, ca, cb, res) in zip(axes[0], results):
            draw(ax, res, na, nb, ca, cb, title, small=True)
        fig.tight_layout()
        fig.savefig(out / f"volcano_{set_name}_overview.png", facecolor="white")
        plt.close(fig)
    S = pd.DataFrame(summary)
    S.to_csv(out / "volcano_summary.csv", index=False)
    pd.set_option("display.width", 220)
    print(S.to_string(index=False))
    print(f"   -> {out}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(prog="python -m tanuki.volcano",
                                 description="Volcano plots from a finished Tanuki run (writes <run>/volcano/).")
    ap.add_argument("folder", nargs="+", help="run folder(s): the pipeline's MASTER_DIR")
    ap.add_argument("--preset", default="ipsc", choices=sorted(DATASETS))
    a = ap.parse_args()
    for f in a.folder:
        run(a.preset, f)
