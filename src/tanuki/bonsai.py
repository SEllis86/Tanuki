"""Bonsai (github.com/dhdegroot/Bonsai-data-representation): input export, optional auto-run, tree layout
and plots. v5 section "AUTO-RUN PIPELINE"."""
from __future__ import annotations

import os
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.cluster import KMeans

from .config import (BONSAI_AUTO_RUN, BONSAI_ENABLED, BONSAI_MAX_CELLS, BONSAI_OUTPUT, BONSAI_PYTHON,
    BONSAI_REPO, BONSAI_RESULTS_FOLDER, BONSAI_SAMPLE_SEED, BONSAI_SAMPLE_STRATIFY,
    BONSAI_STRATIFY_MIN_PER_GROUP, BONSAI_STRATIFY_MODE, BONSAI_TIMEOUT_S, BONSAI_USE_KNN, BONSAI_VERBOSE,
    BONSAI_ZSCORE_CUTOFF, FEATURE_LABEL, MASTER_DIR, N_CLUSTERS, RANDOM_SEED)
from .utils import group_color_map, group_sort_key


# ============================================================
# BONSAI
# ============================================================

def bonsai_cell_id(sample_name, mask_id):
    """Newick-safe unique id per cell - no ':', ';', ',', '(' or ')'."""
    safe = re.sub(r'[^A-Za-z0-9_.-]', '_', str(sample_name))
    return f"{safe}__{int(mask_id)}"


def bonsai_paths():
    results = Path(BONSAI_RESULTS_FOLDER) if BONSAI_RESULTS_FOLDER else (MASTER_DIR / 'bonsai' / 'results')
    return MASTER_DIR / 'bonsai' / 'data', results


def stratified_sample(df, group_col, total, mode='equal', min_per_group=0, seed=42):
    """
    Draw `total` rows spread across the levels of `group_col`.

    Quotas are set, clamped to what each group actually has, then reconciled with the
    budget: any shortfall (because some groups are smaller than their quota) is handed
    back round-robin to groups that still have spare cells, so an 'equal' draw is not
    silently throttled down to smallest_group * n_groups. Any excess is trimmed from
    the largest quotas first, never below min_per_group.

    Returns (sampled_df, report_df).
    """
    sizes = df[group_col].astype(str).value_counts()
    names = list(sizes.index)
    k = len(names)
    if k == 0:
        return df, pd.DataFrame()

    total = int(min(total, len(df)))

    if mode == 'equal':
        base = total // k
        quota = {g: base for g in names}
    else:
        quota = {g: int(round(total * sizes[g] / len(df))) for g in names}

    # every group gets at least the floor, if it has that many
    for g in names:
        quota[g] = int(max(quota[g], min(min_per_group, int(sizes[g]))))
        quota[g] = int(min(quota[g], int(sizes[g])))

    # hand back unused budget to groups that still have cells
    spare = lambda g: int(sizes[g]) - quota[g]
    guard = 0
    while sum(quota.values()) < total and any(spare(g) > 0 for g in names):
        for g in sorted(names, key=lambda x: -spare(x)):
            if sum(quota.values()) >= total:
                break
            if spare(g) > 0:
                quota[g] += 1
        guard += 1
        if guard > total + k:
            break

    # trim any overshoot from the biggest quotas, never below the floor
    guard = 0
    while sum(quota.values()) > total:
        order = sorted(names, key=lambda x: -quota[x])
        trimmed = False
        for g in order:
            if sum(quota.values()) <= total:
                break
            if quota[g] > min(min_per_group, int(sizes[g])) and quota[g] > 0:
                quota[g] -= 1
                trimmed = True
        if not trimmed:
            break
        guard += 1
        if guard > total + k:
            break

    parts = []
    for g in names:
        n = quota[g]
        if n <= 0:
            continue
        block = df[df[group_col].astype(str) == g]
        parts.append(block.sample(n=min(n, len(block)), random_state=seed))

    sampled = pd.concat(parts).sort_index() if parts else df.iloc[:0]

    report = pd.DataFrame({
        'group': names,
        'available': [int(sizes[g]) for g in names],
        'taken': [quota[g] for g in names],
    })
    report['% of group'] = (100 * report['taken'] / report['available']).round(1)
    report = report.sort_values('available', ascending=False)
    return sampled, report


def export_bonsai_input(df, feature_cols, sem_lookup=None, raw_values=None):
    """
    Write a complete Bonsai input folder + config.yaml.

    features.txt            log1p intensities, FEATURES x CELLS, tab separated, no header
    standard_deviations.txt the matching measurement noise on the same log scale
    cellID.txt / geneID.txt one id per row
    bonsai_annotation.csv   metadata for Bonsai-scout / our own colouring

    Bonsai models features as Gaussian, so intensities go in as log1p and the standard
    error on the cell mean is carried through the same transform (delta method:
    sd(log1p(x)) ~ sem(x) / (1 + x)). Where no per-cell spread is available the noise
    falls back to 1/sqrt(pixel_count), i.e. a mean over n pixels is sqrt(n) times
    more certain than one pixel.
    """
    data_folder, results_folder = bonsai_paths()
    data_folder.mkdir(parents=True, exist_ok=True)
    results_folder.mkdir(parents=True, exist_ok=True)

    export_df = df
    if BONSAI_MAX_CELLS and len(df) > int(BONSAI_MAX_CELLS):
        strat_col = BONSAI_SAMPLE_STRATIFY
        if strat_col and strat_col not in df.columns:
            print(f"🌳 Bonsai: '{strat_col}' is not a column; falling back to uniform random.")
            strat_col = None

        if strat_col:
            export_df, report = stratified_sample(
                df, strat_col, int(BONSAI_MAX_CELLS),
                mode=BONSAI_STRATIFY_MODE,
                min_per_group=int(BONSAI_STRATIFY_MIN_PER_GROUP),
                seed=int(BONSAI_SAMPLE_SEED))
            print(f"🌳 Bonsai: subsampling {len(export_df)}/{len(df)} cells, "
                  f"stratified by '{strat_col}' ({BONSAI_STRATIFY_MODE}).")
            print(report.to_string(index=False))
            if BONSAI_STRATIFY_MODE == 'equal':
                print("🌳 NOTE: equal-N per group. Branch structure is comparable between")
                print("   groups, but cluster SIZES no longer reflect true abundance.")
            try:
                report.to_csv(MASTER_DIR / 'bonsai' / 'bonsai_sampling_report.csv', index=False)
            except Exception:
                pass
        else:
            export_df = df.sample(n=int(BONSAI_MAX_CELLS),
                                  random_state=int(BONSAI_SAMPLE_SEED)).sort_index()
            print(f"🌳 Bonsai: subsampling {len(export_df)}/{len(df)} cells "
                  f"(uniform random, BONSAI_MAX_CELLS).")

    cell_ids = [bonsai_cell_id(s, m) for s, m in zip(export_df['sample name'], export_df['mask_id'])]
    if len(set(cell_ids)) != len(cell_ids):
        raise RuntimeError("Bonsai cell ids are not unique - check sample/mask naming.")

    if raw_values is not None:
        # sem_lookup was measured on the raw pixel spread, so the intensities it is
        # paired with have to be the raw ones too; Bonsai weights its likelihood by
        # that noise and a raw/corrected mix mis-states every cell's certainty
        values = np.asarray(raw_values, dtype=float)[export_df.index.values, :]
        print("🌳 Bonsai: exporting RAW (pre-ComBat) intensities to match the measured noise.")
    else:
        values = export_df[feature_cols].astype(float).values
        print("🌳 Bonsai: exporting ComBat-corrected intensities.")
    features = np.log1p(np.clip(values, 0, None))

    # measurement noise, on the log1p scale
    sds = None
    if sem_lookup is not None and not sem_lookup.empty:
        sem_aligned = sem_lookup.reindex(
            pd.MultiIndex.from_arrays([export_df['sample name'].astype(str),
                                       export_df['mask_id'].astype(int)]))
        available = [c for c in feature_cols if c in sem_aligned.columns]
        if available:
            sem_matrix = np.zeros_like(features)
            sem_vals = sem_aligned[available].astype(float).values
            col_pos = {c: i for i, c in enumerate(feature_cols)}
            for j, c in enumerate(available):
                sem_matrix[:, col_pos[c]] = sem_vals[:, j]
            sds = sem_matrix / (1.0 + np.clip(values, 0, None))
            coverage = float(np.isfinite(sds).mean())
            print(f"🌳 Bonsai: using measured per-cell standard errors "
                  f"({len(available)}/{len(feature_cols)} features, {coverage * 100:.0f}% finite).")

    if sds is None or not np.isfinite(sds).any():
        # A flat 1/sqrt(pixel_count) is an ABSOLUTE number with no relation to the
        # intensity scale. On TIC-normalised data a feature sits near 1e-4 while that
        # fallback is near 1e-1, so the noise swamps the signal by ~100x, no feature
        # clears Bonsai's z-score cutoff, and Bonsai exits with 'No gene made the
        # zscore-cutoff'. Scale the fallback to the data instead: relative error
        # 1/sqrt(n) ON the value, carried through log1p by the same delta method.
        print('🌳 Bonsai: no per-cell standard errors found - falling back to a '
              'SCALE-AWARE estimate (value/sqrt(pixel_count)).')
        print('   This is a guess, not a measurement. Re-run stage 3 so real per-cell '
              'standard errors are written, then re-run this script.')
        sds = np.zeros_like(features)

    pixel_col = next((c for c in ('pixel_count', 'area_pixels')
                      if c in export_df.columns), None)
    if pixel_col:
        n_pixels = np.clip(export_df[pixel_col].astype(float).values, 1, None)
        # proportional to the value, so the noise lives on the same scale as the data
        rel = np.clip(values, 0, None) / np.sqrt(n_pixels)[:, None]
        fallback = rel / (1.0 + np.clip(values, 0, None))
    else:
        fallback = 0.1 * np.abs(features)
    sds = np.where(np.isfinite(sds) & (sds > 0), sds, fallback)

    # v5: a zero is a non-detection, not an exactly-known zero. With no pixel spread it got
    # the 1e-6 floor below (60% of all entries), so Bonsai treated every absence as certain.
    # Give zeros at least the feature's detection limit (its smallest non-zero value, on the
    # same log1p scale) as their uncertainty.
    with np.errstate(invalid='ignore'):
        detection = np.log1p(np.nanmin(np.where(values > 0, values, np.nan), axis=0))
    detection = np.where(np.isfinite(detection), detection, np.nanmedian(sds))
    sds = np.where(values <= 0, np.maximum(sds, detection[None, :]), sds)

    # a zero-noise feature would make the likelihood blow up
    floor = np.maximum(1e-6, 0.01 * np.nanmedian(sds))
    sds = np.clip(sds, floor, None)

    # Bonsai keeps a feature only if its spread ACROSS CELLS beats the measurement
    # noise. Check that here rather than discovering it from a one-line exit code.
    with np.errstate(divide='ignore', invalid='ignore'):
        _z = features.std(axis=0) / np.where(sds.mean(axis=0) > 0, sds.mean(axis=0), np.nan)
    _n_pass = int(np.nansum(_z > float(BONSAI_ZSCORE_CUTOFF)))
    print(f'🌳 Bonsai: {_n_pass}/{len(feature_cols)} features clear the z-score cutoff '
          f'of {float(BONSAI_ZSCORE_CUTOFF)} (median z = {np.nanmedian(_z):.2f}).')
    if _n_pass == 0:
        raise RuntimeError(
            f'no feature clears BONSAI_ZSCORE_CUTOFF={float(BONSAI_ZSCORE_CUTOFF)} '
            f'(median z = {np.nanmedian(_z):.3g}). Bonsai would exit with "No gene made '
            f'the zscore-cutoff". The measurement noise is too large relative to the '
            f'signal - check that per-cell standard errors were loaded, or lower the cutoff.')

    np.savetxt(data_folder / 'features.txt', features.T, delimiter='\t', fmt='%.6g')
    np.savetxt(data_folder / 'standard_deviations.txt', sds.T, delimiter='\t', fmt='%.6g')
    (data_folder / 'cellID.txt').write_text("\n".join(cell_ids) + "\n", encoding='utf-8')
    (data_folder / 'geneID.txt').write_text(
        "\n".join(str(c) if str(c).startswith("mz_") else f"mz_{c}"
                  for c in feature_cols) + "\n", encoding='utf-8')

    annotation_cols = [c for c in ['sample name', 'mask_id', 'Individual Sample Type', 'Stim',
                                   'Sample Type + Stim', 'Batch', 'Slide', 'CellType',
                                   'Well', 'location', 'CellLine', 'ClusterUnifiedDens',
                                   'KmeansClusterUMAP', 'KmeansClusterTSNE', 'ClusterNMF',
                                   'ClusterLeiden', 'area_pixels']
                       if c in export_df.columns]
    annotation = export_df[annotation_cols].copy()
    annotation.insert(0, 'cellID', cell_ids)
    annotation.to_csv(data_folder / 'bonsai_annotation.csv', index=False)

    config_path = MASTER_DIR / 'bonsai' / 'config.yaml'
    config_lines = [
        f'dataset: "{MASTER_DIR.name}_msi"',
        f'data_folder: "{data_folder.as_posix()}"',
        f'results_folder: "{results_folder.as_posix()}"',
        'filenames_data: "features.txt,standard_deviations.txt"',
        'input_is_sanity_output: false',
        f'verbose: {str(bool(BONSAI_VERBOSE)).lower()}',
        f'zscore_cutoff: {float(BONSAI_ZSCORE_CUTOFF)}',
        'rescale_by_var: true',
        f'use_knn: {int(BONSAI_USE_KNN)}',
        'nnn_n_randommoves: 1000',
        'nnn_n_randomtrees: 10',
        'UB_ellipsoid_size: 1.0',
        'spr_strategy: "large_tree"',
        'pickup_intermediate: false',
        'tmp_folder: ""',
    ]
    config_path.write_text("\n".join(config_lines) + "\n", encoding='utf-8')

    repo_hint = Path(BONSAI_REPO).as_posix() if BONSAI_REPO else '<path to Bonsai-data-representation>'
    instructions = f"""Bonsai input written by tanuki2026IPSC
=================================================
cells    : {len(export_df)}
features : {len(feature_cols)}  (log1p intensities + standard errors)

Run Bonsai (in its own conda env, see the repo README):

    conda activate bonsai
    python3 {repo_hint}/bonsai/bonsai_main.py --config_filepath "{config_path.as_posix()}" --step all

or in parallel:

    mpiexec -n <CORES> python -m mpi4py {repo_hint}/bonsai/bonsai_main.py \\
        --config_filepath "{config_path.as_posix()}" --step all

Results land in:
    {results_folder.as_posix()}/final_bonsai_*/

Re-run this pipeline afterwards (stages 1-4 are skipped automatically) and the Bonsai
tree plots will be generated from edgeInfo.txt / vertInfo.txt.

For the interactive Bonsai-scout app, point --annotation_path at:
    {(data_folder / 'bonsai_annotation.csv').as_posix()}
"""
    (MASTER_DIR / 'bonsai' / 'RUN_BONSAI.txt').write_text(instructions, encoding='utf-8')

    print(f"🌳 Bonsai input written: {len(export_df)} cells x {len(feature_cols)} features "
          f"-> {data_folder}")
    print(f"🌳 Config: {config_path}  (see bonsai/RUN_BONSAI.txt for the command)")
    return config_path


def run_bonsai(config_path):
    """Run bonsai_main.py as a subprocess. Returns True when it exits cleanly."""
    if not BONSAI_REPO:
        print("🌳 BONSAI_REPO is not set - skipping the auto-run.")
        return False
    main_script = Path(BONSAI_REPO) / 'bonsai' / 'bonsai_main.py'
    if not main_script.exists():
        print(f"🌳 {main_script} not found - skipping the auto-run.")
        return False

    python_exe = str(BONSAI_PYTHON) if BONSAI_PYTHON else sys.executable
    cmd = [python_exe, str(main_script), '--config_filepath', str(config_path), '--step', 'all']
    print(f"🌳 Running Bonsai: {' '.join(cmd)}")
    try:
        mode = str(BONSAI_OUTPUT).lower()
        if mode == 'quiet':
            with open(os.devnull, 'w') as devnull:
                completed = subprocess.run(cmd, cwd=str(BONSAI_REPO), timeout=BONSAI_TIMEOUT_S,
                                           stdout=devnull, stderr=subprocess.STDOUT)
        elif mode == 'file':
            log_path = Path(config_path).parent / 'bonsai_run.log'
            log_path.parent.mkdir(parents=True, exist_ok=True)
            print(f"🌳 Bonsai output -> {log_path} (set BONSAI_OUTPUT='console' to see it live).")
            with open(log_path, 'w', encoding='utf-8', errors='replace') as logf:
                completed = subprocess.run(cmd, cwd=str(BONSAI_REPO), timeout=BONSAI_TIMEOUT_S,
                                           stdout=logf, stderr=subprocess.STDOUT)
        else:
            completed = subprocess.run(cmd, cwd=str(BONSAI_REPO), timeout=BONSAI_TIMEOUT_S)
        if completed.returncode != 0:
            print(f"🌳 Bonsai exited with code {completed.returncode}.")
            return False
        print("🌳 Bonsai finished.")
        return True
    except subprocess.TimeoutExpired:
        print(f"🌳 Bonsai timed out after {BONSAI_TIMEOUT_S} s.")
        return False
    except Exception as e:
        print(f"🌳 Could not run Bonsai: {e}")
        return False


def find_bonsai_result(results_folder=None):
    """Newest 'final_bonsai_*' folder that actually holds a tree."""
    if results_folder is None:
        _, results_folder = bonsai_paths()
    results_folder = Path(results_folder)
    if not results_folder.exists():
        return None
    candidates = [d for d in results_folder.rglob('final_bonsai_*')
                  if d.is_dir() and (d / 'edgeInfo.txt').exists() and (d / 'vertInfo.txt').exists()]
    if not candidates:
        return None
    newest = max(candidates, key=lambda d: (d / 'edgeInfo.txt').stat().st_mtime)
    # v5: a tree OLDER than the current export was built from other cells - a failed or
    # skipped Bonsai run used to leave the previous tree to be drawn as if it were current
    data_folder, _ = bonsai_paths()
    features = Path(data_folder) / 'features.txt'
    if features.exists() and (newest / 'edgeInfo.txt').stat().st_mtime < features.stat().st_mtime:
        print(f"🌳 Newest tree ({newest.name}) predates the current Bonsai export - not using it.")
        return None
    return newest


def _read_bonsai_table(path):
    """edgeInfo.txt / vertInfo.txt carry no header and may be tab or space separated."""
    df = pd.read_csv(path, sep=r'\s+', header=None, engine='python', comment='#')
    # tolerate a header row if a future version adds one
    if df.shape[0] and not str(df.iloc[0, 0]).lstrip('-').replace('.', '', 1).isdigit():
        df = df.iloc[1:].reset_index(drop=True)
    return df


def load_bonsai_tree(result_dir):
    """
    Read a Bonsai result into (edges, vertex_names).

    edgeInfo.txt : vertInd, vertInd, edge_length   (no header)
    vertInfo.txt : vertInd, vertName, ...          (no header; extra columns tolerated)
    """
    result_dir = Path(result_dir)
    edge_df = _read_bonsai_table(result_dir / 'edgeInfo.txt')
    if edge_df.shape[1] < 3:
        raise RuntimeError(f"edgeInfo.txt has {edge_df.shape[1]} columns, expected 3.")
    edges = [(int(float(a)), int(float(b)), float(length))
             for a, b, length in edge_df.iloc[:, :3].itertuples(index=False)]

    vert_df = _read_bonsai_table(result_dir / 'vertInfo.txt')
    vert_names = {}
    if vert_df.shape[1] >= 2:
        # first column is vertInd; vertName is the first non-numeric-looking column
        name_col = vert_df.shape[1] - 1
        for col in range(1, vert_df.shape[1]):
            sample = vert_df[col].astype(str)
            if not sample.str.match(r'^-?\d+(\.\d+)?$').all():
                name_col = col
                break
        for vert_ind, name in zip(vert_df[0], vert_df[name_col]):
            vert_names[int(float(vert_ind))] = str(name)

    print(f"🌳 Bonsai tree loaded from {result_dir.name}: "
          f"{len(edges)} edges, {len(vert_names)} named vertices.")
    return edges, vert_names


def bonsai_radial_layout(edges):
    """
    Equal-angle radial layout of an unrooted tree, edge lengths respected.

    Each subtree gets an angular wedge in proportion to how many leaves it holds, and a
    child sits at its parent's position plus edge_length along the middle of its wedge -
    the standard way phylogenies are drawn, so distances along the drawing follow the
    tree metric rather than a neighbour embedding.
    """
    adjacency = defaultdict(list)
    for a, b, length in edges:
        adjacency[a].append((b, length))
        adjacency[b].append((a, length))
    if not adjacency:
        return {}, []

    def dfs_order(start):
        seen = {start: None}
        seq, stack = [start], [start]
        while stack:
            v = stack.pop()
            for w, _ in adjacency[v]:
                if w not in seen:
                    seen[w] = v
                    seq.append(w)
                    stack.append(w)
        return seq, seen

    # Root at the tree CENTROID - the vertex whose removal leaves the smallest largest
    # component. A Bonsai tree is unrooted, and centring it this way keeps a long
    # caterpillar-shaped tree from being drawn off to one side.
    probe_order, probe_parent = dfs_order(next(iter(adjacency)))
    n_vertices = len(probe_order)
    subtree = {v: 1 for v in probe_order}
    for v in reversed(probe_order):
        p = probe_parent[v]
        if p is not None:
            subtree[p] += subtree[v]

    def largest_component(v):
        parts = [subtree[w] for w, _ in adjacency[v] if probe_parent.get(w) == v]
        parts.append(n_vertices - subtree[v])
        return max(parts)

    root = min(probe_order, key=largest_component)

    order, parent = dfs_order(root)
    children = {v: [(w, length) for w, length in adjacency[v] if parent.get(w) == v] for v in order}

    leaf_counts = {}
    for v in reversed(order):
        kids = children[v]
        leaf_counts[v] = 1 if not kids else sum(leaf_counts[w] for w, _ in kids)

    coords = {root: (0.0, 0.0)}
    wedges = {root: (0.0, 2.0 * np.pi)}
    for v in order:
        start, end = wedges[v]
        kids = children[v]
        total = sum(leaf_counts[w] for w, _ in kids) or 1
        cursor = start
        for w, length in kids:
            span = (end - start) * leaf_counts[w] / total
            wedges[w] = (cursor, cursor + span)
            mid = cursor + span / 2.0
            x, y = coords[v]
            coords[w] = (x + length * float(np.cos(mid)), y + length * float(np.sin(mid)))
            cursor += span

    segments = [(coords[a][0], coords[a][1], coords[b][0], coords[b][1])
                for a, b, _ in edges if a in coords and b in coords]
    return coords, segments


def attach_bonsai_coordinates(df, result_dir):
    """Add Bonsai1/Bonsai2/ClusterBonsai to df. Returns (df, edge_segments) or (df, None)."""
    try:
        edges, vert_names = load_bonsai_tree(result_dir)
        coords, segments = bonsai_radial_layout(edges)
    except Exception as e:
        print(f"🌳 Could not build the Bonsai layout: {e}")
        return df, None

    if not coords:
        print("🌳 Bonsai tree is empty - skipping the tree plots.")
        return df, None

    name_to_xy = {vert_names[v]: coords[v] for v in coords if v in vert_names}
    cell_ids = [bonsai_cell_id(s, m) for s, m in zip(df['sample name'], df['mask_id'])]
    xs = np.full(len(df), np.nan)
    ys = np.full(len(df), np.nan)
    for i, cid in enumerate(cell_ids):
        xy = name_to_xy.get(cid)
        if xy is not None:
            xs[i], ys[i] = xy

    matched = int(np.isfinite(xs).sum())
    print(f"🌳 Bonsai: matched {matched}/{len(df)} cells onto tree leaves.")
    if matched == 0:
        print("🌳 No cell ids matched vertInfo.txt - was this tree built from a different export?")
        return df, None
    # v5: the tree must be the one built from THIS export, not just share some cell ids
    try:
        exported = set((bonsai_paths()[0] / 'cellID.txt').read_text(encoding='utf-8').split())
        if exported and len(exported & set(name_to_xy)) < 0.95 * len(exported):
            print(f"🌳 Tree leaves cover only {len(exported & set(name_to_xy))}/{len(exported)} of the "
                  f"exported cells - built from a different export; not using it.")
            return df, None
    except OSError:
        pass

    df["Bonsai1"], df["Bonsai2"] = xs, ys
    placed = np.isfinite(xs) & np.isfinite(ys)
    n_clusters = int(max(2, min(N_CLUSTERS, placed.sum())))
    labels = np.full(len(df), -1)
    labels[placed] = KMeans(n_clusters=n_clusters, n_init=10, random_state=RANDOM_SEED).fit_predict(
        np.column_stack([xs[placed], ys[placed]]))
    df["ClusterBonsai"] = labels
    return df, segments


def create_bonsai_plot(df, segments, group_col, title, color_map=None, order_list=None,
                       feature_type="bonsai_tree", marker_size=6, font_size=20):
    """The tree skeleton with the cells drawn on their own leaves."""
    df_plot = df[np.isfinite(df["Bonsai1"]) & np.isfinite(df["Bonsai2"])].copy()
    if df_plot.empty:
        print(f"🌳 Nothing to plot for '{title}'.")
        return

    fig = go.Figure()
    if segments:
        edge_x, edge_y = [], []
        for x0, y0, x1, y1 in segments:
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
        fig.add_trace(go.Scatter(x=edge_x, y=edge_y, mode='lines',
                                 line=dict(color='rgba(120,120,120,0.45)', width=1),
                                 hoverinfo='skip', name='tree', showlegend=False))

    # v5: cluster IDs are labels, never a gradient (ClusterLeiden easily has > 12 levels)
    numeric = (pd.api.types.is_numeric_dtype(df_plot[group_col]) and df_plot[group_col].nunique() > 12
               and 'Cluster' not in str(group_col))
    if numeric:
        fig.add_trace(go.Scatter(
            x=df_plot["Bonsai1"], y=df_plot["Bonsai2"], mode='markers',
            marker=dict(size=marker_size, color=df_plot[group_col].astype(float),
                        colorscale='Viridis', showscale=True,
                        colorbar=dict(title=str(group_col))),
            text=df_plot['sample name'] if 'sample name' in df_plot.columns else None,
            name=str(group_col)))
    else:
        groups = df_plot[group_col].astype(str)
        if order_list is None:
            order_list = sorted(groups.unique(), key=group_sort_key)
        color_map = group_color_map(order_list, color_map)
        for group in order_list:
            subset = df_plot[groups == group]
            if subset.empty:
                continue
            fig.add_trace(go.Scatter(
                x=subset["Bonsai1"], y=subset["Bonsai2"], mode='markers',
                marker=dict(size=marker_size,
                            color=color_map.get(group) if color_map else None),
                text=subset['sample name'] if 'sample name' in subset.columns else None,
                name=str(group)))

    fig.update_layout(title=f"{title} (N={len(df_plot)})", template="plotly_white",
                      font=dict(size=font_size), height=900, width=1100,
                      xaxis=dict(visible=False), yaxis=dict(visible=False, scaleanchor="x"),
                      legend=dict(itemsizing='constant'))

    base = f"{title.lower().replace(' ', '_').replace('/', '_')}_{feature_type}"
    fig.write_html(f"{base}.html", include_plotlyjs='cdn')
    try:
        fig.write_image(f"{base}.png", scale=2)
    except Exception as e:
        print(f"⚠️ PNG export skipped for '{title}' ({e}) - HTML still written.")


def create_projection_comparison_html(df, segments, group_col, color_map=None, order_list=None,
                                      filename="projection_comparison_with_bonsai.html"):
    """UMAP / t-SNE / DensMAP / Bonsai side by side, coloured the same way."""
    panels = [("UMAP1", "UMAP2", "UMAP (mass)"),
              ("TSNE1", "TSNE2", "t-SNE (mass)"),
              ("DensMAP_Corrected1", "DensMAP_Corrected2", f"DensMAP corrected ({FEATURE_LABEL})"),
              ("Bonsai1", "Bonsai2", "Bonsai tree (mass)")]
    panels = [p for p in panels if p[0] in df.columns and p[1] in df.columns]
    if len(panels) < 2:
        return

    rows = int(np.ceil(len(panels) / 2))
    fig = make_subplots(rows=rows, cols=2, subplot_titles=[p[2] for p in panels])

    groups = df[group_col].astype(str)
    if order_list is None:
        order_list = sorted(groups.unique(), key=group_sort_key)
    color_map = group_color_map(order_list, color_map)   # v5: same colour in every panel

    for idx, (x_col, y_col, _) in enumerate(panels):
        row, col = idx // 2 + 1, idx % 2 + 1
        if x_col == "Bonsai1" and segments:
            edge_x, edge_y = [], []
            for x0, y0, x1, y1 in segments:
                edge_x.extend([x0, x1, None])
                edge_y.extend([y0, y1, None])
            fig.add_trace(go.Scatter(x=edge_x, y=edge_y, mode='lines',
                                     line=dict(color='rgba(120,120,120,0.35)', width=1),
                                     hoverinfo='skip', showlegend=False), row=row, col=col)
        for group in order_list:
            subset = df[groups == group]
            if subset.empty or subset[x_col].isna().all():
                continue
            fig.add_trace(go.Scatter(
                x=subset[x_col], y=subset[y_col], mode='markers',
                marker=dict(size=4, color=color_map.get(group) if color_map else None),
                name=str(group), legendgroup=str(group), showlegend=(idx == 0),
                hovertext=subset['sample name'] if 'sample name' in subset.columns else None),
                row=row, col=col)

    fig.update_layout(title_text=f"Projection comparison, coloured by {group_col}",
                      template="plotly_white", height=520 * rows, width=1500,
                      legend=dict(itemsizing='constant'))
    fig.write_html(filename, include_plotlyjs='cdn')
    print(f"🌳 Wrote {filename}")


def run_bonsai_stage(df, feature_cols, sem_lookup, color_map, legend_order,
                     raw_values=None):
    """
    Export -> (optionally) run -> load -> lay out -> plot. Never fatal: without a Bonsai
    result the rest of the analysis is unaffected.
    """
    if not BONSAI_ENABLED:
        return df, None

    print("\n🌳 ===== BONSAI =====")
    segments = None
    try:
        config_path = export_bonsai_input(df, feature_cols, sem_lookup, raw_values)
    except Exception as e:
        print(f"🌳 Bonsai export failed: {e}")
        return df, None

    if BONSAI_AUTO_RUN and not run_bonsai(config_path):
        # v5: do not fall back to whatever older tree is on disk
        print("🌳 Bonsai did not finish - no tree plots this run.")
        return df, None

    result_dir = find_bonsai_result()
    if result_dir is None:
        _, results_folder = bonsai_paths()
        print(f"🌳 No 'final_bonsai_*' result under {results_folder} yet - "
              f"run Bonsai (see bonsai/RUN_BONSAI.txt) and re-run this script to get the tree plots.")
        return df, None

    df, segments = attach_bonsai_coordinates(df, result_dir)
    if segments is None:
        return df, None

    create_bonsai_plot(df, segments, "Individual Sample Type", "Bonsai Individual Samples",
                       color_map=color_map, order_list=legend_order)
    create_bonsai_plot(df, segments, "ClusterBonsai", "Bonsai Tree Clusters")
    if "ClusterUnifiedDens" in df.columns:
        create_bonsai_plot(df, segments, "ClusterUnifiedDens", "Bonsai colored by DensMAP clusters")
    if "ClusterLeiden" in df.columns:
        create_bonsai_plot(df, segments, "ClusterLeiden", "Bonsai colored by Leiden clusters")
    if "Batch" in df.columns:
        create_bonsai_plot(df, segments, "Batch", "Bonsai colored by Batch")
    if "location" in df.columns:
        create_bonsai_plot(df, segments, "location", "Bonsai colored by edge or middle")
    create_projection_comparison_html(df, segments, "Individual Sample Type",
                                      color_map=color_map, order_list=legend_order)
    return df, segments
