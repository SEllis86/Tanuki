"""Analysis-friendly CSV export of the RAW per-cell data from a finished Tanuki run (called by the pipeline
at the end of stage 5; RAW_EXPORT_ENABLED).

Writes <run folder>/export/:
  cells_raw.csv        one row per measured cell (every cell stage 3 wrote out, not only those that passed
                       the analysis filters): ids, sample labels, morphology, passed_pipeline_filters,
                       the pipeline's clusters / embedding coordinates (filtered cells: blank), then one
                       column per m/z = raw TIC-normalised mean intensity of the cell (NOT batch-corrected)
  cells_raw.parquet    the same table, much faster to load in Python / R (pandas, arrow)
  samples_raw.csv      one row per sample (well): labels, cell counts, mean of its cells that passed the filters
  features.csv         one row per m/z: m/z value, used in the pipeline analysis?, detection rate and mean
                       intensity overall and per group
  README.txt           what every column means

    python -m tanuki.export "<run folder>"
"""
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

# per-sample labels (constant within a sample) carried from cleaned_feature_matrix.csv: source -> export name.
# folder / raw are filled in by preset() from the run folder.
DATASETS = {
    "ipsc": dict(script="tanuki",
                 labels={"CellType": "cell_type", "Batch": "cell_type_line", "Stim": "stim", "Slide": "slide",
                         "Well": "well_position", "Individual Sample Type": "sample_type",
                         "Sample Type + Stim": "sample_type_stim"},
                 group="cell_type", um2_per_px=1.30 ** 2),
}
MORPH = ["location", "area_pixels", "area_native_pixels", "area_um2", "centroid_row", "centroid_col", "perimeter",
         "extent", "eccentricity", "major_axis_length", "minor_axis_length", "solidity", "branch_points",
         "number_trunks", "number_non_trunk_branches", "number_branch_ends", "total_object_skeleton_length",
         "mask_source"]
ANALYSIS = ["ClusterLeiden", "ClusterUnifiedDens", "KmeansClusterUMAP", "KmeansClusterTSNE", "KmeansClusterPCA",
            "ClusterNMF", "NMF_dominant", "ClusterMorphOnly", "DensMAP_Uncorrected1", "DensMAP_Uncorrected2",
            "DensMAP_UncorrMassOnly1", "DensMAP_UncorrMassOnly2", "DensMAP_MorphOnly1", "DensMAP_MorphOnly2",
            "DensMAP_Corrected1", "DensMAP_Corrected2", "UMAP1", "UMAP2", "TSNE1", "TSNE2", "PCA1", "PCA2", "PCA3",
            "ClusterBonsai", "Bonsai1", "Bonsai2"]
MZ_RE = re.compile(r"(mz_)?(\d+\.\d+)")


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


def export(key, folder, script=None):
    cfg = preset(key, folder, script)
    t0 = time.time()
    folder, out = cfg["folder"], cfg["folder"] / "export"
    out.mkdir(exist_ok=True)
    print(f"\n=== {key}: {folder}")
    clean_cols = pd.read_csv(folder / "cleaned_feature_matrix.csv", nrows=0).columns.tolist()
    clean_mz = {MZ_RE.fullmatch(c).group(2) for c in clean_cols if MZ_RE.fullmatch(c)}
    lab_src = [c for c in cfg["labels"] if c in clean_cols]
    ana = [c for c in ANALYSIS if c in clean_cols]
    clean = pd.read_csv(folder / "cleaned_feature_matrix.csv", usecols=["sample name", "mask_id"] + lab_src + ana,
                        low_memory=False, dtype={c: str for c in lab_src})

    names = pq.read_schema(cfg["raw"]).names
    mz_src = [c for c in names if MZ_RE.fullmatch(c)]
    mz_vals = np.array([float(MZ_RE.fullmatch(c).group(2)) for c in mz_src])
    order = np.argsort(mz_vals)
    mz_src, mz_vals = [mz_src[i] for i in order], mz_vals[order]
    mz_out = [f"mz_{v:.4f}" for v in mz_vals]
    morph = [c for c in MORPH if c in names]
    raw = pq.read_table(cfg["raw"], columns=["sample name", "mask_id"] + morph + mz_src).to_pandas()
    raw = raw.rename(columns=dict(zip(mz_src, mz_out)))

    # labels are per sample: take them from the analysed cells of that sample
    lab = clean.groupby("sample name")[lab_src].agg(lambda s: s.mode().iat[0] if s.notna().any() else np.nan)
    lab = lab.rename(columns=cfg["labels"])
    missing = sorted(set(raw["sample name"]) - set(lab.index))
    if missing:
        print(f"   {len(missing)} sample(s) have no analysed cells, labels left blank: {missing[:5]}")
    cells = raw.merge(lab, left_on="sample name", right_index=True, how="left")
    keep = clean[["sample name", "mask_id"] + ana].assign(passed_pipeline_filters=True)
    cells = cells.merge(keep, on=["sample name", "mask_id"], how="left", validate="one_to_one")
    cells["passed_pipeline_filters"] = cells["passed_pipeline_filters"].notna() & cells["passed_pipeline_filters"].astype("boolean").fillna(False).astype(bool)
    if "cell_type_line" in cells:
        cells.insert(cells.columns.get_loc("cell_type_line") + 1, "cell_line",
                     cells["cell_type_line"].str.split("_").str[-1])
    if cfg["um2_per_px"] and "area_pixels" in cells and "area_um2" not in cells:
        cells.insert(cells.columns.get_loc("area_pixels") + 1, "area_um2", cells["area_pixels"] * cfg["um2_per_px"])
    cells.insert(0, "cell_id", cells["sample name"].astype(str) + "__" + cells["mask_id"].astype(str))
    cells = cells.rename(columns={"sample name": "sample"})
    front = ["cell_id", "sample", "mask_id"] + [c for c in lab.columns.tolist() + ["cell_line"] if c in cells] + \
            ["passed_pipeline_filters"] + [c for c in morph + ["area_um2"] if c in cells and c not in ("sample",)]
    front = list(dict.fromkeys(front))
    cols = front + [c for c in ana if c in cells] + mz_out
    cells = cells[cols]
    cells.to_csv(out / "cells_raw.csv", index=False, float_format="%.6g")
    cells.to_parquet(out / "cells_raw.parquet", index=False)

    passed = cells[cells.passed_pipeline_filters]
    lab_cols = [c for c in lab.columns.tolist() + ["cell_line"] if c in cells]
    smp = cells.groupby("sample")[lab_cols].first()
    smp.insert(0, "n_cells_measured", cells.groupby("sample").size())
    smp.insert(1, "n_cells_passed_filters", passed.groupby("sample").size().reindex(smp.index).fillna(0).astype(int))
    smp = smp.join(passed.groupby("sample")[mz_out].mean())
    smp.reset_index().to_csv(out / "samples_raw.csv", index=False, float_format="%.6g")

    X = passed[mz_out]
    feat = pd.DataFrame({"column": mz_out, "mz": mz_vals,
                         "used_in_pipeline_analysis": [f"{v:.4f}" in {f"{float(m):.4f}" for m in clean_mz} for v in mz_vals],
                         "detected_fraction_all": (X > 0).mean().to_numpy(), "mean_all": X.mean().to_numpy()})
    grp = cfg["group"]
    if grp in passed:
        for g, sub in passed.groupby(grp):
            feat[f"detected_fraction_{g}"] = (sub[mz_out] > 0).mean().to_numpy()
            feat[f"mean_{g}"] = sub[mz_out].mean().to_numpy()
    feat.to_csv(out / "features.csv", index=False, float_format="%.6g")

    readme = f"""Tanuki raw data export - {key}  ({time.strftime('%Y-%m-%d %H:%M')})
Source run: {folder}
Pipeline script: {cfg['script']}

cells_raw.csv / cells_raw.parquet   {len(cells):,} cells x {len(mz_out):,} m/z ({int(cells.passed_pipeline_filters.sum()):,} passed the analysis filters)
  cell_id                   <sample>__<mask_id>, unique
  sample                    folder / region name of the well (one MSI acquisition); use this, not the well
                            position, as the sample identifier
  mask_id                   segmentation mask number within the sample (matches PREMSI_seg.npy / the Dash viewer)
  {', '.join(lab.columns)}  labels of the sample, as used by the pipeline
  passed_pipeline_filters   True = cell is in cleaned_feature_matrix.csv (per-cell-type area bounds, empty-row
                            cleaning); False = measured but filtered out of the analysis
  location ... mask_source  morphology from stage 3 (pixels are segmentation pixels{'; area_um2 = area_pixels x ' + str(round(cfg['um2_per_px'], 3)) if cfg['um2_per_px'] else ''});
                            mask_source: cellpose / cellpose_merged / msi_rescue
  Cluster*, DensMAP*, UMAP*, TSNE*, PCA*, NMF_dominant, Bonsai*   the pipeline's clusters and embedding coordinates
                            (cells that passed the filters only; Bonsai: its subsample only)
  mz_<value>                raw per-cell MEAN intensity at that m/z: TIC-normalised, averaged over the cell's
                            pixels, NOT batch-corrected, NOT scaled. 0 = not detected in that cell.
samples_raw.csv             one row per sample: labels, n_cells_measured, n_cells_passed_filters, and the mean
                            of mz_<value> over the sample's cells that passed the filters ("pseudo-bulk")
features.csv                one row per m/z: used_in_pipeline_analysis (passed the zero-rate filter and entered
                            the embeddings), detection rate and mean intensity over the cells that passed the
                            filters, overall and per {grp}

Load in Python:  pd.read_parquet("cells_raw.parquet")      in R:  arrow::read_parquet("cells_raw.parquet")
"""
    (out / "README.txt").write_text(readme, encoding="utf-8")
    sizes = {p.name: f"{p.stat().st_size / 1e6:.0f} MB" for p in out.iterdir()}
    print(f"   {len(cells):,} cells x {len(mz_out):,} m/z, {len(smp)} samples -> {out}  {sizes}  ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(prog="python -m tanuki.export",
                                 description="Raw per-cell CSV / parquet export of a finished Tanuki run "
                                             "(writes <run>/export/).")
    ap.add_argument("folder", nargs="+", help="run folder(s): the pipeline's MASTER_DIR")
    ap.add_argument("--preset", default="ipsc", choices=sorted(DATASETS))
    a = ap.parse_args()
    for f in a.folder:
        export(a.preset, f)
