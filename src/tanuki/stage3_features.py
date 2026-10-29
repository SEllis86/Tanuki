"""Stage 3 - per-cell morphology and per-cell mean / sum / SEM intensities for every m/z. Runs in a
process pool."""
from __future__ import annotations

import gc
import multiprocessing as mp
import time
import warnings
from pathlib import Path

import cv2
import h5py
import numpy as np
import pandas as pd
import scipy.sparse
from skimage.measure import regionprops
from skimage.morphology import skeletonize

from .config import (MASTER_DIR, MAX_WORKERS_STAGE3, MIN_CELL_AREA_PX, SEM_EFFECTIVE_N, STAGE3_SLAB_MASSES,
    USE_MORPHOLOGY_FEATURES, WRITE_PIXEL_LONG_TABLE)
from .utils import min_area_for_folder, msi_upscale_area, mz_col_name, save_df_parquet


# ============================================================
# STAGE 3 - POST-CELLPOSE
# ============================================================

def compute_morphology_and_multiple_intensities(masks, cubes_dict, image_for_cellpose, masses_of_interest,
                                                min_area_px=MIN_CELL_AREA_PX, native_msi_shape=None):
    # notes: Vectorized morphology to eliminate looping OpenCV calls per-cell.
    # notes: Added safety block to snap old masks to new array shapes, preventing bincount errors.
    # notes: Added edge detection to tag cells touching the boundaries of the image.

    target_shape = image_for_cellpose.shape[:2]
    if masks.shape != target_shape:
        masks = cv2.resize(
            masks.astype(np.float32),
            (target_shape[1], target_shape[0]),
            interpolation=cv2.INTER_NEAREST
        ).astype(np.int32)

    # FIX: previously `label(masks > 0, connectivity=1)` binarised the segmentation and
    # re-derived components, which merged every pair of touching Cellpose cells into one
    # object and destroyed the link back to PREMSI_seg.npy. Instead compact the existing
    # Cellpose IDs to a contiguous 0..N range (required for np.bincount) while keeping a
    # lookup back to the original IDs.
    raw_ids = masks.astype(np.int64)
    unique_ids, compact = np.unique(raw_ids, return_inverse=True)
    compact = compact.reshape(raw_ids.shape).astype(np.int32)

    # np.unique sorts, so background (0) lands at position 0. If the frame has no
    # background at all, shift everything up so label 0 stays reserved for background.
    if unique_ids.size > 0 and unique_ids[0] != 0:
        compact = compact + 1
        unique_ids = np.concatenate(([0], unique_ids))

    labels_img = compact

    edge_labels = set(np.unique(labels_img[0, :])) | \
                  set(np.unique(labels_img[-1, :])) | \
                  set(np.unique(labels_img[:, 0])) | \
                  set(np.unique(labels_img[:, -1]))
    if 0 in edge_labels:
        edge_labels.remove(0)

    props = regionprops(labels_img)
    morph_rows = []

    # FIX: this used to skeletonise the whole foreground as one binary, so any two touching
    # cells fused into a single connected region and their branch counts / skeleton lengths
    # bled across the shared border. That systematically inflates branch metrics in dense
    # fields, which is exactly where the neuron samples live. Drop the one-pixel seam where
    # a foreground pixel has a DIFFERENT non-zero label in its 3x3 neighbourhood, so each
    # cell is skeletonised in isolation. Two morphology ops, negligible cost.
    k3 = np.ones((3, 3), np.uint8)
    lab_f = labels_img.astype(np.float32)
    fg = labels_img > 0
    big = float(labels_img.max()) + 1.0
    neigh_max = cv2.dilate(np.where(fg, lab_f, 0.0), k3)          # ignores background
    neigh_min = cv2.erode(np.where(fg, lab_f, big), k3)           # ignores background
    contact_seam = fg & (neigh_max != neigh_min)

    # skeletonize and the two morphology kernels are the expensive part of stage 3 and
    # are only needed for the shape descriptors
    if USE_MORPHOLOGY_FEATURES:
        boundaries = (fg & ~contact_seam).astype(np.uint8)
        skel = skeletonize(boundaries).astype(np.uint8)

        kernel_conv = np.ones((3, 3), dtype=np.uint8)
        neighbor_count = cv2.filter2D(skel, -1, kernel_conv, borderType=cv2.BORDER_CONSTANT)

        branch_ends_img = (skel > 0) & (neighbor_count == 2)
        branch_pts_img = (skel > 0) & (neighbor_count > 3)

        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        seed_mask_img = cv2.morphologyEx(boundaries, cv2.MORPH_OPEN, kernel_open).astype(bool)
        trunks_img = branch_pts_img & seed_mask_img

    flat_labels = labels_img.ravel()

    if USE_MORPHOLOGY_FEATURES:
        total_skel_per_cell = np.bincount(flat_labels, weights=skel.ravel())
        branch_ends_per_cell = np.bincount(flat_labels, weights=branch_ends_img.ravel())
        branch_pts_per_cell = np.bincount(flat_labels, weights=branch_pts_img.ravel())
        trunks_per_cell = np.bincount(flat_labels, weights=trunks_img.ravel())
    else:
        total_skel_per_cell = np.zeros(int(labels_img.max()) + 1, dtype=np.float64)

    for r in props:
        area = r.area
        if area < min_area_px: continue
        label_id = r.label

        if label_id >= len(total_skel_per_cell): continue

        row = {
            # FIX: mask_id is the true Cellpose ID again, so rows join back to PREMSI_seg.npy.
                "mask_id": int(unique_ids[label_id]),
                "_label_idx": int(label_id),
                "location": "edge" if label_id in edge_labels else "middle",
                "area_pixels": area,
                "centroid_row": r.centroid[0],
                "centroid_col": r.centroid[1],
        }
        if USE_MORPHOLOGY_FEATURES:
            row.update({
                "perimeter": r.perimeter,
                "extent": r.extent,
                "eccentricity": r.eccentricity,
                "major_axis_length": r.major_axis_length,
                "minor_axis_length": r.minor_axis_length,
                "solidity": r.solidity,
                "branch_points": int(branch_pts_per_cell[label_id]),
                "number_trunks": int(trunks_per_cell[label_id]),
                "number_non_trunk_branches": int(branch_pts_per_cell[label_id] - trunks_per_cell[label_id]),
                "number_branch_ends": int(branch_ends_per_cell[label_id]),
                "total_object_skeleton_length": int(total_skel_per_cell[label_id]),
            })
        morph_rows.append(row)

    df_shapes = pd.DataFrame(morph_rows)
    results = {}

    if df_shapes.empty:
        for method_name in cubes_dict.keys():
            results[method_name] = {"mean": pd.DataFrame(), "sum": pd.DataFrame(),
                                    "sem": pd.DataFrame(), "pixels": pd.DataFrame()}
        return df_shapes, results

    areas = np.bincount(flat_labels)
    safe_areas = np.where(areas == 0, 1, areas)

    # v5: effective number of INDEPENDENT pixels per cell for the standard error. The cube is
    # the native MSI raster bilinearly upsampled ~240x, so a cell's upsampled pixels are not
    # independent samples; dividing the variance by their count made every SEM ~15x too
    # small, and Bonsai then treated each value as almost exactly known.
    upscale_area = msi_upscale_area(image_for_cellpose, native_msi_shape) if SEM_EFFECTIVE_N else 1.0

    for method_name, cube in cubes_dict.items():
        n_channels = int(cube.shape[2])
        # FIX: bincount results are indexed by the compact label, but the reported mask_id
        # must be the original Cellpose ID. Keep the two separate.
        label_idx = df_shapes['_label_idx'].to_numpy(dtype=np.int64)
        out_ids = df_shapes['mask_id'].to_numpy(dtype=np.int64)
        keep = areas[label_idx] > 0
        valid_lbls = label_idx[keep]
        valid_out_ids = out_ids[keep]
        n_valid = len(valid_lbls)
        if n_valid == 0 or n_channels == 0:
            results[method_name] = {"mean": pd.DataFrame(), "sum": pd.DataFrame(),
                                    "sem": pd.DataFrame(), "pixels": pd.DataFrame()}
            continue

        # v5: per-cell sums and sums of squares for every mass, one slab of
        # STAGE3_SLAB_MASSES masses at a time: a sparse (cells x pixels) one-hot matrix times
        # a (pixels x masses) slab. The cube is read from the h5 slab by slab instead of being
        # loaded whole (~6 GB per worker), and there is no Python loop of two full-image
        # bincounts per mass. Same sums as before, to floating-point rounding.
        row_of = np.full(int(flat_labels.max()) + 1, -1, dtype=np.int64)
        row_of[valid_lbls] = np.arange(n_valid)
        pix_rows = row_of[flat_labels]
        in_cell = np.flatnonzero(pix_rows >= 0)
        onehot = scipy.sparse.csr_matrix(
            (np.ones(in_cell.size), (pix_rows[in_cell], in_cell)), shape=(n_valid, flat_labels.size))
        sums = np.zeros((n_valid, n_channels), dtype=np.float64)
        sumsq = np.zeros((n_valid, n_channels), dtype=np.float64)
        step = max(1, int(STAGE3_SLAB_MASSES))
        for c0 in range(0, n_channels, step):
            c1 = min(n_channels, c0 + step)
            slab = np.asarray(cube[:, :, c0:c1], dtype=np.float64).reshape(-1, c1 - c0)
            sums[:, c0:c1] = onehot @ slab
            sumsq[:, c0:c1] = onehot @ (slab * slab)
            del slab

        n_px = areas[valid_lbls].astype(np.float64)[:, None]
        means = sums / n_px
        # Spread of the pixels INSIDE each cell -> standard error on the cell mean. Bonsai
        # weights its likelihood by exactly this.
        var = np.maximum(sumsq / n_px - means * means, 0.0)
        n_eff = np.maximum(n_px / max(float(upscale_area), 1.0), 1.0)
        sems = np.sqrt(var / n_eff)

        # FIX: string m/z column names via the shared mz_col_name helper (raw float labels
        # did not survive a parquet round-trip reliably); first occurrence wins if two masses
        # share a name at MZ_COL_PRECISION
        mz_vals = [mz_col_name(masses_of_interest[c]) if c < len(masses_of_interest) else mz_col_name(c)
                   for c in range(n_channels)]
        first = {}
        for i, name in enumerate(mz_vals):
            first.setdefault(name, i)
        keep_cols = list(first.values())
        cols = list(first.keys())

        df_mean = pd.DataFrame(means[:, keep_cols], columns=cols)
        df_mean.insert(0, "mask_id", valid_out_ids)
        df_sum = pd.DataFrame(sums[:, keep_cols], columns=cols)
        df_sum.insert(0, "mask_id", valid_out_ids)
        df_sem = pd.DataFrame(sems[:, keep_cols], columns=cols)
        df_sem.insert(0, "mask_id", valid_out_ids)
        df_mean = df_mean.merge(df_shapes.drop(columns=["_label_idx"], errors="ignore"),
                                on="mask_id", how="left")

        # the long one-row-per-(cell, mass) table only when asked for (WRITE_PIXEL_LONG_TABLE)
        if WRITE_PIXEL_LONG_TABLE:
            df_int = pd.DataFrame({
                "mask_id": np.tile(valid_out_ids, n_channels),
                "mz": np.repeat(np.array(mz_vals, dtype=object), n_valid),
                "channel_index": np.repeat(np.arange(n_channels), n_valid),
                "mean_intensity": means.T.ravel(),
                "sum_intensity": sums.T.ravel(),
                "sem_intensity": sems.T.ravel(),
                "pixel_count": np.tile(areas[valid_lbls], n_channels),
            })
        else:
            df_int = pd.DataFrame()

        results[method_name] = {"mean": df_mean, "sum": df_sum, "sem": df_sem,
                                "pixels": df_int}

    # FIX: internal bookkeeping column, never written to disk.
    df_shapes = df_shapes.drop(columns=["_label_idx"], errors="ignore")

    return df_shapes, results


STAGE3_OUTPUTS = ("cell_morphology_features.parquet", "mask_id_averages_raw.parquet",
                  "mask_id_sums_raw.parquet", "mask_id_sems_raw.parquet", "pixel_intensities.parquet")


def stage3_worker(subdir: Path):
    try:
        # notes: Updated to read everything directly from the single unified HDF5 file
        stage1_h5 = subdir / "PREMSI_stage1_ready.h5"
        seg_path = subdir / "PREMSI_seg.npy"

        # v5: a folder that fails (or has lost its masks) must not leave the PREVIOUS run's
        # parquets behind for concat to pick up - their mask_ids belong to other masks
        for _name in STAGE3_OUTPUTS:
            try:
                (subdir / _name).unlink()
            except FileNotFoundError:
                pass

        if not stage1_h5.exists() or not seg_path.exists():
            return True

        print(f"[S3] ▶ {subdir.name}")
        # FIX: removed os.chdir here too; every path below is absolute.

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")

            seg_data_raw = np.load(seg_path, allow_pickle=True)
            seg_source_map = {}
            if isinstance(seg_data_raw.item(), dict):
                masks = seg_data_raw.item()["masks"]
                # v2: per-label provenance written by stage 2 ('cellpose_merged', 'msi_rescue')
                seg_source_map = dict(seg_data_raw.item().get("mask_source", {}) or {})
            else:
                masks = seg_data_raw.item()

            # v5: the cube stays on disk and is read slab by slab inside the function
            with h5py.File(stage1_h5, 'r') as stage1:
                image_for_cellpose = stage1["image_for_cellpose"][:]
                masses_of_interest = stage1["masses_of_interest"][:]
                native_msi_shape = (tuple(int(v) for v in stage1.attrs["native_msi_shape"])
                                    if "native_msi_shape" in stage1.attrs else None)
                cubes_dict = {"raw": stage1["raw_cube"]}
                df_shapes, results_dict = compute_morphology_and_multiple_intensities(
                    masks, cubes_dict, image_for_cellpose, masses_of_interest,
                    min_area_px=min_area_for_folder(subdir.name), native_msi_shape=native_msi_shape
                )

        # Per-well QC. n_segmented is the RAW Cellpose count straight off PREMSI_seg.npy,
        # before the area gate; n_after_gate is what stage 3 actually wrote. Both are
        # constant per sample and ride along on every row so the final matrix can be
        # pivoted by well in Excel without joining anything.
        _min_area_used = min_area_for_folder(subdir.name)
        _u = np.unique(masks)
        n_segmented = int((_u > 0).sum())
        n_after_gate = int(len(df_shapes))
        print(f'[S3] {subdir.name}: {n_segmented} segmented -> {n_after_gate} past the '
              f'{_min_area_used}px area gate '
              f'({100 * n_after_gate / max(n_segmented, 1):.0f}%)')

        sample_label = subdir.name
        # v2: how each mask came to be - plain Cellpose, merged fragments, or MSI rescue
        _src = lambda m: seg_source_map.get(int(m), 'cellpose')
        if not df_shapes.empty and 'mask_id' in df_shapes.columns:
            df_shapes["mask_source"] = df_shapes["mask_id"].map(_src)
            _counts = df_shapes["mask_source"].value_counts().to_dict()
            if len(_counts) > 1:
                print(f"[S3] {subdir.name}: mask_source {_counts}")
        df_shapes["sample name"] = sample_label
        df_shapes["well_cells_segmented"] = n_segmented
        df_shapes["well_cells_after_area_gate"] = n_after_gate
        save_df_parquet(df_shapes, str(subdir / "cell_morphology_features.parquet"))

        for method, dfs in results_dict.items():
            dfs["mean"]["sample name"] = sample_label
            dfs["sum"]["sample name"] = sample_label
            if not dfs["mean"].empty and 'mask_id' in dfs["mean"].columns:
                dfs["mean"]["mask_source"] = dfs["mean"]["mask_id"].map(_src)
            if not dfs.get("sem", pd.DataFrame()).empty:
                dfs["sem"]["sample name"] = sample_label
                save_df_parquet(dfs["sem"], str(subdir / f"mask_id_sems_{method}.parquet"))
            if not dfs["mean"].empty:
                dfs["mean"]["well_cells_segmented"] = n_segmented
                dfs["mean"]["well_cells_after_area_gate"] = n_after_gate
            save_df_parquet(dfs["mean"], str(subdir / f"mask_id_averages_{method}.parquet"))
            save_df_parquet(dfs["sum"], str(subdir / f"mask_id_sums_{method}.parquet"))

            if method == "raw" and WRITE_PIXEL_LONG_TABLE and not dfs["pixels"].empty:
                dfs["pixels"]["sample name"] = sample_label
                save_df_parquet(dfs["pixels"], str(subdir / "pixel_intensities.parquet"))

        del cubes_dict, df_shapes, results_dict, seg_data_raw, masks
        gc.collect()

        return True

    except Exception as e:
        print(f"[S3] ERROR in {subdir.name}: {e}")
        return False


def run_stage3_postprocess_parallel():
    t0 = time.time()
    subdirs = [d for d in MASTER_DIR.iterdir() if d.is_dir()]

    print(f"\n[S3] Launching PASS 1 on {len(subdirs)} folders with {MAX_WORKERS_STAGE3} workers...")
    # v5: maxtasksperchild=1 recycled every worker after one folder (~9 s of module re-import
    # each time) to free a ~6 GB cube; the cube is no longer loaded whole, so keep them
    with mp.Pool(processes=MAX_WORKERS_STAGE3) as pool:
        results = pool.map(stage3_worker, subdirs, chunksize=1)

    failed_dirs = [subdirs[i] for i, res in enumerate(results) if res is False]

    if failed_dirs:
        print(f"\n[S3] {len(failed_dirs)} folders failed. Requeueing in SAFE MODE (1 Worker)...")
        with mp.Pool(processes=1, maxtasksperchild=1) as safe_pool:
            retry_results = safe_pool.map(stage3_worker, failed_dirs)

        permanent_fails = [failed_dirs[i] for i, res in enumerate(retry_results) if res is False]
        if permanent_fails:
            print(f"[S3] {len(permanent_fails)} folders permanently failed.")
        else:
            print("[S3] All requeued folders recovered successfully!")

    print(f"[S3] DONE in {time.time() - t0:.1f} s")
