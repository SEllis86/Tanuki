import os
import gc
import time
import warnings
import numpy as np
import pandas as pd
import multiprocessing as mp
from pathlib import Path
from skimage.measure import regionprops, label
from skimage.morphology import skeletonize, binary_opening, disk
from scipy.ndimage import convolve

from config import MASTER_DIR, MAX_WORKERS_STAGE3
from utils import save_df_npz

def compute_morphology_and_intensities(masks, resized_data, image_for_cellpose, masses_of_interest):
    labels_img = label(masks.astype(np.int32) > 0, connectivity=1)
    props, morph_rows = regionprops(labels_img), []
    N, skel = resized_data.shape[2], skeletonize((labels_img > 0).astype(np.uint8))
    kernel = np.ones((3, 3), dtype=np.uint8)

    for r in props:
        if r.area < 50: continue
        coords = (labels_img == r.label)
        region_skel = skel & coords
        neighbor_count = convolve(region_skel.astype(np.uint8), kernel, mode='constant', cval=0)
        branch_points_mask = (region_skel) & (neighbor_count > 3)
        
        morph_rows.append({
            "mask_id": r.label, "area_pixels": r.area, "perimeter": r.perimeter, "extent": r.extent,
            "eccentricity": r.eccentricity, "major_axis_length": r.major_axis_length, "minor_axis_length": r.minor_axis_length,
            "solidity": r.solidity, "branch_points": int(np.sum(branch_points_mask)),
            "number_trunks": int(np.sum(branch_points_mask & binary_opening(coords, disk(3)))),
            "number_non_trunk_branches": int(np.sum(branch_points_mask) - np.sum(branch_points_mask & binary_opening(coords, disk(3)))),
            "number_branch_ends": int(np.sum((region_skel) & (neighbor_count == 2))),
            "total_object_skeleton_length": int(np.sum(region_skel)),
            "centroid_row": r.centroid[0], "centroid_col": r.centroid[1],
        })

    df_shapes = pd.DataFrame(morph_rows)
    intens_rows, flat_masks = [], labels_img.ravel()
    areas = np.bincount(flat_masks)
    safe_areas = np.where(areas == 0, 1, areas)

    for c in range(N):
        sums, means = np.bincount(flat_masks, weights=resized_data[:, :, c].ravel()), np.bincount(flat_masks, weights=resized_data[:, :, c].ravel()) / safe_areas
        for mid in df_shapes['mask_id']:
            intens_rows.append({"mask_id": mid, "mz": float(masses_of_interest[c]) if c < len(masses_of_interest) else float(c), "channel_index": c, "mean_intensity": float(means[mid]), "sum_intensity": float(sums[mid]), "pixel_count": int(areas[mid])})

    df_int = pd.DataFrame(intens_rows)
    df_mean = df_int.pivot_table(index="mask_id", columns="mz", values="mean_intensity", aggfunc="first").reset_index().merge(df_shapes, on="mask_id", how="left") if not df_int.empty else pd.DataFrame()
    df_sum = df_int.pivot_table(index="mask_id", columns="mz", values="sum_intensity", aggfunc="first").reset_index() if not df_int.empty else pd.DataFrame()
    return df_shapes, df_int, df_mean, df_sum

def stage3_worker(subdir: Path):
    stage1_npz, seg_path = subdir / "PREMSI_stage1_ready.npz", subdir / "PREMSI_seg.npy"
    if not stage1_npz.exists() or not seg_path.exists(): return True

    os.chdir(subdir)
    with np.load(stage1_npz, allow_pickle=True) as stage1:
        df_shapes, df_pixels, df_mean, df_sum = compute_morphology_and_intensities(np.load(seg_path, allow_pickle=True).item()["masks"], stage1["resized_data"], stage1["image_for_cellpose"], stage1["masses_of_interest"])

    for df in (df_shapes, df_pixels, df_mean, df_sum): df["sample name"] = subdir.name
    save_df_npz(df_pixels, "pixel_intensities.npz"); save_df_npz(df_shapes, "cell_morphology_features.npz")
    save_df_npz(df_mean, "mask_id_averages_with_metadata.npz"); save_df_npz(df_sum, "mask_id_sums_with_metadata.npz")
    return True

def run_stage3_postprocess_parallel():
    t0 = time.time()
    subdirs = [d for d in MASTER_DIR.iterdir() if d.is_dir()]
    with mp.Pool(processes=MAX_WORKERS_STAGE3, maxtasksperchild=1) as pool: pool.map(stage3_worker, subdirs)
    print(f"[S3] DONE in {time.time() - t0:.1f} s")
