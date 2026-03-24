import os
import sys
import warnings
import traceback
import numpy as np
import pandas as pd
import cv2
import tifffile
from pathlib import Path
from scipy.spatial import cKDTree
from skimage.measure import regionprops
from skimage.transform import resize
from config import CSV_PATH

def should_skip_subdir(subdir: Path) -> bool:
    return (subdir / "PREMSI_stage1_ready.npz").exists()

def warn_with_traceback(message, category, filename, lineno, file=None, line=None):
    log = file if hasattr(file, 'write') else sys.stderr
    traceback.print_stack(file=log)
    log.write(warnings.formatwarning(message, category, filename, lineno, line))

def refine_registration_from_centroids(mask, reference_image):
    regions = regionprops(mask)
    mask_centroids = np.array([r.centroid[::-1] for r in regions]) 
    ref = reference_image.copy()
    _, ref_bin = cv2.threshold(ref, 50, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(ref_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    ref_centroids = []
    for cnt in contours:
        M = cv2.moments(cnt)
        if M['m00'] != 0:
            ref_centroids.append((int(M['m10'] / M['m00']), int(M['m01'] / M['m00'])))
    ref_centroids = np.array(ref_centroids)

    if len(mask_centroids) < 3 or len(ref_centroids) < 3:
        return None

    tree = cKDTree(ref_centroids)
    distances, indices = tree.query(mask_centroids)
    matched_ref = ref_centroids[indices]
    
    affine_matrix, _ = cv2.estimateAffine2D(
        mask_centroids.astype(np.float32),
        matched_ref.astype(np.float32),
        method=cv2.RANSAC
    )
    return affine_matrix

def import_mass_list(csv_path=CSV_PATH, precision=4, return_str=False):
    df = pd.read_csv(csv_path, header=0, usecols=[0], sep=None, engine='python')
    df = df.fillna(0)
    try:
        float_masses = [round(float(row[0]), precision) for row in df.values.tolist()]
    except Exception:
        float_masses = df.iloc[:, 0].astype(float).round(precision).tolist()

    unique_masses = []
    seen = set()
    for m in float_masses:
        if m not in seen:
            unique_masses.append(m)
            seen.add(m)

    if return_str:
        return [format(m, f'.{precision}f') for m in unique_masses]
    return unique_masses

def save_df_npz(df: pd.DataFrame, path: str):
    path_str = str(path)
    temp_path = path_str + ".tmp.npz"

    if df is None or df.empty or df.shape[1] == 0:
        np.savez_compressed(temp_path, data=np.array([], dtype=float), columns=np.array([], dtype=object))
        os.replace(temp_path, path_str)
        return

    df = df.copy()
    if df.columns.duplicated().any():
        new_cols, seen = [], {}
        for col in df.columns:
            if col not in seen:
                seen[col] = 0; new_cols.append(col)
            else:
                seen[col] += 1; new_cols.append(f"{col}_{seen[col]}")
        df.columns = new_cols

    for col in df.select_dtypes(include=["object"]):
        df[col] = df[col].astype(str)

    rec = df.to_records(index=False)
    np.savez_compressed(temp_path, data=rec, columns=np.array(df.columns.tolist(), dtype=object))
    os.replace(temp_path, path_str)

def load_df_npz(path: str) -> pd.DataFrame:
    with np.load(path, allow_pickle=True) as npz:
        if 'data' in npz:
            return pd.DataFrame.from_records(npz['data'])
        return pd.DataFrame()

def resize_larger_to_match_smaller(fixed_np, moving_np, tolerance=0.2):
    fixed_shape, moving_shape = fixed_np.shape[:2], moving_np.shape[:2]
    fixed_area = fixed_shape[0] * fixed_shape[1]
    moving_area = moving_shape[0] * moving_shape[1]

    larger_np, smaller_np, label = (fixed_np, moving_np, "fixed") if fixed_area > moving_area else (moving_np, fixed_np, "moving")
    height_diff = abs(fixed_shape[0] - moving_shape[0]) / max(fixed_shape[0], moving_shape[0])
    width_diff = abs(fixed_shape[1] - moving_shape[1]) / max(fixed_shape[1], moving_shape[1])

    if height_diff > tolerance or width_diff > tolerance:
        resized = resize(larger_np, output_shape=smaller_np.shape if larger_np.ndim == 3 else smaller_np.shape[:2], preserve_range=True, anti_aliasing=True).astype(larger_np.dtype)
        return (resized, smaller_np) if label == "fixed" else (smaller_np, resized)
    return fixed_np, moving_np

def load_rgb_tiff(path):
    img = tifffile.imread(path)
    if img.ndim == 3 and img.shape[0] < img.shape[1] and img.shape[0] < img.shape[2]:
        img = np.moveaxis(img, 0, -1)
    if img.ndim == 2:
        img = img[..., np.newaxis]
    if img.ndim == 3:
        return img
    raise ValueError(f"Unsupported TIFF shape: {img.shape} in file {path}")
