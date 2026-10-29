"""Shared helpers: m/z column names, cell type / slide from a folder name, the mass list, parquet I/O,
TIFF loading, stage-1 bookkeeping. (v5: top of the script and "SHARED HELPERS & MASS EXTRACTION")"""
from __future__ import annotations

import functools
import os
import re
import sys
import traceback
import warnings
from pathlib import Path

import cv2
import h5py
import numpy as np
import pandas as pd
import plotly.express as px
import tifffile
from scipy.spatial import cKDTree
from skimage.measure import regionprops

from .config import (H5_COMPRESSION, MIN_CELL_AREA_PX, MZ_COL_PRECISION, NEURO_MIN_CELL_AREA_PX,
    REREGISTER_OLD_MMI_FOLDERS, STAGE1_FORMAT_VERSION, STAGE3_NEURO_MIN_AREA_KEYWORDS)


def mz_col_name(mz) -> str:
    # FIX: pivot_table used to emit raw float column labels. Whether those survived a
    # parquet round-trip as floats or strings depended on the pyarrow version, and
    # detect_mass_columns then matched neither. Always name mass columns as strings.
    return f"mz_{float(mz):.{MZ_COL_PRECISION}f}"


def cell_type_of(name) -> str:
    """
    v3: MICRO / ASTRO / NEURO / OTHER from a folder name, case-insensitive, checked in
    that order (the same order the labelling has always used). The single source for the
    per-cell-type area bounds, the per-cell-type zero-rate filter, the CellType column
    and the cell-type half of the Batch label. 'Neurons_11302' matches 'neuro'.
    """
    s = str(name).lower()
    for key in ('micro', 'astro', 'neuro'):
        if key in s:
            return key.upper()
    return 'OTHER'


def slide_id_of(name) -> str:
    """
    v3: the slide token (S1B2, S2B1, ...) from a folder name, upper-cased, or
    'UnknownSlide'. Bounded on non-alphanumerics so it cannot match inside a longer word.
    COMBAT_BATCH_COL = 'Slide' uses it as the ComBat batch.
    """
    m = re.search(r'(?<![A-Za-z0-9])(S\d+B\d+)(?![A-Za-z0-9])', str(name), re.IGNORECASE)
    return m.group(1).upper() if m else "UnknownSlide"


def _registered_by_phase_method(subdir: Path) -> bool:
    """True when this folder's QC file was written by the phase-correlation registration."""
    qc = subdir / "QC_registration_accuracy.txt"
    if not qc.exists():
        return False
    try:
        return "Method: phase" in qc.read_text(errors='replace')
    except Exception:
        return False


def should_skip_subdir(subdir: Path) -> bool:
    # notes: Updated to look for the new unified .h5 file
    if not (subdir / "PREMSI_stage1_ready.h5").exists():
        return False
    if REREGISTER_OLD_MMI_FOLDERS and not _registered_by_phase_method(subdir):
        return False
    # v5: an h5 written in an older format (8-bit imgs_cp before format 2) is redone
    if stage1_format_of(subdir) < STAGE1_FORMAT_VERSION:
        return False
    return True


def stage1_format_of(subdir: Path) -> int:
    """v5: the stage1_format attr of a folder's stage 1 h5; 0 when absent or unreadable."""
    try:
        with h5py.File(subdir / "PREMSI_stage1_ready.h5", 'r') as f:
            return int(f.attrs.get("stage1_format", 0))
    except Exception:
        return 0


def stage1_run_id_of(subdir: Path) -> str:
    """v5: the stage1_run_id attr of a folder's stage 1 h5; '' when absent or unreadable."""
    try:
        with h5py.File(subdir / "PREMSI_stage1_ready.h5", 'r') as f:
            return str(f.attrs.get("stage1_run_id", ""))
    except Exception:
        return ""


def h5_compression_kwargs():
    """v5: h5py dataset kwargs for H5_COMPRESSION ('lzf', 'gzip' or None)."""
    if not H5_COMPRESSION:
        return {}
    return {"compression": H5_COMPRESSION}


def warn_with_traceback(message, category, filename, lineno, file=None, line=None):
    log = file if hasattr(file, 'write') else sys.stderr
    traceback.print_stack(file=log)
    log.write(warnings.formatwarning(message, category, filename, lineno, line))



def refine_registration_from_centroids(mask, reference_image):
    regions = regionprops(mask)
    mask_centroids = np.array([r.centroid[::-1] for r in regions])  # (x,y)

    ref = reference_image.copy()
    _, ref_bin = cv2.threshold(ref, 50, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(ref_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    ref_centroids = []
    for cnt in contours:
        M = cv2.moments(cnt)
        if M['m00'] != 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            ref_centroids.append((cx, cy))

    ref_centroids = np.array(ref_centroids)

    if len(mask_centroids) < 3 or len(ref_centroids) < 3:
        print("⚠️ Not enough centroids for refinement")
        return None

    tree = cKDTree(ref_centroids)
    distances, indices = tree.query(mask_centroids)

    matched_ref = ref_centroids[indices]
    matched_mask = mask_centroids

    affine_matrix, _ = cv2.estimateAffine2D(
        matched_mask.astype(np.float32),
        matched_ref.astype(np.float32),
        method=cv2.RANSAC
    )

    return affine_matrix


def import_mass_list(csv_path, precision=4, return_str=False):
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
    else:
        return unique_masses




@functools.lru_cache(maxsize=None)
def _mass_list(csv_path):
    return tuple(import_mass_list(csv_path, precision=4, return_str=False))


def get_masses_of_interest():
    """The mass list from CSV_PATH. v5 read it into the module global masses_of_interest at import;
    here it is read on first use (once per process), so the CSV path can be set per run."""
    from . import config
    return list(_mass_list(str(config.CSV_PATH)))


def save_df_parquet(df: pd.DataFrame, path: str):
    # notes: Fast columnar saving via PyArrow
    path_str = str(path)
    temp_path = path_str + ".tmp.parquet"

    if df is None or df.empty or df.shape[1] == 0:
        pd.DataFrame().to_parquet(temp_path, engine='pyarrow', index=False)
        os.replace(temp_path, path_str)
        return

    needs_copy = False
    if df.columns.duplicated().any():
        needs_copy = True
    elif not df.select_dtypes(include=["object"]).empty:
        needs_copy = True

    if needs_copy:
        df = df.copy()

    if df.columns.duplicated().any():
        new_cols = []
        seen = {}
        for col in df.columns:
            if col not in seen:
                seen[col] = 0
                new_cols.append(col)
            else:
                seen[col] += 1
                new_cols.append(f"{col}_{seen[col]}")
        df.columns = new_cols

    for col in df.select_dtypes(include=["object"]):
        df[col] = df[col].astype(str)

    df.to_parquet(temp_path, engine='pyarrow', index=False)
    os.replace(temp_path, path_str)


def load_df_parquet(path: str) -> pd.DataFrame:
    # notes: Instant loading via PyArrow
    try:
        return pd.read_parquet(path, engine='pyarrow')
    except Exception:
        return pd.DataFrame()


def load_rgb_tiff(path):
    img = tifffile.imread(path)
    if img.ndim == 3 and img.shape[0] < img.shape[1] and img.shape[0] < img.shape[2]:
        img = np.moveaxis(img, 0, -1)
    if img.ndim == 2:
        img = img[..., np.newaxis]
    if img.ndim == 3:
        return img
    raise ValueError(f"Unsupported TIFF shape: {img.shape} in file {path}")


# ============================================================
# COLOUR HELPERS (shared by the analysis and the Bonsai plots)
# ============================================================

def discrete_palette(n):
    """Enough distinct colours for n groups (plotly's default sequence repeats after 10)."""
    if n <= 10:
        return list(px.colors.qualitative.Plotly)
    if n <= 24:
        return list(px.colors.qualitative.Dark24)
    return [px.colors.sample_colorscale('Turbo', [i / max(n - 1, 1)])[0] for i in range(n)]


def group_sort_key(v):
    """Numbers in numeric order (cluster 2 before 10), everything else alphabetically after."""
    try:
        return (0, float(v), "")
    except (TypeError, ValueError):
        return (1, 0.0, str(v))


def group_color_map(groups, color_map=None):
    """
    One fixed colour per group, so a group has the SAME colour in every panel and every plot.
    Without an explicit map, plotly colours traces by their position in each figure, so in
    a two-panel figure a batch changed colour from one panel to the next.
    """
    names = sorted({str(g) for g in groups}, key=group_sort_key)
    palette = discrete_palette(len(names))
    out = {g: palette[i % len(palette)] for i, g in enumerate(names)}
    if color_map:
        out.update({str(k): v for k, v in color_map.items()})
    return out


# ============================================================
# STAGE 2 / 3 SHARED (here so that the stage 3 workers do not import Cellpose)
# ============================================================

def min_area_for_folder(folder_name) -> int:
    """
    The stage 3 minimum-area gate for this sample: NEURO_MIN_CELL_AREA_PX for a folder
    matching STAGE3_NEURO_MIN_AREA_KEYWORDS, MIN_CELL_AREA_PX otherwise.
    """
    name_lower = str(folder_name).lower()
    if any(k.lower() in name_lower for k in STAGE3_NEURO_MIN_AREA_KEYWORDS):
        return NEURO_MIN_CELL_AREA_PX
    return MIN_CELL_AREA_PX


def msi_upscale_area(image_for_cellpose, native_msi_shape):
    """Upscaled pixels per native MSI pixel: converts MSI-pixel settings into image pixels."""
    if native_msi_shape and native_msi_shape[0] > 0 and native_msi_shape[1] > 0:
        return (image_for_cellpose.shape[0] * image_for_cellpose.shape[1]) / float(
            native_msi_shape[0] * native_msi_shape[1])
    return 200.0
