"""Stage 4 - concatenation of the per-well stage 3 tables into the run-level parquet / CSV files."""
from __future__ import annotations

import gc
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

from .config import MASTER_DIR, WRITE_PIXEL_LONG_TABLE
from .utils import load_df_parquet, save_df_parquet


# ============================================================
# FIXED CONCAT FUNCTIONS
# ============================================================

def _safe_load_parquet(path, sample_name):
    try:
        df = load_df_parquet(path)
        if df.empty: return pd.DataFrame()
        df = df.copy()
        if "sample name" not in df.columns: df["sample name"] = sample_name
        return df
    except Exception as e:
        print(f"Skipping corrupted file: {path} - Error: {e}")
        return pd.DataFrame()


def _concat_union(dfs) -> pd.DataFrame:
    valid_dfs = [df for df in dfs if not df.empty]
    if not valid_dfs: return pd.DataFrame()
    return pd.concat(valid_dfs, axis=0, ignore_index=True)


def concat_parquet_files(master_directory: Path, target_filename: str) -> pd.DataFrame:
    file_paths = [os.path.join(root, target_filename) for root, _, files in os.walk(master_directory) if
                  target_filename in files]
    if not file_paths: return pd.DataFrame()
    # FIX: a bare ThreadPoolExecutor() defaults to min(32, cpu+4) workers, i.e. up to 32
    # whole parquet files decompressed into RAM simultaneously. Cap it.
    with ThreadPoolExecutor(max_workers=min(8, len(file_paths))) as ex:
        dfs_iterator = ex.map(lambda ps: _safe_load_parquet(ps[0], ps[1]),
                              [(p, Path(p).parent.name) for p in file_paths])
        return _concat_union(list(dfs_iterator))


def _append_parquet_to_csv(file_paths: list, output_csv: Path):
    first_file = True
    for path in file_paths:
        try:
            sample_name = Path(path).parent.name
            df = load_df_parquet(path)
            if df.empty:
                continue

            df = df.copy()
            if "sample name" not in df.columns:
                df["sample name"] = sample_name

            df.to_csv(output_csv, mode='a', header=first_file, index=False)
            first_file = False

            del df
            gc.collect()
        except Exception as e:
            print(f"Skipping corrupted file during CSV append: {path} - Error: {e}")


def run_concat_stage():
    print("[CONCAT] Starting concat stage for all methods...")
    basename = MASTER_DIR.name

    # v5: the long (cell x mass) CSV is 1-2 GB, takes minutes to write to M: and nothing in
    # this script reads it - only on request
    if WRITE_PIXEL_LONG_TABLE:
        try:
            print("[CONCAT] Streaming massive pixel arrays directly to CSV...")
            pixel_files = [os.path.join(root, "pixel_intensities.parquet") for root, _, files in os.walk(MASTER_DIR) if
                           "pixel_intensities.parquet" in files]
            if pixel_files:
                output_csv = MASTER_DIR / f"{basename}_concatenated_mask_id_pixels.csv"
                if os.path.exists(output_csv):
                    os.remove(output_csv)
                _append_parquet_to_csv(pixel_files, output_csv)
        except Exception as e:
            print(f"Error streaming massive pixel save: {e}")
    else:
        print("[CONCAT] WRITE_PIXEL_LONG_TABLE is off - skipping the long pixel CSV.")

    # notes: Non-salted iteration so we only process raw cubes.
    for method in ["raw"]:
        df_avg = concat_parquet_files(MASTER_DIR, f"mask_id_averages_{method}.parquet")
        df_sum = concat_parquet_files(MASTER_DIR, f"mask_id_sums_{method}.parquet")
        df_sem = concat_parquet_files(MASTER_DIR, f"mask_id_sems_{method}.parquet")

        save_df_parquet(df_avg, str(MASTER_DIR / f"{basename}_concatenated_averages_{method}.parquet"))
        save_df_parquet(df_sum, str(MASTER_DIR / f"{basename}_concatenated_sums_{method}.parquet"))

        if method == "raw":
            save_df_parquet(df_avg, str(MASTER_DIR / f"{basename}_concatenated_mask_id_averages.parquet"))
            df_avg.to_csv(str(MASTER_DIR / f"{basename}_concatenated_mask_id_averages.csv"), index=False, na_rep='0')
            save_df_parquet(df_sum, str(MASTER_DIR / f"{basename}_concatenated_mask_id_sums.parquet"))
            # per-cell standard errors: Bonsai's measurement-noise input
            save_df_parquet(df_sem, str(MASTER_DIR / f"{basename}_concatenated_mask_id_sems.parquet"))

    print("[CONCAT] Combined files saved.")
