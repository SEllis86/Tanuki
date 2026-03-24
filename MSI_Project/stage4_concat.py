import os
import gc
import zipfile
import pandas as pd
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

from config import MASTER_DIR
from utils import load_df_npz, import_mass_list, save_df_npz

def _safe_load_npz(path, sample_name):
    try:
        df = load_df_npz(path)
        if not df.empty and "sample name" not in df.columns: df["sample name"] = sample_name
        return df
    except Exception: return pd.DataFrame()

def _concat_union(dfs: list) -> pd.DataFrame:
    return pd.concat([df.reindex(columns=sorted({col for df in dfs for col in df.columns})) for df in dfs if not df.empty], axis=0, ignore_index=True).reset_index(drop=True) if dfs else pd.DataFrame()

def concat_directory(master_dir: Path, filename: str) -> pd.DataFrame:
    paths = [(os.path.join(r, filename), Path(r).parent.name) for r, _, f in os.walk(master_dir) if filename in f]
    with ThreadPoolExecutor() as ex: return _concat_union(list(ex.map(lambda p: _safe_load_npz(p[0], p[1]), paths)))

def run_concat_stage():
    os.chdir(MASTER_DIR)
    basename = os.path.basename(os.getcwd())
    
    final_df_pixels = concat_directory(MASTER_DIR, "pixel_intensities.npz")
    try: final_df_pixels.to_csv(f"{basename}_concatenated_mask_id_pixels.csv", index=False)
    except Exception: pass
    del final_df_pixels; gc.collect()

    final_df_avg = concat_directory(MASTER_DIR, "mask_id_averages_with_metadata.npz")
    masses = [str(float(m)) for m in import_mass_list(return_str=True)]
    exist_mass = [m for m in masses if m in final_df_avg.columns]
    
    if exist_mass: final_df_avg = final_df_avg.loc[~(final_df_avg[exist_mass].fillna(0) == 0).all(axis=1)]
    try: save_df_npz(final_df_avg, f"{basename}_concatenated_mask_id_averages.npz")
    except Exception: pass
    final_df_avg.to_csv(f"{basename}_concatenated_mask_id_averages.csv", index=False, na_rep='0')

    final_df_sums = concat_directory(MASTER_DIR, "mask_id_sums_with_metadata.npz")
    exist_sum = [m for m in masses if m in final_df_sums.columns]
    if exist_sum: final_df_sums = final_df_sums.loc[~(final_df_sums[exist_sum] == 0).all(axis=1)]
    try: save_df_npz(final_df_sums, f"{basename}_concatenated_mask_id_sums.npz")
    except Exception: final_df_sums.to_csv(f"{basename}_concatenated_mask_id_sums.csv", index=False)
