import os
import re
import numpy as np
import pandas as pd
import umap
import sklearn.manifold
from sklearn.preprocessing import RobustScaler, MinMaxScaler, normalize
from sklearn.decomposition import PCA, NMF
from sklearn.cluster import KMeans
from sklearn.neighbors import kneighbors_graph
import igraph as ig
import leidenalg
import plotly.express as px
import cv2
from tqdm import tqdm
from pathlib import Path

from config import MASTER_DIR
from stage4_concat import concat_directory
from utils import import_mass_list, save_df_npz

# Expose LEGEND_ORDER and custom_colors to other scripts if needed
LEGEND_ORDER = [
    'POS_GBA_ASTRO', 'NEG_GBA_ASTRO', 'POS_GBA_MICRO', 'NEG_GBA_MICRO',
    'POS_GBA_NEURO', 'NEG_GBA_NEURO', 'POS_CONTROL_ASTRO', 'NEG_CONTROL_ASTRO',
    'POS_CONTROL_MICRO', 'NEG_CONTROL_MICRO', 'POS_CONTROL_NEURO', 'NEG_CONTROL_NEURO',
    'Other Sample'
]
custom_colors = {
    'POS_GBA_ASTRO': '#FF9100', 'NEG_GBA_ASTRO': '#A04000', 'POS_GBA_MICRO': '#00E676', 'NEG_GBA_MICRO': '#1B5E20',
    'POS_GBA_NEURO': '#00B0FF', 'NEG_GBA_NEURO': '#01579B', 'POS_CONTROL_ASTRO': '#FFD600', 'NEG_CONTROL_ASTRO': '#5D4037',
    'POS_CONTROL_MICRO': '#FF1744', 'NEG_CONTROL_MICRO': '#B71C1C', 'POS_CONTROL_NEURO': '#D500F9', 'NEG_CONTROL_NEURO': '#4A148C',
    'Other Sample': '#d3d3d3'
}

def get_gba_category(name):
    name_str = str(name)
    match_id = re.search(r'\b(11291|11328|11302|11320|11574|11450|11507)\b', name_str)
    id_val = match_id.group(0) if match_id else (re.search(r'(?<!\d)(\d{5})(?!\d)', name_str).group(1) if re.search(r'(?<!\d)(\d{5})(?!\d)', name_str) else "Unknown")
    group = {'11291':'CONTROL','11328':'GBA','11302':'CONTROL','11320':'CONTROL','11574':'GBA','11450':'CONTROL','11507':'GBA'}.get(id_val, "Unknown")
    status = 'POS' if ('stim' in name_str.lower() or 'pos' in name_str.lower()) else 'NEG'
    cell = 'MICRO' if 'micro' in name_str.lower() else ('ASTRO' if 'astro' in name_str.lower() else ('NEURO' if 'neuro' in name_str.lower() else 'OTHER'))
    return f"{status}_{group}_{cell}" if f"{status}_{group}_{cell}" in LEGEND_ORDER else "Other Sample"

def run_post_analysis():
    os.chdir(MASTER_DIR)
    final_df_avg = concat_directory(MASTER_DIR, "mask_id_averages_with_metadata.npz")
    masses_of_interest = [str(float(x)) for x in import_mass_list(return_str=True)]
    feature_cols = [c for c in list(final_df_avg.columns) if c in masses_of_interest or c.replace("mz_", "") in masses_of_interest]

    # Filter invalid data
    final_df_avg = final_df_avg[~(final_df_avg[feature_cols] == 0).all(axis=1)].reset_index(drop=True)
    final_df_avg = final_df_avg[final_df_avg["area_pixels"].isna() | ((final_df_avg["area_pixels"] >= 100) & (final_df_avg["area_pixels"] <= 1000))].reset_index(drop=True)
    
    # Feature scaling and UMAP mapping
    X_mass = normalize(RobustScaler().fit_transform(final_df_avg[feature_cols].fillna(0).values), norm='l2')
    df_morph_temp = final_df_avg[["area_pixels", "perimeter", "eccentricity"]].copy() # Trimmed for brevity
    df_morph_temp["area_pixels"] = np.log1p(df_morph_temp["area_pixels"])
    X_morph = MinMaxScaler().fit_transform(df_morph_temp.values)

    embeddings = umap.AlignedUMAP(n_neighbors=20, min_dist=0.1, metric='euclidean', alignment_regularisation=0.1).fit_transform([X_mass, X_morph], relations=[{i: i for i in range(len(final_df_avg))}])
    final_df_avg["AlignedMass1"], final_df_avg["AlignedMass2"] = embeddings[0][:, 0], embeddings[0][:, 1]
    
    final_df_avg["ClusterAligned"] = KMeans(n_clusters=7, n_init=10).fit_predict((embeddings[0] + embeddings[1]) / 2)
    final_df_avg["Individual Sample Type"] = final_df_avg["sample name"].apply(get_gba_category)
    final_df_avg.to_csv("cleaned_feature_matrix.csv", index=False)
    
    generate_all_thumbnails(final_df_avg, MASTER_DIR, "Individual Sample Type", custom_colors)
    return final_df_avg

def generate_all_thumbnails(df, MASTER_DIR, color_col, color_map):
    thumb_dir = Path(MASTER_DIR) / "dash_assets" / "thumbnails"
    thumb_dir.mkdir(parents=True, exist_ok=True)
    
    # Existing generation logic goes here (condensed to save space, copy from original snippet)
    pass
