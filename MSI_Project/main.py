import time
import argparse
from pathlib import Path

# Import config FIRST so we can overwrite its variables before other modules load them
import config

if __name__ == "__main__":
    # --- CLI Argument Setup ---
    parser = argparse.ArgumentParser(description="Multimodal Spatial Imaging (MSI) Processing Pipeline")
    parser.add_argument(
        "--input_dir", 
        type=str, 
        help="Path to the master directory containing sample subfolders. Defaults to ./data/ALL THE GBA_v2"
    )
    parser.add_argument(
        "--csv_path", 
        type=str, 
        help="Path to the masses CSV file. Defaults to ./data/BIGmatched_peaks_raw.csv"
    )
    
    args = parser.parse_args()

    # --- Override Config Paths if CLI arguments are provided ---
    if args.input_dir:
        config.MASTER_DIR = Path(args.input_dir)
    if args.csv_path:
        config.CSV_PATH = Path(args.csv_path)

    print(f"📂 Master Directory: {config.MASTER_DIR}")
    print(f"📄 CSV Path: {config.CSV_PATH}")
    
    # --- Lazy import pipeline stages to ensure they use the updated config ---
    from stage1_preprocess import run_stage1_preprocess_parallel
    from stage2_segmentation import run_stage2_cellpose_serial
    from stage3_features import run_stage3_postprocess_parallel
    from stage4_concat import run_concat_stage
    from analysis import run_post_analysis, custom_colors, LEGEND_ORDER
    from dash_viewer import run_dash_interactive_viewer

    t0 = time.time()
    
    print("\n🚀 Starting MSI Pipeline...")
    
    run_stage1_preprocess_parallel()
    run_stage2_cellpose_serial()
    run_stage3_postprocess_parallel()
    run_concat_stage()
    
    print("\n📊 Pipeline Complete. Starting Analysis...")
    final_df = run_post_analysis()
    
    print(f"\n✅ Total elapsed time: {time.time() - t0:.1f} seconds")
    
    print("\n🌐 Launching Interactive Dashboard...")
    run_dash_interactive_viewer(
        df=final_df, 
        default_color_col="Individual Sample Type",
        color_map=custom_colors,
        legend_order=LEGEND_ORDER,
        master_dir=config.MASTER_DIR,
        port=8050
    )
