<img width="300" height="400" alt="Tanuki" src="https://github.com/user-attachments/assets/94f1d7ac-7f63-4cfd-8298-425bb9787b90" />

# Tanuki

**Github repository of Tanuki for high throughput single cell MSI and microscopy data processing and analysis.**

## Multimodal Spatial Imaging (MSI) Processing Pipeline

This repository contains a high-performance, parallelized Python pipeline for processing, segmenting, and analyzing Multimodal Spatial Imaging (MSI) and imzML mass spectrometry data. 

The pipeline handles rigid and non-rigid image registration, deep-learning-based cellular segmentation (via Cellpose on GPU), morphological feature extraction, massively parallel spectral intensity mapping, and advanced dimensionality reduction (UMAP, DensMAP, t-SNE, NMF). It concludes by launching a local interactive Dash application for single-cell spectral inspection.

## 📂 Project Structure

```text
my_msi_project/
│
├── main.py                     # CLI Entry point to run the entire pipeline
├── config.py                   # Environment variables and default path configurations
├── utils.py                    # Shared I/O and mathematical helper functions
├── stage1_preprocess.py        # Image registration (SimpleITK) and imzML parsing
├── stage2_segmentation.py      # GPU-accelerated Cellpose segmentation
├── stage3_features.py          # Morphology and intensity extraction
├── stage4_concat.py            # Global dataframe merging and NPZ consolidation
├── analysis.py                 # Dimensionality reduction, clustering, and thumbnail generation
├── dash_viewer.py              # Interactive Dash dashboard for visual analysis
├── requirements.txt            # Python package dependencies
└── README.md                   # Project documentation
```

## ⚙️ Requirements & Installation

1. **Python:** Requires Python 3.9 or higher.
2. **GPU Support:** For Stage 2 (Cellpose), an NVIDIA GPU with CUDA support is highly recommended to prevent severe bottlenecks.
3. **Dependencies:** Install the required Python packages using `pip`:

```bash
pip install -r requirements.txt
```

*Note: You may need to install PyTorch separately depending on your specific CUDA version. Visit the [PyTorch website](https://pytorch.org/get-started/locally/) for the correct installation command for your system.*

## 🚀 Usage

The pipeline is designed to run automatically from end to end using `main.py`. You can point the script to your data directories using command-line arguments.

### Preparing Your Data
1. **Master Directory:** This folder should contain subdirectories for each of your samples. Each sample folder must contain the pre/post `.tif` images and the corresponding `.imzML` data.
2. **Masses CSV:** A `.csv` file containing the specific *m/z* values you want to extract.

### Running the Pipeline

Open your terminal or command prompt and run:

```bash
python main.py --input_dir "path/to/your/master_directory" --csv_path "path/to/your/masses.csv"
```

**Example:**
```bash
python main.py --input_dir "F:\Mika\GBA_data" --csv_path "F:\Mika\matched_peaks.csv"
```

If you run `python main.py` without arguments, it will fall back to the default paths specified in `config.py` (which default to a local `./data/` folder).

## 🧠 Pipeline Stages Explained

* **Stage 1 (Preprocessing):** Uses multiprocessing to perform Euler2D rigid and B-Spline non-rigid registration on pre/post MSI `.tif` images via SimpleITK. It then parses the massive `.imzML` files, normalizes against Total Ion Current (TIC), and safely resizes memory-heavy arrays.
* **Stage 2 (Segmentation):** Serializes processing to the GPU, using a pre-loaded PyTorch Cellpose model to generate highly accurate cellular masks from the aligned images.
* **Stage 3 (Feature Extraction):** Uses parallel processing to compute region properties (area, eccentricity, skeletonization metrics) and extracts mean/sum pixel intensities for all targeted *m/z* channels across every identified cell mask.
* **Stage 4 (Concatenation):** Safely merges thousands of `.npz` arrays into master pandas DataFrames, handling duplicate columns and memory limits gracefully.
* **Analysis & Interactive Viewer:** Performs robust scaling, AlignedUMAP (combining spectral and morphological spaces), Leiden clustering, and NMF. Finally, it generates single-cell BGR thumbnails and hosts a local Plotly Dash web app (`http://127.0.0.1:8050`) to interactively explore the spatial and spectral data.

## ⚠️ Troubleshooting & Memory Limits

This pipeline handles extremely large arrays (often >5GB per sample in memory). 
* Ensure your system has adequate RAM (64GB+ recommended for large datasets).
* The script heavily utilizes garbage collection (`gc.collect()`) and in-place operations to prevent Out-Of-Memory (OOM) errors.
* OpenCV's max image pixel limit is overridden in `config.py`. 
* If a folder fails during Stage 1 or 3, the script automatically catches it and requeues it using a safe single-threaded fallback mode.
