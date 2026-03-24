import os
import multiprocessing as mp
from pathlib import Path

# =========================================================================
# CRITICAL ENVIRONMENT VARIABLES
# =========================================================================
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = "10000000000"

total_cores = mp.cpu_count()
usable_cores = max(1, total_cores - 2)
cores_str = str(usable_cores)

os.environ["OMP_NUM_THREADS"] = cores_str
os.environ["OPENBLAS_NUM_THREADS"] = cores_str
os.environ["MKL_NUM_THREADS"] = cores_str
os.environ["VECLIB_MAXIMUM_THREADS"] = cores_str
os.environ["NUMEXPR_NUM_THREADS"] = cores_str

# =========================================================================
# PATH SETTINGS (Defaults)
# =========================================================================
# Base directory of this script
BASE_DIR = Path(__file__).resolve().parent

# Default relative paths (can be overridden at runtime via main.py)
MASTER_DIR = BASE_DIR / "data" / "ALL THE GBA_v2"
CSV_PATH = BASE_DIR / "data" / "BIGmatched_peaks_raw.csv"
SKIP_FILENAME = 'imzml_square_0.png'

# =========================================================================
# PERFORMANCE SETTINGS
# =========================================================================
safe_ram_workers = 8
MAX_WORKERS_STAGE1 = min(safe_ram_workers, usable_cores)
MAX_WORKERS_STAGE3 = min(safe_ram_workers, usable_cores)
