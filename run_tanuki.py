"""Run the pipeline from a checkout without installing it:

    python run_tanuki.py --input_dir "D:\\data\\GBA run"

(same options as the tanuki command; pip install -e . gives you that command instead)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from tanuki.main import main  # noqa: E402

# the guard matters: the stage 1 / stage 3 worker processes re-import this file on Windows
if __name__ == "__main__":
    main()
