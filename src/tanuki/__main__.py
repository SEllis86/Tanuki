"""python -m tanuki ... (same as the tanuki command)"""
from .main import main

# the guard matters: the stage 1 / stage 3 worker processes re-import this module on Windows
if __name__ == "__main__":
    main()
