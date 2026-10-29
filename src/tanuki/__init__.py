"""Tanuki: single-cell MALDI-MSI + optical microscopy pipeline for iPSC-derived neurons, astrocytes and
microglia. The package is tanuki2026IPSC_phasecorr_reg_v5.py split into modules; see README.md.

    tanuki --input_dir RUN_FOLDER          (or: python -m tanuki ...)
"""
__version__ = "5.0.0"

# First, before anything imports numpy / OpenCV / pyplot: sets the thread-count environment variables,
# the OpenCV image size limit and the Agg backend, and applies overrides handed down to worker processes.
from . import config  # noqa: E402,F401
