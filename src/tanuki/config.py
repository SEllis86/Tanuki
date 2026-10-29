"""Settings for the Tanuki pipeline: v5's USER SETTINGS block, unchanged except for the paths.

Every setting is an UPPER_CASE name below, with the reasoning behind its default. A run overrides
them without editing this file (see RUN-TIME OVERRIDES at the end):
    tanuki --input_dir RUN --csv_path MASSES.csv --config my_run.json --set COMBAT_ENABLED=true
    tanuki --show-config                                  (prints every setting and its value)

Importing this module also sets the thread-count environment variables, the OpenCV image size
limit and matplotlib's Agg backend. Those must be in place before numpy, OpenCV or pyplot are
imported, which is why the package's __init__ imports this module first.
"""
from __future__ import annotations

import json
import multiprocessing as mp
import os
import sys
import time
import warnings
from pathlib import Path

# v5: the emoji in the progress messages raise UnicodeEncodeError under the Windows cp1252
# console as soon as output is piped or redirected to a file; replace instead of crashing.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(errors="replace")
    except Exception:
        pass

# =========================================================================
# CRITICAL: SET ALL ENVIRONMENT VARIABLES *BEFORE* IMPORTING LIBRARIES
# =========================================================================
# Disable OpenCV's massive image protection limit
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = "10000000000"

# 1. Dynamically detect the total number of logical processors
total_cores = mp.cpu_count()
usable_cores = max(1, total_cores - 2)
cores_str = str(usable_cores)

# Force underlying linear algebra libraries to use the dynamic core count
# CRITICAL: Lock underlying math libraries to 1 thread to prevent NxN thread explosion
# when running our 8 independent multiprocessing workers.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
# =========================================================================

if mp.current_process().name == 'MainProcess':
    print(f"🖥️ System detected {total_cores} cores. Allocating {cores_str} threads for heavy math.")

import matplotlib

matplotlib.use("Agg")  # headless / parallel safe

t0 = time.time()
warnings.filterwarnings("ignore")

# ============================================================
# USER SETTINGS
# ============================================================

# The run folder: one sub-folder per well / acquisition (see README). Set it per run with --input_dir
# (or MASTER_DIR in a --config file); a relative path is taken from the current directory.
MASTER_DIR = Path('data')
# The mass list (first column = m/z; a 'name' column labels the volcano tables).
# None = <MASTER_DIR>/BIGmatched_peaks_raw.csv.
CSV_PATH = None
SKIP_FILENAME = 'imzml_square_0.png'

safe_ram_workers = 8

MAX_WORKERS_STAGE1 = min(safe_ram_workers, usable_cores)
MAX_WORKERS_STAGE3 = min(safe_ram_workers, usable_cores)

# FIX: every tuning constant that used to be a magic number buried mid-function now
# lives here, so the two area filters and the two registration thresholds can be seen
# to agree (or be deliberately made to disagree) at a glance.
RANDOM_SEED = 42

# ---- registration channel ----
# Which channel of the PRE and POST optical tiffs drives the alignment. None = pick the
# BRIGHTFIELD channel of each image by its histogram (bright background, dark objects:
# the median sits high in the 1-99 percentile range). On the GBA slides that is channel
# 0 in both scans, but the POST channel 0 is a dark matrix field (score ~0.25) while
# the PRE channel 0 is clean brightfield (0.66-0.91), and a folder saved with another
# channel order would silently break a fixed index. An int forces that channel.
#
# Only the ALIGNMENT uses this channel. The transform it finds is still applied to every
# channel of the moving image, so MovedMSI_multichannel.tif keeps all of them.
REGISTRATION_CHANNEL = None

# Registration quality, in MICROMETRES (see the PRE -> POST OPTICAL REGISTRATION section for how each
# is measured). A folder failing any of these is skipped, not cropped.
REG_MAX_PLACEMENT_UM = 2.5      # transform standard error, LOO / sqrt(n tiles); half a 5 um MSI pixel
REG_MAX_TILE_LOO_UM = 6.0       # per-tile leave-one-out residual; how well one tile localises
REG_MIN_TILES = 4               # fewer agreeing tiles than this and no rotation/scale is believed
REG_MARK_MAX_DIAMETER_UM = 700.0  # the ring marks around a GBA well are 500-650 um across

# Folders registered by the OLD MMI code have a PREMSI_stage1_ready.h5 and would be
# skipped by should_skip_subdir. True re-registers them (their QC file has no
# "Method: phase" line); folders done by THIS script are still skipped.
REREGISTER_OLD_MMI_FOLDERS = True

# v5: stage 1 h5 format. 2 = imgs_cp kept at 16 bit (format 1, the 8-bit imgs_cp scaled by
# one shared max, left dim fluorescence channels with a handful of grey levels). Folders
# whose h5 is older are re-run by stage 1, and stage 2 refuses to segment them.
STAGE1_FORMAT_VERSION = 2
# Compression of the stage 1 h5 cube. 'lzf' writes several times faster than 'gzip' for a
# somewhat larger file; the values are identical either way. None = uncompressed.
H5_COMPRESSION = 'lzf'
# v5 stage 3: masses read from the cube per slab (memory ~ slab x pixels x 16 bytes per worker)
STAGE3_SLAB_MASSES = 64
# v5 stage 3: standard error over the effective number of independent pixels (cell area /
# upsampled pixels per native MSI pixel, at least 1) instead of over the upsampled pixels.
# False = the old, ~15x too small SEM.
SEM_EFFECTIVE_N = True
# v5: write the long one-row-per-(cell, mass) table (pixel_intensities.parquet per folder and
# the 1-2 GB *_concatenated_mask_id_pixels.csv). Nothing in this script reads it.
WRITE_PIXEL_LONG_TABLE = False


# Cell area gates (pixels).
MIN_CELL_AREA_PX = 50           # stage 3: rows below this are never written out
# Folders whose name contains any STAGE3_NEURO_MIN_AREA_KEYWORDS substring (case-insensitive)
# use NEURO_MIN_CELL_AREA_PX at the stage 3 gate instead of MIN_CELL_AREA_PX: a neurite sliver
# under this size is noise, not a cell. v3.3: this has its OWN keyword list. It used to borrow
# the merge skip list, so changing which folders are merged silently moved this gate too.
STAGE3_NEURO_MIN_AREA_KEYWORDS = ('neuro',)
NEURO_MIN_CELL_AREA_PX = 200
ANALYSIS_AREA_MIN_PX = 100      # post-concat analysis filter: default for cell types not listed below
ANALYSIS_AREA_MAX_PX = 2000
# v3: post-concat (min, max) per cell type, in segmentation pixels (~1.3 um/px, so
# 2000 px ~ 3,400 um^2, a disc ~66 um across). None = no bound on that side. The cell
# type comes from cell_type_of() (MICRO / ASTRO / NEURO / OTHER); unlisted types use the
# two defaults above. One 100-2000 px window for everything dropped 60% of the astrocytes:
# flat astrocytes exceed 2000 px routinely, and merge_touching_masks assembles them from
# fragments of up to MERGE_TOUCHING_MASKS_MAX_AREA_PX each. The astro cap therefore starts
# OPEN. The filter prints each type's area percentiles and how many cells each bound
# removed; set a cap from the 95th/99th percentile once the largest astro masks have been
# checked for clumps in the Dash viewer (colour by area_pixels, click the outliers).
ANALYSIS_AREA_BOUNDS_BY_TYPE = {
    'ASTRO': (100, 13000),  # tighten after looking at the largest masks
    'MICRO': (100, 3500),
    'NEURO': (200, 2000),
}

# Feature QC.
ZERO_RATE_THRESHOLD = 0.5       # drop an m/z whose zero rate exceeds this
# v3: judge that zero rate WITHIN each cell type and keep a feature if ANY type passes.
# Pooled over a matrix that is ~80% neurons, a peak in every astrocyte but in no neuron
# reads as ~90% zeros and was dropped, so the astro-specific chemistry never reached the
# embeddings, NMF, Bonsai or the mean spectra. False = the old pooled rule.
ZERO_RATE_PER_CELL_TYPE = True
# A batch with only one SAMPLE cannot have its batch effect estimated, so it normally
# aborts the run as a naming error. Set True when the singletons are the known result of
# upstream failures and you want to analyse what did survive.
ALLOW_SINGLETON_BATCHES = True
# v3: the column ComBat treats as the batch.
#   'Slide' - the S#B# token in the folder name (S1B2, S2B1, ...): the acquisition batch.
#   'Batch' - the old cell type + cell line label. With the Individual Sample Type
#             covariate (group + cell type) this is confounded whenever the lines share a
#             group (all CONTROL on the GBA test set): inmoose refused with "Covariates
#             Cov_0, Cov_1 are confounded with the batches" and nothing was corrected.
# Never the cell line on its own: once GBA lines are in a run the disease group is fixed
# per line, so ComBat would remove the GBA effect itself. 'Batch' is still the label used
# for colouring, the singleton naming check and the Bonsai annotation.
COMBAT_BATCH_COL = 'Slide'
# v5: switch for the batch correction itself. On the GBA test set every cell line of the
# astrocytes and microglia sits on exactly ONE slide (and the 11302 neurons on S3B1), so a
# slide correction cannot tell a slide effect from a cell-line effect and removes both.
#   False - no correction. The "corrected" embeddings are computed on the raw values and every
#           plot says 'not batch-corrected'. Compare lines and cell types on these.
#   True  - ComBat by COMBAT_BATCH_COL. Fine when every slide carries the groups you compare;
#           the design table printed before it runs says which groups are nested in one slide.
COMBAT_ENABLED = False
# Label the plots use for the second DensMAP; set from what actually happened.
COMBAT_LABEL = 'PyComBat corrected'
N_CLUSTERS = 7

# ---- POS/NEG (stim) separation ----
# 'Individual Sample Type' is normally <line group>_<cell type>_<stim status>, e.g.
# GBA_ASTRO_POS vs GBA_ASTRO_NEG, where POS means the folder name contains 'stim' or
# 'pos'. Set False to DROP the stim status from the label entirely, so those two collapse
# into one GBA_ASTRO group everywhere the label is used: the legend, every plot's colour
# map, the mean-spectrum panels, the ComBat covariate, and the Bonsai stratification.
#
# Turning this off HALVES the number of groups, which also means ComBat's covariate has
# half as many levels - that can matter, since a covariate confounded with batch is what
# made ComBat fail on this dataset before. Nothing about the cells themselves changes;
# only how they are labelled and grouped.
SPLIT_POS_NEG = False

# Fiducial square search.
SQUARE_MIN_SIDE_PX = 700
SQUARE_MAX_SIDE_PX = 1500
SQUARE_MAX_SIDE_DIFF_PX = 200
SQUARE_MIN_EXTENT = 0.85

# ---- fiducial square vs imzML cross-check ----
# The imzML header states the MSI grid (pixels) and pixel size (um); the POST TIFF states
# its own um/px. Together they predict how big the ablated square MUST be in optical
# pixels before any search runs. Candidates whose side deviates from that prediction by
# more than SQUARE_MSI_SIZE_TOL are rejected as false positives (a well edge, a gasket
# corner, a stitching tile), and the found-vs-expected comparison is written to
# QC_square_size.txt and the stage 1 h5 for every folder. The gate is skipped (QC only)
# when the TIFF pixel size fell back to DEFAULT_UM_PER_PX, since the prediction is then a
# guess too. Folders processed before this check existed are back-filled at the end of
# stage 1 from their h5 attributes.
SQUARE_USE_MSI_SIZE_GATE = True
SQUARE_MSI_SIZE_TOL = 0.10           # fraction of the expected side

# ---- registration overlap audit ----
# PREMSI_stage1_audit_registration_overlap.png, written per folder right after the
# optical registration: the fixed POST scan in magenta over the registered PRE scan in
# green (overlap = white), a checkerboard of the two, and their absolute difference, all
# on the alignment channel. Folders
# that already have their registered tiffs on disk are back-filled at the end of stage 1
# from FixedMSI.tif / MovedMSI_multichannel.tif, so this needs no re-registration.
REG_AUDIT_ENABLED = True
REG_AUDIT_MAX_PANEL_PX = 1600        # each full-frame panel is shrunk to at most this wide
REG_AUDIT_CHECKER_TILES = 8          # checkerboard tiles across the shorter side

# ---- morphology ----
# The shape descriptors only mean anything when a cell spans enough pixels to HAVE a
# shape. Set False to drop perimeter/extent/eccentricity/axis lengths/solidity and the
# whole skeleton family from the parquet files and from the DensMAP input, leaving the
# embedding driven by mass channels alone. mask_id, area_pixels, location and the
# centroids are ALWAYS kept: the area gate, the Dash cell-crop viewer and the edge/middle
# call all need them. MIN_CELL_AREA_PX is a size gate, not a feature, and still applies.
USE_MORPHOLOGY_FEATURES = True
FEATURE_TAG = 'mass_plus_morph' if USE_MORPHOLOGY_FEATURES else 'mass_only'
FEATURE_LABEL = 'mass+morph' if USE_MORPHOLOGY_FEATURES else 'mass only'

# ---- pre-segmentation smoothing ----
# Equivalent of ImageJ's Gaussian Blur; modern ImageJ takes SIGMA in pixels, so a
# '3 pixel gaussian' is sigma = 3.0 (the pre-1.38 'radius' sense would be 1.2).
# This touches ONLY the image Cellpose segments on. raw_cube, which every intensity in
# the feature matrix comes from, is never smoothed.
CELLPOSE_GAUSSIAN_SIGMA = 0.0          # 0 = off
CELLPOSE_GAUSSIAN_TRUNCATE = 3.5       # ImageJ truncates at ~3.5 sigma for 8/16-bit
# Percentile window for the 8-bit stretch of the Cellpose input. A sparse ion image
# stretched 0-99.99 collapses to near-binary speckle; smoothing first largely fixes it.
CELLPOSE_STRETCH_PERCENTILES = (0.0, 99.99)

# ---- merge over-segmented masks ----
# Cellpose sometimes segments one continuous cell as a chain of small masks along its
# length (a "candy stripe" of different colours in PREMSI_stage2_audit_segmentation.png),
# because the flow field for a long thin process is locally ambiguous. merge_touching_masks
# unions touching mask IDs right after Cellpose returns, so the merge is baked into
# PREMSI_seg.npy and every later stage sees the merged object.
#
# v3.3: the merge runs on NEURON folders ONLY (MERGE_TOUCHING_MASKS_ONLY_KEYWORDS). v2 had
# this the wrong way round: it merged astro and micro and skipped neurons, which joined a
# quarter of the microglia and astrocyte masks (602 and 363 on the GBA set), made the
# objects large enough to hit the analysis area cap, and left the neuron stripes alone.
#   MERGE_TOUCHING_MASKS_ONLY_KEYWORDS  merge only folders whose name contains one of these
#                                       (case-insensitive). Empty tuple = every folder.
#   MERGE_TOUCHING_MASKS_SKIP_KEYWORDS  never merge folders containing one of these (wins
#                                       over ONLY). Empty tuple = no exclusions.
# On a dense neuron network a chain of touching fragments can run away and fuse separate
# neurons. Two guards limit that: the per-fragment area cap and MERGE_MIN_CONTACT_FRAC
# below. The [S2] merge_touching_masks line reports how many were merged per well; check
# the audit image of a dense well before trusting it.
MERGE_TOUCHING_MASKS = True
MERGE_TOUCHING_MASKS_ONLY_KEYWORDS = ('neuro',)
MERGE_TOUCHING_MASKS_SKIP_KEYWORDS = ()
# A mask already this large is very unlikely to be a stray over-segmented sliver - it's
# probably a real, correctly-segmented cell body already - so it is left out of every
# merge rather than risking two genuine neighbouring cells being fused into one. Checked
# against each candidate mask's OWN area (in segmentation/upscaled pixels, before any
# merge), not the running total of a chain, so this caps how big a single fragment can be
# to still be considered a fragment - it does not cap the final size of a merged object.
MERGE_TOUCHING_MASKS_MAX_AREA_PX = 1500  # None = no cap, merge regardless of size
# v2: two touching fragments merge only if their shared border is at least this fraction
# of the SMALLER fragment's perimeter. Two neighbouring cells touch along a short arc
# (< 0.15). v3.4 correction: a fragment that shares its whole width w with its neighbour
# and is L px long scores w / (2 (w + L)), at most 0.5, so 0.30 only joins fragments that
# are SHORTER than about two-thirds of their width (true candy-stripe bands). A fragment
# longer than it is wide scores 0.1-0.25 and is not joined. Lower this (e.g. 0.20) to
# join longer pieces; MERGE_MAX_MERGED_AREA_PX now bounds the result. None = off.
MERGE_MIN_CONTACT_FRAC = 0.30
# v4: how a touching pair is judged.
#   'linear'  - a merge must extend the object, not widen it (the MERGE_LINEAR_* settings
#               below). MERGE_MIN_CONTACT_FRAC is not used.
#   'contact' - the v2/v3 rule above. It rewards side-by-side contact (a long shared edge
#               is a big fraction of the perimeter) and rejects end-to-end contact between
#               pieces longer than they are wide, so on neuron wells it grew wide clouds
#               (one mask covered 86% of A3_Neurons_11302) instead of joining neurites.
MERGE_RULE = 'linear'
# v4 'linear': each object's length and width are those of the rectangle with the same
# pixel spread (second moments), so they are exact for chains of fragments as well and
# cheap to update on every merge. A pair is joined only if
#   1. their shared border is at least MERGE_LINEAR_MIN_CONTACT_OF_WIDTH of the NARROWER
#      piece's width (they abut across their width, not at a corner), and
#   2. the joined object is at most MERGE_LINEAR_MAX_WIDTH_GAIN times as wide as the wider
#      piece, and no wider than MERGE_LINEAR_MAX_WIDTH_PX (unless that piece already was,
#      in which case it may not get wider at all).
# End-to-end pieces leave the width unchanged and pass; side-by-side pieces roughly double
# it and fail, and a cloud cannot form because every step that widens it is refused. The
# pairs that widen the object least are joined first; MERGE_MAX_MERGED_AREA_PX still applies.
# Scale (segmentation px, ~1.3 um/px): 90% of unmerged Cellpose neuron masks on the GBA set
# are narrower than 33 px; the merged clouds over 2000 px had a median width of 46 px.
MERGE_LINEAR_MIN_CONTACT_OF_WIDTH = 0.5
MERGE_LINEAR_MAX_WIDTH_GAIN = 1.25
MERGE_LINEAR_MAX_WIDTH_PX = 35
# v3.4: a cap on the size of the MERGED object, not just of each fragment. The fragment cap
# above stops a big mask from joining, but a chain of small touching fragments could still
# grow without limit: on a confluent well every fragment touches the next and the whole
# field came back as one mask. Pairs are now merged strongest-contact first, and a merge
# that would make the combined object larger than this is refused. 2000 px matches the
# NEURO analysis cap (ANALYSIS_AREA_BOUNDS_BY_TYPE), so a merge never builds an object the
# area filter then throws away; raise the two together. None = no cap (v3.3 behaviour).
MERGE_MAX_MERGED_AREA_PX = 2000
# v3.3: the merge is baked into PREMSI_seg.npy, so a change to the merge settings only takes
# effect when Cellpose runs again. Each seg file now records the merge rule it was made with;
# True = a folder whose recorded rule differs from the current settings (or that has none,
# i.e. was segmented before v3.3) is re-segmented automatically, and only those folders.
# The FIRST v3.3 run therefore re-segments everything once (about 2 h on the GBA set).
# False = keep whatever masks are on disk.
STAGE2_RESEGMENT_IF_MERGE_RULE_CHANGED = True
# v4: PREMSI_seg.npy also stores the UNMERGED Cellpose masks ('masks_raw'). True = a folder
# whose merge rule changed is re-merged from those (plus the MSI rescue, which depends on
# the merge) in seconds instead of re-running Cellpose. Seg files from before v4 have no
# 'masks_raw' and are still re-segmented.
STAGE2_REMERGE_FROM_RAW = True

# ---- stage 2 segmentation input (v2) ----
# Which optical channels Cellpose is shown. 'all' = the 3-channel stack (brightfield
# inverted + the two fluorescence channels); 0/1/2 = that single channel repeated x3.
# Flat cells that are invisible in brightfield still have a stained membrane, so 'all'
# recovers most of the unmasked bright MSI blobs seen in the audit images.
CELLPOSE_INPUT_CHANNELS = 'all'
CELLPOSE_BRIGHTFIELD_CHANNEL = 0        # index of the brightfield channel (inverted); None = none
# Each channel is stretched to its own percentile window before stacking, so a bright
# Hoechst channel no longer flattens the membrane channel to black.
CELLPOSE_CHANNEL_STRETCH_PERCENTILES = (1.0, 99.9)
# Top-hat removes the stitching grid from the fluorescence channels but also removes
# anything LARGER than its kernel. 201 px (260 um) clipped big astrocytes; 401 keeps
# them and still kills the grid. None = no top-hat.
CELLPOSE_TOPHAT_KERNEL_PX = 401
# Second Cellpose pass with the (aligned) 760 ion image as the third channel, so the
# model can complete faint optical outlines from MSI signal. Roughly doubles stage 2
# time and blurrier input can merge neighbours - off until 1 + rescue prove insufficient.
CELLPOSE_MSI_CHANNEL = False
CELLPOSE_MSI_CHANNEL_BLUR_MSI_PX = 1.0  # Gaussian sigma in MSI pixels (softens the blocky upscale)

# ---- MSI-guided rescue (v2) ----
# After segmentation + alignment, Otsu-threshold the 760 ion image; any blob of at least
# MSI_RESCUE_MIN_AREA_MSI_PX native MSI pixels whose area is covered less than
# MSI_RESCUE_MAX_OVERLAP by existing masks becomes a new mask (its uncovered part),
# tagged 'msi_rescue'. Rescued outlines are at MSI resolution (blocky, oversized) and a
# matrix hotspot can be rescued as a cell - filter on mask_source downstream if needed.
MSI_RESCUE_ENABLED = True
MSI_RESCUE_MIN_AREA_MSI_PX = 2.0
MSI_RESCUE_MAX_OVERLAP = 0.30
MSI_RESCUE_OPEN_PX = 3                  # morphological opening on the MSI binary; 0 = off

# ---- stage 2 trial controls (v2) ----
# Non-empty list = stage 2 runs only on these folder names. STAGE2_FORCE_RESEGMENT = True
# re-runs Cellpose even where PREMSI_seg.npy exists (overwriting it). Stage 3 always
# re-measures, so a trial on a few wells flows through to the matrix.
STAGE2_ONLY_FOLDERS = []
# v3: False. The v2 masks are already on disk; True re-runs Cellpose on every folder
# (115 min on the GBA set) even when only the post-concat analysis changed.
STAGE2_FORCE_RESEGMENT = False

# ---- stage 2 audit image (v3.1) ----
# Display colour (R, G, B, each 0-1) of each Cellpose input channel in the TOP panel of
# PREMSI_stage2_audit_segmentation.png. The panel used to show the model input as a plain
# RGB image, which painted the brightfield channel red. Now channel 0 (brightfield,
# inverted) is grey, channel 1 green, channel 2 blue. Display only: Cellpose still gets
# the same 3-channel stack, and imgs_for_model_stage2_audit_segmentation.png still shows
# that stack as-is.
STAGE2_AUDIT_CHANNEL_COLORS = {0: (1.0, 1.0, 1.0), 1: (0.0, 1.0, 0.0), 2: (0.0, 0.0, 1.0)}
# v3.2: show the brightfield channel UN-inverted in that panel (bright field, dark cells,
# as it looks down the microscope), with the fluorescence channels blended over it.
# Cellpose is unaffected: it has always been given the brightfield channel INVERTED
# (CELLPOSE_BRIGHTFIELD_CHANNEL above; set that to None to stop inverting for the model).
STAGE2_AUDIT_BRIGHTFIELD_UNINVERTED = True
# True = folders that already have PREMSI_seg.npy (and are therefore skipped by stage 2)
# get their audit image REDRAWN from disk, without running Cellpose: about a second or two
# per folder. Set False once the audits are up to date.
STAGE2_REDRAW_AUDIT = True

# ---- MSI <-> optical orientation (v2) ----
# Stage 1 rotates the imzML array by MSI_ROT90_K quarter turns (np.rot90 k; -1 = 90 deg
# clockwise, the historical value) before resizing it onto the fiducial square. Stage 2
# then tries the four 90-degree rotations of the OPTICAL layers against the ion image and
# keeps the best overlap. On the GBA set (2026-09-17) every well where that overlap was
# decisive chose cv2 code 0 (90 deg clockwise), i.e. the historical k=-1 is off by one
# quarter turn for this scanner; on neuron wells the four scores are within a few percent
# of each other, so the search picked codes at random and 12 wells came out mis-rotated.
#
# STAGE2_ROTATION_MODE:
#   'search'  - old behaviour, pick the best of four by overlap (unsafe on dense wells)
#   'none'    - never rotate the optical layers; translation search only
#   0 / 1 / 2 - always apply this cv2 rotation code (0 = ROTATE_90_CLOCKWISE, 1 = ROTATE_180,
#               2 = ROTATE_90_COUNTERCLOCKWISE) to the optical layers; translation search only
# With stage 1 output already on disk (k=-1), use 0. If you re-run stage 1 with
# MSI_ROT90_K = 0 instead, the ion image is already in the optical frame and this should
# be 'none'. Either way, one consistent answer for every well.
MSI_ROT90_K = -1
STAGE2_ROTATION_MODE = 0

# ---- per-group mean spectra ----
# Mean spectrum across the full mass list, one panel per label, from the FINAL
# single-cell matrix. 'raw' uses the pre-ComBat intensities (raw_intensities_df): after
# correction the values have been through log1p -> correct -> expm1 and are no longer
# on an intensity scale, so 'corrected' is a QC view, not a figure.
MEAN_SPECTRUM_ENABLED = True
# Every grouping gets its own stacked mean-spectrum figure. Cluster columns come from the
# different clustering methods (missing ones are skipped, e.g. ClusterBonsai without a
# Bonsai result). 'sample name' = one panel per folder; 'Well' = one per well position.
MEAN_SPECTRUM_GROUPS = ['sample name', 'Well', 'Individual Sample Type', 'Batch', 'Slide', 'CellType',
                        'location', 'Stim', 'Sample Type + Stim',
                        'ClusterUnifiedDens', 'KmeansClusterUMAP', 'KmeansClusterTSNE',
                        'KmeansClusterPCA', 'ClusterNMF', 'ClusterLeiden', 'ClusterBonsai',
                        'ClusterMorphOnly']
MEAN_SPECTRUM_SOURCE = 'raw'           # 'raw' | 'corrected' | 'both'
MEAN_SPECTRUM_ANNOTATE_TOP = 5         # label the N tallest peaks per panel; 0 = none
MEAN_SPECTRUM_LOG_Y = False

# Distinctive-peaks sidebar, printed to the right of each panel: the top N peaks that
# make this panel's group stand out from the others in the SAME figure, ranked UNIQUE
# first (only this group has any signal there), then - if fewer than N unique peaks
# exist - filled out with the next-highest-ratio DOMINANT peaks (this group's mean is at
# least MEAN_SPECTRUM_HIGHLIGHT_FOLD times every other group's, even though more than one
# group has some signal). Comparisons ignore a trailing _POS/_NEG on the group name, so
# GBA_ASTRO_POS and GBA_ASTRO_NEG are treated as the same group for this purpose and show
# the identical sidebar - stim status isn't what a "distinctive peak" should hinge on.
# Has no effect on groupings with no POS/NEG suffix (Batch, CellLine, ...): the suffix
# strip is then a no-op and each group is compared against every other as-is.
MEAN_SPECTRUM_HIGHLIGHT_TOP_N = 5
MEAN_SPECTRUM_HIGHLIGHT_FOLD = 2.0     # dominance threshold: >= this many times every other group
MEAN_SPECTRUM_HIGHLIGHT_FLOOR = 0.0     # mean intensity at/below this counts as "not detected"

# ---- mass list de-duplication ----
# Peaks within DEDUP_TOLERANCE_PPM are one peak. The TALLEST is kept and, when
# DEDUP_WIDEN_WINDOW is on, its extraction window is widened to span the whole overlap
# so the peaks it absorbed are integrated into it instead of being silently dropped.
DEDUP_MASS_LIST = False                # the CSV list is usually already de-duplicated
DEDUP_TOLERANCE_PPM = 15.0
DEDUP_WIDEN_WINDOW = True
MASS_WINDOW_PPM = 15.0                 # base half-window per mass

# ---- Bonsai (github.com/dhdegroot/Bonsai-data-representation) ----
# Bonsai reconstructs a maximum-likelihood TREE over the cells instead of a 2-D
# neighbour embedding, so it sits alongside UMAP / DensMAP / t-SNE rather than replacing
# them. It is not a library: it is cloned and run as a CLI in its own env. This pipeline
# always writes a ready-to-run input folder + config, optionally runs it, and draws the
# tree whenever a result exists.
BONSAI_ENABLED = True
BONSAI_REPO   = None   # a clone of Bonsai-data-representation; None = export the input only
BONSAI_PYTHON = None   # python executable of Bonsai's own environment; None = this interpreter
BONSAI_AUTO_RUN = True           # True = run bonsai_main.py from here (~1 h on the GBA set)
BONSAI_RESULTS_FOLDER = None     # None = MASTER_DIR/'bonsai'/'results'
BONSAI_MAX_CELLS = 4000          # subsample before export; None = all
# How that subsample is drawn. None = uniform random, which lets a large group crowd out
# a small one in proportion to their sizes.
#   'equal'        - same number per group, leftover budget handed back to groups that
#                    still have cells. Branch structure becomes comparable BETWEEN
#                    groups, but cluster SIZES no longer reflect true abundance.
#   'proportional' - keeps real proportions, with a floor so no group vanishes.
BONSAI_SAMPLE_STRATIFY = 'Individual Sample Type'   # None = uniform random
BONSAI_STRATIFY_MODE = 'equal'                      # 'equal' | 'proportional'
BONSAI_STRATIFY_MIN_PER_GROUP = 50
BONSAI_SAMPLE_SEED = 42
# Bonsai runs after ComBat, so the corrected values would otherwise be paired with noise
# derived from the RAW pixel counts. Bonsai weights its likelihood by that noise, so the
# two must describe the same numbers.
BONSAI_USE_RAW_INTENSITIES = True
BONSAI_ZSCORE_CUTOFF = 1.0       # Bonsai's own feature signal-to-noise filter
BONSAI_USE_KNN = 10
BONSAI_TIMEOUT_S = None          # subprocess timeout for the auto-run
# Bonsai's own logging, not this pipeline's. It prints a block per merge round and there
# is roughly one round per cell, so a few thousand cells emit tens of thousands of lines.
BONSAI_VERBOSE = False
# Where Bonsai's stdout/stderr goes. 'file' keeps the full log without flooding the
# console, 'console' is the old behaviour, 'quiet' discards it.
BONSAI_OUTPUT = 'file'           # 'file' | 'console' | 'quiet'

# ---- volcano plots + raw data export (2026-10-21) ----
# After the analysis the run also writes, from its own outputs:
#   MASTER_DIR/volcano/  volcano plots, PNG + interactive HTML + CSV per comparison: STIM vs NO_STIM within
#                        each cell type, and cell type vs cell type with both pooled. The replicate is the
#                        donor line: per-donor log2 ratios of the well means, paired by donor, moderated t.
#   MASTER_DIR/export/   the raw per-cell intensities of every measured cell (labels, morphology, clusters,
#                        embeddings) as CSV + parquet, per-well means, a per-m/z table and a README
# The work is done by tanuki.volcano and tanuki.export; both also run on their own on a finished run:
#   python -m tanuki.volcano <run folder>        python -m tanuki.export <run folder>
# False = skip that output.
VOLCANO_ENABLED = True
RAW_EXPORT_ENABLED = True
EXTRA_OUTPUT_PRESET = 'ipsc'   # comparisons + label columns: DATASETS in volcano.py / export.py

# m/z column naming. Stage 3 and the post-concat analysis MUST agree on this, otherwise
# detect_mass_columns silently finds nothing.
MZ_COL_PRECISION = 4



# ============================================================
# RUN-TIME OVERRIDES
# ============================================================
# v5 was configured by editing the settings above. The package keeps them as the defaults and
# lets a run override any of them (every UPPER_CASE name, plus safe_ram_workers) without
# touching this file:
#   tanuki --input_dir RUN --csv_path MASSES.csv --config my_run.json --set COMBAT_ENABLED=true
#   from tanuki import config; config.configure(MASTER_DIR=r"D:\run", BONSAI_ENABLED=False)
# Overrides must be in place BEFORE the stage modules are imported (they copy their settings at
# import, as v5's functions read its globals). The CLI does this for you.
#
# Stages 1 and 3 run in worker processes, which import this module afresh. apply_overrides
# therefore also writes the overrides to the environment variable TANUKI_CONFIG_JSON, and the
# block at the very end of this file re-applies them in every worker.
ENV_VAR = 'TANUKI_CONFIG_JSON'
_PATH_SETTINGS = ('MASTER_DIR', 'CSV_PATH', 'BONSAI_REPO', 'BONSAI_PYTHON', 'BONSAI_RESULTS_FOLDER')
_overrides = {}


def settings() -> dict:
    """{name: current value} of every setting (UPPER_CASE names and safe_ram_workers)."""
    g = globals()
    return {k: g[k] for k in sorted(g)
            if (k.isupper() and not k.startswith('_') and k != 'ENV_VAR') or k == 'safe_ram_workers'}


_DEFAULTS = settings()


def _coerce(name, value, default):
    """JSON / TOML / CLI values back to the type the pipeline expects (Path, tuple, int dict keys)."""
    if value is None:
        return None
    if name in _PATH_SETTINGS:
        return Path(value)
    if isinstance(default, tuple) and isinstance(value, list):
        return tuple(value)
    if isinstance(default, dict) and isinstance(value, dict):
        int_keys = bool(default) and all(isinstance(k, int) for k in default)
        tuple_vals = any(isinstance(v, tuple) for v in default.values())
        out = {}
        for k, v in value.items():
            if int_keys and isinstance(k, str) and k.lstrip('-').isdigit():
                k = int(k)          # JSON turns {0: ...} into {"0": ...}
            out[k] = tuple(v) if tuple_vals and isinstance(v, list) else v
        return out
    return value


def _jsonable(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (tuple, list)):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    return value


def _derive():
    """Settings computed from other settings, recomputed after every override unless set explicitly."""
    global MASTER_DIR, CSV_PATH, BONSAI_REPO, BONSAI_PYTHON, BONSAI_RESULTS_FOLDER
    global MAX_WORKERS_STAGE1, MAX_WORKERS_STAGE3, FEATURE_TAG, FEATURE_LABEL
    # absolute: stage 1 workers chdir into each well and stage 5 chdirs into MASTER_DIR
    MASTER_DIR = Path(os.path.abspath(MASTER_DIR))
    if _overrides.get('CSV_PATH') is None:
        CSV_PATH = MASTER_DIR / 'BIGmatched_peaks_raw.csv'
    CSV_PATH = Path(os.path.abspath(CSV_PATH))
    BONSAI_REPO, BONSAI_PYTHON, BONSAI_RESULTS_FOLDER = (
        Path(os.path.abspath(p)) if p is not None else None
        for p in (BONSAI_REPO, BONSAI_PYTHON, BONSAI_RESULTS_FOLDER))
    if 'MAX_WORKERS_STAGE1' not in _overrides:
        MAX_WORKERS_STAGE1 = min(safe_ram_workers, usable_cores)
    if 'MAX_WORKERS_STAGE3' not in _overrides:
        MAX_WORKERS_STAGE3 = min(safe_ram_workers, usable_cores)
    if 'FEATURE_TAG' not in _overrides:
        FEATURE_TAG = 'mass_plus_morph' if USE_MORPHOLOGY_FEATURES else 'mass_only'
    if 'FEATURE_LABEL' not in _overrides:
        FEATURE_LABEL = 'mass+morph' if USE_MORPHOLOGY_FEATURES else 'mass only'


def apply_overrides(values: dict, export: bool = True):
    """Set settings by name. Unknown names are an error (a typo would otherwise be ignored).
    export=True also hands them to worker processes via TANUKI_CONFIG_JSON."""
    unknown = sorted(set(values) - set(_DEFAULTS))
    if unknown:
        raise KeyError(f"unknown setting(s) {unknown}; see tanuki --show-config for the names")
    g = globals()
    for name, value in values.items():
        g[name] = _coerce(name, value, _DEFAULTS[name])
        _overrides[name] = value
    _derive()
    if export:
        os.environ[ENV_VAR] = json.dumps({k: _jsonable(g[k]) for k in _overrides})


def configure(**values):
    """Override settings from Python, before importing any stage module:
    config.configure(MASTER_DIR=r"D:\\run", BONSAI_AUTO_RUN=False)"""
    apply_overrides(values)


def load_settings_file(path) -> dict:
    """UPPER_CASE settings (and safe_ram_workers) from a .json, .toml or .py file."""
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix == '.json':
        return json.loads(p.read_text(encoding='utf-8'))
    if suffix == '.toml':
        import tomllib
        return tomllib.loads(p.read_text(encoding='utf-8'))
    if suffix == '.py':
        import runpy
        ns = runpy.run_path(str(p))
        return {k: v for k, v in ns.items()
                if (k.isupper() and not k.startswith('_')) or k == 'safe_ram_workers'}
    raise ValueError(f"settings file must be .json, .toml or .py: {p}")


# Worker processes (and any process started with TANUKI_CONFIG_JSON set) pick up the parent's overrides.
if os.environ.get(ENV_VAR):
    apply_overrides(json.loads(os.environ[ENV_VAR]), export=False)
else:
    _derive()
