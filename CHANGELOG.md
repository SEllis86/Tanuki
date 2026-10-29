# Changelog

Version history of the single-file pipeline `tanuki2026IPSC_phasecorr_reg_v5.py` and its predecessors,
as written in the script's header. Version 5 is the code in `src/tanuki`.

## 5.0.0 - package (2026-10-29)

- `tanuki2026IPSC_phasecorr_reg_v5.py` split into the `src/tanuki` package with the same functions.
- Paths come from the command line / settings files (`--input_dir`, `--csv_path`, `--config`, `--set`);
  `--from-stage`, `--no-viewer`, `--show-config`.
- Volcano plots (donor level: paired by donor line, moderated t) and the raw per-cell export run at the end
  of stage 5 (`tanuki.volcano`, `tanuki.export`).
- Replaces the March 2026 `MSI_Project` modules (SimpleITK MMI / B-spline registration, npz tables, AlignedUMAP).

## Script history (newest first)

```text
tanuki2026IPSC_phasecorr_reg_v5.py  (2026-10-19)

= v4 + the fixes from the 2026-10-12 code audit.
  Results that were wrong
  1. ComBat: log1p is linear at TIC-fraction scale and ComBat moved every zero, so the
     "corrected" matrix had 0% zeros and 33% negatives and separated cells MORE by slide.
     Now log(x + half the smallest non-zero value), zeros stay zero, negatives clipped, a
     design table first, per-row ComBatApplied - and COMBAT_ENABLED (off by default here:
     on this set each line sits on one slide, so a slide correction removes line effects).
  2. Optical layers kept at 16 bit (STAGE1_FORMAT_VERSION 2): one shared 8-bit scale left
     dim fluorescence with ~8 grey levels. Older h5 files are redone by stage 1.
  3. Brightfield-only wells: Cellpose got the plane twice in opposite polarities.
  4. Per-cell SEM over the effective number of MSI pixels (SEM_EFFECTIVE_N), not the ~240x
     upsampled ones; zeros get a detection-limit uncertainty in the Bonsai export.
  5. One fixed colour per group in every panel and plot (batches used to swap colours
     between the uncorrected and corrected panel); >10 groups get distinct colours.
  Robustness
  6. Stage 2 re-segments when stage 1 re-ran (stage1_run_id); a failed alignment skips the
     folder instead of saving masks in the wrong frame; stage 3 clears its old parquets
     first; a stale Bonsai tree is no longer drawn.
  7. Bonsai runs after the CSV, QC table and plots; PNG export failures are not fatal.
  Speed
  8. Stage 1 writes the cube channel by channel (lzf), no 6 GB array per worker; stage 3
     reads it in slabs with a sparse matrix (identical means/sums) and keeps its workers;
     no 1-2 GB pixel CSV unless WRITE_PIXEL_LONG_TABLE; SAFE MODE skips folders that fail
     deterministically; the Dash viewer ignores hovers when locked to clicks and caches masks.
  Small: ImageJ micron unit, cell-line regex, legend show/hide per colouring, ratio
  numerator not a denominator, ClusterBonsai seeded, emoji-safe console.

tanuki2026IPSC_phasecorr_reg_v4.py  (2026-10-12)

= v3.4 + three STAGE 2 changes. Stages 1 and 3 and the analysis are unchanged.
  1. Directional neuron merge (MERGE_RULE = 'linear', MERGE_LINEAR_*). The contact rule
     favoured side-by-side contact and refused end-to-end joins of pieces longer than
     wide, so neuron wells grew wide clouds (one mask was 86% of A3_Neurons_11302).
     A merge must now extend an object without widening it.
  2. PREMSI_seg.npy also stores the unmerged Cellpose masks ('masks_raw'); the merge now
     runs after the stage 2 alignment so both share one frame. A changed merge rule is
     re-applied from them without Cellpose (STAGE2_REMERGE_FROM_RAW).
  3. Cellpose gets every CPU core: OMP_NUM_THREADS=1 (meant for the worker pools) also
     held torch to one thread, and the .venv torch is a CPU build.
BONSAI_AUTO_RUN is on. MASTER_DIR is still the "GBA FOR TANUKI - Copy" test folder.

tanuki2026IPSC_phasecorr_reg_v3.py  (2026-10-07; v3.1 2026-10-08)

v3.4: MERGE_MAX_MERGED_AREA_PX caps the size of a merged object; without it a chain of
touching fragments on a confluent well merged into one field-sized mask.
v3.3: merge_touching_masks runs on NEURON folders only (it was astro + micro, neurons
skipped). Seg files record their merge rule and folders made with a different rule are
re-segmented automatically, so the first v3.3 run redoes Cellpose on every folder once.
v3.2: that panel shows brightfield UN-inverted (STAGE2_AUDIT_BRIGHTFIELD_UNINVERTED);
Cellpose still gets it inverted, as before.
v3.1: the top panel of PREMSI_stage2_audit_segmentation.png shows channel 0 in grey,
channel 1 in green and channel 2 in blue (it was plain RGB, so brightfield came out
red). STAGE2_REDRAW_AUDIT redraws that image for already-segmented folders from disk.

= v2 + three POST-CONCAT ANALYSIS changes and one default. Stages 1-3 are unchanged.
  1. Area filter per cell type (ANALYSIS_AREA_BOUNDS_BY_TYPE). The single 100-2000 px
     window dropped 60% of the astrocytes on the GBA set (1228 past stage 3 -> 493 in
     the matrix): 2000 px at ~1.3 um/px is only ~3,400 um^2, and merge_touching_masks
     builds astro objects from fragments of up to 1500 px each. The filter now prints
     the area distribution and the loss per cell type, so the bounds can be set from data.
  2. Zero-rate feature filter per cell type (ZERO_RATE_PER_CELL_TYPE). Pooled over a
     matrix that is ~80% neurons, a peak present in every astrocyte but absent from
     neurons was dropped as "mostly zero". A feature is now kept if ANY cell type passes.
  3. ComBat batch = slide (COMBAT_BATCH_COL = 'Slide', the S#B# token in the folder
     name). The old batch (cell type + cell line) with the covariate (group + cell type)
     was confounded whenever every line shared a group (all CONTROL here), so ComBat
     refused and every "corrected" panel was raw data. The cell line is deliberately NOT
     the batch: once GBA lines are in a run the disease group is fixed per line, and
     ComBat would remove exactly the effect being studied.
  4. STAGE2_FORCE_RESEGMENT defaults to False again (the v2 masks are on disk).
Also: 'Slide' and 'CellType' columns on the matrix, in the Dash colour dropdown, the
per-well QC table, the Bonsai annotation and the mean-spectrum groupings.

tanuki2026IPSC_phasecorr_reg_v2.py  (2026-09-17)

= tanuki2026IPSC_phasecorr_reg.py + four STAGE 2 (segmentation) changes, each behind
a setting in the "stage 2 segmentation" block below:
  1. Cellpose sees all three optical channels (brightfield inverted + CellBrite +
     Hoechst), each stretched on its own percentiles, with a larger top-hat kernel so
     big flat astrocytes survive the stitching-grid removal (CELLPOSE_*).
  2. MSI-guided rescue: bright 760 ion-image blobs that no Cellpose mask covers become
     masks tagged mask_source = 'msi_rescue' (MSI_RESCUE_*). Drawn with a white outline
     in PREMSI_stage2_audit_segmentation.png; the tag rides on every row downstream.
  3. Optional second Cellpose pass with the ion image as the third channel, after the
     optical layers have been aligned to the MSI grid (CELLPOSE_MSI_CHANNEL, off).
  4. Touching-mask merge now also requires the shared border to be a real fraction of
     the smaller fragment's perimeter (MERGE_MIN_CONTACT_FRAC), which lets the area cap
     be raised to catch "candy-stripe" fragments without fusing real neighbours.
Plus STAGE2_ONLY_FOLDERS / STAGE2_FORCE_RESEGMENT for trialling on a few wells.
Everything else is identical to the phasecorr fork.

tanuki2026IPSC_phasecorr_reg.py  (2026-09-17)

Fork of tanuki2026IPSC_oldreg_claudebugs_autocrop_cellposereg.py with the
PRE -> POST optical registration replaced. The Mattes-mutual-information
cascade (moments initialiser, rigid Euler2D, four tiers gated on MMI < -0.5)
never reached its own acceptance threshold on any GBA folder (MMI -0.05 to
-0.25) and was mis-registering wells by a third of a well pitch (locked
onto the tile-scan seam grid one period out) or by a rotation (moments
centroid on a POST scan 936 px wider than its PRE).

It is replaced by the phase-correlation + tile method from the MSI
platform's single-cell pipeline (pre_post_register.py, Shane 2026-09),
ported self-contained into this file (the PRE -> POST OPTICAL REGISTRATION
section). Changes to this script, all in stage 1:
  * alignment channel chosen per image by histogram (brightfield), not a
    fixed index (REGISTRATION_CHANNEL = None; an int still forces it);
  * pixel size read from each TIFF's resolution tags, not assumed;
  * the pass/fail gate is in micrometres (tile leave-one-out residual and
    the transform's placement error), not an MMI number; a folder whose
    fit is not credible is SKIPPED, not cropped with a warning;
  * QC_registration_accuracy.txt lists the whole fit; the audit panels
    carry the fit summary in their header instead of an MMI;
  * folders registered by the old MMI code are re-registered
    (REREGISTER_OLD_MMI_FOLDERS), recognised by their QC file lacking a
    "Method: phase" line, so a re-run of THIS script is still a no-op.
Everything downstream of the registered tiffs (square finder, cube
resize, Cellpose, features, ComBat, Bonsai, Dash) is untouched.
```
