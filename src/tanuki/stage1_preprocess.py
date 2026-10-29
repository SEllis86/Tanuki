"""Stage 1 - pre-Cellpose: registration of the PRE scan onto the POST scan, fiducial square, imzML cube
(PREMSI_stage1_ready.h5), audits and QC files. Runs in a process pool."""
from __future__ import annotations

import gc
import os
import re
import time
import uuid
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import SimpleITK as sitk
import cv2
import h5py
import numpy as np
import tifffile
from pewlib.io.imzml import ImzML
from scipy.ndimage import gaussian_filter

from .config import (CELLPOSE_GAUSSIAN_SIGMA, CELLPOSE_GAUSSIAN_TRUNCATE, CELLPOSE_STRETCH_PERCENTILES,
    MASTER_DIR, MAX_WORKERS_STAGE1, MSI_ROT90_K, REGISTRATION_CHANNEL, REG_AUDIT_CHECKER_TILES,
    REG_AUDIT_ENABLED, REG_AUDIT_MAX_PANEL_PX, SKIP_FILENAME, SQUARE_MAX_SIDE_DIFF_PX, SQUARE_MAX_SIDE_PX,
    SQUARE_MIN_EXTENT, SQUARE_MIN_SIDE_PX, SQUARE_MSI_SIZE_TOL, SQUARE_USE_MSI_SIZE_GATE,
    STAGE1_FORMAT_VERSION)
from .registration import (DEFAULT_UM_PER_PX, PrePostRegistrationError, RegistrationNotCredible,
    brightfield_channel, register_pre_to_post, tiff_um_per_px)
from .utils import get_masses_of_interest, h5_compression_kwargs, load_rgb_tiff, should_skip_subdir


def registration_channel_index(image) -> int:
    """Which channel of a channel-last image the alignment uses. See REGISTRATION_CHANNEL."""
    arr = np.asarray(image)
    if arr.ndim == 2:
        return 0
    if REGISTRATION_CHANNEL is None:
        index, _scores = brightfield_channel(arr)
        return index
    c = int(REGISTRATION_CHANNEL)
    if c >= arr.shape[2]:
        print(f'⚠️ REGISTRATION_CHANNEL={c} but the image has {arr.shape[2]} channel(s); '
              f'using channel 0.')
        c = 0
    return c


def registration_plane(image):
    """The single channel the alignment is judged on. See REGISTRATION_CHANNEL."""
    arr = np.asarray(image)
    if arr.ndim == 2:
        return arr
    if arr.ndim == 3:
        return arr[..., registration_channel_index(arr)]
    raise ValueError(f'Unsupported image shape for registration: {arr.shape}')


def _stretch_signal_bright_uint8(plane):
    """
    1-99 percentile stretch to uint8, then flipped if the background is the bright
    majority (brightfield: dark cells on a white field), so that in every audit panel
    STRUCTURE is bright and empty field is dark. Registration fill (255 white) therefore
    lands on black and never masquerades as signal.
    """
    p = np.asarray(plane).astype(np.float32)
    sub = p[::4, ::4] if p.size > 4_000_000 else p
    lo, hi = np.percentile(sub, (1, 99))
    if hi <= lo:
        hi = lo + 1.0
    n = np.clip((p - lo) / (hi - lo), 0.0, 1.0)
    if float(np.median(sub)) > (lo + hi) / 2.0:
        n = 1.0 - n
    return (n * 255.0).astype(np.uint8)


def write_registration_overlap_audit(fixed_plane, moving_plane, out_path, metric=None, max_panel_px=None,
                                     labels=('POST fixed', 'PRE registered'), note=None):
    """
    Three side-by-side panels judging the optical registration on the alignment channel.
    `note` (the fit summary) is printed in the header; `metric` is kept only so audits
    back-filled from OLD MMI-registered folders can still show their number.
      1. colour overlay  - fixed magenta, registered moving green, overlap white
                           (same convention as PREMSI_stage2_audit_binary_registration.png)
      2. checkerboard    - alternating tiles of fixed / moving; a mis-registration shows as
                           structure that jumps at every tile border
      3. |fixed - moving| on an inferno scale; a good alignment is dark except at edges
    Both planes are stretched with _stretch_signal_bright_uint8 first, so the panels are
    comparable regardless of exposure. Shrunk so each panel is <= max_panel_px wide.
    """
    f8 = _stretch_signal_bright_uint8(fixed_plane)
    m8 = _stretch_signal_bright_uint8(moving_plane)
    if m8.shape != f8.shape:
        m8 = cv2.resize(m8, (f8.shape[1], f8.shape[0]), interpolation=cv2.INTER_AREA)

    max_panel_px = max_panel_px or REG_AUDIT_MAX_PANEL_PX
    longest = max(f8.shape)
    if max_panel_px and longest > max_panel_px:
        sc = max_panel_px / float(longest)
        f8 = cv2.resize(f8, (0, 0), fx=sc, fy=sc, interpolation=cv2.INTER_AREA)
        m8 = cv2.resize(m8, (0, 0), fx=sc, fy=sc, interpolation=cv2.INTER_AREA)
    h, w = f8.shape

    # 1. overlay (BGR): fixed -> blue + red = magenta, moving -> green
    overlay = np.dstack([f8, m8, f8])

    # 2. checkerboard
    tile = max(8, min(h, w) // int(max(1, REG_AUDIT_CHECKER_TILES)))
    yy, xx = np.mgrid[0:h, 0:w]
    checker = ((yy // tile) + (xx // tile)) % 2 == 0
    board = cv2.cvtColor(np.where(checker, f8, m8).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    for y in range(tile, h, tile):
        cv2.line(board, (0, y), (w - 1, y), (80, 80, 80), 1)
    for x in range(tile, w, tile):
        cv2.line(board, (x, 0), (x, h - 1), (80, 80, 80), 1)

    # 3. absolute difference
    diff = cv2.applyColorMap(cv2.absdiff(f8, m8), cv2.COLORMAP_INFERNO)

    gap = np.full((h, 12, 3), 255, dtype=np.uint8)
    canvas = np.hstack([overlay, gap, board, gap, diff])

    font_scale = max(0.5, canvas.shape[1] / 3000.0)
    header_h = int(40 * font_scale) + 20
    header = np.full((header_h, canvas.shape[1], 3), 255, dtype=np.uint8)
    metric_txt = f"   MMI {metric:.4f}" if metric is not None and np.isfinite(metric) else ""
    note_txt = f"   |   {note}" if note else ""
    text = (f"{labels[0]} = magenta, {labels[1]} = green, overlap = white   |   checkerboard {tile}px   |   "
            f"|fixed - moving|{metric_txt}{note_txt}")
    cv2.putText(header, text, (10, header_h - 12), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0),
                max(1, int(font_scale * 1.5)), cv2.LINE_AA)
    cv2.imwrite(str(out_path), np.vstack([header, canvas]))
    return out_path


def _read_reg_metric_from_folder(subdir: Path):
    """MMI from QC_registration_accuracy.txt, else from the MovedMSIFACTOR_MMI_*.tif name."""
    qc = subdir / "QC_registration_accuracy.txt"
    if qc.exists():
        try:
            m = re.search(r'Mattes Mutual Information:\s*([-+0-9.eE]+)', qc.read_text(errors='replace'))
            if m:
                return float(m.group(1))
        except Exception:
            pass
    for f in subdir.glob('MovedMSIFACTOR_MMI_*.tif'):
        m = re.search(r'MMI_([-+0-9.]+)\.tif$', f.name)
        if m:
            try:
                return float(m.group(1))
            except ValueError:
                pass
    return None


def _read_reg_note_from_folder(subdir: Path):
    """One-line fit summary from a QC file written by the phase-correlation registration."""
    qc = subdir / "QC_registration_accuracy.txt"
    if not qc.exists():
        return None
    try:
        text = qc.read_text(errors='replace')
    except Exception:
        return None
    if "Method: phase" not in text:
        return None
    parts = []
    for key in ("Method:", "Credible:", "Tiles used:"):
        for line in text.splitlines():
            if line.startswith(key):
                parts.append(line.strip())
                break
    return "  ".join(parts) if parts else None


def backfill_registration_overlap_audit(subdir: Path):
    """
    Build the full-frame overlap audit for a folder that was registered on an earlier run,
    from the tiffs stage 1 already wrote. Skips folders that have it, or that were never
    registered. Returns True when a file was written.
    """
    out_path = subdir / "PREMSI_stage1_audit_registration_overlap.png"
    fixed_tif = subdir / "FixedMSI.tif"
    moved_tif = subdir / "MovedMSI_multichannel.tif"
    if out_path.exists() or not fixed_tif.exists() or not moved_tif.exists():
        return False
    try:
        fixed_plane = tifffile.imread(str(fixed_tif))
        if fixed_plane.ndim == 3:
            fixed_plane = registration_plane(load_rgb_tiff(str(fixed_tif)))
        moved = tifffile.imread(str(moved_tif))
        if moved.ndim == 3:
            # written channels-first (CYX) by stage 1
            moving_plane = registration_plane(np.moveaxis(moved, 0, -1))
        else:
            moving_plane = moved
        write_registration_overlap_audit(fixed_plane, moving_plane, out_path,
                                         metric=_read_reg_metric_from_folder(subdir),
                                         note=_read_reg_note_from_folder(subdir))
        print(f"[S1] {subdir.name}: back-filled {out_path.name}")
        return True
    except Exception as e:
        print(f"[S1] {subdir.name}: could not back-fill the registration overlap audit ({e})")
        return False


def imzml_expected_square_px(imzml, um_per_px: float):
    """
    (expected_w_px, expected_h_px, info) for the ablated square in the optical frame, or
    (None, None, reason). Stage 1 rotates the MSI array 90 deg clockwise before resizing
    it onto the square, so MSI rows (y) become optical columns: width follows the imzML
    y extent and height the x extent. For the square wells here the two are equal anyway.
    """
    try:
        px_x, px_y = imzml.scan_settings.pixel_size
        n_x, n_y = imzml.image_size
    except Exception as e:
        return None, None, f"imzML has no scan settings ({e})"
    if not px_x or not px_y or not n_x or not n_y or not um_per_px:
        return None, None, "imzML pixel size or grid is zero"
    exp_w = float(n_y) * float(px_y) / float(um_per_px)
    exp_h = float(n_x) * float(px_x) / float(um_per_px)
    info = (f"imzML grid {n_x}x{n_y} px at {px_x:g}x{px_y:g} um = "
            f"{n_x * px_x:g}x{n_y * px_y:g} um; optical {um_per_px:.4f} um/px")
    return exp_w, exp_h, info


def square_size_deviation(w, h, exp_w, exp_h) -> float:
    """Largest relative side error, side lengths sorted so a 90-deg-swapped rect still matches."""
    found = sorted([float(w), float(h)])
    expect = sorted([float(exp_w), float(exp_h)])
    return max(abs(found[0] - expect[0]) / expect[0], abs(found[1] - expect[1]) / expect[1])


def write_square_size_qc(subdir: Path, w, h, exp_w, exp_h, info, gate_active: bool):
    """QC_square_size.txt: found vs expected square side, and whether the gate was live."""
    dev = square_size_deviation(w, h, exp_w, exp_h)
    verdict = ("OK" if dev <= SQUARE_MSI_SIZE_TOL else
               f"WARNING: deviates by more than {SQUARE_MSI_SIZE_TOL:.0%}")
    lines = [
        f"Found square: {float(w):.1f} x {float(h):.1f} optical px",
        f"Expected from imzML: {exp_w:.1f} x {exp_h:.1f} optical px ({info})",
        f"Deviation: {dev:.1%} of the expected side -> {verdict}",
        f"Size gate during search: {'active' if gate_active else 'QC only (optical pixel size was a default)'}",
    ]
    (subdir / "QC_square_size.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return dev


def backfill_square_size_qc(subdir: Path):
    """
    For a folder processed before the imzML cross-check existed: rebuild the QC line from
    the square stored in the h5, the POST TIFF's pixel size and the imzML header, and
    store the expected size on the h5. Returns the deviation, or None if not possible.
    """
    stage1_h5 = subdir / "PREMSI_stage1_ready.h5"
    if not stage1_h5.exists() or (subdir / "QC_square_size.txt").exists():
        return None
    try:
        with h5py.File(stage1_h5, 'r') as f:
            if "square_size_wh" not in f.attrs:
                return None
            w, h = [float(v) for v in f.attrs["square_size_wh"]]
        posts = [x for x in subdir.iterdir() if x.suffix.lower() == '.tif' and 'post' in x.name.lower()]
        if not posts:
            generated = {'fixedmsi.tif', 'movedmsi.tif', 'movedmsiold.tif', 'movedmsi_multichannel.tif',
                         'movedmsi_cropped_multichannel.tif', '760check.tif'}
            posts = [x for x in subdir.iterdir() if x.suffix.lower() == '.tif'
                     and 'pre' not in x.name.lower() and x.name.lower() not in generated
                     and not x.name.lower().startswith(('premsi_', 'movedmsifactor_'))]
        if len(posts) != 1:
            print(f"[S1] {subdir.name}: square QC back-fill skipped (POST tiff ambiguous: {len(posts)})")
            return None
        um_per_px, um_src = tiff_um_per_px(posts[0])
        imzml_files = sorted(x for x in subdir.iterdir() if x.suffix.lower() == '.imzml')
        if not imzml_files:
            return None
        imzml = ImzML.from_file(str(imzml_files[0]))
        exp_w, exp_h, info = imzml_expected_square_px(imzml, um_per_px)
        if exp_w is None:
            print(f"[S1] {subdir.name}: square QC back-fill skipped ({info})")
            return None
        dev = write_square_size_qc(subdir, w, h, exp_w, exp_h, info,
                                   gate_active=not um_src.startswith("default"))
        with h5py.File(stage1_h5, 'a') as f:
            f.attrs["square_expected_wh_px"] = [exp_w, exp_h]
            f.attrs["square_size_dev_frac"] = float(dev)
            f.attrs["msi_pixel_size_um"] = [float(v) for v in imzml.scan_settings.pixel_size]
        flag = "" if dev <= SQUARE_MSI_SIZE_TOL else "  <-- WARNING"
        print(f"[S1] {subdir.name}: square {w:.0f}x{h:.0f} px vs imzML-expected "
              f"{exp_w:.0f}x{exp_h:.0f} px ({dev:.1%}){flag}")
        return dev
    except Exception as e:
        print(f"[S1] {subdir.name}: square QC back-fill failed ({e})")
        return None


# ============================================================
# STAGE 1 - PRE-CELLPOSE
# ============================================================

def register_multichannel_images(fixed_image: np.ndarray, moving_image: np.ndarray,
                                 fixed_um_per_px: float, moving_um_per_px: float):
    """
    Register the PRE scan (moving) onto the POST scan (fixed) and write the same files the
    old MMI cascade wrote, so the square finder and every later stage are unchanged:
        FixedMSI.tif                 POST alignment channel, uint8, own max = 255
        MovedMSI_multichannel.tif    every PRE channel resampled onto the POST grid (CYX)
        MovedMSI.tif / MovedMSIOLD.tif   the PRE alignment channel of the above
        MovedMSIFACTOR_<summary>.tif     same again, the fit summary in the name
    Returns (registered_moving_full [H, W, C] uint8, SimpleITK transform POST -> PRE, fit).

    The method (register_pre_to_post below): brightfield channel of each
    scan by histogram; translation by true phase correlation on high-passed images over a
    ladder of working scales; rotation + one scale from a 6x6 tile grid searched only within
    60 um of that shift; a similarity fitted with outlier rejection and a leave-one-out
    residual; the dark ring marks paired through the fit as an independent check. No
    intensity optimiser, no initialiser, no MMI gate. `fit.credible` is the pass/fail and
    the CALLER decides what to do with a failure.

    Why not Mattes MI: on these pairs (cells + clean brightfield in PRE; matrix grain, a
    dark field and one ablated square in POST) MI never reached its own -0.5 acceptance on
    any folder, locked onto the tile-scan seam grid one period out, and its moments
    initialiser started a rotation whenever the two fields differed in extent.
    """
    fixed_image = np.asarray(fixed_image)
    moving_image = np.asarray(moving_image)
    if fixed_image.ndim == 2:
        fixed_image = fixed_image[..., np.newaxis]
    if moving_image.ndim == 2:
        moving_image = moving_image[..., np.newaxis]

    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(1)

    # Out-of-frame fill is 255.0 in RAW units, exactly as before: on uint16 scans that is
    # near-black after the shared-max normalisation below, on uint8 scans it is white. The
    # square finder downstream was tuned against that behaviour, so it is kept.
    resampled_raw, coverage, final_transform, fit = register_pre_to_post(
        fixed_image, moving_image, fixed_um_per_px, moving_um_per_px,
        channel=REGISTRATION_CHANNEL, fill_value=255.0)

    # POST alignment channel, normalised by its own max (unchanged from the old code).
    f_gray_raw = fixed_image[..., fit.post_channel].astype(np.float32)
    fixed_gray_uint8 = np.clip(f_gray_raw / (float(np.max(f_gray_raw)) + 1e-8), 0, 1) * 255
    tifffile.imwrite("FixedMSI.tif", fixed_gray_uint8.astype(np.uint8))

    # One scale for every PRE channel, computed only over pixels that carry data, so
    # relative brightness between channels survives and the fill does not set the range.
    if not coverage.any():
        print("⚠️ Registration produced no in-bounds overlap; falling back to full-frame normalisation.")
        coverage = np.ones_like(coverage, dtype=bool)
    in_bounds_maxima = [float(np.max(ch[coverage])) for ch in resampled_raw if ch[coverage].size]
    global_max = max(in_bounds_maxima) if in_bounds_maxima else 0.0
    if not np.isfinite(global_max) or global_max <= 0:
        global_max = 1.0

    registered_channels = [
        (np.clip(ch / (global_max + 1e-8), 0.0, 1.0) * 255).astype(np.uint8) for ch in resampled_raw
    ]
    registered_moving_full = np.stack(registered_channels, axis=-1)

    # v5: MovedMSI_multichannel.tif (which becomes imgs_cp, the Cellpose input) keeps the
    # resampled values at 16 bit. Scaling every channel by ONE shared max into uint8 left a
    # dim fluorescence channel with as few as 8 grey levels (channel 1 of the 11450 microglia
    # wells), and Cellpose's per-channel stretch cannot recover levels that are gone. The
    # 8-bit plane files below (MovedMSI.tif / MovedMSIOLD.tif) are what the square finder
    # reads and stay exactly as before.
    multichannel_u16 = np.stack(
        [np.clip(np.rint(ch), 0, 65535).astype(np.uint16) for ch in resampled_raw], axis=-1)
    if multichannel_u16.shape[2] > 1:
        tifffile.imwrite("MovedMSI_multichannel.tif", np.moveaxis(multichannel_u16, -1, 0),
                         imagej=True, metadata={'axes': 'CYX'})
    else:
        tifffile.imwrite("MovedMSI_multichannel.tif", multichannel_u16[:, :, 0], photometric='minisblack')

    plane = registered_channels[fit.pre_channel]
    tifffile.imwrite("MovedMSI.tif", plane, photometric='minisblack')
    tifffile.imwrite("MovedMSIOLD.tif", plane, photometric='minisblack')
    # The old name carried the MMI; this one carries what the fit is judged on.
    summary = (f"{fit.method.replace('+', '_')}_tiles{fit.n_tiles}_LOO{fit.loo_rmse_um:.1f}um"
               f"_placed{fit.placement_error_um:.2f}um" + ("" if fit.credible else "_NOTCREDIBLE"))
    for stale in Path('.').glob('MovedMSIFACTOR_*.tif'):
        try:
            stale.unlink()
        except OSError:
            pass
    tifffile.imwrite(f"MovedMSIFACTOR_{summary}.tif", plane, photometric='minisblack')

    print(f"Registration done: {fit.method}, {fit.n_tiles} tiles, per-tile LOO {fit.loo_rmse_um:.2f} um, "
          f"placed {fit.placement_error_um:.2f} um, scale {fit.scale:.5f}, rotation {fit.rotation_deg:+.3f} deg, "
          f"credible {fit.credible}")
    return registered_moving_full, final_transform, fit


def stage1_worker(subdir: Path):
    if should_skip_subdir(subdir):
        print(f"[S1] SKIPPING {subdir.name} - Already processed with new format")
        return True
    masses_of_interest = get_masses_of_interest()
    try:
        if not any(subdir.glob('*PRE*.tif')):
            print(f"[S1] Skipping {subdir.name} - missing PRE image")
            return True

        print(f"[S1] ▶ {subdir.name}", flush=True)
        os.chdir(subdir)

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")

            file_list = [f for f in os.listdir('.') if
                         f.lower().endswith('.tif') and 'pre' in f.lower() and not f.lower().startswith('premsi_')]
            if len(file_list) == 1:
                moving_image_path = file_list[0]
                moving_image = load_rgb_tiff(moving_image_path)
            elif len(file_list) == 0:
                raise FileNotFoundError("No file containing 'PRE' found.")
            else:
                raise RuntimeError(f"Multiple files containing 'PRE' found: {file_list}")

            file_list = [f for f in os.listdir('.') if f.lower().endswith('.tif') and 'post' in f.lower()]
            if len(file_list) > 1:
                raise RuntimeError(f"Multiple files containing 'POST' found: {file_list}")

            if len(file_list) == 0:
                # Some acquisitions name the post-ablation scan without the word 'POST'.
                # Fall back to the ONLY other optical tiff in the folder, once everything
                # this pipeline writes itself has been excluded - otherwise a re-run would
                # happily register against its own FixedMSI.tif from the previous attempt.
                generated = {
                    'fixedmsi.tif', 'movedmsi.tif', 'movedmsiold.tif',
                    'movedmsi_multichannel.tif', 'movedmsi_cropped_multichannel.tif',
                    '760check.tif',
                }
                generated_prefixes = ('premsi_', 'movedmsifactor_')

                candidates = []
                for f in os.listdir('.'):
                    low = f.lower()
                    if not low.endswith('.tif'):
                        continue
                    if f == moving_image_path:
                        continue
                    if low in generated or low.startswith(generated_prefixes):
                        continue
                    candidates.append(f)

                if len(candidates) == 1:
                    file_list = candidates
                    print(f"[S1] {subdir.name}: no 'POST' in any filename; "
                          f"assuming '{candidates[0]}' is the POST image "
                          f"(PRE = '{moving_image_path}').")
                elif len(candidates) == 0:
                    raise FileNotFoundError(
                        "No file containing 'POST' found, and no other optical tiff to fall "
                        f"back on (PRE = '{moving_image_path}').")
                else:
                    raise RuntimeError(
                        "No file containing 'POST' found, and the fallback is ambiguous: "
                        f"{sorted(candidates)}. Rename the post-ablation scan to include 'POST'.")

            fixed_image_path = file_list[0]
            fixed_image = load_rgb_tiff(fixed_image_path)

            # ---- imzML parse + native-resolution 760check.tif -------------------------
            # Done BEFORE registration/square-finding so a diagnostic ion image survives
            # even when either of those fails. Registration and square detection only
            # affect how the optical image is aligned/cropped; they have nothing to do
            # with whether the imzML file itself can be read, so there is no reason a
            # failure in one should silently delete the other's output. `imzml`, `tic`
            # and `tic_safe` computed here are reused below (not re-parsed) once/if a
            # square is found, so this adds one 760-mass extraction, not a second full
            # XML parse.
            imzml_files = [f for f in os.listdir(subdir) if f.lower().endswith('.imzml')]
            print(f'Found {len(imzml_files)} imzML files in {subdir.name}')
            if not imzml_files:
                raise FileNotFoundError("No imzML files found in subdir.")
            imzml_path = os.path.join(subdir, imzml_files[0])
            try:
                imzml = ImzML.from_file(imzml_path)
            except Exception as xml_err:
                print(f"[S1] XML Parsing Error in {imzml_files[0]}: {xml_err}")
                raise

            data_cellpose = imzml.extract_masses(760.5852, mass_width_ppm=10).astype(np.float32)
            tic = imzml.extract_tic().astype(np.float32)

            # --- ORIENTATION CORRECTION --- (matches the full-mass-list extraction below)
            data_cellpose = np.rot90(data_cellpose, k=MSI_ROT90_K, axes=(0, 1))
            tic = np.rot90(tic, k=MSI_ROT90_K, axes=(0, 1))

            tic_safe = np.where(tic == 0, 1, tic).astype(np.float32)
            if data_cellpose.ndim == 3:
                data_cellpose /= tic_safe[:, :, np.newaxis]
            else:
                data_cellpose /= tic_safe
            data_cellpose[tic == 0] = 0
            np.nan_to_num(data_cellpose, copy=False, nan=0.0)

            # Best-effort: a failure writing the diagnostic tif must never take the rest
            # of the worker down with it. `data_cellpose` is copied before blurring so the
            # untouched, tic-normalised, native-resolution array is still available below
            # for the resize-to-square path if registration and square-finding succeed.
            try:
                _early_cp = data_cellpose.copy()
                if CELLPOSE_GAUSSIAN_SIGMA and CELLPOSE_GAUSSIAN_SIGMA > 0:
                    _early_cp = gaussian_filter(
                        _early_cp, sigma=float(CELLPOSE_GAUSSIAN_SIGMA),
                        truncate=CELLPOSE_GAUSSIAN_TRUNCATE, mode='nearest')
                _p_lo, _p_hi = CELLPOSE_STRETCH_PERCENTILES
                _dmin = np.percentile(_early_cp, _p_lo)
                _dmax = np.percentile(_early_cp, _p_hi)
                if _dmax >= _dmin:
                    _early_8bit = (((_early_cp - _dmin) / (_dmax - _dmin + 1e-8)) * 255).astype(np.uint8)
                else:
                    _early_8bit = np.zeros_like(_early_cp, dtype=np.uint8)
                tifffile.imwrite(os.path.join(subdir, "760check.tif"), _early_8bit)
                print(f"[S1] {subdir.name}: wrote 760check.tif at native MSI resolution "
                      f"(pre-registration diagnostic; overwritten below with the final "
                      f"registered/cropped version if registration and square-finding succeed).")
            except Exception as e:
                print(f"[S1] {subdir.name}: could not write the native-resolution "
                      f"760check.tif ({e}) - continuing anyway.")

            # Pixel sizes come from the TIFFs themselves (LAS X writes px/cm, ImageJ px/um).
            # The registration works in micrometres, so a wrong value here would move
            # every threshold; the source is recorded in the QC file.
            fixed_um, fixed_um_src = tiff_um_per_px(fixed_image_path)
            moving_um, moving_um_src = tiff_um_per_px(moving_image_path)
            if fixed_um_src.startswith("default") or moving_um_src.startswith("default"):
                print(f"[S1] ⚠️ {subdir.name}: pixel size fell back to {DEFAULT_UM_PER_PX} um/px "
                      f"(POST {fixed_um_src}; PRE {moving_um_src})")

            # How big the ablated square must be in the POST frame, from the imzML header.
            # The square is searched for in FixedMSI.tif, which shares the POST pixel grid.
            expected_square_wh = None
            expected_square_info = ""
            square_gate_active = False
            _exp_w, _exp_h, expected_square_info = imzml_expected_square_px(imzml, fixed_um)
            if _exp_w is not None:
                expected_square_wh = (_exp_w, _exp_h)
                square_gate_active = SQUARE_USE_MSI_SIZE_GATE and not fixed_um_src.startswith("default")
                print(f"[S1] {subdir.name}: expected square {_exp_w:.0f}x{_exp_h:.0f} optical px "
                      f"({expected_square_info}); size gate "
                      f"{'ON' if square_gate_active else 'off, QC only'} at +/-{SQUARE_MSI_SIZE_TOL:.0%}")
            else:
                print(f"[S1] {subdir.name}: no imzML-based square size check ({expected_square_info})")

            def square_dims_ok(w_px, h_px) -> bool:
                """Fixed side/aspect gates, plus the imzML-predicted size when available."""
                if not (SQUARE_MIN_SIDE_PX < w_px < SQUARE_MAX_SIDE_PX
                        and SQUARE_MIN_SIDE_PX < h_px < SQUARE_MAX_SIDE_PX
                        and abs(w_px - h_px) <= SQUARE_MAX_SIDE_DIFF_PX):
                    return False
                if square_gate_active and expected_square_wh is not None:
                    dev = square_size_deviation(w_px, h_px, *expected_square_wh)
                    if dev > SQUARE_MSI_SIZE_TOL:
                        print(f"Ignored candidate {w_px:.0f}x{h_px:.0f} px: imzML predicts "
                              f"{expected_square_wh[0]:.0f}x{expected_square_wh[1]:.0f} px ({dev:.0%} off)")
                        return False
                return True

            reg_metric = None   # no MMI any more; kept so the audit calls below read unchanged
            try:
                registered, transform, fit = register_multichannel_images(
                    fixed_image, moving_image, fixed_um, moving_um)

                with open("QC_registration_accuracy.txt", "w") as qcf:
                    qcf.write("Registration: phase correlation + tile similarity fit "
                              "(ported from the MSI platform pre_post_register method)\n")
                    for line in fit.summary_lines():
                        qcf.write(line + "\n")
                    qcf.write(f"PRE pixel size source: {moving_um_src}; POST pixel size source: {fixed_um_src}\n")
                    qcf.write("Alignment channel is used for the fit only; every channel is transformed.\n")

                reg_note = (f"{fit.method}: {fit.n_tiles} tiles, LOO {fit.loo_rmse_um:.1f} um, "
                            f"placed {fit.placement_error_um:.2f} um, scale {fit.scale:.4f}, "
                            f"rot {fit.rotation_deg:+.2f} deg, marks {fit.n_marks_paired}"
                            + ("" if fit.credible else "  NOT CREDIBLE"))
                print(f"[S1] {subdir.name}: {reg_note}")

                # Visual check of the alignment the QC file describes. Best-effort: the
                # audit must never fail the folder - and it is written BEFORE the
                # credibility gate so a rejected fit can still be looked at.
                if REG_AUDIT_ENABLED:
                    try:
                        write_registration_overlap_audit(
                            fixed_image[..., fit.post_channel], registered[..., fit.pre_channel],
                            subdir / "PREMSI_stage1_audit_registration_overlap.png", note=reg_note)
                        print(f"[S1] {subdir.name}: wrote PREMSI_stage1_audit_registration_overlap.png")
                    except Exception as audit_err:
                        print(f"[S1] {subdir.name}: registration overlap audit failed ({audit_err})")

                if not fit.credible:
                    raise RegistrationNotCredible("; ".join(fit.failures()))
            except PrePostRegistrationError as e:
                # The old code wrote a warning and cropped anyway. A transform that is not
                # credible must not feed the square finder: skip the folder and say why.
                print(f"🛑 [S1] {subdir.name}: registration REJECTED ({e}). Skipping this folder.")
                try:
                    with open("QC_registration_accuracy.txt", "a") as qcf:
                        qcf.write(f"REJECTED: {e}\n")
                except Exception:
                    pass
                # v5: deterministic - a SAFE MODE retry would compute the same fit again
                return "REJECTED"
            except Exception as e:
                print(f"Unhandled step execution failure: {e}")
                return False

            # notes: Cloud FUSE and network drives have write-latency. This ensures files exist and are fully flushed before OpenCV reads them.
            wait_time = 0
            while wait_time < 5:
                if os.path.exists('FixedMSI.tif') and os.path.exists('MovedMSIOLD.tif'):
                    if os.path.getsize('FixedMSI.tif') > 0 and os.path.getsize('MovedMSIOLD.tif') > 0:
                        break
                time.sleep(0.5)
                wait_time += 0.5
            else:
                raise FileNotFoundError("Timeout waiting for OS to flush TIFF files to disk.")

            imagepost = cv2.imread('FixedMSI.tif', cv2.IMREAD_GRAYSCALE)
            imagepre = cv2.imread('MovedMSIOLD.tif', cv2.IMREAD_GRAYSCALE)

            square_found = False
            square_list = []

            # FIX: THRESH_BINARY keeps pixels STRICTLY GREATER than the threshold, so a
            # threshold of 255 on a uint8 image returns an all-zero array. Both quick-find
            # thresholds were 255, meaning this entire fast path could never find anything
            # and every folder fell through to the expensive deep search below. 254 matches
            # the value already used correctly in the deep search.
            print("--- Running Initial Quick Find (Threshold 254) ---")
            _, thresh_post = cv2.threshold(imagepost, 254, 255, cv2.THRESH_BINARY)
            _, thresh_pre = cv2.threshold(imagepre, 254, 255, cv2.THRESH_BINARY)
            combined = cv2.add(thresh_post, thresh_pre)

            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            quick_morph = cv2.dilate(combined, kernel, iterations=1)
            quick_morph = cv2.medianBlur(quick_morph, 29)

            contours, _ = cv2.findContours(quick_morph, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            for cnt in contours:
                approx = cv2.approxPolyDP(cnt, 0.015 * cv2.arcLength(cnt, True), True)
                if len(approx) == 4 and cv2.isContourConvex(approx):
                    rect = cv2.minAreaRect(approx)
                    (cx, cy), (w, h), angle = rect

                    if square_dims_ok(w, h):
                        contour_area = cv2.contourArea(cnt)
                        rect_area = w * h

                        if rect_area > 0:
                            extent = contour_area / rect_area
                            if extent > SQUARE_MIN_EXTENT:
                                square_found = True
                                square_list.append(((cx, cy), (w, h), angle))
                                print(f"Square found immediately at ({int(cx)}, {int(cy)}) with extent {extent:.2f}")
                                break
                            else:
                                print(
                                    f"Ignored false positive at ({int(cx)}, {int(cy)}) - Extent too low: {extent:.2f}")

            if not square_found:
                max_normalizations = 3
                normalization_count = 0
                threshold_values = [254, 250, 200, 220, 180, 50]

                while normalization_count < max_normalizations and not square_found:
                    print(f"\n--- Normalization attempt {normalization_count + 1}/{max_normalizations} ---")

                    for thresh_val in threshold_values:
                        print(f"Trying threshold value: {thresh_val}")
                        _, thresholdimagepost = cv2.threshold(imagepost, thresh_val, 255, cv2.THRESH_BINARY)
                        _, thresholdimagepre = cv2.threshold(imagepre, 254, 255, cv2.THRESH_BINARY)

                        combined_image = cv2.add(thresholdimagepost, thresholdimagepre)
                        _, combined_binary = cv2.threshold(combined_image, 254, 255, cv2.THRESH_BINARY)

                        erode_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
                        m_proc = cv2.dilate(combined_binary, erode_kernel, iterations=1)

                        h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 7))
                        m_proc = cv2.erode(m_proc, h_kernel, iterations=1)
                        v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 1))
                        m_proc = cv2.erode(m_proc, v_kernel, iterations=1)

                        big_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))
                        m_proc = cv2.dilate(m_proc, big_kernel, iterations=1)
                        m_proc = cv2.medianBlur(m_proc, 29)
                        _, dilate_binary = cv2.threshold(m_proc, 250, 255, cv2.THRESH_BINARY)

                        max_attempts = 10
                        attempt = 0
                        while attempt < max_attempts and not square_found:
                            contours, _ = cv2.findContours(dilate_binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

                            for cnt in contours:
                                approx = cv2.approxPolyDP(cnt, 0.015 * cv2.arcLength(cnt, True), True)
                                if len(approx) == 4 and cv2.isContourConvex(approx):
                                    rect = cv2.minAreaRect(approx)
                                    (cx, cy), (w, h), angle = rect

                                    if square_dims_ok(w, h):
                                        contour_area = cv2.contourArea(cnt)
                                        rect_area = w * h

                                        if rect_area > 0:
                                            extent = contour_area / rect_area
                                            if extent > SQUARE_MIN_EXTENT:
                                                square_found = True
                                                square_list.append(((cx, cy), (w, h), angle))
                                                print(
                                                    f"Square found during deep search at ({int(cx)}, {int(cy)}) with extent {extent:.2f}")
                                                break

                            if not square_found:
                                dilate_binary = cv2.dilate(dilate_binary, erode_kernel, iterations=1)
                                attempt += 1

                        if square_found:
                            break

                    if not square_found:
                        normalization_count += 1
                        if normalization_count < max_normalizations:
                            print("No squares found - Applying CLAHE normalization...")
                            clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(4, 4))
                            norm_image = clahe.apply(imagepost)
                            cv2.imwrite('FixedMSI.tif', norm_image)
                            imagepost = norm_image
                        else:
                            print("No squares detected after all attempts.")
                            raise ValueError("Final Detection Failed.")

            print("Total squares identified:", len(square_list))
            print('masses imported')

            if len(square_list) > 1:
                raise ValueError("Too many squares detected. Stopping execution.")
            elif len(square_list) == 0:
                raise ValueError("No squares detected. Stopping execution.")
            else:
                print("Square list is valid. Continuing...")

            (center, size, angle) = square_list[0]
            cx, cy = center
            w, h = size

            # FIX: cv2.minAreaRect returns an angle in [-90, 0). A perfectly axis-aligned
            # square reports -90.0, and a square rotated 10 degrees clockwise reports -80.0.
            # Feeding that straight into getRotationMatrix2D below rotated the crop by a
            # spurious 90 degrees. Because the size gate only requires w ~= h, the crop still
            # landed on the square and nothing raised, so the error was completely silent.
            # Fold the angle back into [-45, 45] and swap the side lengths with it.
            if angle < -45.0:
                angle += 90.0
                w, h = h, w
                print(f"Normalised minAreaRect angle to {angle:.2f} deg (was {angle - 90.0:.2f}).")

            square_size_dev = float('nan')
            if expected_square_wh is not None:
                try:
                    square_size_dev = write_square_size_qc(
                        subdir, w, h, expected_square_wh[0], expected_square_wh[1],
                        expected_square_info, gate_active=square_gate_active)
                    flag = "" if square_size_dev <= SQUARE_MSI_SIZE_TOL else "  <-- WARNING: check imzml_square_0.png"
                    print(f"[S1] {subdir.name}: square {w:.0f}x{h:.0f} px vs imzML-expected "
                          f"{expected_square_wh[0]:.0f}x{expected_square_wh[1]:.0f} px "
                          f"({square_size_dev:.1%}){flag}")
                except Exception as qc_err:
                    print(f"[S1] {subdir.name}: square size QC failed ({qc_err})")

            audit_img = cv2.cvtColor(cv2.imread('FixedMSI.tif', cv2.IMREAD_GRAYSCALE), cv2.COLOR_GRAY2BGR)
            box = cv2.boxPoints(square_list[0])
            box = np.intp(box)
            cv2.drawContours(audit_img, [box], 0, (0, 0, 255), 5)

            scale_factor = 0.2
            downsampled_img = cv2.resize(audit_img, (0, 0), fx=scale_factor, fy=scale_factor,
                                         interpolation=cv2.INTER_AREA)
            cv2.imwrite(SKIP_FILENAME, downsampled_img)
            print(f"Saved audit square image: {SKIP_FILENAME}")

            # imzml / data_cellpose / tic / tic_safe were already parsed and computed
            # in the early 760check.tif block above (before registration ran) - no need
            # to re-parse the imzML file or re-extract the 760 channel here. Only the
            # FULL mass list is new work, since only it depends on masses_of_interest.
            data = imzml.extract_masses(masses_of_interest, mass_width_ppm=10).astype(np.float32)

            # --- ORIENTATION CORRECTION ---
            # Rotate MSI data 90 degrees clockwise to match optical scanner native axes
            data = np.rot90(data, k=MSI_ROT90_K, axes=(0, 1))

            if data.ndim == 3:
                data /= tic_safe[:, :, np.newaxis]
            else:
                data /= tic_safe

            data[tic == 0] = 0
            np.nan_to_num(data, copy=False, nan=0.0)
            # v5: the cube is resized channel by channel straight into the h5 below instead of
            # into a (h, w, n_masses) array first - that array was ~6 GB per folder (x8 workers).

            resized_data_cellpose = cv2.resize(data_cellpose, (int(w), int(h)), interpolation=cv2.INTER_LINEAR)
            np.nan_to_num(resized_data_cellpose, copy=False, nan=0.0)

            # Gaussian on the FLOAT image, before the 8-bit stretch: smoothing after
            # quantisation would only smear the quantisation steps. Segmentation input
            # only - resized_data (the quantitative cube) is untouched.
            if CELLPOSE_GAUSSIAN_SIGMA and CELLPOSE_GAUSSIAN_SIGMA > 0:
                resized_data_cellpose = gaussian_filter(
                    resized_data_cellpose, sigma=float(CELLPOSE_GAUSSIAN_SIGMA),
                    truncate=CELLPOSE_GAUSSIAN_TRUNCATE, mode='nearest')

            _p_lo, _p_hi = CELLPOSE_STRETCH_PERCENTILES
            data_min = np.percentile(resized_data_cellpose, _p_lo)
            data_max = np.percentile(resized_data_cellpose, _p_hi)

            if data_max >= data_min:
                norm_8bit = (((resized_data_cellpose - data_min) / (data_max - data_min + 1e-8)) * 255).astype(np.uint8)
            else:
                norm_8bit = np.zeros_like(resized_data_cellpose, dtype=np.uint8)

            _n_levels = int(np.unique(norm_8bit).size)
            if _n_levels <= 8:
                print(f'[S1] WARNING {subdir.name}: Cellpose input has only {_n_levels} grey '
                      f'levels after the {_p_lo}-{_p_hi} percentile stretch. Raise '
                      f'CELLPOSE_GAUSSIAN_SIGMA or widen CELLPOSE_STRETCH_PERCENTILES.')

            tifffile.imwrite(os.path.join(subdir, "760check.tif"), norm_8bit)

            resampled_array = tifffile.imread("MovedMSI_multichannel.tif")
            if resampled_array.ndim == 3:
                resampled_array = np.moveaxis(resampled_array, 0, -1)
            else:
                resampled_array = resampled_array[:, :, np.newaxis]
            # v5: 16-bit now; cv2.getRectSubPix only takes 8-bit or float32, so crop in float32
            # and go back to uint16 for imgs_cp
            resampled_array = resampled_array.astype(np.float32)

            rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated_image = cv2.warpAffine(resampled_array, rotation_matrix,
                                           (resampled_array.shape[1], resampled_array.shape[0]),
                                           flags=cv2.INTER_LINEAR)

            if rotated_image.ndim == 2:
                cropped_array = cv2.getRectSubPix(rotated_image, (int(w), int(h)), center)
            elif rotated_image.ndim == 3:
                cropped_channels = []
                for c in range(rotated_image.shape[2]):
                    ch = rotated_image[:, :, c]
                    cropped_ch = cv2.getRectSubPix(ch, (int(w), int(h)), center)
                    cropped_channels.append(cropped_ch)
                cropped_array = np.stack(cropped_channels, axis=-1)
            else:
                raise ValueError(f"Unexpected image shape for cropping: {rotated_image.shape}")

            # FIX: cv2.warpAffine SILENTLY DROPS a singleton channel - a (h, w, 1) input
            # comes back as plain (h, w). On a single-channel optical image (the
            # Neurons_*_S3B1 folders) that sent us down the ndim == 2 branch above, so
            # cropped_array was 2D, and then:
            #   - np.moveaxis(2D, -1, 0) merely TRANSPOSES rather than adding a channel
            #     axis, so the imagej write below got a 2D array against 3 declared axes
            #     and raised "ImageJ hyperstack shape and axes do not match", failing the
            #     whole folder after registration and square detection had both succeeded;
            #   - imgs_cp (= cropped_array) would also have been 2D, and the h5 write
            #     further down chunks on imgs_cp.shape[-1], which for a 2D array is the
            #     image WIDTH rather than a channel count.
            # Restoring the axis here fixes both, and is a no-op for multi-channel data.
            if cropped_array.ndim == 2:
                cropped_array = cropped_array[:, :, np.newaxis]
            cropped_array = np.clip(np.rint(cropped_array), 0, 65535).astype(np.uint16)

            # Mirror the guard already used for MovedMSI_multichannel.tif above: an
            # ImageJ hyperstack needs a real channel axis, so a single-channel image is
            # written as a plain 2D greyscale page instead of a 1-deep 'CYX' stack.
            if cropped_array.shape[2] > 1:
                tifffile.imwrite("MovedMSI_cropped_multichannel.tif",
                                 np.moveaxis(cropped_array, -1, 0),
                                 imagej=True, metadata={'axes': 'CYX'})
            else:
                tifffile.imwrite("MovedMSI_cropped_multichannel.tif",
                                 cropped_array[:, :, 0],
                                 photometric='minisblack')

            imgs_cp = cropped_array
            image_for_cellpose = norm_8bit.copy()


            print(f"[{subdir.name}] Saving final .h5 cubes...", flush=True)
            stage1_h5 = subdir / "PREMSI_stage1_ready.h5"
            temp_h5 = str(stage1_h5).replace(".h5", "_temp.h5")

            try:
                with h5py.File(temp_h5, 'w') as f:
                    # FIX: reg_metric is None under the phase method (no MMI), and h5py cannot
                    # store None ("Object dtype has no native HDF5 equivalent") - that killed
                    # every folder at the very last step. Store NaN, plus the fit summary.
                    f.attrs["registration_mattes_mi"] = float('nan') if reg_metric is None else float(reg_metric)
                    f.attrs["registration_method"] = "phase correlation + tile similarity fit"
                    f.attrs["registration_summary"] = str(reg_note)
                    f.attrs["registration_placement_um"] = float(fit.placement_error_um)
                    f.attrs["registration_tile_loo_um"] = float(fit.loo_rmse_um)
                    f.attrs["registration_scale"] = float(fit.scale)
                    f.attrs["registration_rotation_deg"] = float(fit.rotation_deg)
                    f.attrs["native_msi_shape"] = [data_cellpose.shape[0], data_cellpose.shape[1]]
                    # fiducial square in FixedMSI.tif coordinates (angle already folded to [-45, 45])
                    f.attrs["square_center_xy"] = [float(center[0]), float(center[1])]
                    f.attrs["square_size_wh"] = [float(w), float(h)]
                    f.attrs["square_angle_deg"] = float(angle)
                    if expected_square_wh is not None:
                        f.attrs["square_expected_wh_px"] = [float(expected_square_wh[0]), float(expected_square_wh[1])]
                        f.attrs["square_size_dev_frac"] = float(square_size_dev)
                    try:
                        f.attrs["msi_pixel_size_um"] = [float(v) for v in imzml.scan_settings.pixel_size]
                    except Exception:
                        pass
                    # v5: what this h5 holds (should_skip_subdir re-runs older formats) and which
                    # run made it (stage 2 re-segments when its masks were cut from another run)
                    f.attrs["stage1_format"] = int(STAGE1_FORMAT_VERSION)
                    f.attrs["stage1_run_id"] = uuid.uuid4().hex
                    # MSI cube resized onto the optical square one mass at a time, written as it
                    # goes; identical values to the old in-memory resize
                    _cube = f.create_dataset("raw_cube", shape=(int(h), int(w), data.shape[2]),
                                             dtype=np.float32, chunks=(256, 256, 1),
                                             **h5_compression_kwargs())
                    for i in range(data.shape[2]):
                        _cube[:, :, i] = cv2.resize(data[:, :, i], (int(w), int(h)),
                                                    interpolation=cv2.INTER_LINEAR)
                    f.create_dataset("imgs_cp", data=imgs_cp, compression="gzip",
                                     chunks=(256, 256, min(3, imgs_cp.shape[-1])))
                    f.create_dataset("image_for_cellpose", data=image_for_cellpose, compression="gzip",
                                     chunks=(256, 256))
                    f.create_dataset("masses_of_interest", data=np.array(masses_of_interest, dtype=float))

                del data
                gc.collect()
                # v5: one atomic replace (remove-then-rename left a window with no h5 at all)
                os.replace(temp_h5, stage1_h5)
                print(f"[S1] Verified and Saved Data Cubes: {stage1_h5.name}", flush=True)
            except Exception as e:
                if os.path.exists(temp_h5): os.remove(temp_h5)
                raise

            tifffile.imwrite("PREMSI_stage1_ref_image.tif", image_for_cellpose.astype(np.uint8))
            return True

    except Exception as e:
        print(f"[S1] ERROR in {subdir.name}: {e}")
        # v5: the square search is deterministic, so a folder with no (or too many) squares
        # fails the same way in SAFE MODE - report it instead of spending minutes retrying
        if str(e).startswith(("Final Detection Failed", "Too many squares", "No squares detected")):
            return "NO_SQUARE"
        return False


def run_stage1_preprocess_parallel():
    t0 = time.time()
    subdirs = [d for d in MASTER_DIR.iterdir() if d.is_dir()]

    print(f"\n[S1] Launching PASS 1 on {len(subdirs)} folders with {MAX_WORKERS_STAGE1} workers...")
    with ProcessPoolExecutor(max_workers=MAX_WORKERS_STAGE1) as executor:
        results = list(executor.map(stage1_worker, subdirs))

    failed_dirs = [subdirs[i] for i, res in enumerate(results) if res is False]
    for _code, _what in (("REJECTED", "registration not credible"), ("NO_SQUARE", "no fiducial square found")):
        _names = [subdirs[i].name for i, res in enumerate(results) if res == _code]
        if _names:
            print(f"[S1] {len(_names)} folder(s) skipped, {_what} (not retried): {_names}")

    if failed_dirs:
        print(f"\n[S1] {len(failed_dirs)} folders failed. Requeueing in SAFE MODE (1 Worker)...")
        with ProcessPoolExecutor(max_workers=1) as safe_executor:
            retry_results = list(safe_executor.map(stage1_worker, failed_dirs))

        permanent_fails = [failed_dirs[i] for i, res in enumerate(retry_results) if res is False]
        if permanent_fails:
            print(f"[S1] {len(permanent_fails)} folders permanently failed. See console for details.")
        else:
            print("[S1] All requeued folders recovered successfully!")

    # Folders registered on an earlier run were skipped above, so give them the overlap
    # audit too, from the registered tiffs already on disk. No-op once every folder has it.
    if REG_AUDIT_ENABLED:
        _todo = [d for d in subdirs
                 if not (d / "PREMSI_stage1_audit_registration_overlap.png").exists()
                 and (d / "FixedMSI.tif").exists() and (d / "MovedMSI_multichannel.tif").exists()]
        if _todo:
            print(f"[S1] Back-filling the registration overlap audit for {len(_todo)} folder(s)...")
            with ProcessPoolExecutor(max_workers=MAX_WORKERS_STAGE1) as audit_executor:
                _done = list(audit_executor.map(backfill_registration_overlap_audit, _todo))
            print(f"[S1] Registration overlap audit written for {sum(bool(x) for x in _done)}/{len(_todo)} folder(s).")

    # Same idea for the imzML square-size cross-check: folders done before it existed get
    # QC_square_size.txt from their stored square, the POST tiff and the imzML header.
    _todo_sq = [d for d in subdirs
                if (d / "PREMSI_stage1_ready.h5").exists() and not (d / "QC_square_size.txt").exists()]
    if _todo_sq:
        print(f"[S1] Back-filling the imzML square-size check for {len(_todo_sq)} folder(s)...")
        with ProcessPoolExecutor(max_workers=MAX_WORKERS_STAGE1) as sq_executor:
            _devs = list(sq_executor.map(backfill_square_size_qc, _todo_sq))
        _bad = [d.name for d, v in zip(_todo_sq, _devs) if v is not None and v > SQUARE_MSI_SIZE_TOL]
        if _bad:
            print(f"[S1] ⚠️ {len(_bad)} folder(s) have a square more than {SQUARE_MSI_SIZE_TOL:.0%} off the "
                  f"imzML-expected size: {_bad}")
        else:
            print("[S1] Square sizes agree with the imzML headers for every back-filled folder.")

    print(f"[S1] DONE in {time.time() - t0:.1f} s")
