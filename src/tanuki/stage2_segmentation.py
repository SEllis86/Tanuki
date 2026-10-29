"""Stage 2 - Cellpose-SAM segmentation (serial), directional neuron merge, MSI-guided rescue,
alignment of the optical layers to the MSI grid, audit images."""
from __future__ import annotations

import io as ioio
import time
from collections import defaultdict
from pathlib import Path

import cv2
import h5py
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from cellpose import models

from .config import (CELLPOSE_BRIGHTFIELD_CHANNEL, CELLPOSE_CHANNEL_STRETCH_PERCENTILES,
    CELLPOSE_INPUT_CHANNELS, CELLPOSE_MSI_CHANNEL, CELLPOSE_MSI_CHANNEL_BLUR_MSI_PX,
    CELLPOSE_TOPHAT_KERNEL_PX, MASTER_DIR, MERGE_LINEAR_MAX_WIDTH_GAIN, MERGE_LINEAR_MAX_WIDTH_PX,
    MERGE_LINEAR_MIN_CONTACT_OF_WIDTH, MERGE_MAX_MERGED_AREA_PX, MERGE_MIN_CONTACT_FRAC, MERGE_RULE,
    MERGE_TOUCHING_MASKS, MERGE_TOUCHING_MASKS_MAX_AREA_PX, MERGE_TOUCHING_MASKS_ONLY_KEYWORDS,
    MERGE_TOUCHING_MASKS_SKIP_KEYWORDS, MSI_RESCUE_ENABLED, MSI_RESCUE_MAX_OVERLAP,
    MSI_RESCUE_MIN_AREA_MSI_PX, MSI_RESCUE_OPEN_PX, RANDOM_SEED, STAGE1_FORMAT_VERSION,
    STAGE2_AUDIT_BRIGHTFIELD_UNINVERTED, STAGE2_AUDIT_CHANNEL_COLORS, STAGE2_FORCE_RESEGMENT,
    STAGE2_ONLY_FOLDERS, STAGE2_REDRAW_AUDIT, STAGE2_REMERGE_FROM_RAW,
    STAGE2_RESEGMENT_IF_MERGE_RULE_CHANGED, STAGE2_ROTATION_MODE, usable_cores)
from .utils import msi_upscale_area, stage1_format_of, stage1_run_id_of


# ============================================================
# STAGE 2 - CELLPOSE (SERIAL ON GPU)
# ============================================================

def _length_width_from_moments(mom):
    """
    v4: (length, width) in px of the rectangle with the same pixel spread as an object, from
    its summed moments [n, sum x, sum y, sum x^2, sum y^2, sum xy]. Moments add, so the shape
    of a merged object (or a whole chain) comes straight from its parts. Each pixel counts
    as a unit square, so a single pixel is 1 x 1 rather than 0 x 0.
    """
    n = float(mom[0])
    mx, my = mom[1] / n, mom[2] / n
    cxx = mom[3] / n - mx * mx + 1.0 / 12.0
    cyy = mom[4] / n - my * my + 1.0 / 12.0
    cxy = mom[5] / n - mx * my
    half_trace = 0.5 * (cxx + cyy)
    spread = float(np.sqrt(max(half_trace * half_trace - (cxx * cyy - cxy * cxy), 0.0)))
    return (float(np.sqrt(12.0 * (half_trace + spread))),
            float(np.sqrt(12.0 * max(half_trace - spread, 1.0 / 12.0))))


def merge_touching_masks(masks, max_area_px=None, min_contact_frac=None, return_groups=False,
                         max_merged_area_px=None, rule='contact', linear_min_contact_of_width=0.5,
                         linear_max_width_gain=1.25, linear_max_width_px=None):
    """
    Union any two Cellpose label IDs that physically touch (share a 4-connected pixel
    border) into one label. Background (0) is never merged into anything.

    Adjacency is found by shifting the label image one pixel in each of the four
    directions and comparing against the original: wherever both the original and the
    shifted pixel are non-zero AND different, those two labels touch. A union-find over
    those pairs collapses whole chains (A touches B touches C) into one group in one pass,
    so a cell fragmented into many small masks along its length comes back as one object.

    max_area_px: a candidate mask already at or above this many pixels is excluded from
    every merge - checked against each mask's OWN area, not the running total of a chain.

    min_contact_frac (v2): merge a pair only if the length of their shared border is at
    least this fraction of the SMALLER mask's perimeter. Consecutive fragments of one
    over-segmented cell share almost a full width (typically 0.4-0.7); two genuinely
    separate cells that happen to touch share a short arc (well under 0.15). With this
    on, the area cap can be raised to catch big fragments without fusing real neighbours.

    max_merged_area_px (v3.4): never let a merge produce an object larger than this. Pairs are
    merged in order of their shared-border fraction, strongest first, so the clearest
    stripe joins happen before the cap is reached.

    rule (v4): 'contact' is everything above. 'linear' replaces min_contact_frac with a
    shape test, so merges extend an object instead of widening it: a pair must share at
    least linear_min_contact_of_width of the narrower piece's width, and the joined object
    (the whole group so far, not just the pair) may be at most linear_max_width_gain times
    as wide as the wider part and no wider than linear_max_width_px, unless that part
    already was, in which case it may not widen at all. Pairs that widen least go first.
    See _length_width_from_moments for how length and width are measured.

    Returns a NEW array with 1..N contiguous labels (0 stays background). With
    return_groups=True also returns {new_label: number of original masks merged into it}.
    """
    labels = np.unique(masks)
    labels = labels[labels != 0]
    if labels.size <= 1:
        return (masks.copy(), {}) if return_groups else masks.copy()

    parent = {int(l): int(l) for l in labels}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    max_label = int(labels.max())
    areas = np.bincount(masks.ravel(), minlength=max_label + 1)

    m = masks
    contact = defaultdict(int)             # unordered (lo, hi) -> touching pixels, both sides summed
    boundary = np.zeros(m.shape, dtype=bool)
    for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        shifted = np.roll(m, shift=(dy, dx), axis=(0, 1))
        valid = np.ones_like(m, dtype=bool)
        if dy == -1:
            valid[-1, :] = False
        elif dy == 1:
            valid[0, :] = False
        if dx == -1:
            valid[:, -1] = False
        elif dx == 1:
            valid[:, 0] = False
        differs = valid & (m != 0) & (m != shifted)
        boundary |= differs
        touching = differs & (shifted != 0)
        if not touching.any():
            continue
        pa = m[touching].astype(np.int64)
        pb = shifted[touching].astype(np.int64)
        lo = np.minimum(pa, pb)
        hi = np.maximum(pa, pb)
        keys, counts = np.unique(lo * (max_label + 1) + hi, return_counts=True)
        for k, c in zip(keys.tolist(), counts.tolist()):
            contact[(int(k // (max_label + 1)), int(k % (max_label + 1)))] += int(c)

    # 4-neighbour boundary pixel count per label ~ its perimeter in pixels
    perimeter = np.bincount(m[boundary].ravel(), minlength=max_label + 1)

    linear = str(rule).lower() == 'linear'
    if linear:
        # v4: per-label moments [n, sum x, sum y, sum x^2, sum y^2, sum xy]
        flat = m.ravel()
        yy, xx = np.indices(m.shape)
        xf = xx.ravel().astype(np.float64)
        yf = yy.ravel().astype(np.float64)
        moments = np.stack([areas.astype(np.float64)] + [
            np.bincount(flat, weights=w, minlength=max_label + 1)
            for w in (xf, yf, xf * xf, yf * yf, xf * yf)], axis=1)
        width = {int(l): _length_width_from_moments(moments[l])[1] for l in labels}

    n_skipped_large = 0
    n_skipped_contact = 0
    n_skipped_total = 0
    n_skipped_shape = 0
    candidates = []                        # (sort key, a, b): lowest key is merged first
    for (a, b), c in contact.items():
        if max_area_px is not None and (areas[a] >= max_area_px or areas[b] >= max_area_px):
            n_skipped_large += 1
            continue
        # each shared border pixel was counted once from either side -> halve
        shared = c / 2.0
        if linear:
            if shared < float(linear_min_contact_of_width) * min(width[a], width[b]):
                n_skipped_contact += 1
                continue
            # how much the pair widens the wider piece: end-to-end ~1.0, side-by-side ~2.0
            pair_width = _length_width_from_moments(moments[a] + moments[b])[1]
            candidates.append((pair_width / max(width[a], width[b]), a, b))
            continue
        pmin = max(1.0, float(min(perimeter[a], perimeter[b])))
        frac = shared / pmin
        if min_contact_frac and frac < float(min_contact_frac):
            n_skipped_contact += 1
            continue
        candidates.append((-frac, a, b))

    # v3.4: strongest contact first (v4 'linear': least widening first), and track each
    # group's running area so a chain of fragments cannot grow past max_merged_area_px
    group_area = {int(l): int(areas[l]) for l in labels}
    group_moments = {int(l): moments[l].copy() for l in labels} if linear else None
    for _key, a, b in sorted(candidates, key=lambda t: t[0]):
        ra, rb = find(a), find(b)
        if ra == rb:
            continue
        if max_merged_area_px is not None and group_area[ra] + group_area[rb] > max_merged_area_px:
            n_skipped_total += 1
            continue
        if linear:
            # judged on the two GROUPS, so a chain is held to the same rule at every step
            joined = group_moments[ra] + group_moments[rb]
            wider = max(_length_width_from_moments(group_moments[ra])[1],
                        _length_width_from_moments(group_moments[rb])[1])
            allowed = wider * float(linear_max_width_gain)
            if linear_max_width_px is not None:
                allowed = min(allowed, max(float(linear_max_width_px), wider))
            if _length_width_from_moments(joined)[1] > allowed:
                n_skipped_shape += 1
                continue
            group_moments[rb] = joined
        parent[ra] = rb
        group_area[rb] += group_area[ra]

    root_to_new = {}
    next_id = 1
    remap = np.zeros(max_label + 1, dtype=np.int32)
    members = defaultdict(int)
    for l in labels:
        r = find(int(l))
        if r not in root_to_new:
            root_to_new[r] = next_id
            next_id += 1
        remap[l] = root_to_new[r]
        members[root_to_new[r]] += 1

    merged = np.where(masks > 0, remap[np.clip(masks, 0, len(remap) - 1)], 0).astype(masks.dtype)
    n_before, n_after = int(labels.size), next_id - 1
    if n_after < n_before or n_skipped_large or n_skipped_contact or n_skipped_total or n_skipped_shape:
        notes = []
        if n_skipped_large:
            notes.append(f"{n_skipped_large} pair(s) skipped for exceeding {max_area_px}px")
        if n_skipped_total:
            notes.append(f"{n_skipped_total} merge(s) refused: the joined object would exceed {max_merged_area_px}px")
        if n_skipped_shape:
            notes.append(f"{n_skipped_shape} merge(s) refused: the joined object would get wider, not longer")
        if n_skipped_contact and linear:
            notes.append(f"{n_skipped_contact} pair(s) skipped for touching across less than "
                         f"{float(linear_min_contact_of_width):.0%} of the narrower piece's width")
        elif n_skipped_contact:
            notes.append(f"{n_skipped_contact} pair(s) skipped for a short shared border (< {min_contact_frac:.0%})")
        note = f" ({'; '.join(notes)})" if notes else ""
        print(f"[S2] merge_touching_masks: {n_before} masks -> {n_after} "
              f"({n_before - n_after} fragment(s) merged into their neighbours){note}.")
    if return_groups:
        return merged, dict(members)
    return merged


def msi_rescue_masks(masks, ion_image, min_area_px, max_overlap=0.30, open_px=3):
    """
    v2: add a mask for every bright ion-image blob that Cellpose left (mostly) uncovered.

    The 760 ion image is Otsu-thresholded and opened to drop speckle; each connected blob
    of at least `min_area_px` upscaled pixels whose area is covered by existing masks by
    less than `max_overlap` contributes its UNCOVERED pixels as a new label. Blobs that
    are mostly covered are the normal delocalisation halo around a segmented cell and are
    left alone. Returns (masks_int32, [new label ids]).
    """
    ion = np.asarray(ion_image)
    if ion.dtype != np.uint8:
        lo, hi = np.percentile(ion, (0.0, 99.99))
        ion = np.clip((ion.astype(np.float32) - lo) / max(hi - lo, 1e-8), 0, 1)
        ion = (ion * 255).astype(np.uint8)
    _, bin_msi = cv2.threshold(ion, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if open_px and int(open_px) > 1:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(open_px), int(open_px)))
        bin_msi = cv2.morphologyEx(bin_msi, cv2.MORPH_OPEN, k)

    n, lab, stats, _ = cv2.connectedComponentsWithStats(bin_msi, connectivity=8)
    out = masks.astype(np.int32).copy()
    covered = out > 0
    next_id = int(out.max()) + 1
    added = []
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < min_area_px:
            continue
        blob = lab == i
        overlap = float(np.count_nonzero(blob & covered)) / float(area)
        if overlap > max_overlap:
            continue
        new_px = blob & ~covered
        if np.count_nonzero(new_px) < min_area_px:
            continue
        out[new_px] = next_id
        added.append(next_id)
        next_id += 1
    return out, added


def merge_applies_to_folder(folder_name) -> bool:
    """v3.3: whether merge_touching_masks runs on this folder. See the merge settings."""
    if not MERGE_TOUCHING_MASKS:
        return False
    name_lower = str(folder_name).lower()
    if any(k.lower() in name_lower for k in MERGE_TOUCHING_MASKS_SKIP_KEYWORDS):
        return False
    if MERGE_TOUCHING_MASKS_ONLY_KEYWORDS:
        return any(k.lower() in name_lower for k in MERGE_TOUCHING_MASKS_ONLY_KEYWORDS)
    return True


def stage2_merge_rule(folder_name) -> str:
    """
    The post-Cellpose rule this folder gets under the current settings, as stored in
    PREMSI_seg.npy: the merge, and (v5) the MSI rescue, which the re-merge path re-applies
    too - a change to the rescue settings used to refresh nothing.
    """
    rescue = (f";rescue={bool(MSI_RESCUE_ENABLED)}:{MSI_RESCUE_MIN_AREA_MSI_PX}:"
              f"{MSI_RESCUE_MAX_OVERLAP}:{MSI_RESCUE_OPEN_PX}")
    if not merge_applies_to_folder(folder_name):
        return "merge=off" + rescue
    rule = (f"merge=on;rule={MERGE_RULE};max_area_px={MERGE_TOUCHING_MASKS_MAX_AREA_PX};"
            f"max_merged_area_px={MERGE_MAX_MERGED_AREA_PX}")
    if str(MERGE_RULE).lower() == 'linear':
        return rule + (f";min_contact_of_width={MERGE_LINEAR_MIN_CONTACT_OF_WIDTH}"
                       f";max_width_gain={MERGE_LINEAR_MAX_WIDTH_GAIN};max_width_px={MERGE_LINEAR_MAX_WIDTH_PX}") + rescue
    return rule + f";min_contact_frac={MERGE_MIN_CONTACT_FRAC}" + rescue


def merge_masks_for_folder(masks, folder_name):
    """
    v4: the stage 2 merge under the current settings, shared by the Cellpose path and the
    re-merge path. Returns (masks, {label: 'cellpose_merged'} for every merged object).
    """
    if not merge_applies_to_folder(folder_name):
        return masks, {}
    merged, groups = merge_touching_masks(
        masks, max_area_px=MERGE_TOUCHING_MASKS_MAX_AREA_PX, min_contact_frac=MERGE_MIN_CONTACT_FRAC,
        return_groups=True, max_merged_area_px=MERGE_MAX_MERGED_AREA_PX, rule=MERGE_RULE,
        linear_min_contact_of_width=MERGE_LINEAR_MIN_CONTACT_OF_WIDTH,
        linear_max_width_gain=MERGE_LINEAR_MAX_WIDTH_GAIN, linear_max_width_px=MERGE_LINEAR_MAX_WIDTH_PX)
    return merged, {int(l): 'cellpose_merged' for l, n in groups.items() if n > 1}


def apply_msi_rescue(masks, mask_source, image_for_cellpose, upscale_area, folder_name):
    """v2 (2): MSI-guided rescue of bright ion-image blobs no mask covers. Returns (masks, rescued ids)."""
    rescued_ids = []
    if not MSI_RESCUE_ENABLED:
        return masks, rescued_ids
    try:
        _min_px = float(MSI_RESCUE_MIN_AREA_MSI_PX) * upscale_area
        _n_before = int((np.unique(masks) > 0).sum())
        masks, rescued_ids = msi_rescue_masks(masks, image_for_cellpose, _min_px,
                                              max_overlap=MSI_RESCUE_MAX_OVERLAP, open_px=MSI_RESCUE_OPEN_PX)
        for rid in rescued_ids:
            mask_source[int(rid)] = 'msi_rescue'
        print(f"[S2] {folder_name}: MSI rescue added {len(rescued_ids)} mask(s) to {_n_before} "
              f"(min {_min_px:.0f} px = {MSI_RESCUE_MIN_AREA_MSI_PX:g} MSI px, "
              f"max existing overlap {MSI_RESCUE_MAX_OVERLAP:.0%}).")
    except Exception as e:
        print(f"[S2] {folder_name}: MSI rescue failed ({e}); continuing without it.")
    return masks, rescued_ids


def stage2_remerge_worker(subdir: Path) -> bool:
    """
    v4: re-apply the merge, and the MSI rescue that depends on it, to the UNMERGED Cellpose
    masks stored in PREMSI_seg.npy, and redraw the audit. No Cellpose. Returns False when
    the seg file has no 'masks_raw' (segmented before v4), so the caller re-segments instead.
    """
    stage1_h5 = subdir / "PREMSI_stage1_ready.h5"
    seg_out = subdir / "PREMSI_seg.npy"
    try:
        seg = np.load(seg_out, allow_pickle=True).item()
        if not isinstance(seg, dict) or "masks_raw" not in seg:
            return False
        masks_raw = np.asarray(seg["masks_raw"]).astype(np.int32)
        with h5py.File(stage1_h5, 'r') as f:
            image_for_cellpose = f["image_for_cellpose"][:]
            imgs_cp = f["imgs_cp_aligned"][:] if "imgs_cp_aligned" in f else f["imgs_cp"][:]
            native_msi_shape = (tuple(int(v) for v in f.attrs["native_msi_shape"])
                                if "native_msi_shape" in f.attrs else None)
    except Exception as e:
        print(f"[S2] {subdir.name}: cannot re-merge from stored masks ({e}).")
        return False

    print(f"[S2] ▶ {subdir.name}  (re-merging the stored Cellpose masks, no Cellpose run)")
    masks, mask_source = merge_masks_for_folder(masks_raw, subdir.name)
    masks, rescued_ids = apply_msi_rescue(masks.astype(np.int32), mask_source, image_for_cellpose,
                                          msi_upscale_area(image_for_cellpose, native_msi_shape), subdir.name)
    np.save(seg_out, {"masks": masks, "masks_raw": masks_raw, "mask_source": mask_source,
                      "merge_rule": stage2_merge_rule(subdir.name),
                      "stage1_run_id": seg.get("stage1_run_id", "")}, allow_pickle=True)
    try:
        img_rgb_8bit = build_cellpose_input(imgs_cp)
        if CELLPOSE_INPUT_CHANNELS == 'all':
            imgs_for_model = img_rgb_8bit
        else:
            single = img_rgb_8bit[:, :, int(CELLPOSE_INPUT_CHANNELS)]
            imgs_for_model = np.stack([single, single, single], axis=-1)
        write_stage2_segmentation_audit(subdir, imgs_for_model, image_for_cellpose, masks, rescued_ids)
    except Exception as e:
        print(f"[S2] Audit plot failed for {subdir.name}: {e}")
    return True


def seg_merge_rule_is_stale(subdir: Path) -> bool:
    """True when PREMSI_seg.npy was made with a different merge rule (or records none)."""
    seg_path = subdir / "PREMSI_seg.npy"
    if not seg_path.exists():
        return False
    try:
        seg = np.load(seg_path, allow_pickle=True).item()
        recorded = seg.get("merge_rule") if isinstance(seg, dict) else None
    except Exception:
        return True
    return recorded != stage2_merge_rule(subdir.name)


def seg_stage1_mismatch(subdir: Path) -> bool:
    """
    v5: True when PREMSI_seg.npy was cut from a DIFFERENT stage 1 h5 than the one on disk
    (stage 1 re-ran, or the seg predates v5). Its masks then belong to another crop, so the
    folder must be re-segmented - re-merging the stored masks would not help.
    """
    seg_path = subdir / "PREMSI_seg.npy"
    if not seg_path.exists():
        return False
    try:
        seg = np.load(seg_path, allow_pickle=True).item()
        recorded = seg.get("stage1_run_id", "") if isinstance(seg, dict) else ""
    except Exception:
        return True
    return recorded != stage1_run_id_of(subdir)


def build_cellpose_input(imgs_cp):
    """
    The 3-channel uint8 stack Cellpose is shown, from the cropped optical layers:
    standardise to 3 channels, stretch each on its own percentiles, top-hat the
    fluorescence channels, invert the brightfield channel. Shared by the stage 2 worker
    and the audit redraw so the two can never drift apart.
    """
    imgs_cp = np.asarray(imgs_cp)
    if imgs_cp.ndim == 2:
        imgs_cp = imgs_cp[:, :, np.newaxis]
    grayscale_channel_idx = CELLPOSE_BRIGHTFIELD_CHANNEL

    h, w, c = imgs_cp.shape

    # 1. Standardize to a 3-channel array (Native RGB order from your PRE image)
    if c >= 3:
        imgs_3ch = imgs_cp[:, :, :3].copy()
    elif c == 2:
        zero_channel = np.zeros((h, w, 1), dtype=imgs_cp.dtype)
        imgs_3ch = np.concatenate([imgs_cp, zero_channel], axis=-1)
    else:
        # FIX: imgs_cp is ALREADY 3-D here - the `if imgs_cp.ndim == 2` guard above
        # guarantees a trailing channel axis - so a single-channel image is (h, w, 1).
        # Indexing it with np.newaxis added a SECOND axis, giving (h, w, 3, 1) rather
        # than (h, w, 3). img_rgb_8bit[:, :, i] was then (h, w, 1) while
        # cv2.morphologyEx returns (h, w), so the tophat assignment below raised
        # "could not broadcast input array from shape (h,w) into shape (h,w,1)" on
        # every single-channel folder. Repeat along the existing axis instead.
        imgs_3ch = np.repeat(imgs_cp, 3, axis=2)

    # 2. v2: stretch EACH channel to its own percentile window before stacking. The old
    # single global maximum let the brightest channel set the scale for all three, so a
    # strong Hoechst channel could push the membrane stain to near black.
    _p_lo, _p_hi = CELLPOSE_CHANNEL_STRETCH_PERCENTILES
    img_rgb_8bit = np.zeros(imgs_3ch.shape[:2] + (3,), dtype=np.uint8)
    for i in range(3):
        ch = imgs_3ch[:, :, i].astype(np.float32)
        lo, hi = np.percentile(ch, (_p_lo, _p_hi))
        if hi > lo:
            img_rgb_8bit[:, :, i] = (np.clip((ch - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8)
        elif ch.max() > 0:
            img_rgb_8bit[:, :, i] = (ch / ch.max() * 255).astype(np.uint8)

    # notes: NEW - Morphological Top-Hat Filter to erase microscope stitching grids
    # notes: The kernel size MUST be larger than your biggest cells.
    # v2: kernel is a setting (None = off) and defaults to 401 px so big flat cells survive.
    if CELLPOSE_TOPHAT_KERNEL_PX:
        _k = int(CELLPOSE_TOPHAT_KERNEL_PX) | 1          # odd
        tophat_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (_k, _k))
        for i in range(3):
            # notes: Only apply this to the raw fluorescent channels, not the brightfield
            if grayscale_channel_idx is not None and i == grayscale_channel_idx:
                continue
            img_rgb_8bit[:, :, i] = cv2.morphologyEx(img_rgb_8bit[:, :, i], cv2.MORPH_TOPHAT, tophat_kernel)

    # 3. Invert the grayscale brightfield channel if one exists (Dark-on-light -> Light-on-dark)
    if grayscale_channel_idx is not None:
        img_rgb_8bit[:, :, grayscale_channel_idx] = 255 - img_rgb_8bit[:, :, grayscale_channel_idx]
    if c == 1:
        # v5: one optical plane was repeated into all three channels above and then processed
        # per channel, so channel 0 came out inverted (brightfield) while 1 and 2 were
        # top-hatted and NOT inverted: the same image twice in opposite polarities
        # (correlation -1.000 on the brightfield-only 11302 wells). Give the model channel 0's
        # result in all three.
        img_rgb_8bit = np.repeat(img_rgb_8bit[:, :, :1], 3, axis=2)
    return img_rgb_8bit


def stage2_audit_display_image(imgs_for_model):
    """
    The model input recoloured for the audit panel. DISPLAY ONLY: Cellpose is still given
    the stack with the brightfield channel inverted (build_cellpose_input).

    v3.2, STAGE2_AUDIT_BRIGHTFIELD_UNINVERTED = True: the brightfield channel is flipped
    back to how the microscope shows it (bright field, dark cells) and drawn as the grey
    base; the other channels are alpha-blended over it in their STAGE2_AUDIT_CHANNEL_COLORS
    colour, opacity = their own intensity. A plain additive mix would clip to white on a
    bright background and hide the fluorescence. False = the v3.1 additive view of the
    inverted stack. When Cellpose was shown ONE channel repeated three times, that plane
    is returned as greyscale (un-inverted too if it is the brightfield channel).
    """
    img = np.asarray(imgs_for_model)
    if img.ndim == 2:
        img = img[:, :, np.newaxis]
    img = img[:, :, :3]
    bf = CELLPOSE_BRIGHTFIELD_CHANNEL
    uninvert = bool(STAGE2_AUDIT_BRIGHTFIELD_UNINVERTED) and bf is not None

    _one_plane = (img.shape[2] >= 3 and np.array_equal(img[:, :, 0], img[:, :, 1])
                  and np.array_equal(img[:, :, 0], img[:, :, 2]))
    if CELLPOSE_INPUT_CHANNELS != 'all' or img.shape[2] < 3 or _one_plane:
        plane = np.ascontiguousarray(img[:, :, 0]).astype(np.uint8)
        if uninvert and (CELLPOSE_INPUT_CHANNELS == bf or (_one_plane and bf == 0)):
            plane = 255 - plane
        return cv2.cvtColor(plane, cv2.COLOR_GRAY2RGB)

    def _colour(ch_idx):
        return np.asarray(STAGE2_AUDIT_CHANNEL_COLORS.get(int(ch_idx), (1.0, 1.0, 1.0)), dtype=np.float32)

    if not uninvert or not (0 <= int(bf) < img.shape[2]):
        out = np.zeros(img.shape[:2] + (3,), dtype=np.float32)
        for ch_idx in range(img.shape[2]):
            if int(ch_idx) in STAGE2_AUDIT_CHANNEL_COLORS:
                out += img[:, :, ch_idx, np.newaxis].astype(np.float32) * _colour(ch_idx)
        return np.clip(out, 0, 255).astype(np.uint8)

    bf = int(bf)
    base = (255.0 - img[:, :, bf].astype(np.float32))[:, :, np.newaxis] * _colour(bf)
    fluor = np.zeros(img.shape[:2] + (3,), dtype=np.float32)
    alpha = np.zeros(img.shape[:2], dtype=np.float32)
    for ch_idx in range(img.shape[2]):
        if ch_idx == bf or ch_idx not in STAGE2_AUDIT_CHANNEL_COLORS:
            continue
        plane = img[:, :, ch_idx].astype(np.float32)
        fluor += plane[:, :, np.newaxis] * _colour(ch_idx)
        alpha = np.maximum(alpha, plane / 255.0)
    out = base * (1.0 - alpha[:, :, np.newaxis]) + fluor
    return np.clip(out, 0, 255).astype(np.uint8)


def get_consistent_seg_overlay(img, msk, lut, outline_labels=None):
    # Normalize base image to 8-bit RGB
    if img.max() <= 1.0:
        base = (img * 255).astype(np.uint8)
    else:
        base = img.astype(np.uint8).copy()

    if base.ndim == 2 or (base.ndim == 3 and base.shape[-1] == 1):
        base = cv2.cvtColor(base, cv2.COLOR_GRAY2RGB)
    elif base.shape[-1] > 3:
        base = base[:, :, :3]

    overlay = base.copy()

    # Instantly map every pixel to its assigned color using the LUT
    color_mapped = lut[msk]

    # Apply semi-transparent fill
    region_mask = msk > 0
    overlay[region_mask] = color_mapped[region_mask]
    cv2.addWeighted(overlay, 0.3, base, 0.7, 0, overlay)  # 30% fill opacity

    # Draw solid boundaries so touching cells remain visually distinct
    msk_shift_d = np.roll(msk, 1, axis=0)
    msk_shift_r = np.roll(msk, 1, axis=1)
    boundaries = (msk > 0) & ((msk != msk_shift_d) | (msk != msk_shift_r))

    overlay[boundaries] = color_mapped[boundaries]

    # v2: rescued (MSI-only) masks get a solid white outline so they stand out
    if outline_labels:
        ol = np.isin(msk, np.asarray(list(outline_labels))).astype(np.uint8)
        cnts, _ = cv2.findContours(ol, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # v3.2: black under white, so the outline also shows on a bright (un-inverted) field
        cv2.drawContours(overlay, cnts, -1, (0, 0, 0), 4)
        cv2.drawContours(overlay, cnts, -1, (255, 255, 255), 2)

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(overlay)
    ax.axis("off")
    buf = ioio.BytesIO()
    plt.savefig(buf, format="png", dpi=200, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def write_stage2_segmentation_audit(subdir, imgs_for_model, image_for_cellpose, masks, rescued_ids=None):
    """
    PREMSI_stage2_audit_segmentation.png: the masks over the optical model input (top,
    recoloured by stage2_audit_display_image) and over the 760 ion image (bottom), with
    the same colour per mask in both and MSI-rescued masks outlined in white.
    """
    masks = np.asarray(masks).astype(np.int32)
    max_mask_id = int(masks.max()) if masks.size else 0
    # FIX: np.random.seed() reseeds the GLOBAL numpy RNG for the rest of the process. Use a
    # local Generator so a cosmetic colour LUT can't silently make anything else downstream
    # deterministic (or repeat identical "random" draws).
    lut_rng = np.random.default_rng(RANDOM_SEED)
    color_lut = lut_rng.integers(50, 255, size=(max_mask_id + 1, 3), dtype=np.uint8)
    color_lut[0] = [0, 0, 0]  # Background (ID 0) must remain black

    # Pass the exact same color LUT to both overlays!
    seg1 = get_consistent_seg_overlay(stage2_audit_display_image(imgs_for_model), masks, color_lut,
                                      outline_labels=rescued_ids)
    seg2 = get_consistent_seg_overlay(image_for_cellpose, masks, color_lut, outline_labels=rescued_ids)

    width = max(seg1.width, seg2.width)
    total_height = seg1.height + seg2.height
    combined = Image.new("RGB", (width, total_height), (255, 255, 255))
    combined.paste(seg1, (0, 0))
    combined.paste(seg2, (0, seg1.height))
    combined.save(Path(subdir) / "PREMSI_stage2_audit_segmentation.png")


def redraw_stage2_audit(subdir: Path) -> bool:
    """
    v3.1: rebuild PREMSI_stage2_audit_segmentation.png for a folder that is already
    segmented, from PREMSI_stage1_ready.h5 and PREMSI_seg.npy. No Cellpose. The optical
    layers come from imgs_cp_aligned (what stage 2 saved after its rotation/shift), so
    they sit on the same grid as the masks. Returns True when the image was written.
    """
    stage1_h5 = subdir / "PREMSI_stage1_ready.h5"
    seg_path = subdir / "PREMSI_seg.npy"
    if not stage1_h5.exists() or not seg_path.exists():
        return False
    try:
        with h5py.File(stage1_h5, 'r') as f:
            image_for_cellpose = f["image_for_cellpose"][:]
            imgs_cp = f["imgs_cp_aligned"][:] if "imgs_cp_aligned" in f else f["imgs_cp"][:]
        seg = np.load(seg_path, allow_pickle=True).item()
        if isinstance(seg, dict):
            masks = seg["masks"]
            source = dict(seg.get("mask_source", {}) or {})
        else:
            masks, source = seg, {}
        masks = np.asarray(masks)
        img_rgb_8bit = build_cellpose_input(imgs_cp)
        if masks.shape[:2] != img_rgb_8bit.shape[:2] or masks.shape[:2] != image_for_cellpose.shape[:2]:
            print(f"[S2] {subdir.name}: audit redraw skipped (mask {masks.shape} vs optical "
                  f"{img_rgb_8bit.shape[:2]} vs ion image {image_for_cellpose.shape[:2]}).")
            return False
        if CELLPOSE_INPUT_CHANNELS == 'all':
            imgs_for_model = img_rgb_8bit
        else:
            single = img_rgb_8bit[:, :, int(CELLPOSE_INPUT_CHANNELS)]
            imgs_for_model = np.stack([single, single, single], axis=-1)
        rescued_ids = [int(k) for k, v in source.items() if v == 'msi_rescue']
        write_stage2_segmentation_audit(subdir, imgs_for_model, image_for_cellpose, masks, rescued_ids)
        return True
    except Exception as e:
        print(f"[S2] {subdir.name}: audit redraw failed ({e})")
        return False


def stage2_cellpose_worker(subdir: Path, model, force=False):
    # notes: Updated to read from the HDF5 file
    stage1_h5 = subdir / "PREMSI_stage1_ready.h5"
    seg_out = subdir / "PREMSI_seg.npy"

    if not stage1_h5.exists():
        return
    if seg_out.exists() and not (STAGE2_FORCE_RESEGMENT or force):
        return
    # v5: never segment an h5 in an older format (8-bit imgs_cp); stage 1 redoes those
    if stage1_format_of(subdir) < STAGE1_FORMAT_VERSION:
        print(f"[S2] {subdir.name}: stage 1 h5 is an older format (< {STAGE1_FORMAT_VERSION}); "
              f"skipped until stage 1 has re-run it.")
        return
    stage1_run_id = stage1_run_id_of(subdir)

    print(f"[S2] ▶ {subdir.name}" + ("  (forced re-segmentation)" if seg_out.exists() else ""))
    # FIX: removed os.chdir here. Every path in this function is already absolute, and a
    # process-wide chdir is a footgun the moment stage 2 is ever parallelised.

    # notes: Set this to 0 (Red), 1 (Green), or 2 (Blue) if you have a brightfield channel.
    # notes: Set to None if all optical channels are standard fluorescence.
    grayscale_channel_idx = CELLPOSE_BRIGHTFIELD_CHANNEL

    # notes: Set to 'all' for full RGB, or 0, 1, or 2 to isolate and stack a single channel
    target_channel_for_cellpose = CELLPOSE_INPUT_CHANNELS

    with h5py.File(stage1_h5, 'r') as data:
        image_for_cellpose = data["image_for_cellpose"][:]
        imgs_cp = data["imgs_cp"][:]
        native_msi_shape = (tuple(int(v) for v in data.attrs["native_msi_shape"])
                            if "native_msi_shape" in data.attrs else None)

    upscale_area = msi_upscale_area(image_for_cellpose, native_msi_shape)

    if imgs_cp.ndim == 2:
        imgs_cp = imgs_cp[:, :, np.newaxis]

    # v3.1: the stretch / top-hat / invert now lives in build_cellpose_input (shared with the audit redraw)
    img_rgb_8bit = build_cellpose_input(imgs_cp)

    # 4. Create the BGR copy strictly for OpenCV to save to your hard drive
    img_bgr_audit = cv2.cvtColor(img_rgb_8bit, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(subdir / "imgs_for_model_stage2_audit_segmentation.png"), img_bgr_audit)

    # 5. Format for CellposeSAM based on channel selection
    if target_channel_for_cellpose == 'all':
        imgs_for_model = img_rgb_8bit.copy()
    elif target_channel_for_cellpose in [0, 1, 2]:
        single_channel = img_rgb_8bit[:, :, target_channel_for_cellpose]
        imgs_for_model = np.stack([single_channel, single_channel, single_channel], axis=-1)
    else:
        raise ValueError("target_channel_for_cellpose must be 0, 1, 2, or 'all'")

    # notes: Dynamically adjust Cellpose flow and probability thresholds based on biological morphology
    folder_name_lower = subdir.name.lower()

    if 'neuro' in folder_name_lower:
        # Neurons: Sprawling networks.
        cp_flow = 0.0
        # Brightfield-only wells (one optical channel, the 11302 Neurons_*_S3B1 set): at -6
        # Cellpose tiles ~98% of the field, background included. Stained wells need -6.
        cp_prob = -2.0 if imgs_cp.shape[2] == 1 else -6.0
        # Force a massive iteration limit so flows can trace all the way back up long axons
        cp_niter = 2500
    elif 'astro' in folder_name_lower or 'micro' in folder_name_lower:
        # Astrocytes / Microglia: Singular cell bodies.
        cp_flow = 0.0
        cp_prob = -6.0
        # None tells Cellpose to just auto-calculate based on cell diameter
        cp_niter = None
    else:
        cp_flow = 0.4
        cp_prob = 0.0
        cp_niter = None

    masks, flows, styles = model.eval(
        imgs_for_model,
        batch_size=64,
        flow_threshold=cp_flow,
        cellprob_threshold=cp_prob,
        niter=cp_niter
    )

    # v4: the merge now runs AFTER the alignment below, on the aligned masks, so the unmerged
    # and merged masks share one frame and both can be saved. The alignment only ever looks
    # at masks > 0, which a merge does not change, so moving the merge leaves it untouched.

    # =========================================================================
    # ⬅️ NEW: ITERATIVE 90-DEGREE ROTATION + SPATIAL WIGGLE ALIGNMENT
    # =========================================================================
    try:
        # notes: PAD-TO-SQUARE so every 90-degree rotation is geometrically exact (no resize/stretch needed).
        orig_h, orig_w = image_for_cellpose.shape[:2]
        square_side = max(orig_h, orig_w)
        pad_top = (square_side - orig_h) // 2
        pad_bottom = square_side - orig_h - pad_top
        pad_left = (square_side - orig_w) // 2
        pad_right = square_side - orig_w - pad_left

        def pad_to_square(arr_2d):
            return cv2.copyMakeBorder(arr_2d, pad_top, pad_bottom, pad_left, pad_right,
                                      cv2.BORDER_CONSTANT, value=0)

        def unpad_from_square(arr_2d):
            return arr_2d[pad_top:pad_top + orig_h, pad_left:pad_left + orig_w]

        image_for_cellpose_sq = pad_to_square(image_for_cellpose)
        _, bin_msi_sq = cv2.threshold(image_for_cellpose_sq, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        bin_msi_bool_sq = bin_msi_sq > 0
        # notes: bin_msi (unpadded) is still needed downstream for the audit overlay image.
        _, bin_msi = cv2.threshold(image_for_cellpose, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        masks_sq = pad_to_square(masks.astype(np.float32)).astype(masks.dtype)

        rotations = [
            (0, None),
            (90, cv2.ROTATE_90_CLOCKWISE),
            (180, cv2.ROTATE_180),
            (270, cv2.ROTATE_90_COUNTERCLOCKWISE)
        ]

        # v2: the rotation is a known, data-set-wide constant (STAGE2_ROTATION_MODE), not
        # something to re-discover per well - on dense wells the four overlap scores are
        # indistinguishable and the search picks at random. 'search' keeps the old path.
        _mode = STAGE2_ROTATION_MODE
        if isinstance(_mode, str) and _mode.lower() == 'none':
            rotations = [(0, None)]
            print(f"[{subdir.name}] Phase 1: rotation fixed to none (STAGE2_ROTATION_MODE='none').")
        elif isinstance(_mode, (int, np.integer)) and not isinstance(_mode, bool):
            rotations = [r for r in rotations if r[1] == int(_mode)]
            if not rotations:
                raise ValueError(f"STAGE2_ROTATION_MODE={_mode!r} is not a cv2 rotation code (0, 1, 2)")
            print(f"[{subdir.name}] Phase 1: rotation fixed to cv2 code {int(_mode)} "
                  f"({rotations[0][0]} deg) (STAGE2_ROTATION_MODE).")
        else:
            print(f"[{subdir.name}] Phase 1: testing 4 rotations on padded square canvas (zero offset)...")

        best_score = -1
        best_rot_code = None
        rotated_candidates = {}
        rotation_scores = {}
        for angle, cv2_rot_code in rotations:
            rot_masks_sq = cv2.rotate(masks_sq, cv2_rot_code) if cv2_rot_code is not None else masks_sq.copy()
            rotated_candidates[cv2_rot_code] = rot_masks_sq

            rot_bin_optical_sq = (rot_masks_sq > 0).astype(np.uint8) * 255
            score = np.sum((rot_bin_optical_sq > 0) & bin_msi_bool_sq)
            rotation_scores[angle] = int(score)

            if score > best_score:
                best_score = score
                best_rot_code = cv2_rot_code

        if len(rotation_scores) > 1:
            _ranked = sorted(rotation_scores.values(), reverse=True)
            _margin = _ranked[0] / max(_ranked[1], 1)
            print(f"[{subdir.name}] Phase 1 overlap by rotation (deg): {rotation_scores}; "
                  f"best/second = {_margin:.2f}" + ("  <-- AMBIGUOUS" if _margin < 1.15 else ""))

        # notes: PHASE 2 - fine-tune translation only for the winning rotation.
        # FIX: the old comment claimed 441 checks, but range(-50, 51, 2) is 51 values, so
        # this was 51*51-1 = 2600 full warpAffine calls per folder. A coarse pass at 8 px
        # followed by a +/-8 px refinement at 1 px is ~330 warps AND lands on a finer
        # optimum than the old 2 px grid could reach.
        winning_rot_masks_sq = rotated_candidates[best_rot_code]

        def _score_shift(tx, ty):
            if tx == 0 and ty == 0:
                shifted = winning_rot_masks_sq
            else:
                M = np.float32([[1, 0, tx], [0, 1, ty]])
                shifted = cv2.warpAffine(winning_rot_masks_sq, M,
                                         (winning_rot_masks_sq.shape[1], winning_rot_masks_sq.shape[0]),
                                         flags=cv2.INTER_NEAREST)
            binar = (shifted > 0).astype(np.uint8) * 255
            return int(np.sum((binar > 0) & bin_msi_bool_sq)), shifted, binar

        best_masks_sq = winning_rot_masks_sq
        best_bin_optical_sq = (winning_rot_masks_sq > 0).astype(np.uint8) * 255
        best_tx, best_ty = 0, 0

        print(f"[{subdir.name}] Phase 2: winning rotation={best_rot_code}, coarse shift search (+/- 48px @ 8px)...")
        for tx in range(-48, 49, 8):
            for ty in range(-48, 49, 8):
                if tx == 0 and ty == 0:
                    continue
                score, shifted, binar = _score_shift(tx, ty)
                # Strict greater-than: only accept a shift that objectively beats the
                # rotation-only baseline.
                if score > best_score:
                    best_score, best_masks_sq, best_bin_optical_sq = score, shifted, binar
                    best_tx, best_ty = tx, ty

        coarse_tx, coarse_ty = best_tx, best_ty
        print(f"[{subdir.name}] Phase 2b: refining around (X:{coarse_tx}, Y:{coarse_ty}) at 1px...")
        for tx in range(coarse_tx - 8, coarse_tx + 9):
            for ty in range(coarse_ty - 8, coarse_ty + 9):
                if tx == coarse_tx and ty == coarse_ty:
                    continue
                score, shifted, binar = _score_shift(tx, ty)
                if score > best_score:
                    best_score, best_masks_sq, best_bin_optical_sq = score, shifted, binar
                    best_tx, best_ty = tx, ty

        if best_rot_code is not None or best_tx != 0 or best_ty != 0:
            print(
                f"[{subdir.name}] Applied corrective rotation ({best_rot_code}) and translation (X:{best_tx}, Y:{best_ty}) to optical layers.")
            masks = unpad_from_square(best_masks_sq)
            bin_optical = unpad_from_square(best_bin_optical_sq)

            M_trans_final = np.float32([[1, 0, best_tx], [0, 1, best_ty]])

            def transform_channel(ch):
                ch_sq = pad_to_square(ch)
                if best_rot_code is not None:
                    ch_sq = cv2.rotate(ch_sq, best_rot_code)
                if best_tx != 0 or best_ty != 0:
                    ch_sq = cv2.warpAffine(ch_sq, M_trans_final, (ch_sq.shape[1], ch_sq.shape[0]),
                                           flags=cv2.INTER_LINEAR)
                return unpad_from_square(ch_sq)

            # Apply identical corrective rotation & translation to the Raw optical arrays (imgs_cp)
            new_imgs_cp = np.zeros_like(imgs_cp)
            for c_idx in range(imgs_cp.shape[2]):
                new_imgs_cp[:, :, c_idx] = transform_channel(imgs_cp[:, :, c_idx])

            imgs_cp = new_imgs_cp

            # FIX: this used to `del f["imgs_cp"]` and rewrite it in place, which broke
            # stage separation three ways: stage 1's output was no longer reproducible from
            # disk, every re-run of stage 2 pushed imgs_cp through another INTER_LINEAR warp
            # so blur compounded, and HDF5 never reclaims deleted space so the file grew each
            # time. Write the aligned copy as a SEPARATE dataset and leave the original alone.
            with h5py.File(stage1_h5, 'a') as f:
                if "imgs_cp_aligned" in f:
                    del f["imgs_cp_aligned"]
                f.create_dataset("imgs_cp_aligned", data=imgs_cp, compression="gzip",
                                 chunks=(256, 256, min(3, imgs_cp.shape[-1])))
                f.attrs["stage2_rot_code"] = -1 if best_rot_code is None else int(best_rot_code)
                f.attrs["stage2_shift_x"] = int(best_tx)
                f.attrs["stage2_shift_y"] = int(best_ty)

            # Apply identical corrective rotation & translation to imgs_for_model for audit plots
            new_imgs_for_model = np.zeros_like(imgs_for_model)
            for c_idx in range(imgs_for_model.shape[2]):
                new_imgs_for_model[:, :, c_idx] = transform_channel(imgs_for_model[:, :, c_idx])
            imgs_for_model = new_imgs_for_model
        else:
            bin_optical = (masks > 0).astype(np.uint8) * 255
            # v5: no correction this time, so an imgs_cp_aligned / stage2_* left by an earlier
            # run would put the viewer's optical crop in a different frame from these masks
            with h5py.File(stage1_h5, 'a') as f:
                if "imgs_cp_aligned" in f:
                    del f["imgs_cp_aligned"]
                for _a in ("stage2_rot_code", "stage2_shift_x", "stage2_shift_y"):
                    if _a in f.attrs:
                        del f.attrs[_a]

    except Exception as e:
        # v5: this used to carry on with the masks in whatever frame they had reached, while
        # the ion image and the cube stay in the MSI frame - stage 3 then measured the wrong
        # pixels with only this line to show for it. Skip the folder instead, and remove an
        # older seg file so stage 3 cannot measure stale masks either.
        print(f"🛑 [S2] {subdir.name}: alignment to the ion image failed ({e}); folder skipped.")
        if seg_out.exists():
            try:
                seg_out.unlink()
            except OSError:
                pass
        return
    # =========================================================================

    masks = masks.astype(np.int32)
    # v2: per-label provenance, carried into stage 3 as the mask_source column
    masks_raw = masks.copy()
    masks, mask_source = merge_masks_for_folder(masks, subdir.name)

    # v2 (3): optional second pass with the ion image as the third channel. Only valid
    # AFTER the alignment above, which put the optical layers onto the MSI grid.
    if CELLPOSE_MSI_CHANNEL:
        try:
            _sigma = float(CELLPOSE_MSI_CHANNEL_BLUR_MSI_PX) * float(np.sqrt(upscale_area))
            _msi_ch = cv2.GaussianBlur(image_for_cellpose, (0, 0), max(_sigma, 0.1)) if _sigma > 0 else image_for_cellpose
            _msi_ch = cv2.resize(_msi_ch, (imgs_for_model.shape[1], imgs_for_model.shape[0]))
            imgs_pass2 = imgs_for_model.copy()
            imgs_pass2[:, :, 2] = _msi_ch.astype(np.uint8)
            print(f"[S2] {subdir.name}: second Cellpose pass with the 760 ion image as channel 3...")
            masks2, _, _ = model.eval(imgs_pass2, batch_size=64, flow_threshold=cp_flow,
                                      cellprob_threshold=cp_prob, niter=cp_niter)
            print(f"[S2] {subdir.name}: MSI-channel pass gave {int((np.unique(masks2) > 0).sum())} masks "
                  f"(optical-only pass had {int((np.unique(masks_raw) > 0).sum())}); using the MSI-channel result.")
            masks_raw = masks2.astype(np.int32)
            masks, mask_source = merge_masks_for_folder(masks_raw.copy(), subdir.name)
            imgs_for_model = imgs_pass2
        except Exception as e:
            print(f"[S2] {subdir.name}: MSI-channel pass failed ({e}); keeping the optical-only masks.")

    # v2 (2): MSI-guided rescue of bright ion-image blobs no mask covers
    masks, rescued_ids = apply_msi_rescue(masks.astype(np.int32), mask_source, image_for_cellpose,
                                          upscale_area, subdir.name)

    # v4: masks_raw = Cellpose's own masks (aligned, before merge and rescue), so a change to
    # the merge settings can be re-applied without Cellpose (STAGE2_REMERGE_FROM_RAW)
    seg_data = {"masks": masks, "masks_raw": masks_raw, "mask_source": mask_source,
                "merge_rule": stage2_merge_rule(subdir.name),
                "stage1_run_id": stage1_run_id}
    np.save(seg_out, seg_data, allow_pickle=True)

    # v3.1: top panel recoloured (channel 0 grey, 1 green, 2 blue), see STAGE2_AUDIT_CHANNEL_COLORS
    try:
        write_stage2_segmentation_audit(subdir, imgs_for_model, image_for_cellpose, masks, rescued_ids)
    except Exception as e:
        print(f"[S2] Audit plot failed for {subdir.name}: {e}")

    # notes: Final binary mask vs binary 760m/z registration check
    try:
        # Create a color blind friendly composite (BGR format for OpenCV)
        # Optical = Green (0, 255, 0)
        # MSI = Magenta (255, 0, 255)
        # Overlap = White (255, 255, 255)
        reg_audit = np.zeros((image_for_cellpose.shape[0], image_for_cellpose.shape[1], 3), dtype=np.uint8)
        reg_audit[:, :, 1] = bin_optical  # Green
        reg_audit[:, :, 0] = bin_msi  # Blue (part of Magenta)
        reg_audit[:, :, 2] = bin_msi  # Red (part of Magenta)

        cv2.imwrite(str(subdir / "PREMSI_stage2_audit_binary_registration.png"), reg_audit)
    except Exception as e:
        print(f"[S2] Binary registration audit failed for {subdir.name}: {e}")


def run_stage2_cellpose_serial():
    t0 = time.time()
    subdirs = [d for d in MASTER_DIR.iterdir() if d.is_dir()]

    print("\n[S2] Initializing PyTorch Cellpose Model on GPU...")
    import torch
    torch.backends.cudnn.benchmark = True
    # v4: the OMP_NUM_THREADS=1 set at the top (for the stage 1/3 worker pools) also held
    # torch to ONE thread, and the .venv torch is a CPU build, so Cellpose ran on one core.
    # Stage 2 is serial in this process, so give torch the cores: on the GBA set a neuron
    # well went from ~12 min to ~40 s.
    torch.set_num_threads(usable_cores)
    print(f"[S2] torch: {torch.get_num_threads()} CPU thread(s), CUDA available: {torch.cuda.is_available()}")

    model = models.CellposeModel(gpu=True)
    if STAGE2_ONLY_FOLDERS:
        _only = {str(n) for n in STAGE2_ONLY_FOLDERS}
        subdirs = [d for d in subdirs if d.name in _only]
        print(f"[S2] STAGE2_ONLY_FOLDERS set: restricting stage 2 to {len(subdirs)} folder(s): "
              f"{[d.name for d in subdirs]}")
    print(f"[S2] Model loaded successfully. Processing {len(subdirs)} folders...")

    n_redrawn = 0
    n_stale = 0
    n_remerged = 0
    n_new_stage1 = 0
    for subdir in subdirs:
        has_seg = (subdir / "PREMSI_seg.npy").exists()
        has_h5 = (subdir / "PREMSI_stage1_ready.h5").exists()
        # v5: masks cut from a different stage 1 h5 belong to another crop - always re-segment
        new_stage1 = has_seg and has_h5 and not STAGE2_FORCE_RESEGMENT and seg_stage1_mismatch(subdir)
        stale = new_stage1 or (has_seg and has_h5 and not STAGE2_FORCE_RESEGMENT
                               and STAGE2_RESEGMENT_IF_MERGE_RULE_CHANGED and seg_merge_rule_is_stale(subdir))
        if new_stage1:
            n_new_stage1 += 1
            print(f"[S2] {subdir.name}: stage 1 output changed since these masks were made; re-segmenting.")
        elif stale:
            n_stale += 1
            if STAGE2_REMERGE_FROM_RAW and stage2_remerge_worker(subdir):
                n_remerged += 1
                continue
            print(f"[S2] {subdir.name}: masks on disk were made with a different merge rule; "
                  f"re-segmenting with '{stage2_merge_rule(subdir.name)}'.")
        if has_seg and not STAGE2_FORCE_RESEGMENT and not stale:
            if STAGE2_REDRAW_AUDIT:
                n_redrawn += int(redraw_stage2_audit(subdir))
            continue
        stage2_cellpose_worker(subdir, model, force=stale)
    if n_stale:
        print(f"[S2] {n_stale} folder(s) had a changed merge rule: {n_remerged} re-merged from their stored "
              f"Cellpose masks, {n_stale - n_remerged} re-segmented (STAGE2_RESEGMENT_IF_MERGE_RULE_CHANGED).")
    if n_new_stage1:
        print(f"[S2] {n_new_stage1} folder(s) re-segmented because stage 1 re-ran under them.")
    if n_redrawn:
        print(f"[S2] Redrew PREMSI_stage2_audit_segmentation.png for {n_redrawn} already-segmented "
              f"folder(s) (STAGE2_REDRAW_AUDIT; no Cellpose run).")

    print(f"[S2] DONE in {time.time() - t0:.1f} s")
