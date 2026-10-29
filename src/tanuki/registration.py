"""PRE -> POST optical registration by phase correlation + tiles (v5 section "PRE -> POST OPTICAL
REGISTRATION"; the full method description is _REGISTRATION_METHOD_DOC below)."""
from __future__ import annotations

from dataclasses import dataclass, field as _dc_field

import numpy as np

from .config import REG_MARK_MAX_DIAMETER_UM, REG_MAX_PLACEMENT_UM, REG_MAX_TILE_LOO_UM, REG_MIN_TILES


# ============================================================
# PRE -> POST OPTICAL REGISTRATION (was ipsc_prepost_registration.py)
# ============================================================

_REGISTRATION_METHOD_DOC = """PRE -> POST optical registration for the iPSC single-cell pipeline.

Self-contained (numpy, scipy, opencv, tifffile, SimpleITK): a port of the method
in the MSI platform's `pre_post_register.py` (Shane, 2026-09), which was
measured on this exact class of image pair - one well photographed before
matrix (cells visible) and after MSI (matrix grain and the ablated area
visible). The two scans share almost no texture, so an intensity optimiser
started from a centroid has nothing reliable to descend on. What they DO share
is large-scale structure: the ring of dark marks around the well, the gasket
outline, debris. This module uses only that.

    1. Brightfield channel of each scan, chosen by its histogram (bright
       background, dark objects), never by a fixed index.
    2. Translation by TRUE phase correlation on high-passed images, tried at
       several working scales; the sharpest peak wins and a flat surface is a
       refusal, not an answer.
    3. Rotation and scale from a 6x6 grid of tiles, each phase-correlated
       against its own patch of the other scan and searched only within
       TILE_SEARCH_TOLERANCE_UM of the whole-well shift - so the tile-scan
       seam grid, which is periodic in BOTH scans, cannot win one period out.
    4. A similarity (rotation + translation + one scale) fitted to the tile
       displacements with outlier rejection and a leave-one-out residual.
    5. The dark marks around the well, paired through the fit with a tight
       tolerance, as an independent check the tiles cannot fake.

Frames: PRE/POST are each scan's own native pixels, (y, x). The fitted
transform maps POST native -> PRE native. With POST as the FIXED image and
PRE as the MOVING one (this pipeline's convention) that is exactly the
direction SimpleITK's resampler wants, see `to_sitk_transform`.

Measured on the GBA FOR TANUKI folders, 2026-09-16 (1.30 um/px, ~8.6 mm
field): phase peaks 20-87x the surrounding surface, 30-36 of 36 tiles used,
per-tile LOO 1.4-3.5 um, placement 0.2-0.6 um, scale within 0.05 % of unity,
rotation under 0.07 deg. The MMI cascade this replaces mis-registered the
same wells by a third of a well pitch (seam-grid alias) or by a rotation
(moments initialiser on a POST scan 936 px wider than its PRE).
"""
# ----------------------------------------------------------------------------
# Tuning. Values are Shane's, measured on the real S1B2 well A4 at 0.65 um/px,
# and re-checked here at 1.30 um/px. Everything is in micrometres so the same
# numbers hold at either pixel size.
# ----------------------------------------------------------------------------

#: Working resolutions (um/px) tried for the coarse translation; the sharpest
#: correlation peak wins.
WORKING_UM_PER_PX_LADDER = (4.0, 5.0, 6.5, 8.0)

#: A correlation peak flatter than this (ratio to the 99.9th percentile of the
#: surface) means the two images share nothing locatable.
MIN_PEAK_SHARPNESS = 2.0

#: High-pass scale (um) before correlating: illumination and matrix haze are
#: broader than this, marks, gasket edges and debris are not.
DEFAULT_HIGH_PASS_UM = 120.0

#: Each tile's own peak is searched only this far (um) from the whole-well
#: shift. Across one tile the scale difference moves a feature by under 10
#: um; anything further is an alias (the tile-scan seam grid one period away).
TILE_SEARCH_TOLERANCE_UM = 60.0

#: Fewer tiles than this and no similarity is fitted.
MIN_PAIRED_TILES = REG_MIN_TILES

#: Per-tile leave-one-out RMSE (um) beyond which the tiles disagree too much.
MAX_TILE_LOO_UM = REG_MAX_TILE_LOO_UM

#: The transform's own standard error, LOO / sqrt(n_tiles), beyond which the
#: fit is not credible. Half an MSI pixel at a 5 um raster.
MAX_PLACEMENT_ERROR_UM = REG_MAX_PLACEMENT_UM

#: A dark mark pairs across the two scans through the fit only if its
#: counterpart lands within this (um). Coarse on purpose: a mark's apparent
#: outline differs between the scans (matrix over it in one), so its centroid
#: moves by tens of micrometres. Marks confirm the well-level placement.
MARK_CHECK_TOLERANCE_UM = 40.0

#: Dark-mark size range (um). The ring marks on the GBA slides are 500-650 um,
#: which is why the upper bound is above Shane's 450.
MARK_MIN_DIAMETER_UM = 60.0
MARK_MAX_DIAMETER_UM = REG_MARK_MAX_DIAMETER_UM

#: If a TIFF carries no resolution tag this is assumed, with a warning.
DEFAULT_UM_PER_PX = 1.30


class PrePostRegistrationError(RuntimeError):
    """The two scans could not be registered to each other."""


class RegistrationNotCredible(PrePostRegistrationError):
    """A transform was fitted but does not meet the credibility thresholds."""


# ----------------------------------------------------------------------------
# Image helpers
# ----------------------------------------------------------------------------

def tiff_um_per_px(path) -> tuple[float, str]:
    """`(um/px, source)` from a TIFF's resolution tags.

    LAS X writes px/cm (unit 3); ImageJ writes px/um with unit NONE and
    `unit=micron` in its description. Anything else falls back to
    DEFAULT_UM_PER_PX and says so.
    """
    import tifffile

    try:
        with tifffile.TiffFile(str(path)) as tif:
            page = tif.pages[0]
            if "XResolution" not in page.tags:
                return DEFAULT_UM_PER_PX, "default (no XResolution tag)"
            num, den = page.tags["XResolution"].value
            if not num or not den:
                return DEFAULT_UM_PER_PX, "default (zero resolution tag)"
            px_per_unit = num / den
            unit = int(page.tags["ResolutionUnit"].value) if "ResolutionUnit" in page.tags else 1
            if unit == 3:
                return 1e4 / px_per_unit, "tag (px/cm)"
            if unit == 2:
                return 25400.0 / px_per_unit, "tag (px/inch)"
            # Unit NONE: ImageJ convention is px/um when its metadata says micron.
            # v5: 1/1 is what an UNcalibrated save writes - treat it as no tag rather than as
            # 1.0 um/px, which would also switch the square-size gate on with a wrong size
            if abs(px_per_unit - 1.0) < 1e-9:
                return DEFAULT_UM_PER_PX, "default (resolution 1/1, uncalibrated)"
            desc = ""
            if tif.imagej_metadata:
                desc = str(tif.imagej_metadata.get("unit", ""))
            # v5: ImageJ stores the micro sign escaped, literally 'µm', so test for that too
            _unit = desc.lower().strip().replace("\\u00b5", "µ")
            if "micron" in _unit or "µm" in _unit or _unit == "um":
                return 1.0 / px_per_unit, "tag (ImageJ px/um)"
            value = 1.0 / px_per_unit
            if 0.05 < value < 50.0:
                return value, "tag (unit NONE, assumed px/um)"
            return DEFAULT_UM_PER_PX, "default (unit NONE, tag not plausible)"
    except Exception as exc:  # noqa: BLE001 - any failure here is a missing tag
        return DEFAULT_UM_PER_PX, f"default ({type(exc).__name__})"


def brightfield_scores(stack_channel_last: np.ndarray) -> list[float]:
    """Per-channel position of the median within the 1-99 percentile range.

    Brightfield transmits light: background bright, objects dark, so the
    median sits high (> 0.5). Fluorescence is the other way round (~0).
    Dimensionless, so comparable across channels with different exposures.
    """
    array = np.asarray(stack_channel_last)
    if array.ndim == 2:
        return [1.0]
    scores = []
    for index in range(array.shape[2]):
        channel = array[::4, ::4, index].astype(np.float64)
        low, median, high = np.percentile(channel, [1, 50, 99])
        scores.append(float((median - low) / max(high - low, 1e-9)))
    return scores


def brightfield_channel(stack_channel_last: np.ndarray) -> tuple[int, list[float]]:
    """Index of the brightfield-most channel, plus every channel's score.

    Not the sharpest channel: on the real pre-MSI scans the tile-scan
    stitching seams made a fluorescence channel score highest by Laplacian
    energy, and registering on it failed.
    """
    scores = brightfield_scores(stack_channel_last)
    return int(np.argmax(scores)), scores


def shrink(image: np.ndarray, factor: int) -> np.ndarray:
    """Block-mean downsample by an integer factor (trims the remainder)."""
    factor = int(max(1, factor))
    array = np.asarray(image, dtype=np.float64)
    if factor == 1:
        return array
    height = array.shape[0] // factor * factor
    width = array.shape[1] // factor * factor
    if height == 0 or width == 0:
        return array
    return array[:height, :width].reshape(height // factor, factor, width // factor, factor).mean(axis=(1, 3))


def high_pass(image: np.ndarray, um_per_px: float, scale_um: float = DEFAULT_HIGH_PASS_UM) -> np.ndarray:
    """`image` minus its own broad background."""
    from scipy import ndimage

    array = np.asarray(image, dtype=np.float64)
    return array - ndimage.gaussian_filter(array, max(1.0, scale_um / um_per_px))


# ----------------------------------------------------------------------------
# Stage 1: translation by phase correlation
# ----------------------------------------------------------------------------

def _phase_correlate(pre_gray, pre_um, post_gray, post_um, working_um, high_pass_um,
                     prior_shift_um=None, prior_tolerance_um=None):
    """One phase correlation at one working scale.

    Returns `((dy, dx) um placing POST within PRE, peak sharpness)`.
    TRUE phase correlation (every frequency's amplitude normalised away), not
    cross-correlation: unnormalised, the score grows with how much bright
    structure overlaps at all and prefers piling the two well interiors on
    top of each other.
    """
    pre_factor = max(1, int(round(working_um / pre_um)))
    post_factor = max(1, int(round(working_um / post_um)))
    pre_work = high_pass(shrink(pre_gray, pre_factor), pre_um * pre_factor, high_pass_um)
    post_work = high_pass(shrink(post_gray, post_factor), post_um * post_factor, high_pass_um)
    scale_um = pre_um * pre_factor

    height = max(pre_work.shape[0], post_work.shape[0])
    width = max(pre_work.shape[1], post_work.shape[1])
    fixed = np.zeros((height, width))
    fixed[: pre_work.shape[0], : pre_work.shape[1]] = pre_work
    moving = np.zeros((height, width))
    moving[: post_work.shape[0], : post_work.shape[1]] = post_work
    spectrum = np.fft.fft2(fixed) * np.conj(np.fft.fft2(moving))
    magnitude = np.abs(spectrum)
    surface = np.fft.ifft2(
        np.where(magnitude > 0, spectrum / np.where(magnitude == 0, 1.0, magnitude), 0.0)
    ).real

    if prior_shift_um is not None and prior_tolerance_um is not None:
        # The surface is periodic (a peak at -k reads as height - k), so the
        # window is built modulo the surface; sharpness is still judged
        # against the WHOLE surface.
        centre_y = int(round(prior_shift_um[0] / scale_um)) % height
        centre_x = int(round(prior_shift_um[1] / scale_um)) % width
        radius = max(2, int(round(prior_tolerance_um / scale_um)))
        rows = (np.arange(height) - centre_y + height // 2) % height - height // 2
        cols = (np.arange(width) - centre_x + width // 2) % width - width // 2
        window = (np.abs(rows)[:, None] <= radius) & (np.abs(cols)[None, :] <= radius)
        searched = np.where(window, surface, -np.inf)
    else:
        searched = surface
    peak = int(np.argmax(searched))
    peak_y, peak_x = np.unravel_index(peak, surface.shape)
    sharpness = float(surface[peak_y, peak_x] / max(float(np.percentile(surface, 99.9)), 1e-12))

    def sub_pixel(values, index):
        before = values[(index - 1) % len(values)]
        after = values[(index + 1) % len(values)]
        centre = values[index]
        denominator = before - 2.0 * centre + after
        if abs(denominator) < 1e-12:
            return 0.0
        return float(np.clip(0.5 * (before - after) / denominator, -0.5, 0.5))

    offset_y = peak_y + sub_pixel(surface[:, peak_x], peak_y)
    offset_x = peak_x + sub_pixel(surface[peak_y, :], peak_x)
    if offset_y > height / 2:
        offset_y -= height
    if offset_x > width / 2:
        offset_x -= width
    return (float(offset_y * scale_um), float(offset_x * scale_um)), sharpness


def coarse_shift_um(pre_gray, pre_um, post_gray, post_um, *, ladder=WORKING_UM_PER_PX_LADDER,
                    high_pass_um=DEFAULT_HIGH_PASS_UM):
    """`((dy, dx) um, sharpness, working um/px)` placing POST within PRE.

    Several working scales are tried and the sharpest peak wins: which scale
    works is a property of the images, and the sharpness says which to believe.
    """
    best = None
    for working in ladder:
        shift, sharpness = _phase_correlate(pre_gray, pre_um, post_gray, post_um, working, high_pass_um)
        if best is None or sharpness > best[1]:
            best = (shift, sharpness, working)
    return best


# ----------------------------------------------------------------------------
# Stage 2: rotation and scale from tiles
# ----------------------------------------------------------------------------

@dataclass
class BlockShift:
    centre_post_yx: tuple[float, float]  # tile centre, POST native px
    shift_um: tuple[float, float]        # this tile's own POST-within-PRE shift
    sharpness: float


def block_shifts(pre_gray, pre_um, post_gray, post_um, coarse_shift, *, tiles=6, tile_um=None,
                 working_um=2.6, min_sharpness=MIN_PEAK_SHARPNESS) -> list[BlockShift]:
    """Phase-correlate a grid of POST tiles, each against its own place in PRE.

    A systematic variation of the tile shifts across the field IS the scale
    and rotation. Tiles whose peak is flat are dropped rather than fitted.
    """
    shifts: list[BlockShift] = []
    if tile_um is None:
        extent_um = min(post_gray.shape) * post_um
        tile_um = max(300.0, 0.7 * extent_um / tiles)
    tile_post_px = int(round(tile_um / post_um))
    if tile_post_px < 8:
        return shifts
    height, width = post_gray.shape
    margin_y = max(0, (height - tiles * tile_post_px) // (tiles + 1))
    margin_x = max(0, (width - tiles * tile_post_px) // (tiles + 1))
    for row in range(tiles):
        for column in range(tiles):
            y0 = margin_y + row * (tile_post_px + margin_y)
            x0 = margin_x + column * (tile_post_px + margin_x)
            if y0 + tile_post_px > height or x0 + tile_post_px > width:
                continue
            post_tile = post_gray[y0: y0 + tile_post_px, x0: x0 + tile_post_px]
            centre_post = np.array([y0 + tile_post_px / 2.0, x0 + tile_post_px / 2.0])
            centre_pre_um = centre_post * post_um + np.asarray(coarse_shift, dtype=np.float64)
            half_pre_px = int(round((0.5 * tile_um + 0.15 * tile_um + 60.0) / pre_um))
            centre_pre_px = centre_pre_um / pre_um
            py0 = int(round(centre_pre_px[0])) - half_pre_px
            px0 = int(round(centre_pre_px[1])) - half_pre_px
            py1, px1 = py0 + 2 * half_pre_px, px0 + 2 * half_pre_px
            if py0 < 0 or px0 < 0 or py1 > pre_gray.shape[0] or px1 > pre_gray.shape[1]:
                continue
            pre_tile = pre_gray[py0:py1, px0:px1]
            expected = (
                float(coarse_shift[0]) - (py0 * pre_um - y0 * post_um),
                float(coarse_shift[1]) - (px0 * pre_um - x0 * post_um),
            )
            (dy, dx), sharpness = _phase_correlate(
                pre_tile, pre_um, post_tile, post_um, working_um, DEFAULT_HIGH_PASS_UM,
                prior_shift_um=expected, prior_tolerance_um=TILE_SEARCH_TOLERANCE_UM,
            )
            if not np.isfinite(sharpness) or sharpness < min_sharpness:
                continue
            shifts.append(BlockShift(
                centre_post_yx=(float(centre_post[0]), float(centre_post[1])),
                shift_um=(float(dy + py0 * pre_um - y0 * post_um), float(dx + px0 * pre_um - x0 * post_um)),
                sharpness=float(sharpness),
            ))
    return shifts


def fit_similarity_yx(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Least-squares similarity (rotation, one scale, translation) mapping
    `source` (N, 2) onto `target` (N, 2), both (y, x). Returns 3x3 for
    `matrix @ [y, x, 1]`. Umeyama's closed form, reflection disallowed.
    """
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    mean_s = source.mean(axis=0)
    mean_t = target.mean(axis=0)
    centred_s = source - mean_s
    centred_t = target - mean_t
    covariance = centred_t.T @ centred_s / len(source)
    u, singular, vt = np.linalg.svd(covariance)
    sign = np.ones(2)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        sign[-1] = -1.0
    rotation = u @ np.diag(sign) @ vt
    variance_s = (centred_s ** 2).sum() / len(source)
    scale = float((singular * sign).sum() / max(variance_s, 1e-12))
    matrix = np.eye(3)
    matrix[:2, :2] = scale * rotation
    matrix[:2, 2] = mean_t - scale * rotation @ mean_s
    return matrix


def fit_from_block_shifts(shifts: list[BlockShift], pre_um: float, post_um: float, rounds: int = 3):
    """Similarity POST -> PRE (native px) from the tile displacements.

    Returns `(transform, rmse_um, loo_rmse_um, n_used, n_dropped)` or None
    when too few tiles agree. Each tile is one correspondence: its centre and
    where that centre lands in PRE. Outliers beyond 3x the median residual
    (never tighter than MAX_TILE_LOO_UM) are dropped and the fit repeated.
    """
    if len(shifts) < MIN_PAIRED_TILES:
        return None
    source = np.array([block.centre_post_yx for block in shifts])
    target = np.array([
        (np.asarray(block.centre_post_yx) * post_um + np.asarray(block.shift_um)) / pre_um for block in shifts
    ])
    keep = np.ones(len(shifts), dtype=bool)
    dropped = 0
    for _round in range(rounds):
        matrix = fit_similarity_yx(source[keep], target[keep])
        mapped = (matrix[:2, :2] @ source[keep].T).T + matrix[:2, 2]
        residuals = np.hypot(*(mapped - target[keep]).T) * pre_um
        limit = max(3.0 * float(np.median(residuals)), MAX_TILE_LOO_UM)
        survivors = residuals <= limit
        if survivors.all() or survivors.sum() < MIN_PAIRED_TILES:
            break
        dropped += int((~survivors).sum())
        indices = np.flatnonzero(keep)
        keep[indices[~survivors]] = False
    matrix = fit_similarity_yx(source[keep], target[keep])
    mapped = (matrix[:2, :2] @ source[keep].T).T + matrix[:2, 2]
    rmse_um = float(np.sqrt(np.mean(np.sum((mapped - target[keep]) ** 2, axis=1)))) * pre_um
    kept_source = source[keep]
    kept_target = target[keep]
    loo = []
    if len(kept_source) > MIN_PAIRED_TILES:
        for i in range(len(kept_source)):
            mask = np.ones(len(kept_source), dtype=bool)
            mask[i] = False
            sub = fit_similarity_yx(kept_source[mask], kept_target[mask])
            predicted = sub[:2, :2] @ kept_source[i] + sub[:2, 2]
            loo.append(float(np.hypot(*(predicted - kept_target[i]))) * pre_um)
    loo_rmse = float(np.sqrt(np.mean(np.square(loo)))) if loo else float("nan")
    return matrix, rmse_um, loo_rmse, int(keep.sum()), dropped


# ----------------------------------------------------------------------------
# Stage 3: dark marks as an independent check
# ----------------------------------------------------------------------------

def detect_dark_marks(image, um_per_px, *, min_diameter_um=MARK_MIN_DIAMETER_UM,
                      max_diameter_um=MARK_MAX_DIAMETER_UM, working_um=5.0, threshold_sigmas=3.0,
                      max_blobs=200) -> np.ndarray:
    """Dark-blob centroids `(N, 2)` in this image's own native pixels: the ring
    of marks around the well plus debris, which both scans see."""
    from scipy import ndimage

    factor = max(1, int(round(working_um / um_per_px)))
    coarse = shrink(image, factor)
    scale_um = um_per_px * factor
    detail = high_pass(coarse, scale_um, max_diameter_um)
    median = float(np.median(detail))
    sigma = max(float(np.median(np.abs(detail - median))) * 1.4826, 1e-9)
    mask = detail < median - threshold_sigmas * sigma
    if not mask.any():
        return np.zeros((0, 2))
    labels, count = ndimage.label(mask)
    if count == 0:
        return np.zeros((0, 2))
    areas = np.bincount(labels.ravel())[1:]
    min_area = np.pi * (0.5 * min_diameter_um / scale_um) ** 2
    max_area = np.pi * (0.5 * max_diameter_um / scale_um) ** 2
    keep = np.flatnonzero((areas >= min_area) & (areas <= max_area)) + 1
    if keep.size == 0:
        return np.zeros((0, 2))
    if keep.size > max_blobs:
        keep = keep[np.argsort(areas[keep - 1])[::-1][:max_blobs]]
    centroids = np.array(ndimage.center_of_mass(mask, labels, keep), dtype=np.float64)
    return (centroids + 0.5) * factor - 0.5


def pair_marks(pre_marks, post_marks, transform, pre_um, tolerance_um=MARK_CHECK_TOLERANCE_UM):
    """Mutually-nearest pairs (POST mark, PRE mark) within `tolerance_um`
    through `transform`, and their RMSE in um. Mutual, not one-way: one-way
    nearest neighbour pairs everything with something."""
    if len(pre_marks) == 0 or len(post_marks) == 0:
        return 0, float("nan")
    mapped = (transform[:2, :2] @ np.asarray(post_marks).T).T + transform[:2, 2]
    distances = np.hypot(mapped[:, 0, None] - pre_marks[None, :, 0], mapped[:, 1, None] - pre_marks[None, :, 1])
    nearest_pre = np.argmin(distances, axis=1)
    nearest_post = np.argmin(distances, axis=0)
    tolerance_px = tolerance_um / pre_um
    paired = [
        distances[post_index, pre_index]
        for post_index, pre_index in enumerate(nearest_pre)
        if nearest_post[pre_index] == post_index and distances[post_index, pre_index] <= tolerance_px
    ]
    if not paired:
        return 0, float("nan")
    return len(paired), float(np.sqrt(np.mean(np.square(paired)))) * pre_um


# ----------------------------------------------------------------------------
# The fit
# ----------------------------------------------------------------------------

@dataclass
class PrePostFit:
    """POST -> PRE in native pixels, (y, x), plus its quality."""

    transform: np.ndarray
    method: str                       # "phase" | "phase+tiles"
    n_tiles: int
    n_tiles_dropped: int
    rmse_um: float
    loo_rmse_um: float
    placement_error_um: float         # loo / sqrt(n_tiles): how well the TRANSFORM is placed
    shift_um: tuple[float, float]
    peak_sharpness: float
    working_um_per_px: float
    scale: float                      # physical POST->PRE scale, 1.0 = the pixel sizes agree
    rotation_deg: float
    n_marks_pre: int = 0
    n_marks_post: int = 0
    n_marks_paired: int = 0
    mark_rmse_um: float = float("nan")
    pre_channel: int = 0
    post_channel: int = 0
    pre_scores: list[float] = _dc_field(default_factory=list)
    post_scores: list[float] = _dc_field(default_factory=list)
    pre_um_per_px: float = float("nan")
    post_um_per_px: float = float("nan")
    notes: list[str] = _dc_field(default_factory=list)

    @property
    def credible(self) -> bool:
        if self.n_tiles < MIN_PAIRED_TILES:
            return False
        if not np.isfinite(self.placement_error_um) or not np.isfinite(self.loo_rmse_um):
            return False
        return bool(self.placement_error_um <= MAX_PLACEMENT_ERROR_UM and self.loo_rmse_um <= MAX_TILE_LOO_UM)

    def failures(self) -> list[str]:
        out = []
        if self.n_tiles < MIN_PAIRED_TILES:
            out.append(f"only {self.n_tiles} tiles agreed (need {MIN_PAIRED_TILES})")
        if not np.isfinite(self.loo_rmse_um):
            out.append("no leave-one-out residual could be measured")
        elif self.loo_rmse_um > MAX_TILE_LOO_UM:
            out.append(f"per-tile LOO {self.loo_rmse_um:.1f} um > {MAX_TILE_LOO_UM:.1f}")
        if np.isfinite(self.placement_error_um) and self.placement_error_um > MAX_PLACEMENT_ERROR_UM:
            out.append(f"placement {self.placement_error_um:.2f} um > {MAX_PLACEMENT_ERROR_UM:.2f}")
        return out

    def summary_lines(self) -> list[str]:
        lines = [
            f"Method: {self.method}",
            f"Credible: {'yes' if self.credible else 'NO (' + '; '.join(self.failures()) + ')'}",
            f"Coarse shift POST within PRE (dy, dx): {self.shift_um[0]:+.1f}, {self.shift_um[1]:+.1f} um",
            f"Phase peak sharpness: {self.peak_sharpness:.2f}x at {self.working_um_per_px:.1f} um/px",
            f"Tiles used: {self.n_tiles} (dropped {self.n_tiles_dropped}); per-tile RMSE {self.rmse_um:.2f} um; "
            f"LOO {self.loo_rmse_um:.2f} um; placement {self.placement_error_um:.2f} um",
            f"Scale POST->PRE: {self.scale:.5f}; rotation: {self.rotation_deg:+.3f} deg",
            f"Dark marks: {self.n_marks_paired} paired of {self.n_marks_pre} PRE / {self.n_marks_post} POST "
            f"within {MARK_CHECK_TOLERANCE_UM:.0f} um" + (f", RMSE {self.mark_rmse_um:.1f} um" if self.n_marks_paired else ""),
            f"PRE: channel {self.pre_channel} of scores {[round(s, 2) for s in self.pre_scores]} at {self.pre_um_per_px:.4f} um/px",
            f"POST: channel {self.post_channel} of scores {[round(s, 2) for s in self.post_scores]} at {self.post_um_per_px:.4f} um/px",
        ]
        lines += [f"Note: {note}" for note in self.notes]
        return lines


def register_pre_post(pre_gray, pre_um, post_gray, post_um, *, tile_grid=6) -> PrePostFit:
    """Fit POST -> PRE for one well, in each scan's own native pixels."""
    pre_gray = np.asarray(pre_gray, dtype=np.float64)
    post_gray = np.asarray(post_gray, dtype=np.float64)
    (shift_y, shift_x), sharpness, working = coarse_shift_um(pre_gray, pre_um, post_gray, post_um)
    notes = [f"phase correlation peak {sharpness:.2f}x the surrounding surface at {working:.1f} um/px"]
    if not np.isfinite(sharpness) or sharpness < MIN_PEAK_SHARPNESS:
        raise PrePostRegistrationError(
            f"the two scans' correlation surface is flat (peak {sharpness:.2f}x, need {MIN_PEAK_SHARPNESS:.1f}x): "
            "they share no locatable structure - check that both cover the same well and that the "
            "brightfield channel was chosen"
        )

    transform = np.eye(3)
    transform[0, 0] = transform[1, 1] = post_um / pre_um
    transform[:2, 2] = (shift_y / pre_um, shift_x / pre_um)
    method = "phase"
    n_tiles = dropped = 0
    rmse_um = loo_um = float("nan")

    tiles = block_shifts(pre_gray, pre_um, post_gray, post_um, (shift_y, shift_x), tiles=tile_grid)
    notes.append(f"{len(tiles)} of {tile_grid * tile_grid} tiles correlated sharply enough to use")
    fitted = fit_from_block_shifts(tiles, pre_um, post_um)
    if fitted is not None:
        candidate, candidate_rmse, candidate_loo, candidate_n, dropped = fitted
        if np.isfinite(candidate_loo) and candidate_loo <= MAX_TILE_LOO_UM:
            transform, rmse_um, loo_um, n_tiles, method = candidate, candidate_rmse, candidate_loo, candidate_n, "phase+tiles"
            if dropped:
                notes.append(f"{dropped} tiles were rejected as outliers before the final fit")
        else:
            n_tiles = candidate_n
            loo_um = candidate_loo
            notes.append(
                f"the tile fit was rejected (per-tile LOO {candidate_loo:.1f} um > {MAX_TILE_LOO_UM:.1f}); "
                "kept the whole-well translation, which assumes the two stated pixel sizes are exact"
            )
    else:
        notes.append("too few tiles agreed to fit a rotation and scale; kept the whole-well translation")

    pre_marks = detect_dark_marks(pre_gray, pre_um)
    post_marks = detect_dark_marks(post_gray, post_um)
    if len(post_marks) < MIN_PAIRED_TILES:
        # The POST scan is a dark matrix field whose grain sets a large noise sigma, so
        # the ring marks do not clear 3 sigma at 5 um/px (0 found on the real micro and
        # neuro wells). Averaging the grain away at 20 um/px and asking for 2 sigma
        # finds them (43 and 9). This only affects the CHECK, never the fit.
        post_marks = detect_dark_marks(post_gray, post_um, working_um=20.0, threshold_sigmas=2.0)
        notes.append("POST marks detected at 20 um/px, 2 sigma (too few at the default)")
    n_paired, mark_rmse = pair_marks(pre_marks, post_marks, transform, pre_um)

    placement = float(loo_um / np.sqrt(n_tiles)) if n_tiles and np.isfinite(loo_um) and method == "phase+tiles" else float("nan")
    return PrePostFit(
        transform=transform,
        method=method,
        n_tiles=n_tiles,
        n_tiles_dropped=dropped,
        rmse_um=rmse_um,
        loo_rmse_um=loo_um,
        placement_error_um=placement,
        shift_um=(shift_y, shift_x),
        peak_sharpness=sharpness,
        working_um_per_px=working,
        scale=float(np.hypot(*transform[:2, 0]) * pre_um / post_um),
        # (y, x) order: a rotation reads [[cos, sin], [-sin, cos]], so the angle is -m[1, 0].
        rotation_deg=float(np.degrees(np.arctan2(-transform[1, 0], transform[0, 0]))),
        n_marks_pre=len(pre_marks),
        n_marks_post=len(post_marks),
        n_marks_paired=n_paired,
        mark_rmse_um=mark_rmse,
        pre_um_per_px=pre_um,
        post_um_per_px=post_um,
        notes=notes,
    )


# ----------------------------------------------------------------------------
# Glue for the pipeline: channel-last uint16 arrays in, resampled PRE out
# ----------------------------------------------------------------------------

def to_sitk_transform(transform_yx: np.ndarray):
    """The 3x3 POST -> PRE (y, x) matrix as a SimpleITK 2D affine.

    `sitk.Resample(moving=PRE, fixed=POST, T)` samples PRE at T(fixed point),
    i.e. T must map POST physical points to PRE physical points, in (x, y).
    With spacing 1 and origin 0 on both images that is the same matrix with
    the axes swapped.
    """
    import SimpleITK as sitk

    t = np.asarray(transform_yx, dtype=np.float64)
    affine = sitk.AffineTransform(2)
    affine.SetMatrix((float(t[1, 1]), float(t[1, 0]), float(t[0, 1]), float(t[0, 0])))
    affine.SetTranslation((float(t[1, 2]), float(t[0, 2])))
    return affine


def register_pre_to_post(fixed_image: np.ndarray, moving_image: np.ndarray, fixed_um: float, moving_um: float,
                         *, channel: int | None = None, fill_value: float = 255.0):
    """Register PRE (moving) onto POST (fixed) and resample every PRE channel.

    `fixed_image`/`moving_image` are channel-last (H, W, C). `channel` forces
    the alignment channel; None picks the brightfield channel of each image
    by its histogram. Returns `(registered_channels, coverage, sitk_transform,
    fit)` where `registered_channels` is a list of float32 arrays on the POST
    grid with `fill_value` outside PRE's footprint and `coverage` is the
    boolean mask of genuinely resampled pixels.
    """
    import SimpleITK as sitk

    fixed_image = np.asarray(fixed_image)
    moving_image = np.asarray(moving_image)
    if fixed_image.ndim == 2:
        fixed_image = fixed_image[..., None]
    if moving_image.ndim == 2:
        moving_image = moving_image[..., None]

    if channel is None:
        post_channel, post_scores = brightfield_channel(fixed_image)
        pre_channel, pre_scores = brightfield_channel(moving_image)
    else:
        post_channel = min(int(channel), fixed_image.shape[2] - 1)
        pre_channel = min(int(channel), moving_image.shape[2] - 1)
        post_scores = brightfield_scores(fixed_image)
        pre_scores = brightfield_scores(moving_image)

    fit = register_pre_post(moving_image[..., pre_channel], moving_um, fixed_image[..., post_channel], fixed_um)
    fit.pre_channel, fit.post_channel = pre_channel, post_channel
    fit.pre_scores, fit.post_scores = pre_scores, post_scores

    sitk_transform = to_sitk_transform(fit.transform)
    reference = sitk.GetImageFromArray(np.zeros(fixed_image.shape[:2], dtype=np.float32))
    reference.SetSpacing([1.0, 1.0])
    reference.SetOrigin([0.0, 0.0])

    ones = sitk.GetImageFromArray(np.ones(moving_image.shape[:2], dtype=np.float32))
    ones.SetSpacing([1.0, 1.0])
    ones.SetOrigin([0.0, 0.0])
    coverage = sitk.GetArrayFromImage(
        sitk.Resample(ones, reference, sitk_transform, sitk.sitkNearestNeighbor, 0.0, sitk.sitkFloat32)
    ) > 0.5

    registered = []
    for index in range(moving_image.shape[2]):
        channel_image = sitk.GetImageFromArray(moving_image[..., index].astype(np.float32))
        channel_image.SetSpacing([1.0, 1.0])
        channel_image.SetOrigin([0.0, 0.0])
        resampled = sitk.Resample(channel_image, reference, sitk_transform, sitk.sitkLinear, float(fill_value),
                                  sitk.sitkFloat32)
        registered.append(sitk.GetArrayFromImage(resampled))
    return registered, coverage, sitk_transform, fit
