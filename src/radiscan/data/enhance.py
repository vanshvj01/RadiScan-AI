"""Automatic radiograph clean-up: find the image, crop to it, reduce noise.

    from radiscan.data.enhance import enhance
    result = enhance(rgb_uint8)
    result.image      # cleaned uint8 RGB, ready for CLAHE + the model
    result.steps      # what was actually done, in order, for the UI and report

Three steps, each one conservative and each one able to decline:

1. **Locate the film** in a phone photograph (a radiograph held up to a camera,
   common when a scan is shared over WhatsApp) and crop to it, discarding the
   hand, chair and room behind it. A digital radiograph has no such background
   and this step does nothing. It crops; it does not correct perspective, so a
   steeply angled photo stays skewed.
2. **Crop to the patient**: strip black or white padding, collimator borders
   and empty margins, keeping the full thorax. It deliberately does NOT crop to
   the lungs: cardiomegaly needs the heart, effusion needs the costophrenic
   angles, fracture needs ribs and clavicles, support devices sit anywhere.
3. **Denoise** with non-local means at low strength. Strong denoising would
   also erase what matters most and is hardest to see: the hair-thin pleural
   line of a pneumothorax and small nodules.

Every step is fail-safe. If it cannot find what it is looking for, or what it
found is implausible (too small, too thin), it returns its input unchanged and
says so. A clean-up step that guesses wrong would hide anatomy from the model,
which is worse than not cleaning up at all.

Training and inference call this same function, and the checkpoint records
whether it was on, so the model never sees one kind of image in training and
another in deployment.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

# Below this fraction of the frame, a detected "film" or "body" is more likely
# a false detection (a label, a lead marker) than the real thing.
MIN_FILM_AREA = 0.20
MIN_BODY_AREA = 0.35
# Only crop when it removes something worthwhile; trimming 1% is just jitter.
MIN_USEFUL_TRIM = 0.04
# Rows/columns flatter than this (grey levels) are padding, not anatomy.
UNIFORM_STD = 6.0


@dataclass
class EnhanceResult:
    image: np.ndarray
    steps: list[str] = field(default_factory=list)
    crop_box: tuple[int, int, int, int] | None = None  # x0, y0, x1, y1 in the input
    film_detected: bool = False


def _is_colour(image: np.ndarray, tol: float = 6.0) -> bool:
    """True for a colour photo; False for a greyscale radiograph saved as RGB."""
    if image.ndim == 2:
        return False
    px = image[:: max(1, image.shape[0] // 256), :: max(1, image.shape[1] // 256)].astype(np.int16)
    spread = px.max(axis=2) - px.min(axis=2)
    return float(np.percentile(spread, 90)) > tol


def _gray(image: np.ndarray) -> np.ndarray:
    return image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)


def _fill_holes(mask: np.ndarray) -> np.ndarray:
    """Fill regions of a 0/1 mask that cannot be reached from the image border."""
    h, w = mask.shape
    outside = (1 - mask).astype(np.uint8)
    seed = np.zeros((h + 2, w + 2), np.uint8)
    for x, y in ((0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1)):
        if outside[y, x] == 1:
            cv2.floodFill(outside, seed, (x, y), 2)
    filled = mask.copy()
    filled[outside == 1] = 1
    return filled


def find_film(gray: np.ndarray) -> tuple[int, int, int, int] | None:
    """Box around a radiograph photographed against a background, or None.

    A developed film is dark (exposed background) with bright anatomy inside,
    held against a usually lighter room. So: take the largest dark region,
    fill the bright anatomy back in, and keep the rows and columns that region
    mostly covers. Thin bridges into dark parts of the background are dropped
    because they cover few rows or columns.

    Returns None for a digital radiograph -- the checks below are written so
    that the black air around a patient is not mistaken for a film.
    """
    h, w = gray.shape
    scale = min(1.0, 800.0 / max(h, w))
    small = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    sh, sw = small.shape
    blur = cv2.GaussianBlur(small, (7, 7), 0)
    _, dark = cv2.threshold(blur, 0, 1, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    dark = cv2.morphologyEx(
        dark.astype(np.uint8),
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)),
    )
    n, lab, stats, _ = cv2.connectedComponentsWithStats(dark)
    if n < 2:
        return None
    region = (lab == 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))).astype(np.uint8)
    region = _fill_holes(region)

    cov_c, cov_r = region.mean(axis=0), region.mean(axis=1)
    cols = np.where(cov_c > 0.5 * cov_c.max())[0]
    rows = np.where(cov_r > 0.5 * cov_r.max())[0]
    if cols.size == 0 or rows.size == 0:
        return None
    x0, x1, y0, y1 = int(cols[0]), int(cols[-1]) + 1, int(rows[0]), int(rows[-1]) + 1

    area = (x1 - x0) * (y1 - y0) / (sh * sw)
    if not MIN_FILM_AREA <= area <= 0.90:
        return None
    # The film must sit inside a real background: touching all four edges of
    # the frame means there is nothing around it to remove.
    touches = (x0 <= 2) + (y0 <= 2) + (x1 >= sw - 2) + (y1 >= sh - 2)
    if touches >= 3:
        return None
    # ...and that background must be clearly brighter than the film. Around a
    # digital radiograph, what lies outside a dark box is more dark air.
    inside = np.zeros_like(region, bool)
    inside[y0:y1, x0:x1] = True
    if blur[~inside].mean() < blur[inside].mean() + 25:
        return None

    inv = 1.0 / scale
    return int(x0 * inv), int(y0 * inv), int(min(w, x1 * inv)), int(min(h, y1 * inv))


def body_box(gray: np.ndarray) -> tuple[int, int, int, int] | None:
    """Bounding box of the patient, excluding uniform padding and borders."""
    h, w = gray.shape
    # Trim rows/columns that are (almost) constant: black or white padding,
    # collimator bands. Variation, not brightness, is what marks anatomy.
    #
    # The threshold is absolute, not relative to the image: only rows and
    # columns with essentially no content (std of a few grey levels, i.e.
    # padding plus JPEG noise) are removed. A faint but real structure -- the
    # lung apices, a thin patient's flank -- always varies more than that.
    row_std = gray.std(axis=1)
    col_std = gray.std(axis=0)
    rows = np.where(row_std > UNIFORM_STD)[0]
    cols = np.where(col_std > UNIFORM_STD)[0]
    if rows.size == 0 or cols.size == 0:
        return None
    y0, y1 = int(rows[0]), int(rows[-1]) + 1
    x0, x1 = int(cols[0]), int(cols[-1]) + 1

    # Small margin so the edge of the anatomy is never clipped.
    my, mx = int(0.02 * h), int(0.02 * w)
    y0, y1 = max(0, y0 - my), min(h, y1 + my)
    x0, x1 = max(0, x0 - mx), min(w, x1 + mx)

    if (y1 - y0) * (x1 - x0) < MIN_BODY_AREA * h * w:
        return None
    return x0, y0, x1, y1


def estimate_noise(gray: np.ndarray) -> float:
    """Noise standard deviation in grey levels (Immerkaer's fast estimator).

    Convolves with a kernel that cancels smooth structure and edges to first
    order, leaving mostly noise; robust enough to set a denoising strength.
    """
    k = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], np.float32)
    resp = cv2.filter2D(gray.astype(np.float32), -1, k)[1:-1, 1:-1]
    return float(np.sqrt(np.pi / 2) * np.abs(resp).mean() / 6.0)


def denoise(image: np.ndarray, strength: float | None = None) -> np.ndarray:
    """Non-local-means denoising on luminance, strength matched to the noise.

    ``strength=None`` estimates the noise and uses about that much, clamped to
    [3, 10]. The ceiling is the safety limit: above it NLM starts to erase
    hair-thin structures such as a pneumothorax's pleural line.
    """
    gray = _gray(image)
    h = strength if strength is not None else float(np.clip(estimate_noise(gray), 3.0, 10.0))
    out = cv2.fastNlMeansDenoising(gray, None, h=h, templateWindowSize=7, searchWindowSize=15)
    return cv2.cvtColor(out, cv2.COLOR_GRAY2RGB)


def enhance(
    image: np.ndarray,
    find_film_in_photo: bool = True,
    crop: bool = True,
    reduce_noise: bool = True,
    max_side: int | None = 1024,
    denoise_strength: float | None = None,
) -> EnhanceResult:
    """Run the clean-up pipeline. Never raises on an odd image; it declines instead."""
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    steps: list[str] = []

    # Radiographs carry no colour. A phone photo does (white balance, the
    # screen's tint), and it is noise to the model.
    is_photo = _is_colour(image)
    if is_photo:
        steps.append("converted colour photo to greyscale")
    image = cv2.cvtColor(_gray(image), cv2.COLOR_GRAY2RGB)

    # Film detection runs only on colour input. A scanner or PACS export is
    # greyscale; a phone photo of a film is colour. Without this gate the dark
    # lung fields of a digital radiograph, framed by brighter ribs and
    # mediastinum, look exactly like a dark film against a bright room, and the
    # "film" found is one lung.
    film = False
    if find_film_in_photo and is_photo:
        fb = find_film(_gray(image))
        if fb is not None:
            x0, y0, x1, y1 = fb
            image, film = image[y0:y1, x0:x1].copy(), True
            steps.append("located the X-ray film in the photo and removed the background")

    box = None
    if crop:
        b = body_box(_gray(image))
        if b is not None:
            x0, y0, x1, y1 = b
            h, w = image.shape[:2]
            if 1 - ((x1 - x0) * (y1 - y0)) / (h * w) >= MIN_USEFUL_TRIM:
                image, box = image[y0:y1, x0:x1].copy(), b
                steps.append("cropped empty borders around the patient")

    # Bound the cost of denoising and everything after it. The model sees
    # 320 px; 1024 keeps detail for the displayed image and saliency overlay.
    if max_side and max(image.shape[:2]) > max_side:
        f = max_side / max(image.shape[:2])
        image = cv2.resize(image, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)

    if reduce_noise:
        image = denoise(image, denoise_strength)
        steps.append("reduced noise (non-local means, strength matched to the image)")

    if not steps:
        steps.append("no clean-up needed")
    return EnhanceResult(image=image, steps=steps, crop_box=box, film_detected=film)
