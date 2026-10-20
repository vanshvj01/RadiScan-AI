"""Image-quality gate, run before a radiograph is interpreted.

A confident probability computed from an unreadable film is worse than no
probability at all. This module checks the things a radiographer would reject a
film for -- resolution, exposure, contrast, blur, and being a phone photo of a
film rather than the original. The learned aleatoric confidence is recorded
alongside, but does not raise issues (see ``CONFIDENCE_FAIR``).

The result never blocks a prediction. It is attached to the result and shown
to the reader, and a ``poor`` film adds a retake recommendation to the report:
whether to act on the analysis stays the clinician's decision.

Thresholds are deliberately conservative heuristics, chosen so that a normal
digital chest radiograph passes. They are not validated against radiographer
judgement; say so if they are quoted in the report.
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

GOOD, FAIR, POOR = "good", "fair", "poor"
_RANK = {GOOD: 0, FAIR: 1, POOR: 2}

# Resolution (shorter side, pixels).
MIN_SIDE_POOR = 256
MIN_SIDE_FAIR = 512
# Exposure (mean grey level, 0-255) and clipped-pixel fractions.
DARK_MEAN, BRIGHT_MEAN = 35.0, 220.0
CLIPPED_FRACTION = 0.55
# Contrast: spread between the 1st and 99th percentile grey levels.
CONTRAST_POOR, CONTRAST_FAIR = 35.0, 70.0
# Sharpness: variance of the Laplacian at a fixed 512 px working size.
BLUR_POOR, BLUR_FAIR = 12.0, 30.0
# Mean aleatoric confidence across findings, when the model has the head.
# Recorded, but NOT used to raise an issue: on the trained CheXpert checkpoint
# the head's mean output sat in a narrow 0.42-0.44 band on real films and was
# *higher* (0.46) on pure noise, so it does not separate readable films from
# unreadable ones. Set a threshold here only after it is shown to.
CONFIDENCE_FAIR: float | None = None


def _issue(code: str, level: str, message: str) -> dict[str, str]:
    return {"code": code, "level": level, "message": message}


def assess_quality(
    image: np.ndarray,
    preprocessing_steps: list[str] | None = None,
    model_confidence: np.ndarray | None = None,
) -> dict[str, Any]:
    """Return ``{"status", "issues", "metrics", "recommendation"}``.

    ``image`` is the upload as loaded (uint8, grey or RGB). ``preprocessing_steps``
    are the clean-up stage's notes, used to detect a photographed film.
    ``model_confidence`` is the confidence head's per-finding output, if any.
    """
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    if gray.dtype != np.uint8:
        gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    h, w = gray.shape[:2]
    min_side = int(min(h, w))
    mean = float(gray.mean())
    dark = float((gray < 10).mean())
    bright = float((gray > 245).mean())
    p1, p99 = (float(v) for v in np.percentile(gray, [1, 99]))
    contrast = p99 - p1

    scale = 512.0 / max(h, w)
    work = cv2.resize(gray, (max(1, int(w * scale)), max(1, int(h * scale)))) if scale < 1 else gray
    sharpness = float(cv2.Laplacian(work, cv2.CV_64F).var())

    issues: list[dict[str, str]] = []

    if min_side < MIN_SIDE_POOR:
        issues.append(_issue("low_resolution", POOR, f"Very low resolution ({w}x{h} px)."))
    elif min_side < MIN_SIDE_FAIR:
        issues.append(_issue("low_resolution", FAIR, f"Low resolution ({w}x{h} px)."))

    if mean < DARK_MEAN or dark > CLIPPED_FRACTION:
        issues.append(
            _issue("underexposed", POOR, "Image is very dark; it may be under-exposed or inverted.")
        )
    elif mean > BRIGHT_MEAN or bright > CLIPPED_FRACTION:
        issues.append(_issue("overexposed", POOR, "Image is washed out; it may be over-exposed."))

    if contrast < CONTRAST_POOR:
        issues.append(_issue("low_contrast", POOR, "Very low contrast between structures."))
    elif contrast < CONTRAST_FAIR:
        issues.append(_issue("low_contrast", FAIR, "Low contrast between structures."))

    if sharpness < BLUR_POOR:
        issues.append(_issue("blurred", POOR, "Image is blurred; fine detail is lost."))
    elif sharpness < BLUR_FAIR:
        issues.append(_issue("blurred", FAIR, "Image is slightly soft."))

    steps = " ".join(preprocessing_steps or []).lower()
    if "photo" in steps or "located the x-ray film" in steps:
        issues.append(
            _issue(
                "photo_of_film",
                FAIR,
                "This is a photograph of a film, not the original image; glare and "
                "perspective can hide findings.",
            )
        )

    mean_conf = None
    if model_confidence is not None and np.size(model_confidence):
        mean_conf = float(np.mean(model_confidence))
        if CONFIDENCE_FAIR is not None and mean_conf < CONFIDENCE_FAIR:
            issues.append(
                _issue(
                    "model_low_confidence",
                    FAIR,
                    "The model reports low confidence on this film (possible rotation, "
                    "positioning or annotation clutter).",
                )
            )

    status = max((i["level"] for i in issues), key=_RANK.__getitem__, default=GOOD)
    recommendation = {
        GOOD: "Image quality is adequate for analysis.",
        FAIR: "Image quality is acceptable but limited; interpret with caution.",
        POOR: "Image quality is poor. Repeat the radiograph before relying on this analysis.",
    }[status]

    return {
        "status": status,
        "issues": issues,
        "recommendation": recommendation,
        "metrics": {
            "width": int(w),
            "height": int(h),
            "mean_intensity": round(mean, 1),
            "dark_fraction": round(dark, 3),
            "bright_fraction": round(bright, 3),
            "contrast": round(contrast, 1),
            "sharpness": round(sharpness, 1),
            "model_confidence": None if mean_conf is None else round(mean_conf, 3),
        },
    }
