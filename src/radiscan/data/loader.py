"""Image ingestion: DICOM, PNG, JPEG -> uint8 RGB ndarray."""

from __future__ import annotations

import io
import logging
from pathlib import Path
from typing import Any

import cv2
import numpy as np

log = logging.getLogger(__name__)

SUPPORTED_EXTS = {".dcm", ".dicom", ".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


class ImageLoadError(RuntimeError):
    """Raised when an image cannot be decoded. Never silently substituted."""


def load_image(
    source: str | Path | bytes, filename: str | None = None
) -> tuple[np.ndarray, dict[str, Any]]:
    """Load any supported radiograph format into uint8 RGB.

    Returns ``(image, metadata)``. Unlike the v1 loader, a missing or corrupt
    file raises :class:`ImageLoadError` -- it never returns a blank black frame.
    That substitution is what let v1 "successfully" train on 223,000 black
    rectangles and report a 0.5 AUC with no error anywhere in the logs.
    """
    if isinstance(source, bytes):
        name = filename or ""
        if name.lower().endswith((".dcm", ".dicom")) or source[128:132] == b"DICM":
            return _load_dicom_bytes(source)
        return _decode_buffer(source), {"source": "bytes", "format": Path(name).suffix}

    path = Path(source)
    if not path.exists():
        raise ImageLoadError(f"Image not found: {path}")
    ext = path.suffix.lower()
    if ext not in SUPPORTED_EXTS:
        raise ImageLoadError(f"Unsupported format {ext!r}. Supported: {sorted(SUPPORTED_EXTS)}")

    if ext in (".dcm", ".dicom"):
        return _load_dicom_bytes(path.read_bytes(), str(path))

    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ImageLoadError(f"Failed to decode image: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB), {
        "source": str(path),
        "format": ext,
        "shape": image.shape,
    }


def _decode_buffer(buf: bytes) -> np.ndarray:
    arr = np.frombuffer(buf, np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image is None:
        raise ImageLoadError("Failed to decode image buffer.")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def _load_dicom_bytes(buf: bytes, name: str = "<buffer>") -> tuple[np.ndarray, dict[str, Any]]:
    try:
        import pydicom
    except ImportError as exc:  # pragma: no cover
        raise ImageLoadError("pydicom is required to read DICOM files.") from exc

    ds = pydicom.dcmread(io.BytesIO(buf), force=True)
    if not hasattr(ds, "pixel_array"):
        raise ImageLoadError(f"DICOM {name} has no pixel data.")

    pixels = ds.pixel_array.astype(np.float32)

    # Apply rescale slope/intercept if present, then window if specified.
    slope = float(getattr(ds, "RescaleSlope", 1.0) or 1.0)
    intercept = float(getattr(ds, "RescaleIntercept", 0.0) or 0.0)
    pixels = pixels * slope + intercept

    center = getattr(ds, "WindowCenter", None)
    width = getattr(ds, "WindowWidth", None)
    if center is not None and width is not None:
        center = float(center[0] if hasattr(center, "__iter__") else center)
        width = float(width[0] if hasattr(width, "__iter__") else width)
        lo, hi = center - width / 2, center + width / 2
        pixels = np.clip(pixels, lo, hi)

    lo, hi = float(pixels.min()), float(pixels.max())
    pixels = (pixels - lo) / (hi - lo + 1e-8)

    # MONOCHROME1 means low values are bright -- invert so bone reads white.
    if str(getattr(ds, "PhotometricInterpretation", "")).upper() == "MONOCHROME1":
        pixels = 1.0 - pixels

    image = (pixels * 255).astype(np.uint8)
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    meta = {
        "source": name,
        "format": ".dcm",
        "modality": str(getattr(ds, "Modality", "")),
        "view_position": str(getattr(ds, "ViewPosition", "")),
        "study_date": str(getattr(ds, "StudyDate", "")),
        "photometric": str(getattr(ds, "PhotometricInterpretation", "")),
        "shape": image.shape,
    }
    return image, meta


def anonymize_dicom(path: str | Path, out_path: str | Path) -> Path:
    """Strip patient-identifying tags. Required before any data leaves a site."""
    import pydicom

    ds = pydicom.dcmread(str(path), force=True)
    for tag in (
        "PatientName",
        "PatientID",
        "PatientBirthDate",
        "PatientAddress",
        "OtherPatientIDs",
        "InstitutionName",
        "InstitutionAddress",
        "ReferringPhysicianName",
        "AccessionNumber",
    ):
        if tag in ds:
            setattr(ds, tag, "")
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    ds.save_as(str(out))
    return out
