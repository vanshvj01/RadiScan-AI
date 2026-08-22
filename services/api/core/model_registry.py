"""Process-wide model singleton.

Loading a checkpoint takes seconds; doing it per request would be absurd. This
holds one predictor per process, loaded lazily on first use so the API can boot
(and serve ``/health``) even when no checkpoint exists yet -- ``/health`` then
tells you exactly that, instead of the process dying at import time.
"""

from __future__ import annotations

import logging
import threading
from typing import Any

from radiscan.inference.predictor import RadiScanPredictor, load_predictor
from radiscan.models.network import MissingCheckpointError
from services.api.core.settings import get_settings

log = logging.getLogger(__name__)

_lock = threading.Lock()
_predictor: RadiScanPredictor | None = None
_load_error: str | None = None


def get_predictor() -> RadiScanPredictor:
    global _predictor, _load_error
    if _predictor is not None:
        return _predictor

    with _lock:
        if _predictor is not None:
            return _predictor
        settings = get_settings()
        try:
            _predictor = load_predictor(
                settings.checkpoint_path,
                device=settings.device,
                allow_untrained=settings.allow_untrained,
            )
            _predictor.mc_samples = settings.mc_dropout_samples
            _load_error = None
            log.info(
                "Model ready: %s | %d labels | device=%s",
                _predictor.model_version,
                len(_predictor.labels),
                _predictor.device,
            )
        except MissingCheckpointError as exc:
            _load_error = str(exc)
            raise
    return _predictor


def status() -> dict[str, Any]:
    """Non-throwing health probe."""
    settings = get_settings()
    if _predictor is not None:
        return {
            "loaded": True,
            "model_version": _predictor.model_version,
            "device": str(_predictor.device),
            "labels": list(_predictor.labels),
            "warnings": (
                ["Serving an UNTRAINED model; every output is meaningless."]
                if _predictor.model_version == "untrained-demo"
                else []
            ),
        }
    return {
        "loaded": False,
        "model_version": None,
        "device": settings.device,
        "labels": None,
        "warnings": [
            _load_error or f"No checkpoint loaded yet (expected at {settings.checkpoint_path})."
        ],
    }


def reset() -> None:
    """Test hook."""
    global _predictor, _load_error
    with _lock:
        _predictor = None
        _load_error = None
