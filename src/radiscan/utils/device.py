"""Device selection that never crashes on a CPU-only machine."""

from __future__ import annotations

import logging
import os

import torch

log = logging.getLogger(__name__)


def resolve_device(preference: str | None = None) -> torch.device:
    """Resolve a torch device.

    Order of precedence: explicit ``preference`` -> ``RADISCAN_DEVICE`` env var
    -> best available. A request for ``cuda`` on a machine without CUDA logs a
    warning and falls back to CPU instead of raising -- the original repo
    hard-defaulted to ``cuda`` and crashed on every laptop.
    """
    want = (preference or os.getenv("RADISCAN_DEVICE") or "auto").lower()

    if want == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    if want.startswith("cuda") and not torch.cuda.is_available():
        log.warning("CUDA requested but unavailable; falling back to CPU.")
        return torch.device("cpu")
    if want == "mps" and not (
        getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
    ):
        log.warning("MPS requested but unavailable; falling back to CPU.")
        return torch.device("cpu")
    return torch.device(want)


def describe_device(device: torch.device) -> str:
    if device.type == "cuda":
        idx = device.index or 0
        props = torch.cuda.get_device_properties(idx)
        return f"cuda:{idx} ({props.name}, {props.total_memory / 1e9:.1f} GB)"
    return device.type


def amp_supported(device: torch.device) -> bool:
    return device.type == "cuda"
