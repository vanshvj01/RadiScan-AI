"""Shared CLI plumbing."""

from __future__ import annotations

import argparse
import logging
import sys


def setup_logging(verbose: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )
    for noisy in ("PIL", "matplotlib", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def add_common_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--config", default="configs/chexpert_base.yaml", help="YAML config path")
    parser.add_argument("--device", default=None, help="cpu | cuda | cuda:0 | mps | auto")
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser


def parse_overrides(pairs: list[str] | None) -> dict:
    """Turn ``--set train.epochs=5 data.batch_size=8`` into a dict."""
    out: dict = {}
    for pair in pairs or []:
        if "=" not in pair:
            raise SystemExit(f"Bad override {pair!r}; expected key=value")
        key, _, raw = pair.partition("=")
        out[key.strip()] = _coerce(raw.strip())
    return out


def _coerce(value: str):
    low = value.lower()
    if low in ("true", "false"):
        return low == "true"
    if low in ("none", "null"):
        return None
    for cast in (int, float):
        try:
            return cast(value)
        except ValueError:
            continue
    return value
