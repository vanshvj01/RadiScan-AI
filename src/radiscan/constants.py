"""Canonical label sets.

The single most dangerous class of bug in a multi-label medical classifier is a
*silent* label-order mismatch: the head is trained on dataset A's ordering and
served with dataset B's names. Nothing crashes -- every disease name in the
output is simply wrong.

This module is the single source of truth. Every checkpoint written by
:mod:`radiscan.training` embeds the exact label list used to train it, and
:func:`radiscan.models.network.load_model` refuses to load a checkpoint whose
embedded labels disagree with the labels the caller asked for.
"""

from __future__ import annotations

from typing import Final

# --- NIH ChestX-ray14 -------------------------------------------------------
CHESTXRAY14_LABELS: Final[tuple[str, ...]] = (
    "Atelectasis",
    "Cardiomegaly",
    "Effusion",
    "Infiltration",
    "Mass",
    "Nodule",
    "Pneumonia",
    "Pneumothorax",
    "Consolidation",
    "Edema",
    "Emphysema",
    "Fibrosis",
    "Pleural_Thickening",
    "Hernia",
)

# --- CheXpert (Stanford) ----------------------------------------------------
CHEXPERT_LABELS: Final[tuple[str, ...]] = (
    "No Finding",
    "Enlarged Cardiomediastinum",
    "Cardiomegaly",
    "Lung Opacity",
    "Lung Lesion",
    "Edema",
    "Consolidation",
    "Pneumonia",
    "Atelectasis",
    "Pneumothorax",
    "Pleural Effusion",
    "Pleural Other",
    "Fracture",
    "Support Devices",
)

# The five CheXpert "competition" labels -- the ones Stanford actually scores.
# Report mean AUC over these separately; it is the number comparable to
# published leaderboards.
CHEXPERT_COMPETITION_LABELS: Final[tuple[str, ...]] = (
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    "Pleural Effusion",
)

LABEL_SETS: Final[dict[str, tuple[str, ...]]] = {
    "chexpert": CHEXPERT_LABELS,
    "chestxray14": CHESTXRAY14_LABELS,
}

RISK_LEVELS: Final[tuple[str, ...]] = ("low", "moderate", "high", "critical")

IMAGENET_MEAN: Final[tuple[float, float, float]] = (0.485, 0.456, 0.406)
IMAGENET_STD: Final[tuple[float, float, float]] = (0.229, 0.224, 0.225)


def get_labels(dataset: str) -> tuple[str, ...]:
    """Return the canonical label tuple for ``dataset``."""
    key = dataset.lower().replace("-", "").replace("_", "")
    if key in ("chexpert",):
        return CHEXPERT_LABELS
    if key in ("chestxray14", "nih", "nihchestxray14"):
        return CHESTXRAY14_LABELS
    raise ValueError(f"Unknown dataset {dataset!r}. Known: {sorted(LABEL_SETS)}")
