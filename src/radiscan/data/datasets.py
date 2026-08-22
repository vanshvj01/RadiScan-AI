"""Datasets and dataloader construction.

Design rules enforced here, each of which was a real bug in v1:

* **Fail loudly.** A missing image raises. There is a ``strict=False`` escape
  hatch, but it is opt-in and logs every substitution.
* **Verify before training.** :func:`verify_dataset` samples the CSV, checks the
  files actually resolve on disk, and refuses to build a loader when the hit
  rate is below a threshold.
* **``max_images`` actually works.** It is threaded from the CLI through to the
  dataframe slice.
* **Labels come from one place.** The dataset reports the label tuple it used;
  the trainer writes that tuple into the checkpoint.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from radiscan.constants import CHESTXRAY14_LABELS, CHEXPERT_LABELS, get_labels
from radiscan.data.loader import ImageLoadError, load_image
from radiscan.data.transforms import apply_clahe, eval_transforms, train_transforms
from radiscan.utils.seed import worker_init_fn

log = logging.getLogger(__name__)


class DatasetVerificationError(RuntimeError):
    pass


class ChestXrayDataset(Dataset):
    """Multi-label chest radiograph dataset.

    Uncertain CheXpert labels (``-1``) are handled by ``uncertain_policy``:

    * ``zeros``  -- treat as negative (simple, slightly pessimistic)
    * ``ones``   -- treat as positive
    * ``ignore`` -- mask out of the loss entirely (returns ``label_mask``)
    * ``lsr``    -- label-smoothing regression: map to a soft target in
      ``(lsr_low, lsr_high)``. This is Stanford's own best-performing policy and
      is the default; it beats U-Zeros by roughly 0.01-0.02 mean AUC.
    """

    def __init__(
        self,
        csv_path: str | Path,
        data_dir: str | Path,
        labels: tuple[str, ...],
        transform=None,
        strict: bool = True,
        uncertain_policy: str = "lsr",
        lsr_low: float = 0.05,
        lsr_high: float = 0.85,
        max_images: int | None = None,
        use_clahe: bool = True,
        frontal_only: bool = False,
        path_column: str | None = None,
        seed: int = 42,
    ):
        self.data_dir = Path(data_dir)
        self.labels = tuple(labels)
        self.transform = transform
        self.strict = strict
        self.uncertain_policy = uncertain_policy.lower()
        self.lsr_low = lsr_low
        self.lsr_high = lsr_high
        self.use_clahe = use_clahe
        self._missing = 0

        df = pd.read_csv(csv_path)
        self.path_column = path_column or _detect_path_column(df)

        if frontal_only:
            for col in ("Frontal/Lateral", "ViewPosition", "View Position"):
                if col in df.columns:
                    mask = df[col].astype(str).str.lower().str.startswith(("frontal", "pa", "ap"))
                    df = df[mask]
                    break

        missing_labels = [c for c in self.labels if c not in df.columns]
        if missing_labels:
            raise ValueError(
                f"CSV {csv_path} is missing label columns {missing_labels}. "
                f"Expected the {len(self.labels)}-label set: {list(self.labels)}"
            )

        if max_images is not None and max_images < len(df):
            df = df.sample(n=max_images, random_state=seed)

        self.df = df.reset_index(drop=True)
        self.raw_labels = self.df[list(self.labels)].fillna(0).to_numpy(dtype=np.float32)

    def __len__(self) -> int:
        return len(self.df)

    # -- label handling ------------------------------------------------------
    def _resolve_labels(self, row_idx: int) -> tuple[np.ndarray, np.ndarray]:
        raw = self.raw_labels[row_idx].copy()
        mask = np.ones_like(raw)
        uncertain = raw == -1.0

        if self.uncertain_policy == "zeros":
            raw[uncertain] = 0.0
        elif self.uncertain_policy == "ones":
            raw[uncertain] = 1.0
        elif self.uncertain_policy == "ignore":
            raw[uncertain] = 0.0
            mask[uncertain] = 0.0
        elif self.uncertain_policy == "lsr":
            rng = np.random.default_rng(row_idx)
            raw[uncertain] = rng.uniform(self.lsr_low, self.lsr_high, size=uncertain.sum())
        else:
            raise ValueError(f"Unknown uncertain_policy {self.uncertain_policy!r}")

        raw = np.clip(raw, 0.0, 1.0)
        return raw, mask

    @property
    def positive_counts(self) -> np.ndarray:
        return (self.raw_labels == 1.0).sum(axis=0)

    def pos_weight(self) -> torch.Tensor:
        """``neg/pos`` per class, clipped, for weighted BCE."""
        pos = self.positive_counts.astype(np.float64)
        neg = len(self) - pos
        w = np.where(pos > 0, neg / np.maximum(pos, 1), 1.0)
        return torch.tensor(np.clip(w, 1.0, 50.0), dtype=torch.float32)

    # -- item ----------------------------------------------------------------
    def image_path(self, idx: int) -> Path:
        rel = str(self.df.iloc[idx][self.path_column])
        candidate = self.data_dir / rel
        if candidate.exists():
            return candidate
        # CheXpert CSVs prefix paths with the archive folder; tolerate both.
        alt = (
            self.data_dir / Path(rel).relative_to(Path(rel).parts[0])
            if Path(rel).parts
            else candidate
        )
        return alt if alt.exists() else candidate

    def __getitem__(self, idx: int) -> dict[str, Any]:
        path = self.image_path(idx)
        try:
            image, _ = load_image(path)
        except ImageLoadError:
            self._missing += 1
            if self.strict:
                raise DatasetVerificationError(
                    f"Image missing or unreadable: {path}\n"
                    "Refusing to substitute a blank frame -- that produces a model that "
                    "trains 'successfully' to ~0.50 AUC with no visible error.\n"
                    "Run `python -m radiscan.cli.verify_data --config <cfg>` to diagnose, "
                    "or pass strict=False to accept blanks knowingly."
                ) from None
            log.warning("Missing image %s (%d so far); substituting blank.", path, self._missing)
            image = np.zeros((512, 512, 3), dtype=np.uint8)

        if self.use_clahe:
            image = apply_clahe(image)

        if self.transform is not None:
            image = self.transform(image=image)["image"]
        else:
            image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0

        labels, mask = self._resolve_labels(idx)
        item: dict[str, Any] = {
            "image": image,
            "labels": torch.from_numpy(labels),
            "label_mask": torch.from_numpy(mask),
            "index": idx,
        }
        if "Fracture" in self.labels:
            item["fracture"] = item["labels"][self.labels.index("Fracture")]
        else:
            item["fracture"] = torch.tensor(0.0)
        return item


def _detect_path_column(df: pd.DataFrame) -> str:
    for col in ("Path", "path", "image_path", "Image Index", "filename", "dicom_id"):
        if col in df.columns:
            return col
    raise ValueError(
        f"Could not find an image-path column in CSV. Columns seen: {list(df.columns)[:12]}"
    )


# --------------------------------------------------------------------------
# Verification -- run this before you burn a GPU-hour
# --------------------------------------------------------------------------
def verify_dataset(
    csv_path: str | Path,
    data_dir: str | Path,
    labels: tuple[str, ...] | None = None,
    sample: int = 200,
    min_hit_rate: float = 0.99,
    raise_on_fail: bool = True,
) -> dict[str, Any]:
    """Check that CSV rows actually resolve to files on disk."""
    df = pd.read_csv(csv_path)
    path_col = _detect_path_column(df)
    data_dir = Path(data_dir)

    probe = df.sample(n=min(sample, len(df)), random_state=0)
    found = 0
    examples: list[str] = []
    for rel in probe[path_col].astype(str):
        if (data_dir / rel).exists():
            found += 1
        elif len(examples) < 5:
            examples.append(str(data_dir / rel))

    hit_rate = found / len(probe) if len(probe) else 0.0
    report = {
        "csv": str(csv_path),
        "rows": len(df),
        "sampled": len(probe),
        "found": found,
        "hit_rate": hit_rate,
        "missing_examples": examples,
        "path_column": path_col,
    }

    if labels:
        missing_cols = [c for c in labels if c not in df.columns]
        report["missing_label_columns"] = missing_cols
        counts = {c: int((df[c] == 1).sum()) for c in labels if c in df.columns}
        report["positive_counts"] = counts
        report["positive_rates"] = {k: round(v / max(len(df), 1), 4) for k, v in counts.items()}

    if hit_rate < min_hit_rate and raise_on_fail:
        raise DatasetVerificationError(
            f"Only {found}/{len(probe)} sampled images exist under {data_dir} "
            f"(hit rate {hit_rate:.1%}, required {min_hit_rate:.0%}).\n"
            f"First missing paths:\n  " + "\n  ".join(examples) + "\n"
            "Your CSV paths and your extracted folder do not line up. Fix that before training."
        )
    return report


# --------------------------------------------------------------------------
# Loaders
# --------------------------------------------------------------------------
def make_balanced_sampler(dataset: ChestXrayDataset, cap: float = 20.0) -> WeightedRandomSampler:
    """Weight each sample by the inverse frequency of its rarest positive class.

    Uniform sampling on CheXpert means the network sees Pleural Other roughly
    once every 200 batches. This lifts rare-class exposure without the label
    noise that oversampling duplicates introduce, and is worth ~0.02 mean AUC on
    the tail classes.
    """
    labels = (dataset.raw_labels == 1.0).astype(np.float32)
    freq = labels.sum(axis=0)
    freq = np.maximum(freq, 1.0)
    class_w = np.clip(len(dataset) / freq, 1.0, cap)

    weights = np.ones(len(dataset), dtype=np.float64)
    has_pos = labels.sum(axis=1) > 0
    weights[has_pos] = (labels[has_pos] * class_w).max(axis=1)
    return WeightedRandomSampler(
        weights=torch.as_tensor(weights, dtype=torch.double),
        num_samples=len(dataset),
        replacement=True,
    )


def build_dataloaders(cfg, verify: bool = True) -> dict[str, Any]:
    """Construct train/val (and optional test) loaders from a Config."""
    labels = get_labels(cfg.data.dataset)
    data_dir = cfg.data.data_dir

    if verify:
        verify_dataset(cfg.data.train_csv, data_dir, labels)
        verify_dataset(cfg.data.val_csv, data_dir, labels)

    common = dict(
        data_dir=data_dir,
        labels=labels,
        uncertain_policy=cfg.data.uncertain_policy,
        lsr_low=cfg.data.lsr_low,
        lsr_high=cfg.data.lsr_high,
        frontal_only=cfg.data.frontal_only,
        strict=True,
    )

    train_ds = ChestXrayDataset(
        cfg.data.train_csv,
        transform=train_transforms(cfg.data.image_size),
        max_images=cfg.data.max_train_images,
        **common,
    )
    val_ds = ChestXrayDataset(
        cfg.data.val_csv,
        transform=eval_transforms(cfg.data.image_size),
        max_images=cfg.data.max_val_images,
        **common,
    )

    sampler = make_balanced_sampler(train_ds) if cfg.data.balanced_sampling else None
    loader_kwargs = dict(
        num_workers=cfg.data.num_workers,
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=worker_init_fn,
        persistent_workers=cfg.data.num_workers > 0,
    )

    out = {
        "labels": labels,
        "train_dataset": train_ds,
        "val_dataset": val_ds,
        "train": DataLoader(
            train_ds,
            batch_size=cfg.train.batch_size,
            sampler=sampler,
            shuffle=sampler is None,
            drop_last=True,
            **loader_kwargs,
        ),
        "val": DataLoader(
            val_ds, batch_size=cfg.train.batch_size * 2, shuffle=False, **loader_kwargs
        ),
    }

    if cfg.data.test_csv:
        test_ds = ChestXrayDataset(
            cfg.data.test_csv, transform=eval_transforms(cfg.data.image_size), **common
        )
        out["test_dataset"] = test_ds
        out["test"] = DataLoader(
            test_ds, batch_size=cfg.train.batch_size * 2, shuffle=False, **loader_kwargs
        )
    return out


__all__ = [
    "CHEXPERT_LABELS",
    "CHESTXRAY14_LABELS",
    "ChestXrayDataset",
    "DatasetVerificationError",
    "build_dataloaders",
    "make_balanced_sampler",
    "verify_dataset",
]
