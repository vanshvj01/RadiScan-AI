"""Shared fixtures. Everything runs on synthetic data -- no dataset download,
so CI is green on a fresh checkout."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from radiscan.config import Config
from radiscan.constants import CHEXPERT_LABELS
from radiscan.models.network import RadiScanNet


@pytest.fixture(scope="session")
def labels():
    return CHEXPERT_LABELS


@pytest.fixture
def model(labels):
    """Function-scoped on purpose: several tests mutate weights or toggle
    dropout, and a shared instance would leak that state across tests."""
    torch.manual_seed(0)
    return RadiScanNet(labels, backbone="densenet121", pretrained=False, image_size=224).eval()


@pytest.fixture
def batch():
    return torch.randn(2, 3, 224, 224)


@pytest.fixture
def synthetic_dataset(tmp_path, labels):
    """A tiny on-disk dataset: 24 real PNG files plus a matching CSV."""
    import cv2

    img_dir = tmp_path / "images"
    img_dir.mkdir()
    rng = np.random.default_rng(0)

    rows = []
    for i in range(24):
        rel = f"images/patient{i // 3:04d}_study1.png"
        arr = rng.integers(0, 255, (128, 128, 3), dtype=np.uint8)
        cv2.imwrite(str(tmp_path / rel), arr)
        row = {"Path": rel, "Frontal/Lateral": "Frontal"}
        for j, label in enumerate(labels):
            row[label] = float(rng.choice([0, 1, -1], p=[0.7, 0.2, 0.1])) if j else 0.0
        rows.append(row)

    csv_path = tmp_path / "data.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    return {"root": tmp_path, "csv": csv_path, "labels": labels}


@pytest.fixture
def smoke_config(synthetic_dataset, tmp_path):
    cfg = Config()
    cfg.data.data_dir = str(synthetic_dataset["root"])
    cfg.data.train_csv = str(synthetic_dataset["csv"])
    cfg.data.val_csv = str(synthetic_dataset["csv"])
    cfg.data.image_size = 128
    cfg.data.num_workers = 0
    cfg.model.pretrained = False
    cfg.train.epochs = 1
    cfg.train.batch_size = 4
    cfg.train.amp = False
    cfg.train.ema_decay = 0.0
    cfg.train.warmup_epochs = 0
    cfg.train.mixup_alpha = 0.0
    cfg.train.checkpoint_dir = str(tmp_path / "ckpt")
    cfg.train.log_dir = str(tmp_path / "logs")
    return cfg


@pytest.fixture
def xray_png(tmp_path):
    import cv2

    path = tmp_path / "xray.png"
    rng = np.random.default_rng(1)
    cv2.imwrite(str(path), rng.integers(0, 255, (256, 256, 3), dtype=np.uint8))
    return path
