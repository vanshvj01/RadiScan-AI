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


# --- API fixtures: shared by test_api.py and test_sharing.py -----------------
@pytest.fixture
def client(tmp_path, monkeypatch):
    """The whole API in-process: real SQLite, untrained model, no Redis."""
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path / 'test.db'}")
    monkeypatch.setenv("SECRET_KEY", "test-secret-key-not-for-production-use-1234")
    monkeypatch.setenv("DEBUG", "true")
    monkeypatch.setenv("ALLOW_UNTRAINED", "true")
    # Point at a path that cannot exist, so these tests always exercise the
    # untrained-demo path. Without this, a real checkpoint left in the default
    # location by scripts/demo.py silently changes what is under test.
    monkeypatch.setenv("CHECKPOINT_PATH", str(tmp_path / "no-such-checkpoint.pth"))
    monkeypatch.setenv("UPLOAD_DIR", str(tmp_path / "uploads"))
    monkeypatch.setenv("HEATMAP_DIR", str(tmp_path / "heatmaps"))
    monkeypatch.setenv("CELERY_EAGER", "true")
    monkeypatch.setenv("MC_DROPOUT_SAMPLES", "3")

    from services.api.core import model_registry
    from services.api.core import settings as settings_mod

    settings_mod.get_settings.cache_clear()
    model_registry.reset()

    # Modules that snapshot settings or the engine at import time must be
    # reloaded, in dependency order, so the test env vars actually take effect.
    import importlib

    import services.api.core.tasks as tasks_mod
    import services.api.main as main_mod
    from services.api.db import session as session_mod
    from services.api.routes import auth as auth_mod
    from services.api.routes import health as health_mod
    from services.api.routes import predictions as pred_mod
    from services.api.routes import reminders as reminders_mod
    from services.api.routes import reports as reports_mod
    from services.api.routes import reviews as reviews_mod
    from services.api.routes import sharing as sharing_mod
    from services.api.routes import worklist as worklist_mod

    for mod in (
        session_mod,
        tasks_mod,
        auth_mod,
        pred_mod,
        reports_mod,
        sharing_mod,
        reviews_mod,
        worklist_mod,
        reminders_mod,
        health_mod,
        main_mod,
    ):
        importlib.reload(mod)

    with TestClient(main_mod.app) as c:
        yield c


@pytest.fixture
def png_bytes():
    import cv2

    arr = np.random.default_rng(0).integers(0, 255, (256, 256, 3), dtype=np.uint8)
    ok, buf = cv2.imencode(".png", arr)
    assert ok
    return buf.tobytes()
