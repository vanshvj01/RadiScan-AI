"""One-command demo: synthetic data -> trained model -> running app.

    python scripts/demo.py

Generates a small synthetic dataset with a *learnable* signal, builds
patient-disjoint splits, trains for a few epochs on CPU, calibrates, evaluates,
and leaves a real checkpoint in ``models/checkpoints/`` so the API and dashboard
have something to serve.

This exists so you can verify the entire pipeline end to end in about five
minutes, before committing to an 11 GB download and a GPU session. The resulting
model is trained on synthetic shapes, not radiographs -- it proves the plumbing,
it is not a result. Nothing here goes in your report.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from radiscan.constants import CHEXPERT_LABELS

ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / "data" / "demo"


def make_dataset(n_patients: int = 150, size: int = 160, seed: int = 0) -> Path:
    """Synthetic radiograph-shaped images with a signal the model can actually find.

    Each label is tied to a bright disc in a fixed image region, so a working
    training loop must reach an AUC well above 0.5 -- roughly 0.85-0.90 in six
    CPU epochs. That is the point: if the demo finishes near 0.5, the pipeline
    is broken, not the data.
    """
    img_dir = DEMO / "images"
    if img_dir.exists():
        shutil.rmtree(DEMO)
    img_dir.mkdir(parents=True)

    rng = np.random.default_rng(seed)
    n_labels = len(CHEXPERT_LABELS)
    grid = int(np.ceil(np.sqrt(n_labels)))
    cell = size // grid

    rows = []
    for p in range(n_patients):
        for study in range(2):
            labels = (rng.random(n_labels) < 0.25).astype(float)

            # Lung-field-ish background: a bright ellipse pair on dark ground.
            img = rng.integers(20, 60, (size, size, 3), dtype=np.uint8)
            cv2.ellipse(
                img, (size // 3, size // 2), (size // 6, size // 3), 0, 0, 360, (110,) * 3, -1
            )
            cv2.ellipse(
                img, (2 * size // 3, size // 2), (size // 6, size // 3), 0, 0, 360, (110,) * 3, -1
            )

            # One region per label, saturated when that label is positive.
            # The marks are deliberately large and high-contrast: they must
            # survive RandomResizedCrop, rotation and CoarseDropout, or the demo
            # measures the augmentation pipeline rather than the training loop.
            for j in range(n_labels):
                if labels[j] == 0:
                    continue
                r, c = divmod(j, grid)
                cy, cx = int((r + 0.5) * cell), int((c + 0.5) * cell)
                cv2.circle(img, (cx, cy), int(cell * 0.42), (255, 255, 255), -1)

            img = np.clip(img.astype(np.int16) + rng.normal(0, 6, img.shape), 0, 255).astype(
                np.uint8
            )

            rel = f"images/patient{p:05d}_study{study}.png"
            cv2.imwrite(str(DEMO / rel), img)

            row = {"Path": rel, "Frontal/Lateral": "Frontal"}
            row.update(dict(zip(CHEXPERT_LABELS, labels)))
            rows.append(row)

    csv_path = DEMO / "all.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"  {len(rows)} synthetic images written to {DEMO}")
    return csv_path


def run(cmd: list[str], step: str) -> None:
    print(f"\n{'=' * 68}\n  {step}\n{'=' * 68}")
    result = subprocess.run([sys.executable, *cmd], cwd=ROOT)
    if result.returncode != 0:
        raise SystemExit(f"\nStep failed: {step}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the RadiScan demo pipeline")
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--patients", type=int, default=250)
    parser.add_argument(
        "--keep-data", action="store_true", help="do not delete data/demo afterwards"
    )
    args = parser.parse_args()

    print("=" * 68)
    print("  RadiScan demo -- synthetic data, real pipeline")
    print("=" * 68)
    print("\nGenerating synthetic dataset...")
    csv_path = make_dataset(n_patients=args.patients)

    run(
        [
            "-m",
            "radiscan.cli.prepare_data",
            "--source",
            str(csv_path),
            "--out-dir",
            str(DEMO / "splits"),
            "--val-size",
            "60",
            "--test-size",
            "60",
        ],
        "1/4  Patient-disjoint splits",
    )

    cfg = DEMO / "demo.yaml"
    cfg.write_text(f"""data:
  dataset: chexpert
  data_dir: {DEMO.as_posix()}
  train_csv: {(DEMO / 'splits' / 'train.csv').as_posix()}
  val_csv: {(DEMO / 'splits' / 'val.csv').as_posix()}
  test_csv: {(DEMO / 'splits' / 'test.csv').as_posix()}
  image_size: 160
  uncertain_policy: lsr
  num_workers: 0
  balanced_sampling: true
  frontal_only: true
model:
  backbone: densenet121
  pretrained: true
  dropout: 0.2
loss:
  kind: asl
train:
  epochs: {args.epochs}
  batch_size: 16
  lr: 0.0005
  warmup_epochs: 0
  scheduler: cosine
  amp: true
  ema_decay: 0.99
  mixup_alpha: 0.0
  freeze_backbone_epochs: 0
  early_stopping_patience: 99
  checkpoint_dir: models/checkpoints
  log_dir: logs/demo
inference:
  mc_dropout_samples: 10
  tta_hflip: true
""")

    run(["-m", "radiscan.cli.verify_data", "--config", str(cfg)], "2/4  Verify data integrity")
    run(["-m", "radiscan.cli.train", "--config", str(cfg)], "3/4  Train (CPU, a few minutes)")
    run(
        [
            "-m",
            "radiscan.cli.evaluate",
            "--config",
            str(cfg),
            "--split",
            "test",
            "--out",
            "results/demo",
        ],
        "4/4  Evaluate",
    )

    sample = next((DEMO / "images").glob("*.png"))
    run(
        [
            "-m",
            "radiscan.cli.predict",
            "--config",
            str(cfg),
            "--image",
            str(sample),
            "--heatmaps",
            "results/demo/heatmaps",
        ],
        "Bonus  Single-image prediction with Grad-CAM",
    )

    if not args.keep_data:
        shutil.rmtree(DEMO, ignore_errors=True)

    print(f"""
{'=' * 68}
  Demo complete.
{'=' * 68}

  Checkpoint : models/checkpoints/best_model.pth
  Results    : results/demo/   (tables, ROC curves, calibration, heatmaps)

  Mean AUC should be comfortably above 0.85 -- the synthetic signal is easy.
  If it is near 0.50, something in the pipeline is broken; that is exactly
  what this demo is for.

  Now start the application:

    uvicorn services.api.main:app --reload
    cd web && npm install && npm run dev

  Then open http://localhost:3000, register an account, and upload any
  chest X-ray image.

  Reminder: this model was trained on synthetic shapes. It is proof the
  system works, not a result. Delete the checkpoint before training for real:

    rm models/checkpoints/best_model.pth
""")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
