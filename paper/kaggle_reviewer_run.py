"""One-shot Kaggle job for the reviewer revision.

In a Kaggle notebook (GPU on, Internet on, CheXpert attached):

    !git clone -q -b paper-revision https://github.com/vanshvj01/RadiScan-AI.git /kaggle/working/repo
    %cd /kaggle/working/repo
    !python paper/kaggle_reviewer_run.py

Steps
1. Rebuild the original patient-disjoint split (same command, same seed, same
   sizes as notebooks/02_train_on_kaggle.ipynb), so the test split is the one
   behind the originally reported numbers.
2. Use an attached checkpoint (any ``best_model.pth`` under /kaggle/input);
   otherwise retrain with the original configuration.
3. Run ``paper.reviewer_experiments`` (R1-R4, R6-R11).
4. Run ``paper.run_ablations`` (R5).
5. Zip everything to /kaggle/working/reviewer_bundle.zip and print the JSON.

Each step is skipped if its output already exists, so a re-run resumes.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
WORK = Path("/kaggle/working")
SPLITS = WORK / "data" / "splits"
OUT = WORK / "reviewer"


def sh(cmd: str, check=True) -> int:
    print(f"\n$ {cmd}", flush=True)
    env = dict(os.environ, PYTHONPATH=f"{REPO / 'src'}:{REPO}:" + os.environ.get("PYTHONPATH", ""))
    return subprocess.run(cmd, shell=True, check=check, cwd=REPO, env=env).returncode


def shallow_find(name_or_rel: str, root: str = "/kaggle/input", max_depth: int = 6):
    """Find files without walking image folders.

    Path.rglob over /kaggle/input visits every one of CheXpert's 224k images on
    Kaggle's network mount and takes hours. Prune patient/study folders and
    stop at a fixed depth instead.
    """
    rel = name_or_rel.split("/")
    hits = []
    base_depth = root.rstrip("/").count("/")
    for dirpath, dirnames, filenames in os.walk(root):
        depth = dirpath.count("/") - base_depth
        dirnames[:] = [d for d in dirnames if not d.startswith(("patient", "study")) and d not in ("valid", "train", "repo", "test") or depth < 1]
        dirnames[:] = [d for d in dirnames if not d.startswith(("patient", "study"))]
        if depth >= max_depth:
            dirnames[:] = []
        if rel[-1] in filenames and (len(rel) == 1 or Path(dirpath).name == rel[-2]):
            hits.append(Path(dirpath) / rel[-1])
    return sorted(hits)


def find_data():
    import pandas as pd

    cands = [c for c in shallow_find("train.csv") if "splits" not in c.parts]
    assert cands, "attach the CheXpert dataset"
    train_csv = cands[0]
    rel = str(pd.read_csv(train_csv, nrows=1)["Path"].iloc[0])
    for root in [train_csv.parent, train_csv.parent.parent, *train_csv.parent.parent.iterdir()]:
        if not root.is_dir():
            continue
        if (root / rel).exists():
            return train_csv, root, False
        if (root / rel.split("/", 1)[-1]).exists():
            return train_csv, root, True
    raise SystemExit(f"cannot resolve {rel}")


def main() -> int:
    t_start = time.time()
    try:  # Kaggle images ship these; a pip resolve there can take very long
        import albumentations, pydicom, timm  # noqa: F401
    except ImportError:
        sh("pip install -q timm albumentations pydicom 2>&1 | tail -1", check=False)
    import pandas as pd

    train_csv, data_root, strip = find_data()
    print("data root:", data_root, "strip:", strip)
    master = WORK / "data" / "master.csv"
    master.parent.mkdir(parents=True, exist_ok=True)
    if not master.exists():
        df = pd.read_csv(train_csv)
        if strip:
            df["Path"] = df["Path"].str.split("/", n=1).str[-1]
        df.to_csv(master, index=False)

    attached_splits = shallow_find("splits/test.csv")
    if attached_splits and not (SPLITS / "test.csv").exists():
        # the original run's own split files: identical test set by construction
        shutil.copytree(attached_splits[0].parent, SPLITS, dirs_exist_ok=True)
        print("using attached original splits from", attached_splits[0].parent)
    if not (SPLITS / "test.csv").exists():
        sh(
            f"python -m radiscan.cli.prepare_data --source {master} --out-dir {SPLITS} "
            "--frontal-only --train-size 40000 --val-size 4000 --test-size 4000 --seed 42"
        )

    cfg = WORK / "reviewer.yaml"
    cfg.write_text(
        f"""data:
  dataset: chexpert
  data_dir: {data_root}
  train_csv: {SPLITS}/train.csv
  val_csv: {SPLITS}/val.csv
  test_csv: {SPLITS}/test.csv
  image_size: 320
  uncertain_policy: lsr
  lsr_low: 0.05
  lsr_high: 0.85
  num_workers: 4
  balanced_sampling: true
  frontal_only: true
  auto_crop: true
model:
  backbone: densenet121
  pretrained: true
  dropout: 0.2
  fracture_head: true
  confidence_head: true
loss:
  kind: asl
train:
  epochs: 25
  batch_size: 64
  lr: 0.0001
  head_lr_mult: 10.0
  weight_decay: 0.0001
  warmup_epochs: 2
  scheduler: cosine
  amp: true
  ema_decay: 0.999
  grad_clip: 5.0
  early_stopping_patience: 6
  freeze_backbone_epochs: 1
  mixup_alpha: 0.2
  seed: 42
  checkpoint_dir: {WORK}/checkpoints
  log_dir: {WORK}/logs
  monitor: val_mean_auc
inference:
  mc_dropout_samples: 20
  tta_hflip: true
"""
    )

    attached = shallow_find("best_model.pth")
    ckpt = WORK / "checkpoints" / "best_model.pth"
    timing = {}
    if attached:
        ckpt = attached[0]
        print("using attached checkpoint", ckpt)
    elif not ckpt.exists():
        t0 = time.time()
        sh(f"python -m radiscan.cli.train --config {cfg}")
        timing["main_training_minutes"] = (time.time() - t0) / 60

    OUT.mkdir(parents=True, exist_ok=True)
    if not (OUT / "results.json").exists():
        t0 = time.time()
        sh(
            f"python -m paper.reviewer_experiments --config {cfg} --checkpoint {ckpt} "
            f"--out {OUT} --s-max 50 --batch-size 64 --workers 4"
        )
        timing["reviewer_experiments_minutes"] = (time.time() - t0) / 60

    t0 = time.time()
    if os.environ.get("SKIP_ABLATION") != "1":
      sh(
        f"python -m paper.run_ablations --config {cfg} --out {WORK / 'ablation'} "
        "--train-images 20000 --epochs 8",
        check=False,
      )
    timing["ablation_minutes"] = (time.time() - t0) / 60
    try:
        import torch

        timing["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    except Exception:
        pass
    timing["total_minutes"] = (time.time() - t_start) / 60
    (OUT / "kaggle_timing.json").write_text(json.dumps(timing, indent=2))

    bundle = WORK / "reviewer_bundle"
    if bundle.exists():
        shutil.rmtree(bundle)
    bundle.mkdir()
    for f in OUT.iterdir():
        if f.name != "cache.npz":
            shutil.copy(f, bundle / f.name)
    for f in (WORK / "ablation").glob("ablation.*"):
        shutil.copy(f, bundle / f.name)
    for f in (WORK / "logs").glob("*.json") if (WORK / "logs").exists() else []:
        shutil.copy(f, bundle / f"main_{f.name}")
    shutil.copy(cfg, bundle / "reviewer.yaml")
    for f in [h for n in ("history.json", "final_summary.json") for h in shallow_find("logs/" + n)]:
        shutil.copy(f, bundle / f"orig_{f.name}")
    shutil.make_archive(str(WORK / "reviewer_bundle"), "zip", bundle)
    print("\n==== REVIEWER RESULTS ====")
    print((OUT / "results.json").read_text()[:20000])
    abl = WORK / "ablation" / "ablation.csv"
    if abl.exists():
        print(abl.read_text())
    print(json.dumps(timing, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
