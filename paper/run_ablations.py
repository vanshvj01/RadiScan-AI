"""R5: component ablations under one fixed, reduced training budget.

    python -m paper.run_ablations --config <cfg.yaml> --out results/ablation \
        --train-images 20000 --epochs 8

Every arm uses the same patient-disjoint split, the same training subset (same
seed), the same schedule and the same evaluation (test split, uncertain labels
masked, MC-20 posterior mean, each arm's own validation-tuned thresholds and
temperature). Only the named component changes. The reference arm ("full") is
retrained at the same reduced budget, so the comparison is controlled; it is
*not* the 25-epoch model reported in Table I.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from paper.reviewer_experiments import (
    collect,
    ece_brier,
    load_checkpoint,
    make_dataset,
    masks,
    posterior,
)
from radiscan.config import Config
from radiscan.constants import CHEXPERT_COMPETITION_LABELS
from radiscan.evaluation.metrics import per_class_metrics, summarize

ARMS: dict[str, list[str]] = {
    "full": [],
    "no_asl": ["loss.kind=bce"],  # class-weighted BCE (pos_weight) replaces ASL
    "no_balanced_sampling": ["data.balanced_sampling=false"],
    "no_label_smoothing": ["data.uncertain_policy=zeros"],  # U-Zeros
    "no_mixup": ["train.mixup_alpha=0.0"],
    "no_confidence_head": ["model.confidence_head=false"],
    "none_of_the_four": [
        "loss.kind=bce",
        "data.balanced_sampling=false",
        "data.uncertain_policy=zeros",
        "train.mixup_alpha=0.0",
    ],
}


def evaluate_arm(cfg: Config, ckpt: Path, device, s: int = 20) -> dict:
    model, meta = load_checkpoint(ckpt, device)
    labels = list(model.labels)
    ds = make_dataset(cfg, cfg.data.test_csv, labels)
    data = collect(model, ds, device, s, 64, cfg.data.num_workers, check=False)
    post = posterior(data["mc"], float(model.temperature.item()))
    y, m = masks(data["raw"])
    rows = per_class_metrics(y, post["mu"], labels, thresholds=meta.get("thresholds", {}), mask=m)
    ov = summarize(rows)
    cx5 = summarize(rows, subset=CHEXPERT_COMPETITION_LABELS)
    cal = ece_brier(y, post["mu"], m)
    return {
        "mean_auc": ov["mean_auc"],
        "mean_ap": ov["mean_ap"],
        "chexpert5_auc": cx5["mean_auc"],
        "macro_f1": ov["mean_f1"],
        "ece": cal["ece"],
        "brier": cal["brier"],
        "best_epoch": meta.get("epoch"),
        "temperature": float(model.temperature.item()),
        "per_class_auc": {r["label"]: r["auc"] for r in rows},
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", default="results/ablation")
    ap.add_argument("--train-images", type=int, default=20000)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--arms", nargs="*", default=list(ARMS))
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = {}
    res_path = out / "ablation.json"
    if res_path.exists():
        results = json.loads(res_path.read_text())

    for arm in args.arms:
        if arm in results:
            print(f"[skip] {arm} already evaluated")
            continue
        arm_dir = out / arm
        sets = ARMS[arm] + [
            f"train.epochs={args.epochs}",
            f"train.early_stopping_patience={args.epochs}",
            f"data.max_train_images={args.train_images}",
            f"train.seed={args.seed}",
            f"train.checkpoint_dir={arm_dir / 'ckpt'}",
            f"train.log_dir={arm_dir / 'logs'}",
        ]
        if args.batch_size:
            sets.append(f"train.batch_size={args.batch_size}")
        cmd = [sys.executable, "-m", "radiscan.cli.train", "--config", args.config, "--set", *sets]
        print("\n>>>", arm, " ".join(cmd), flush=True)
        subprocess.run(cmd, check=True)

        cfg = Config.from_yaml(args.config)
        r = evaluate_arm(cfg, arm_dir / "ckpt" / "best_model.pth", device)
        r["overrides"] = ARMS[arm]
        results[arm] = r
        res_path.write_text(json.dumps(results, indent=2, default=float))
        print(arm, {k: v for k, v in r.items() if k != "per_class_auc"}, flush=True)

    rows = []
    ref = results.get("full", {})
    for arm, r in results.items():
        row = {"arm": arm, **{k: v for k, v in r.items() if k not in ("per_class_auc", "overrides")}}
        if ref:
            row["delta_auc"] = r["mean_auc"] - ref["mean_auc"]
            row["delta_ap"] = r["mean_ap"] - ref["mean_ap"]
        rows.append(row)
    pd.DataFrame(rows).to_csv(out / "ablation.csv", index=False)
    print(pd.DataFrame(rows).round(4).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
