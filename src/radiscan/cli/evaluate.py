"""Evaluation: produces every table and figure the report needs, in one command.

    python -m radiscan.cli.evaluate --config configs/chexpert_base.yaml

Writes to ``results/``:

* ``per_class_metrics.csv``   -- Table 1 of the report
* ``summary.json``            -- headline numbers
* ``uncertainty_quartiles.csv`` -- the money table
* ``roc_curves.png``, ``pr_curves.png``, ``calibration.png``,
  ``per_class_auc.png``, ``uncertainty_quartiles.png``,
  ``selective_prediction.png``, ``training_curves.png``
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

from radiscan.cli._common import add_common_args, parse_overrides, setup_logging
from radiscan.config import Config
from radiscan.constants import CHEXPERT_COMPETITION_LABELS
from radiscan.data.datasets import build_dataloaders
from radiscan.evaluation import plots
from radiscan.evaluation.metrics import (
    expected_calibration_error,
    per_class_metrics,
    selective_prediction_curve,
    summarize,
    uncertainty_quartiles,
)
from radiscan.models.network import load_model
from radiscan.modules.uncertainty import predict_with_uncertainty
from radiscan.utils.device import resolve_device
from radiscan.utils.seed import set_seed

log = logging.getLogger(__name__)


@torch.no_grad()
def run_inference(model, loader, device, mc_samples: int, tta: bool):
    """Collect calibrated probabilities and MC-Dropout uncertainty over a split."""
    probs, stds, targets, masks = [], [], [], []
    for batch in tqdm(loader, desc="evaluating"):
        images = batch["image"].to(device)
        out = predict_with_uncertainty(
            model, images, n_samples=mc_samples, tta_hflip=tta, calibrated=True
        )
        probs.append(out["mean"])
        stds.append(out["std"])
        targets.append(batch["labels"].numpy())
        masks.append(batch["label_mask"].numpy())
    return (
        np.concatenate(probs),
        np.concatenate(stds),
        np.concatenate(targets),
        np.concatenate(masks),
    )


def main(argv: list[str] | None = None) -> int:
    parser = add_common_args(argparse.ArgumentParser(description="Evaluate RadiScan"))
    parser.add_argument("--set", nargs="*", dest="overrides")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--split", default="val", choices=["val", "test"])
    parser.add_argument("--out", default="results")
    parser.add_argument("--mc-samples", type=int, default=None)
    args = parser.parse_args(argv)
    setup_logging(args.verbose)

    cfg = Config.from_yaml(args.config).override(**parse_overrides(args.overrides))
    set_seed(cfg.train.seed)
    device = resolve_device(args.device)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt = args.checkpoint or str(Path(cfg.train.checkpoint_dir) / "best_model.pth")
    model, meta = load_model(ckpt, device=device)
    labels = list(model.labels)
    thresholds = dict(meta.get("thresholds", {}))
    mc = args.mc_samples or cfg.inference.mc_dropout_samples
    log.info(
        "Checkpoint %s | %d labels | T=%.3f | MC=%d",
        ckpt,
        len(labels),
        float(model.temperature.item()),
        mc,
    )

    loaders = build_dataloaders(cfg, verify=False)
    loader = loaders.get(args.split) or loaders["val"]

    probs, stds, targets, masks = run_inference(model, loader, device, mc, cfg.inference.tta_hflip)

    # -- tables --------------------------------------------------------------
    rows = per_class_metrics(targets, probs, labels, thresholds=thresholds, mask=masks)
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "per_class_metrics.csv", index=False)

    overall = summarize(rows)
    competition = summarize(rows, subset=CHEXPERT_COMPETITION_LABELS)
    ece, _ = expected_calibration_error(targets, probs)

    quartiles = uncertainty_quartiles(targets, probs, stds, thresholds, labels)
    selective = selective_prediction_curve(targets, probs, stds)

    monotonic = all(
        quartiles[i]["accuracy"] >= quartiles[i + 1]["accuracy"] for i in range(len(quartiles) - 1)
    )

    summary = {
        "checkpoint": ckpt,
        "split": args.split,
        "n_samples": int(len(targets)),
        "mc_dropout_samples": mc,
        "tta_hflip": cfg.inference.tta_hflip,
        "temperature": float(model.temperature.item()),
        "overall": overall,
        "chexpert_competition_5": competition,
        "ece": ece,
        "uncertainty_is_informative": bool(monotonic),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    # -- figures -------------------------------------------------------------
    plots.roc_curves(targets, probs, labels, out_dir)
    plots.pr_curves(targets, probs, labels, out_dir)
    plots.reliability_diagram(targets, probs, out_dir)
    plots.per_class_bars(rows, out_dir)
    plots.uncertainty_vs_accuracy(quartiles, out_dir)
    plots.selective_prediction(selective, out_dir)

    history_path = Path(cfg.train.log_dir) / "history.json"
    if history_path.exists():
        plots.training_curves(json.loads(history_path.read_text()), out_dir)

    # -- console -------------------------------------------------------------
    print(
        "\n"
        + df[["label", "support", "auc", "ap", "precision", "recall", "f1"]].to_string(index=False)
    )
    print("\n" + "=" * 62)
    print(f"  Mean AUC (all {len(labels)})      : {overall['mean_auc']:.4f}")
    print(f"  Mean AP                       : {overall['mean_ap']:.4f}")
    print(f"  Mean F1 @ tuned thresholds    : {overall['mean_f1']:.4f}")
    print(f"  Mean AUC (CheXpert-5)         : {competition['mean_auc']:.4f}")
    print(f"  Expected calibration error    : {ece:.4f}")
    print("=" * 62)
    print("\nUncertainty stratification (Table 3):")
    for q in quartiles:
        print(
            f"  {q['quartile']:<24} n={q['n']:>7,}  acc={q['accuracy']:.4f}  mean sigma={q['mean_uncertainty']:.4f}"
        )
    print(
        "\n  -> Accuracy falls monotonically with uncertainty: "
        + (
            "YES -- the uncertainty estimates carry real signal."
            if monotonic
            else "NO -- report this honestly; it is a legitimate negative result."
        )
    )
    print(f"\nAll tables and figures written to {out_dir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
