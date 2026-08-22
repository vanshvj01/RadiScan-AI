"""Training entry point.

    python -m radiscan.cli.train --config configs/chexpert_base.yaml
    python -m radiscan.cli.train --config configs/chexpert_base.yaml \
        --set train.epochs=3 data.max_train_images=2000 train.batch_size=8

After the last epoch this script also:

1. re-runs validation with the best (EMA) weights,
2. fits a temperature on the validation logits,
3. tunes a per-class operating threshold,
4. writes both back into ``best_model.pth``.

So the served checkpoint is calibrated and has operating points baked in --
there is no separate "remember to calibrate" step to forget.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

from radiscan.cli._common import add_common_args, parse_overrides, setup_logging
from radiscan.config import Config
from radiscan.data.datasets import build_dataloaders
from radiscan.evaluation.metrics import optimal_thresholds, per_class_metrics, summarize
from radiscan.models.network import build_model, count_parameters, load_model
from radiscan.training.calibration import calibrate_model, collect_logits
from radiscan.training.trainer import Trainer
from radiscan.utils.device import describe_device, resolve_device
from radiscan.utils.seed import set_seed

log = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> int:
    parser = add_common_args(argparse.ArgumentParser(description="Train RadiScan"))
    parser.add_argument("--set", nargs="*", dest="overrides", help="key=value config overrides")
    parser.add_argument("--resume", default=None, help="checkpoint to resume from")
    parser.add_argument(
        "--no-verify", action="store_true", help="skip dataset verification (unwise)"
    )
    parser.add_argument("--deterministic", action="store_true")
    args = parser.parse_args(argv)
    setup_logging(args.verbose)

    cfg = Config.from_yaml(args.config).override(**parse_overrides(args.overrides))
    set_seed(cfg.train.seed, deterministic=args.deterministic)
    device = resolve_device(args.device)
    log.info("Device: %s", describe_device(device))

    loaders = build_dataloaders(cfg, verify=not args.no_verify)
    labels = loaders["labels"]
    log.info(
        "Data: %d train / %d val | %d labels | balanced_sampling=%s",
        len(loaders["train_dataset"]),
        len(loaders["val_dataset"]),
        len(labels),
        cfg.data.balanced_sampling,
    )

    if args.resume:
        model, _ = load_model(args.resume, device=device, expected_labels=labels)
        log.info("Resumed from %s", args.resume)
    else:
        model = build_model(cfg, labels)

    total, trainable = count_parameters(model)
    log.info(
        "Model: %s | %.2fM params (%.2fM trainable)",
        cfg.model.backbone,
        total / 1e6,
        trainable / 1e6,
    )

    trainer = Trainer(
        model,
        cfg,
        loaders["train"],
        loaders["val"],
        device=device,
        pos_weight=loaders["train_dataset"].pos_weight(),
    )
    result = trainer.fit()
    log.info(
        "Training finished in %.1f min. Best %s = %.4f",
        result["elapsed_minutes"],
        cfg.train.monitor,
        result["best_metric"],
    )

    # -- post-training: calibrate + tune thresholds --------------------------
    best_path = Path(result["best_checkpoint"])
    if not best_path.exists():
        log.error("No best checkpoint was written; skipping calibration.")
        return 1

    best_model, meta = load_model(best_path, device=device, expected_labels=labels)
    logits, targets, mask = collect_logits(best_model, loaders["val"], device)

    calib = calibrate_model(best_model, logits, targets, mask)
    probs = 1 / (1 + np.exp(-logits / calib["temperature"]))
    thresholds = optimal_thresholds(targets, probs, labels, objective="f1")
    rows = per_class_metrics(targets, probs, labels, thresholds=thresholds, mask=mask)
    summary = summarize(rows)

    best_model.save(
        best_path,
        epoch=meta.get("epoch"),
        metrics={**summary, **calib},
        thresholds=thresholds,
        config=cfg.to_dict(),
        model_version=f"{cfg.model.backbone}-{meta.get('epoch', '?')}ep",
    )
    Path(cfg.train.log_dir).mkdir(parents=True, exist_ok=True)
    (Path(cfg.train.log_dir) / "final_summary.json").write_text(
        json.dumps(
            {
                "summary": summary,
                "calibration": calib,
                "thresholds": thresholds,
                "per_class": rows,
                "training": {k: v for k, v in result.items() if k != "history"},
            },
            indent=2,
        )
    )

    print("\n" + "=" * 62)
    print(f"  Best checkpoint : {best_path}")
    print(f"  Mean AUC        : {summary['mean_auc']:.4f}")
    print(f"  Mean AP         : {summary['mean_ap']:.4f}")
    print(f"  Mean F1 @ tuned : {summary['mean_f1']:.4f}")
    print(
        f"  Temperature     : {calib['temperature']:.3f}  (ECE {calib['ece_before']:.4f} -> {calib['ece_after']:.4f})"
    )
    print("=" * 62)
    print("\nNext: python -m radiscan.cli.evaluate --config", args.config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
