"""Training loop.

Accuracy techniques enabled by default, and why each is here:

======================  ==================================================
Technique               Rationale
======================  ==================================================
Asymmetric Loss         Decouples pos/neg focusing under 2% prevalence
320px input             Small pathologies survive downsampling
Balanced sampling       Rare classes actually appear in a batch
Mixup (multi-label)     Regularises a small fine-tuning set
EMA weights             Free ~0.01 AUC, smooths the endgame
Warmup + cosine         Protects pretrained features from head gradients
Layer-wise LR           Heads learn 10x faster than the backbone
AMP (fp16/bf16)         2-3x throughput -> more epochs in the same budget
Grad accumulation       Large effective batch on small VRAM
TTA hflip               ~0.003-0.008 AUC at inference, no retraining
Temperature scaling     Fixes systematic overconfidence post-hoc
Per-class thresholds    Doubles F1 on the tail vs. a single 0.5
======================  ==================================================
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from radiscan.evaluation.metrics import per_class_metrics, summarize
from radiscan.models.losses import MultiTaskLoss
from radiscan.models.network import RadiScanNet, count_parameters
from radiscan.training.ema import ModelEMA
from radiscan.training.schedulers import build_scheduler
from radiscan.utils.device import amp_supported, describe_device, resolve_device

log = logging.getLogger(__name__)


@dataclass
class EarlyStopping:
    patience: int = 6
    mode: str = "max"
    min_delta: float = 1e-4
    best: float = field(default=float("-inf"))
    counter: int = 0

    def __post_init__(self):
        if self.mode == "min":
            self.best = float("inf")

    def step(self, value: float) -> tuple[bool, bool]:
        """Return ``(improved, should_stop)``."""
        if np.isnan(value):
            self.counter += 1
            return False, self.counter >= self.patience
        improved = (
            value > self.best + self.min_delta
            if self.mode == "max"
            else value < self.best - self.min_delta
        )
        if improved:
            self.best = value
            self.counter = 0
        else:
            self.counter += 1
        return improved, self.counter >= self.patience


def mixup_multilabel(
    images: torch.Tensor, targets: torch.Tensor, alpha: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Mixup adapted for multi-label targets.

    For multi-label, mixing targets by *maximum* rather than by linear
    interpolation preserves the semantics: if either source image has effusion,
    the blended image plausibly shows effusion. Linear interpolation would teach
    the model that a clearly-present finding deserves a 0.5 target.
    """
    if alpha <= 0:
        return images, targets
    lam = float(np.random.beta(alpha, alpha))
    idx = torch.randperm(images.size(0), device=images.device)
    mixed = lam * images + (1 - lam) * images[idx]
    mixed_targets = torch.maximum(targets, targets[idx])
    return mixed, mixed_targets


class Trainer:
    def __init__(
        self,
        model: RadiScanNet,
        cfg,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: str | torch.device | None = None,
        pos_weight: torch.Tensor | None = None,
    ):
        self.cfg = cfg
        self.device = (
            resolve_device(device if isinstance(device, str) else None)
            if not isinstance(device, torch.device)
            else device
        )
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.labels = model.labels

        self.criterion = MultiTaskLoss(
            kind=cfg.loss.kind,
            disease_weight=cfg.loss.disease_weight,
            fracture_weight=cfg.loss.fracture_weight,
            pos_weight=None if pos_weight is None else pos_weight.to(self.device),
            asl_gamma_neg=cfg.loss.asl_gamma_neg,
            asl_gamma_pos=cfg.loss.asl_gamma_pos,
            asl_clip=cfg.loss.asl_clip,
            focal_gamma=cfg.loss.focal_gamma,
        ).to(self.device)

        self.optimizer = torch.optim.AdamW(
            model.param_groups(cfg.train.lr, cfg.train.head_lr_mult, cfg.train.weight_decay)
        )
        steps_per_epoch = max(len(train_loader) // max(cfg.train.accumulate_steps, 1), 1)
        self.scheduler, self.sched_interval = build_scheduler(self.optimizer, cfg, steps_per_epoch)

        self.use_amp = cfg.train.amp and amp_supported(self.device)
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)
        self.ema = (
            ModelEMA(model, cfg.train.ema_decay).to(self.device) if cfg.train.ema_decay else None
        )
        self.stopper = EarlyStopping(patience=cfg.train.early_stopping_patience, mode="max")

        self.ckpt_dir = Path(cfg.train.checkpoint_dir)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir = Path(cfg.train.log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.history: list[dict[str, Any]] = []

        total, trainable = count_parameters(model)
        log.info(
            "Trainer ready | device=%s | params=%.2fM (trainable %.2fM) | amp=%s | ema=%s",
            describe_device(self.device),
            total / 1e6,
            trainable / 1e6,
            self.use_amp,
            bool(self.ema),
        )

    # -- one epoch -----------------------------------------------------------
    def train_epoch(self, epoch: int) -> dict[str, float]:
        self.model.train()
        if epoch < self.cfg.train.freeze_backbone_epochs:
            self.model.set_backbone_trainable(False)
        elif epoch == self.cfg.train.freeze_backbone_epochs:
            self.model.set_backbone_trainable(True)

        accum = max(self.cfg.train.accumulate_steps, 1)
        running: dict[str, float] = {}
        n_batches = 0
        pbar = tqdm(
            self.train_loader, desc=f"epoch {epoch + 1}/{self.cfg.train.epochs}", leave=False
        )

        self.optimizer.zero_grad(set_to_none=True)
        for step, batch in enumerate(pbar):
            images = batch["image"].to(self.device, non_blocking=True)
            targets = batch["labels"].to(self.device, non_blocking=True)
            batch_gpu = {
                "labels": targets,
                "label_mask": batch["label_mask"].to(self.device, non_blocking=True),
                "fracture": batch["fracture"].to(self.device, non_blocking=True),
            }

            if self.cfg.train.mixup_alpha > 0:
                images, batch_gpu["labels"] = mixup_multilabel(
                    images, targets, self.cfg.train.mixup_alpha
                )

            with torch.autocast(device_type=self.device.type, enabled=self.use_amp):
                outputs = self.model(images)
                loss, parts = self.criterion(outputs, batch_gpu)
                loss = loss / accum

            self.scaler.scale(loss).backward()

            if (step + 1) % accum == 0:
                if self.cfg.train.grad_clip:
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.train.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
                if self.ema:
                    self.ema.update(self.model)
                if self.scheduler is not None and self.sched_interval == "step":
                    self.scheduler.step()

            n_batches += 1
            for k, v in parts.items():
                running[k] = running.get(k, 0.0) + v
            pbar.set_postfix(loss=f"{running['total'] / n_batches:.4f}")

        return {f"train_{k}": v / max(n_batches, 1) for k, v in running.items()}

    # -- validation ----------------------------------------------------------
    @torch.no_grad()
    def validate(self, use_ema: bool = True) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
        model = self.ema.module if (use_ema and self.ema) else self.model
        model.eval()

        all_probs, all_targets, all_masks = [], [], []
        total_loss, n = 0.0, 0

        for batch in tqdm(self.val_loader, desc="validate", leave=False):
            images = batch["image"].to(self.device, non_blocking=True)
            batch_gpu = {
                "labels": batch["labels"].to(self.device, non_blocking=True),
                "label_mask": batch["label_mask"].to(self.device, non_blocking=True),
                "fracture": batch["fracture"].to(self.device, non_blocking=True),
            }
            with torch.autocast(device_type=self.device.type, enabled=self.use_amp):
                outputs = model(images)
                loss, _ = self.criterion(outputs, batch_gpu)
            total_loss += float(loss.detach()) * images.size(0)
            n += images.size(0)
            all_probs.append(outputs["disease_probs"].float().cpu().numpy())
            all_targets.append(batch["labels"].numpy())
            all_masks.append(batch["label_mask"].numpy())

        probs = np.concatenate(all_probs)
        targets = np.concatenate(all_targets)
        masks = np.concatenate(all_masks)

        rows = per_class_metrics(targets, probs, self.labels, mask=masks)
        summary = summarize(rows)
        metrics = {
            "val_loss": total_loss / max(n, 1),
            "val_mean_auc": summary["mean_auc"],
            "val_mean_ap": summary["mean_ap"],
        }
        return metrics, targets, probs

    # -- full run ------------------------------------------------------------
    def fit(self) -> dict[str, Any]:
        best_path = self.ckpt_dir / "best_model.pth"
        last_path = self.ckpt_dir / "last.pth"
        best_metric = float("-inf")
        start = time.time()

        for epoch in range(self.cfg.train.epochs):
            t0 = time.time()
            train_metrics = self.train_epoch(epoch)
            val_metrics, targets, probs = self.validate()

            if self.scheduler is not None and self.sched_interval == "epoch":
                self.scheduler.step(val_metrics["val_mean_auc"])

            record = {
                "epoch": epoch + 1,
                "lr": self.optimizer.param_groups[0]["lr"],
                "seconds": round(time.time() - t0, 1),
                **train_metrics,
                **val_metrics,
            }
            self.history.append(record)
            log.info(
                "epoch %d | train %.4f | val %.4f | mAUC %.4f | mAP %.4f | %.0fs",
                epoch + 1,
                train_metrics.get("train_total", float("nan")),
                val_metrics["val_loss"],
                val_metrics["val_mean_auc"],
                val_metrics["val_mean_ap"],
                record["seconds"],
            )

            monitored = val_metrics[self.cfg.train.monitor]
            improved, should_stop = self.stopper.step(monitored)

            model_to_save = self.ema.module if self.ema else self.model
            model_to_save.save(
                last_path,
                epoch=epoch + 1,
                metrics=val_metrics,
                config=self.cfg.to_dict(),
            )
            if improved:
                model_to_save.save(
                    best_path,
                    epoch=epoch + 1,
                    metrics=val_metrics,
                    config=self.cfg.to_dict(),
                )
                log.info("  new best %s=%.4f -> %s", self.cfg.train.monitor, monitored, best_path)
                best_metric = monitored

            (self.log_dir / "history.json").write_text(json.dumps(self.history, indent=2))

            if should_stop:
                log.info("Early stopping at epoch %d (best %.4f).", epoch + 1, best_metric)
                break

        return {
            "best_metric": best_metric,
            "best_checkpoint": str(best_path),
            "history": self.history,
            "elapsed_minutes": round((time.time() - start) / 60, 2),
        }
