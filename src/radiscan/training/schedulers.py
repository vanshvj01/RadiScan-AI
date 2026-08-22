"""Learning-rate schedules."""

from __future__ import annotations

import math

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR, ReduceLROnPlateau


def cosine_with_warmup(
    optimizer: Optimizer,
    warmup_steps: int,
    total_steps: int,
    min_lr_ratio: float = 0.01,
) -> LambdaLR:
    """Linear warmup then cosine decay.

    Warmup matters more than it looks with a pretrained backbone: the randomly
    initialised heads produce large gradients in the first few hundred steps and
    will wreck ImageNet features if the full LR is applied immediately.
    """

    def lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1 + math.cos(math.pi * progress))
        return min_lr_ratio + (1 - min_lr_ratio) * cosine

    return LambdaLR(optimizer, lr_lambda)


def build_scheduler(optimizer: Optimizer, cfg, steps_per_epoch: int):
    kind = cfg.train.scheduler.lower()
    if kind == "cosine":
        total = steps_per_epoch * cfg.train.epochs
        warmup = steps_per_epoch * cfg.train.warmup_epochs
        return cosine_with_warmup(optimizer, warmup, total), "step"
    if kind == "plateau":
        return ReduceLROnPlateau(optimizer, mode="max", factor=0.3, patience=2), "epoch"
    return None, "none"
