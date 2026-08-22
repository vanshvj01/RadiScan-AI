"""Exponential moving average of weights.

An EMA shadow copy of the parameters is one of the cheapest accuracy gains
available: it costs one extra copy of the model in memory, no extra compute in
the backward pass, and typically buys 0.005-0.015 mean AUC because it smooths
out the noise of the final few high-LR steps. Evaluate and checkpoint the EMA
weights, not the raw ones.
"""

from __future__ import annotations

from copy import deepcopy

import torch
import torch.nn as nn


class ModelEMA:
    def __init__(self, model: nn.Module, decay: float = 0.999, warmup_steps: int = 1000):
        self.module = deepcopy(model).eval()
        for p in self.module.parameters():
            p.requires_grad_(False)
        self.decay = decay
        self.warmup_steps = warmup_steps
        self.steps = 0

    def _current_decay(self) -> float:
        # Ramp the decay in, so early (fast-moving) weights are not over-smoothed.
        if self.warmup_steps <= 0:
            return self.decay
        return min(self.decay, (1 + self.steps) / (10 + self.steps))

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        self.steps += 1
        d = self._current_decay()
        msd = model.state_dict()
        for key, value in self.module.state_dict().items():
            if not value.dtype.is_floating_point:
                value.copy_(msd[key])
                continue
            value.mul_(d).add_(msd[key].detach(), alpha=1.0 - d)

    def state_dict(self):
        return self.module.state_dict()

    def to(self, device):
        self.module.to(device)
        return self
