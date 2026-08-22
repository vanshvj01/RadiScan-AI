"""Post-hoc calibration.

Deep networks are systematically overconfident (Guo et al., ICML 2017). For a
system whose entire premise is "trust the number and the uncertainty around it",
shipping an uncalibrated model is a contradiction. Temperature scaling fixes it
with a single scalar, fitted on the validation set, that never changes the
ranking -- so AUC is untouched while ECE typically drops by 3-5x.
"""

from __future__ import annotations

import logging

import numpy as np
import torch
import torch.nn.functional as F

from radiscan.evaluation.metrics import expected_calibration_error

log = logging.getLogger(__name__)


def fit_temperature(
    logits: np.ndarray,
    targets: np.ndarray,
    mask: np.ndarray | None = None,
    max_iter: int = 200,
    lr: float = 0.01,
) -> float:
    """Fit a single temperature T minimising BCE on held-out validation logits."""
    z = torch.tensor(logits, dtype=torch.float32)
    y = torch.tensor((targets >= 0.5).astype(np.float32))
    m = torch.tensor(mask.astype(np.float32)) if mask is not None else torch.ones_like(y)

    log_t = torch.zeros(1, requires_grad=True)  # optimise log T for positivity
    optimizer = torch.optim.LBFGS([log_t], lr=lr, max_iter=max_iter)

    def closure():
        optimizer.zero_grad()
        loss = F.binary_cross_entropy_with_logits(z / log_t.exp(), y, reduction="none")
        loss = (loss * m).sum() / m.sum().clamp(min=1.0)
        loss.backward()
        return loss

    optimizer.step(closure)
    temperature = float(log_t.exp().item())
    return max(min(temperature, 10.0), 0.1)


def calibrate_model(
    model,
    logits: np.ndarray,
    targets: np.ndarray,
    mask: np.ndarray | None = None,
) -> dict:
    """Fit T, write it into the model buffer, and report the ECE improvement."""
    before_ece, _ = expected_calibration_error(targets, 1 / (1 + np.exp(-logits)))
    temperature = fit_temperature(logits, targets, mask)
    after_ece, _ = expected_calibration_error(targets, 1 / (1 + np.exp(-logits / temperature)))

    model.temperature.fill_(temperature)
    log.info(
        "Temperature scaling: T=%.3f | ECE %.4f -> %.4f (%.1fx better)",
        temperature,
        before_ece,
        after_ece,
        before_ece / max(after_ece, 1e-9),
    )
    return {
        "temperature": temperature,
        "ece_before": before_ece,
        "ece_after": after_ece,
    }


@torch.no_grad()
def collect_logits(model, loader, device) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run the validation set once and return raw logits, targets, mask."""
    model.eval().to(device)
    logits, targets, masks = [], [], []
    for batch in loader:
        out = model(batch["image"].to(device))
        logits.append(out["disease_logits"].float().cpu().numpy())
        targets.append(batch["labels"].numpy())
        masks.append(batch["label_mask"].numpy())
    return np.concatenate(logits), np.concatenate(targets), np.concatenate(masks)
