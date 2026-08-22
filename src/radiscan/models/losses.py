"""Losses tuned for extreme multi-label class imbalance.

Why Asymmetric Loss (ASL) is the default rather than Focal Loss:

Focal Loss applies the same ``(1-p)^gamma`` down-weighting to positives and
negatives. In chest X-ray datasets some pathologies are under 2% positive, so
the *negatives* are the overwhelming majority and the *positives* are the rare
signal you cannot afford to down-weight at all. ASL (Ridnik et al., ICCV 2021)
decouples the two focusing parameters -- ``gamma_neg=4, gamma_pos=0`` -- and adds
a probability-shifting clip that hard-discards very-easy negatives entirely. On
multi-label benchmarks it consistently beats focal by 1-3 mAP points, and it is
the single cheapest accuracy win available in this pipeline.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class AsymmetricLoss(nn.Module):
    """Asymmetric Loss for multi-label classification (Ridnik et al., 2021)."""

    def __init__(
        self,
        gamma_neg: float = 4.0,
        gamma_pos: float = 0.0,
        clip: float = 0.05,
        eps: float = 1e-8,
        disable_grad_focal: bool = True,
    ):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps
        self.disable_grad_focal = disable_grad_focal

    def forward(
        self, logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        p = torch.sigmoid(logits)
        p_pos = p
        p_neg = 1.0 - p

        # Probability shifting: treat negatives below `clip` as already solved.
        if self.clip > 0:
            p_neg = (p_neg + self.clip).clamp(max=1.0)

        loss_pos = targets * torch.log(p_pos.clamp(min=self.eps))
        loss_neg = (1 - targets) * torch.log(p_neg.clamp(min=self.eps))
        loss = loss_pos + loss_neg

        if self.gamma_neg > 0 or self.gamma_pos > 0:
            # The focusing weight is treated as a constant w.r.t. the gradient
            # (as in the reference implementation). Restore the *previous* grad
            # mode rather than unconditionally re-enabling it -- forcing it on
            # would silently break every caller inside `torch.no_grad()`.
            prev = torch.is_grad_enabled()
            torch.set_grad_enabled(prev and not self.disable_grad_focal)
            pt = p_pos * targets + p_neg * (1 - targets)
            gamma = self.gamma_pos * targets + self.gamma_neg * (1 - targets)
            weight = torch.pow(1 - pt, gamma)
            torch.set_grad_enabled(prev)
            loss = loss * weight

        loss = -loss
        if mask is not None:
            loss = loss * mask
            denom = mask.sum().clamp(min=1.0)
            return loss.sum() / denom
        return loss.mean()


class FocalLoss(nn.Module):
    """Standard multi-label focal loss, kept for the ablation table."""

    def __init__(self, gamma: float = 2.0, alpha: float = 0.25, eps: float = 1e-8):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.eps = eps

    def forward(
        self, logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        p = torch.sigmoid(logits).clamp(self.eps, 1 - self.eps)
        ce = -(targets * torch.log(p) + (1 - targets) * torch.log(1 - p))
        pt = p * targets + (1 - p) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        loss = alpha_t * torch.pow(1 - pt, self.gamma) * ce
        if mask is not None:
            loss = loss * mask
            return loss.sum() / mask.sum().clamp(min=1.0)
        return loss.mean()


class MaskedBCE(nn.Module):
    """BCE-with-logits that honours an ignore mask and optional pos_weight."""

    def __init__(self, pos_weight: torch.Tensor | None = None):
        super().__init__()
        self.register_buffer("pos_weight", pos_weight)

    def forward(
        self, logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        loss = F.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=self.pos_weight, reduction="none"
        )
        if mask is not None:
            loss = loss * mask
            return loss.sum() / mask.sum().clamp(min=1.0)
        return loss.mean()


class ConfidencePenalty(nn.Module):
    """Devries & Taylor confidence objective.

    The model may 'ask for a hint': predictions are interpolated toward the
    ground truth in proportion to the predicted confidence ``c``, and a budget
    term penalises asking. A model that is genuinely sure keeps ``c`` high and
    pays nothing; a model that is guessing lowers ``c`` and gets an easier
    classification loss at a cost. This is what makes the confidence head
    trained rather than decorative.
    """

    def __init__(self, budget: float = 0.3, eps: float = 1e-8):
        super().__init__()
        self.budget = budget
        self.eps = eps

    def forward(
        self, probs: torch.Tensor, confidence: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        c = confidence.clamp(self.eps, 1 - self.eps)
        hinted = c * probs + (1 - c) * targets
        hinted = hinted.clamp(self.eps, 1 - self.eps)
        task = -(targets * torch.log(hinted) + (1 - targets) * torch.log(1 - hinted)).mean()
        penalty = -torch.log(c).mean()
        return task + self.budget * penalty


class MultiTaskLoss(nn.Module):
    """Combines disease, fracture and confidence objectives."""

    def __init__(
        self,
        kind: str = "asl",
        disease_weight: float = 1.0,
        fracture_weight: float = 0.3,
        confidence_weight: float = 0.1,
        pos_weight: torch.Tensor | None = None,
        asl_gamma_neg: float = 4.0,
        asl_gamma_pos: float = 0.0,
        asl_clip: float = 0.05,
        focal_gamma: float = 2.0,
    ):
        super().__init__()
        kind = kind.lower()
        if kind == "asl":
            self.disease_loss: nn.Module = AsymmetricLoss(asl_gamma_neg, asl_gamma_pos, asl_clip)
        elif kind == "focal":
            self.disease_loss = FocalLoss(gamma=focal_gamma)
        elif kind == "bce":
            self.disease_loss = MaskedBCE(pos_weight=pos_weight)
        else:
            raise ValueError(f"Unknown loss kind {kind!r}; expected asl|focal|bce")

        self.fracture_loss = MaskedBCE()
        self.confidence_loss = ConfidencePenalty()
        self.disease_weight = disease_weight
        self.fracture_weight = fracture_weight
        self.confidence_weight = confidence_weight

    def forward(
        self, outputs: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, dict[str, float]]:
        targets = batch["labels"]
        mask = batch.get("label_mask")

        disease = self.disease_loss(outputs["disease_logits"], targets, mask)
        total = self.disease_weight * disease
        parts = {"disease": float(disease.detach())}

        if "fracture_logit" in outputs and "fracture" in batch:
            frac = self.fracture_loss(
                outputs["fracture_logit"].squeeze(-1), batch["fracture"].float()
            )
            total = total + self.fracture_weight * frac
            parts["fracture"] = float(frac.detach())

        if "confidence" in outputs and self.confidence_weight > 0:
            probs = torch.sigmoid(outputs["disease_logits"].detach())
            conf = self.confidence_loss(probs, outputs["confidence"], targets)
            total = total + self.confidence_weight * conf
            parts["confidence"] = float(conf.detach())

        parts["total"] = float(total.detach())
        return total, parts
