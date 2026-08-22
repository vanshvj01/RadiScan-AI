"""Task-specific prediction heads."""

from __future__ import annotations

import torch
import torch.nn as nn


class ClassifierHead(nn.Module):
    """Two-layer MLP head with dropout kept live for MC-Dropout inference."""

    def __init__(self, in_features: int, num_outputs: int, hidden: int = 512, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_outputs),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ConfidenceHead(nn.Module):
    """Learned per-class *aleatoric* confidence.

    This is distinct from the MC-Dropout uncertainty, which is epistemic. The
    head predicts, for each class, how noisy the model believes this particular
    image is -- a value in (0, 1) that is trained with the confidence-penalty
    objective from Devries & Taylor (2018).
    """

    def __init__(self, in_features: int, num_outputs: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, 256),
            nn.GELU(),
            nn.Linear(256, num_outputs),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MultiTaskHead(nn.Module):
    """Disease (multi-label) + fracture (binary) + confidence."""

    def __init__(
        self,
        in_features: int,
        num_diseases: int,
        dropout: float = 0.2,
        fracture: bool = True,
        confidence: bool = True,
    ):
        super().__init__()
        self.disease = ClassifierHead(in_features, num_diseases, dropout=dropout)
        self.fracture = (
            ClassifierHead(in_features, 1, hidden=256, dropout=dropout) if fracture else None
        )
        self.confidence = ConfidenceHead(in_features, num_diseases) if confidence else None

    def forward(self, feats: torch.Tensor) -> dict[str, torch.Tensor]:
        out: dict[str, torch.Tensor] = {"disease_logits": self.disease(feats)}
        if self.fracture is not None:
            out["fracture_logit"] = self.fracture(feats)
        if self.confidence is not None:
            out["confidence"] = self.confidence(feats)
        return out
