"""RadiScan network: CNN (+ optional ViT) trunk -> multi-task heads.

Checkpoint contract
-------------------
Every checkpoint this module writes carries its own metadata:

``{"state_dict", "labels", "backbone", "image_size", "use_transformer",
  "temperature", "thresholds", "metrics", "radiscan_version"}``

:func:`load_model` reconstructs the architecture *from the checkpoint* rather
than from whatever config the caller happens to be holding, and raises if the
embedded labels disagree with the labels requested. That makes the label
mismatch that silently corrupted the v1 pipeline structurally impossible.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from radiscan.constants import get_labels
from radiscan.models.backbone import Backbone
from radiscan.models.heads import MultiTaskHead

log = logging.getLogger(__name__)

RADISCAN_VERSION = "2.0.0"


class RadiScanNet(nn.Module):
    def __init__(
        self,
        labels: tuple[str, ...] | list[str],
        backbone: str = "densenet121",
        pretrained: bool = True,
        use_transformer: bool = False,
        vit_name: str = "vit_small_patch16_224",
        dropout: float = 0.2,
        fracture_head: bool = True,
        confidence_head: bool = True,
        image_size: int = 320,
    ):
        super().__init__()
        self.labels = tuple(labels)
        self.image_size = image_size
        self.backbone_name = backbone
        self.use_transformer = use_transformer

        self.cnn = Backbone(backbone, pretrained=pretrained)
        width = self.cnn.out_features

        self.vit: Backbone | None = None
        if use_transformer:
            self.vit = Backbone(vit_name, pretrained=pretrained)
            width += self.vit.out_features

        self.fuse = nn.Sequential(nn.LayerNorm(width), nn.Dropout(dropout))
        self.head = MultiTaskHead(
            width,
            num_diseases=len(self.labels),
            dropout=dropout,
            fracture=fracture_head,
            confidence=confidence_head,
        )

        # Post-hoc calibration, filled in after training. Identity until then.
        self.register_buffer("temperature", torch.ones(1))

    # -- forward -------------------------------------------------------------
    def trunk(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.cnn(x)
        if self.vit is not None:
            feats = torch.cat([feats, self.vit(x)], dim=1)
        return self.fuse(feats)

    def forward(self, x: torch.Tensor, calibrated: bool = False) -> dict[str, torch.Tensor]:
        out = self.head(self.trunk(x))
        logits = out["disease_logits"]
        if calibrated:
            logits = logits / self.temperature.clamp(min=1e-3)
            out["disease_logits"] = logits
        out["disease_probs"] = torch.sigmoid(logits)
        if "fracture_logit" in out:
            out["fracture_prob"] = torch.sigmoid(out["fracture_logit"])
        return out

    # -- parameter groups for layer-wise LR ----------------------------------
    def param_groups(self, base_lr: float, head_mult: float = 10.0, weight_decay: float = 1e-4):
        backbone_params, head_params = [], []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            (head_params if name.startswith(("head.", "fuse.")) else backbone_params).append(param)
        return [
            {"params": backbone_params, "lr": base_lr, "weight_decay": weight_decay},
            {"params": head_params, "lr": base_lr * head_mult, "weight_decay": weight_decay},
        ]

    def set_backbone_trainable(self, flag: bool) -> None:
        for p in self.cnn.parameters():
            p.requires_grad = flag
        if self.vit is not None:
            for p in self.vit.parameters():
                p.requires_grad = flag

    # -- checkpointing -------------------------------------------------------
    def checkpoint_payload(self, **extra: Any) -> dict[str, Any]:
        return {
            "radiscan_version": RADISCAN_VERSION,
            "state_dict": self.state_dict(),
            "labels": list(self.labels),
            "backbone": self.backbone_name,
            "use_transformer": self.use_transformer,
            "image_size": self.image_size,
            "temperature": float(self.temperature.item()),
            **extra,
        }

    def save(self, path: str | Path, **extra: Any) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.checkpoint_payload(**extra), path)


class MissingCheckpointError(FileNotFoundError):
    """Raised with an actionable message instead of a bare FileNotFoundError."""


class LabelMismatchError(ValueError):
    pass


def build_model(cfg, labels: tuple[str, ...] | None = None) -> RadiScanNet:
    """Construct a fresh model from a :class:`radiscan.config.Config`."""
    labels = labels or get_labels(cfg.data.dataset)
    return RadiScanNet(
        labels=labels,
        backbone=cfg.model.backbone,
        pretrained=cfg.model.pretrained,
        use_transformer=cfg.model.use_transformer,
        vit_name=cfg.model.vit_name,
        dropout=cfg.model.dropout,
        fracture_head=cfg.model.fracture_head,
        confidence_head=cfg.model.confidence_head,
        image_size=cfg.data.image_size,
    )


def load_model(
    path: str | Path,
    device: torch.device | str = "cpu",
    expected_labels: tuple[str, ...] | None = None,
    strict: bool = True,
) -> tuple[RadiScanNet, dict[str, Any]]:
    """Load a checkpoint, rebuilding the architecture from its own metadata."""
    path = Path(path)
    if not path.exists():
        raise MissingCheckpointError(
            f"No checkpoint at {path}.\n"
            "Train one first:\n"
            "  python -m radiscan.cli.train --config configs/chexpert_base.yaml\n"
            "or run the API with RADISCAN_ALLOW_UNTRAINED=1 for a UI-only demo "
            "(outputs will be meaningless and are labelled 'untrained-demo')."
        )

    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if "state_dict" not in ckpt or "labels" not in ckpt:
        raise ValueError(
            f"{path} is not a RadiScan v2 checkpoint (missing 'state_dict'/'labels'). "
            "v1 checkpoints carry no label metadata and cannot be served safely."
        )

    labels = tuple(ckpt["labels"])
    if expected_labels is not None and tuple(expected_labels) != labels:
        raise LabelMismatchError(
            "Checkpoint label set does not match the requested one -- refusing to "
            "serve, because every disease name in the output would be wrong.\n"
            f"  checkpoint: {list(labels)}\n"
            f"  requested : {list(expected_labels)}"
        )

    model = RadiScanNet(
        labels=labels,
        backbone=ckpt.get("backbone", "densenet121"),
        pretrained=False,
        use_transformer=ckpt.get("use_transformer", False),
        image_size=ckpt.get("image_size", 320),
    )
    missing, unexpected = model.load_state_dict(ckpt["state_dict"], strict=strict)
    if missing or unexpected:
        log.warning("State dict mismatch. missing=%s unexpected=%s", missing, unexpected)

    if "temperature" in ckpt:
        model.temperature.fill_(float(ckpt["temperature"]))

    model.to(device).eval()
    meta = {k: v for k, v in ckpt.items() if k != "state_dict"}
    return model, meta


def count_parameters(model: nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable
