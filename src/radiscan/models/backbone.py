"""Feature extractors.

Backbones are created through ``timm`` so the whole modern zoo is reachable by
name (``densenet121``, ``convnext_tiny``, ``tf_efficientnetv2_s``,
``resnet50``...). Each backbone exposes:

* ``out_features`` -- pooled embedding width, needed to size the heads
* ``forward``      -- pooled features, shape ``(B, out_features)``
* ``forward_features`` -- the final spatial map, shape ``(B, C, H, W)``,
  which is what Grad-CAM hooks onto

Keeping the CAM target as an explicit attribute (``cam_layer``) rather than a
hard-coded string like ``features.denseblock4`` is what lets the backbone be
swapped without silently breaking explainability.
"""

from __future__ import annotations

import torch
import torch.nn as nn

try:  # timm is the preferred path
    import timm

    _HAS_TIMM = True
except ImportError:  # pragma: no cover - exercised only in minimal installs
    _HAS_TIMM = False

import torchvision.models as tvm


class Backbone(nn.Module):
    """Wraps a timm (or torchvision fallback) feature extractor."""

    def __init__(self, name: str = "densenet121", pretrained: bool = True, in_chans: int = 3):
        super().__init__()
        self.name = name

        if _HAS_TIMM:
            self.body = timm.create_model(
                name,
                pretrained=pretrained,
                num_classes=0,  # strip the classifier
                global_pool="",  # keep the spatial map
                in_chans=in_chans,
            )
            self.out_features = int(self.body.num_features)
        else:
            self.body, self.out_features = _torchvision_fallback(name, pretrained)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self._feature_map: torch.Tensor | None = None

    # -- Grad-CAM plumbing ---------------------------------------------------
    @property
    def cam_layer(self) -> nn.Module:
        """The module whose output Grad-CAM differentiates against.

        We deliberately return the *last convolutional stage*, discovered by
        walking the module tree backwards, rather than naming a layer that only
        exists on DenseNet.
        """
        last_conv: nn.Module | None = None
        for module in self.body.modules():
            if isinstance(module, nn.Conv2d):
                last_conv = module
        if last_conv is None:
            raise RuntimeError(f"No Conv2d found in backbone {self.name!r}; Grad-CAM unavailable.")
        return last_conv

    # -- forward -------------------------------------------------------------
    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.body(x)
        if feats.ndim == 3:  # ViT-style (B, N, C) -> fake a square map
            b, n, c = feats.shape
            side = int(n**0.5)
            if side * side == n:
                feats = feats.transpose(1, 2).reshape(b, c, side, side)
        self._feature_map = feats
        return feats

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.forward_features(x)
        if feats.ndim == 4:
            return self.pool(feats).flatten(1)
        return feats


def _torchvision_fallback(name: str, pretrained: bool) -> tuple[nn.Module, int]:
    weights = "DEFAULT" if pretrained else None
    if name == "densenet121":
        m = tvm.densenet121(weights=weights)
        return m.features, 1024
    if name == "resnet50":
        m = tvm.resnet50(weights=weights)
        return nn.Sequential(*list(m.children())[:-2]), 2048
    if name in ("efficientnet_b4", "tf_efficientnetv2_s"):
        m = tvm.efficientnet_b4(weights=weights)
        return m.features, 1792
    raise ValueError(
        f"Backbone {name!r} requires timm. Install it with `pip install timm`, "
        "or choose one of: densenet121, resnet50, efficientnet_b4."
    )
