"""Grad-CAM and Grad-CAM++ saliency.

Grad-CAM (Selvaraju et al., ICCV 2017) weights the final convolutional feature
maps by the gradient of a class score, giving a coarse map of "where the network
looked". Grad-CAM++ (Chattopadhay et al., WACV 2018) replaces the global-average
gradient weighting with a per-pixel second-order weighting, which localises
multiple instances of a finding better -- relevant here because nodules and
infiltrates are frequently multifocal.

Honest caveat to carry into the report: these maps show *correlation with the
score*, not causation. The canonical failure mode is a model latching onto chest
drains, pacemaker leads, or the laterality marker rather than the pathology.
Quantifying localisation requires bounding boxes and a pointing-game metric.
"""

from __future__ import annotations

from collections.abc import Iterable

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class GradCAM:
    """Grad-CAM with forward/backward hooks, usable as a context manager."""

    def __init__(
        self, model: nn.Module, target_layer: nn.Module | None = None, plus_plus: bool = False
    ):
        self.model = model
        self.plus_plus = plus_plus
        self.target_layer = target_layer or self._default_layer(model)
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None
        self._handles: list = []

    @staticmethod
    def _default_layer(model: nn.Module) -> nn.Module:
        cnn = getattr(model, "cnn", None)
        if cnn is not None and hasattr(cnn, "cam_layer"):
            return cnn.cam_layer
        last = None
        for module in model.modules():
            if isinstance(module, nn.Conv2d):
                last = module
        if last is None:
            raise RuntimeError("No convolutional layer found; Grad-CAM unavailable.")
        return last

    def __enter__(self) -> GradCAM:
        self._handles.append(self.target_layer.register_forward_hook(self._save_activation))
        self._handles.append(self.target_layer.register_full_backward_hook(self._save_gradient))
        return self

    def __exit__(self, *exc) -> None:
        self.remove()

    def remove(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def _save_activation(self, _module, _inp, output) -> None:
        self.activations = output.detach()

    def _save_gradient(self, _module, _grad_in, grad_out) -> None:
        self.gradients = grad_out[0].detach()

    # -- core ----------------------------------------------------------------
    def _weights(self) -> torch.Tensor:
        grads, acts = self.gradients, self.activations
        assert grads is not None and acts is not None

        if not self.plus_plus:
            return grads.mean(dim=(2, 3), keepdim=True)

        g2, g3 = grads.pow(2), grads.pow(3)
        denom = 2 * g2 + (acts * g3).sum(dim=(2, 3), keepdim=True)
        alpha = g2 / torch.where(denom != 0, denom, torch.ones_like(denom))
        return (alpha * F.relu(grads)).sum(dim=(2, 3), keepdim=True)

    def generate(
        self,
        image: torch.Tensor,
        class_indices: Iterable[int],
        output_size: tuple[int, int] | None = None,
    ) -> dict[int, np.ndarray]:
        """Return ``{class_index: heatmap}``, each normalised to ``[0, 1]``.

        ``image`` must be a single image of shape ``(1, 3, H, W)``.
        """
        if image.dim() == 3:
            image = image.unsqueeze(0)
        if image.size(0) != 1:
            raise ValueError("GradCAM.generate expects a single image (batch size 1).")

        size = output_size or (image.shape[2], image.shape[3])
        was_training = self.model.training
        self.model.eval()
        heatmaps: dict[int, np.ndarray] = {}

        for idx in class_indices:
            self.model.zero_grad(set_to_none=True)
            with torch.enable_grad():
                inp = image.clone().requires_grad_(True)
                out = self.model(inp)
                score = out["disease_logits"][0, idx]
                score.backward(retain_graph=False)

            if self.activations is None or self.gradients is None:
                raise RuntimeError(
                    "Hooks captured nothing -- is the target layer on the forward path?"
                )

            cam = (self._weights() * self.activations).sum(dim=1, keepdim=True)
            cam = F.relu(cam)
            cam = F.interpolate(cam, size=size, mode="bilinear", align_corners=False)
            cam = cam[0, 0].cpu().numpy()

            lo, hi = float(cam.min()), float(cam.max())
            heatmaps[int(idx)] = (cam - lo) / (hi - lo + 1e-8)

        if was_training:
            self.model.train()
        return heatmaps


def overlay_heatmap(
    image: np.ndarray, heatmap: np.ndarray, alpha: float = 0.4, colormap: int = cv2.COLORMAP_JET
) -> np.ndarray:
    """Blend a ``[0,1]`` heatmap over a uint8 RGB image."""
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    h, w = image.shape[:2]
    hm = cv2.resize(heatmap.astype(np.float32), (w, h))
    hm_u8 = np.uint8(255 * np.clip(hm, 0, 1))
    colored = cv2.applyColorMap(hm_u8, colormap)
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    return np.uint8(np.clip((1 - alpha) * image + alpha * colored, 0, 255))


def save_overlay(image: np.ndarray, heatmap: np.ndarray, path, alpha: float = 0.4) -> str:
    from pathlib import Path

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    blended = overlay_heatmap(image, heatmap, alpha)
    cv2.imwrite(str(out), cv2.cvtColor(blended, cv2.COLOR_RGB2BGR))
    return str(out)
