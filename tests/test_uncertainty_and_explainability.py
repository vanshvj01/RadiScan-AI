from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn

from radiscan.modules.explainability import GradCAM, overlay_heatmap
from radiscan.modules.uncertainty import (
    enable_mc_dropout,
    predict_with_uncertainty,
    reliability_score,
)


# --- MC Dropout ------------------------------------------------------------
def test_mc_dropout_enables_only_dropout(model):
    model.eval()
    n = enable_mc_dropout(model)
    assert n > 0
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            assert not module.training, "BatchNorm must stay in eval mode"
    model.eval()


def test_uncertainty_is_nonzero_with_dropout(model, batch):
    out = predict_with_uncertainty(model, batch, n_samples=10, tta_hflip=False)
    assert out["std"].shape == (2, len(model.labels))
    assert out["std"].mean() > 0, "MC Dropout produced zero variance"


def test_confidence_interval_brackets_the_mean(model, batch):
    out = predict_with_uncertainty(model, batch, n_samples=15, tta_hflip=False)
    assert np.all(out["ci_low"] <= out["mean"] + 1e-6)
    assert np.all(out["mean"] <= out["ci_high"] + 1e-6)


def test_entropy_is_within_binary_bounds(model, batch):
    out = predict_with_uncertainty(model, batch, n_samples=5, tta_hflip=False)
    assert np.all(out["entropy"] >= -1e-6)
    assert np.all(out["entropy"] <= np.log(2) + 1e-6)


def test_mutual_information_never_negative(model, batch):
    out = predict_with_uncertainty(model, batch, n_samples=10, tta_hflip=False)
    assert np.all(out["mutual_information"] >= 0)


def test_tta_runs_and_changes_nothing_structural(model, batch):
    out = predict_with_uncertainty(model, batch, n_samples=4, tta_hflip=True)
    assert out["mean"].shape == (2, len(model.labels))


def test_model_returns_to_eval_after_sampling(model, batch):
    model.eval()
    predict_with_uncertainty(model, batch, n_samples=3)
    assert not model.training


def test_reliability_score_is_bounded():
    std = np.array([0.0, 0.1, 0.5])
    ent = np.array([0.0, 0.3, 0.69])
    r = reliability_score(std, ent)
    assert np.all(r >= 0) and np.all(r <= 1)
    assert r[0] > r[-1], "Lower uncertainty must mean higher reliability"


# --- Grad-CAM --------------------------------------------------------------
def test_gradcam_output_shape_and_range(model):
    x = torch.randn(1, 3, 224, 224)
    with GradCAM(model) as cam:
        maps = cam.generate(x, [0, 3])
    assert set(maps) == {0, 3}
    for hm in maps.values():
        assert hm.shape == (224, 224)
        assert hm.min() >= 0.0 and hm.max() <= 1.0 + 1e-6


def test_gradcam_plusplus_runs(model):
    x = torch.randn(1, 3, 224, 224)
    with GradCAM(model, plus_plus=True) as cam:
        maps = cam.generate(x, [1])
    assert maps[1].shape == (224, 224)


def test_gradcam_rejects_batches(model):
    with GradCAM(model) as cam, pytest.raises(ValueError):
        cam.generate(torch.randn(4, 3, 224, 224), [0])


def test_gradcam_removes_hooks(model):
    layer = GradCAM._default_layer(model)
    before = len(layer._forward_hooks)
    with GradCAM(model) as cam:
        cam.generate(torch.randn(1, 3, 224, 224), [0])
    assert len(layer._forward_hooks) == before, "Hooks leaked; repeated calls would compound"


def test_gradcam_leaves_model_in_eval(model):
    model.eval()
    with GradCAM(model) as cam:
        cam.generate(torch.randn(1, 3, 224, 224), [0])
    assert not model.training


def test_overlay_shape():
    img = np.random.randint(0, 255, (100, 120, 3), dtype=np.uint8)
    hm = np.random.rand(50, 60)
    out = overlay_heatmap(img, hm)
    assert out.shape == img.shape and out.dtype == np.uint8
