from __future__ import annotations

import pytest
import torch

from radiscan.constants import CHESTXRAY14_LABELS
from radiscan.models.network import (
    LabelMismatchError,
    MissingCheckpointError,
    RadiScanNet,
    count_parameters,
    load_model,
)


def test_forward_shapes(model, batch, labels):
    out = model(batch)
    assert out["disease_logits"].shape == (2, len(labels))
    assert out["disease_probs"].shape == (2, len(labels))
    assert out["fracture_prob"].shape == (2, 1)
    assert out["confidence"].shape == (2, len(labels))


def test_probabilities_are_valid(model, batch):
    probs = model(batch)["disease_probs"]
    assert torch.all(probs >= 0) and torch.all(probs <= 1)


def test_multilabel_not_softmax(model, batch):
    """Independent sigmoids must NOT sum to one -- a patient can have several
    findings at once. A softmax head here would be a modelling error."""
    probs = model(batch)["disease_probs"]
    assert not torch.allclose(probs.sum(dim=1), torch.ones(2), atol=0.05)


def test_parameter_count_is_reasonable(model):
    total, trainable = count_parameters(model)
    assert 5e6 < total < 30e6
    assert trainable == total


def test_param_groups_split_head_and_backbone(model):
    groups = model.param_groups(base_lr=1e-4, head_mult=10.0)
    assert len(groups) == 2
    assert groups[1]["lr"] == pytest.approx(1e-3)


def test_checkpoint_roundtrip_preserves_labels(model, tmp_path):
    path = tmp_path / "ckpt.pth"
    model.save(path, epoch=3)
    loaded, meta = load_model(path)
    assert loaded.labels == model.labels
    assert meta["epoch"] == 3
    assert meta["backbone"] == model.backbone_name


def test_load_rejects_mismatched_labels(model, tmp_path):
    """The v1 bug: CheXpert-trained head served under NIH names. Every output
    label would be wrong and nothing would crash. Now it crashes."""
    path = tmp_path / "ckpt.pth"
    model.save(path)
    with pytest.raises(LabelMismatchError):
        load_model(path, expected_labels=CHESTXRAY14_LABELS)


def test_missing_checkpoint_is_actionable(tmp_path):
    with pytest.raises(MissingCheckpointError) as exc:
        load_model(tmp_path / "nope.pth")
    assert "Train one first" in str(exc.value) or "train" in str(exc.value).lower()


def test_temperature_scaling_changes_probs_not_ranking(model, batch):
    raw = model(batch, calibrated=False)["disease_probs"]
    model.temperature.fill_(2.0)
    scaled = model(batch, calibrated=True)["disease_probs"]
    model.temperature.fill_(1.0)
    assert not torch.allclose(raw, scaled)
    assert torch.equal(raw.argsort(dim=1), scaled.argsort(dim=1))


def test_eval_mode_is_deterministic(model, batch):
    model.eval()
    with torch.no_grad():
        assert torch.allclose(model(batch)["disease_probs"], model(batch)["disease_probs"])


@pytest.mark.parametrize("backbone", ["densenet121", "resnet50"])
def test_alternate_backbones_build(backbone, labels):
    net = RadiScanNet(labels, backbone=backbone, pretrained=False, image_size=128)
    out = net(torch.randn(1, 3, 128, 128))
    assert out["disease_logits"].shape == (1, len(labels))
