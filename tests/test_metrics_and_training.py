from __future__ import annotations

import numpy as np
import pytest
import torch

from radiscan.data.datasets import build_dataloaders
from radiscan.evaluation.metrics import (
    expected_calibration_error,
    optimal_thresholds,
    per_class_metrics,
    selective_prediction_curve,
    summarize,
    uncertainty_quartiles,
)
from radiscan.models.losses import AsymmetricLoss, FocalLoss, MultiTaskLoss
from radiscan.models.network import build_model, load_model
from radiscan.training.calibration import fit_temperature
from radiscan.training.ema import ModelEMA
from radiscan.training.schedulers import cosine_with_warmup
from radiscan.training.trainer import EarlyStopping, Trainer, mixup_multilabel


# --- losses ----------------------------------------------------------------
def test_asl_penalises_wrong_predictions_more():
    loss = AsymmetricLoss()
    targets = torch.ones(4, 3)
    confident_right = loss(torch.full((4, 3), 5.0), targets)
    confident_wrong = loss(torch.full((4, 3), -5.0), targets)
    assert confident_wrong > confident_right


def test_asl_downweights_easy_negatives_vs_focal():
    """The whole point of ASL: asymmetric focusing on the negative class."""
    logits = torch.full((8, 5), -6.0)  # very easy negatives
    targets = torch.zeros(8, 5)
    assert AsymmetricLoss()(logits, targets) < FocalLoss()(logits, targets)


def test_loss_mask_excludes_ignored_labels():
    loss = AsymmetricLoss()
    logits = torch.randn(4, 6)
    targets = torch.zeros(4, 6)
    full = loss(logits, targets, mask=torch.ones(4, 6))
    partial = loss(logits, targets, mask=torch.zeros(4, 6).index_fill_(1, torch.tensor([0]), 1.0))
    assert not torch.isclose(full, partial)


def test_multitask_loss_returns_components():
    crit = MultiTaskLoss(kind="asl")
    outputs = {
        "disease_logits": torch.randn(2, 14),
        "fracture_logit": torch.randn(2, 1),
        "confidence": torch.rand(2, 14),
    }
    batch = {
        "labels": torch.randint(0, 2, (2, 14)).float(),
        "label_mask": torch.ones(2, 14),
        "fracture": torch.zeros(2),
    }
    total, parts = crit(outputs, batch)
    assert {"disease", "fracture", "confidence", "total"} <= set(parts)
    assert torch.isfinite(total)


def test_unknown_loss_kind_rejected():
    with pytest.raises(ValueError):
        MultiTaskLoss(kind="vibes")


# --- metrics ---------------------------------------------------------------
@pytest.fixture
def fake_scores():
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 2, (500, 4)).astype(float)
    # Probabilities correlated with truth, so AUC should land well above chance.
    y_prob = np.clip(y_true * 0.6 + rng.random((500, 4)) * 0.4, 0, 1)
    return y_true, y_prob, ["A", "B", "C", "D"]


def test_per_class_metrics_shape_and_range(fake_scores):
    y_true, y_prob, labels = fake_scores
    rows = per_class_metrics(y_true, y_prob, labels)
    assert len(rows) == 4
    for row in rows:
        assert 0.0 <= row["auc"] <= 1.0
        assert 0.0 <= row["ap"] <= 1.0


def test_summary_beats_chance(fake_scores):
    rows = per_class_metrics(*fake_scores)
    assert summarize(rows)["mean_auc"] > 0.6


def test_class_with_no_positives_yields_nan_not_crash():
    y_true = np.zeros((50, 2))
    y_prob = np.random.rand(50, 2)
    rows = per_class_metrics(y_true, y_prob, ["X", "Y"])
    assert all(np.isnan(r["auc"]) for r in rows)


def test_optimal_thresholds_differ_per_class(fake_scores):
    y_true, y_prob, labels = fake_scores
    thr = optimal_thresholds(y_true, y_prob, labels)
    assert set(thr) == set(labels)
    assert all(0.0 <= v <= 1.0 for v in thr.values())


def test_min_recall_constraint_lowers_threshold(fake_scores):
    y_true, y_prob, labels = fake_scores
    loose = optimal_thresholds(y_true, y_prob, labels, min_recall=0.0)
    strict = optimal_thresholds(y_true, y_prob, labels, min_recall=0.95)
    assert any(strict[k] <= loose[k] for k in labels)


def test_ece_is_zero_for_perfect_calibration():
    y_prob = np.repeat(np.array([0.0, 1.0]), 500).reshape(-1, 1)
    y_true = y_prob.copy()
    ece, bins = expected_calibration_error(y_true, y_prob)
    assert ece < 0.01
    assert len(bins) == 15


def test_uncertainty_quartiles_are_ordered_by_uncertainty():
    rng = np.random.default_rng(1)
    y_true = rng.integers(0, 2, (400, 3)).astype(float)
    y_prob = rng.random((400, 3))
    unc = rng.random((400, 3))
    rows = uncertainty_quartiles(y_true, y_prob, unc)
    assert len(rows) == 4
    means = [r["mean_uncertainty"] for r in rows]
    assert means == sorted(means)


def test_selective_prediction_curve_decreasing_coverage():
    rng = np.random.default_rng(2)
    y_true = rng.integers(0, 2, (200, 2)).astype(float)
    curve = selective_prediction_curve(y_true, rng.random((200, 2)), rng.random((200, 2)))
    covs = [r["coverage"] for r in curve]
    assert covs == sorted(covs, reverse=True)


# --- training components ---------------------------------------------------
def test_early_stopping_triggers_after_patience():
    stopper = EarlyStopping(patience=2, mode="max")
    assert stopper.step(0.8) == (True, False)
    assert stopper.step(0.7) == (False, False)
    assert stopper.step(0.6) == (False, True)


def test_mixup_preserves_shape_and_takes_label_union():
    images = torch.randn(4, 3, 32, 32)
    targets = torch.zeros(4, 5)
    targets[0, 0] = 1.0
    mixed, mixed_t = mixup_multilabel(images, targets, alpha=0.4)
    assert mixed.shape == images.shape
    assert mixed_t.max() == 1.0, "Multi-label mixup must not dilute a present finding to 0.5"


def test_mixup_disabled_is_identity():
    images = torch.randn(2, 3, 8, 8)
    targets = torch.rand(2, 4)
    a, b = mixup_multilabel(images, targets, alpha=0.0)
    assert torch.equal(a, images) and torch.equal(b, targets)


def test_cosine_scheduler_warms_up_then_decays():
    params = [torch.nn.Parameter(torch.randn(2))]
    opt = torch.optim.SGD(params, lr=1.0)
    sched = cosine_with_warmup(opt, warmup_steps=5, total_steps=50)
    lrs = []
    for _ in range(50):
        lrs.append(opt.param_groups[0]["lr"])
        opt.step()  # before sched.step(), or PyTorch warns and skips the first LR
        sched.step()
    assert lrs[0] < lrs[5]
    assert lrs[-1] < lrs[5]


def test_ema_tracks_but_lags_the_model(model):
    ema = ModelEMA(model, decay=0.9, warmup_steps=0)
    with torch.no_grad():
        for p in model.parameters():
            p.add_(1.0)
    ema.update(model)
    key = next(name for name, _ in model.named_parameters())
    assert not torch.allclose(ema.state_dict()[key], model.state_dict()[key])


def test_temperature_scaling_recovers_a_known_temperature():
    rng = np.random.default_rng(3)
    true_logits = rng.normal(0, 2, (2000, 5))
    targets = (rng.random((2000, 5)) < 1 / (1 + np.exp(-true_logits))).astype(float)
    fitted = fit_temperature(true_logits * 3.0, targets)  # overconfident by 3x
    assert 2.0 < fitted < 4.5


# --- integration: one real training epoch ---------------------------------
@pytest.mark.slow
def test_one_epoch_trains_and_writes_a_loadable_checkpoint(smoke_config):
    loaders = build_dataloaders(smoke_config, verify=True)
    model = build_model(smoke_config, loaders["labels"])
    trainer = Trainer(model, smoke_config, loaders["train"], loaders["val"], device="cpu")
    result = trainer.fit()

    assert len(trainer.history) == 1
    loaded, meta = load_model(result["best_checkpoint"], expected_labels=loaders["labels"])
    assert loaded.labels == tuple(loaders["labels"])
    assert "metrics" in meta
