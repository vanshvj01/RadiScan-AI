from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from radiscan.data.datasets import (
    ChestXrayDataset,
    DatasetVerificationError,
    make_balanced_sampler,
    verify_dataset,
)
from radiscan.data.loader import ImageLoadError, load_image
from radiscan.data.transforms import apply_clahe, eval_transforms, train_transforms


# --- the bug that mattered most -------------------------------------------
def test_missing_image_raises_instead_of_returning_black(synthetic_dataset, tmp_path):
    """v1 logged a warning and returned a blank frame. Training then 'succeeded'
    on 223k black images at ~0.50 AUC with no error anywhere."""
    df = pd.read_csv(synthetic_dataset["csv"])
    df.loc[0, "Path"] = "images/does_not_exist.png"
    bad_csv = tmp_path / "bad.csv"
    df.to_csv(bad_csv, index=False)

    ds = ChestXrayDataset(
        bad_csv, synthetic_dataset["root"], synthetic_dataset["labels"], strict=True
    )
    with pytest.raises(DatasetVerificationError):
        _ = ds[0]


def test_non_strict_mode_is_opt_in(synthetic_dataset, tmp_path):
    df = pd.read_csv(synthetic_dataset["csv"])
    df.loc[0, "Path"] = "images/gone.png"
    bad_csv = tmp_path / "bad.csv"
    df.to_csv(bad_csv, index=False)

    ds = ChestXrayDataset(
        bad_csv, synthetic_dataset["root"], synthetic_dataset["labels"], strict=False
    )
    assert ds[0]["image"] is not None  # blank substitution, knowingly accepted


def test_verify_dataset_detects_path_mismatch(synthetic_dataset, tmp_path):
    df = pd.read_csv(synthetic_dataset["csv"])
    df["Path"] = "CheXpert-v1.0-small/" + df["Path"]  # the real-world mismatch
    bad_csv = tmp_path / "prefixed.csv"
    df.to_csv(bad_csv, index=False)

    with pytest.raises(DatasetVerificationError, match="hit rate"):
        verify_dataset(bad_csv, synthetic_dataset["root"], synthetic_dataset["labels"])


def test_verify_dataset_passes_on_good_data(synthetic_dataset):
    report = verify_dataset(
        synthetic_dataset["csv"], synthetic_dataset["root"], synthetic_dataset["labels"]
    )
    assert report["hit_rate"] == 1.0
    assert "positive_rates" in report


# --- max_images actually works --------------------------------------------
def test_max_images_is_respected(synthetic_dataset):
    ds = ChestXrayDataset(
        synthetic_dataset["csv"],
        synthetic_dataset["root"],
        synthetic_dataset["labels"],
        max_images=5,
    )
    assert len(ds) == 5


# --- uncertain-label policies ---------------------------------------------
@pytest.mark.parametrize("policy", ["zeros", "ones", "ignore", "lsr"])
def test_uncertain_policies(synthetic_dataset, policy):
    ds = ChestXrayDataset(
        synthetic_dataset["csv"],
        synthetic_dataset["root"],
        synthetic_dataset["labels"],
        uncertain_policy=policy,
        transform=eval_transforms(128),
    )
    item = ds[0]
    assert torch.all(item["labels"] >= 0) and torch.all(item["labels"] <= 1)
    if policy == "ignore":
        assert item["label_mask"].min() <= 1.0


def test_unknown_policy_rejected(synthetic_dataset):
    ds = ChestXrayDataset(
        synthetic_dataset["csv"],
        synthetic_dataset["root"],
        synthetic_dataset["labels"],
        uncertain_policy="wishful",
    )
    with pytest.raises(ValueError):
        _ = ds[0]


# --- transforms ------------------------------------------------------------
def test_clahe_preserves_shape_and_dtype():
    img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
    out = apply_clahe(img)
    assert out.shape == img.shape and out.dtype == np.uint8


def test_eval_transform_output_is_normalized_tensor():
    img = np.random.randint(0, 255, (300, 400, 3), dtype=np.uint8)
    t = eval_transforms(224)(image=img)["image"]
    assert t.shape == (3, 224, 224) and t.dtype == torch.float32
    assert -4 < float(t.mean()) < 4  # ImageNet-normalised, not raw 0-255


def test_train_transform_is_stochastic():
    img = np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8)
    tf = train_transforms(224)
    assert not torch.allclose(tf(image=img)["image"], tf(image=img)["image"])


# --- sampling --------------------------------------------------------------
def test_balanced_sampler_upweights_rare_classes(synthetic_dataset):
    ds = ChestXrayDataset(
        synthetic_dataset["csv"], synthetic_dataset["root"], synthetic_dataset["labels"]
    )
    sampler = make_balanced_sampler(ds)
    assert len(sampler) == len(ds)
    assert float(torch.as_tensor(sampler.weights).max()) >= 1.0


def test_pos_weight_is_bounded(synthetic_dataset):
    ds = ChestXrayDataset(
        synthetic_dataset["csv"], synthetic_dataset["root"], synthetic_dataset["labels"]
    )
    w = ds.pos_weight()
    assert w.shape == (len(synthetic_dataset["labels"]),)
    assert float(w.min()) >= 1.0 and float(w.max()) <= 50.0


# --- loader ----------------------------------------------------------------
def test_load_image_reads_png(xray_png):
    img, meta = load_image(xray_png)
    assert img.ndim == 3 and img.shape[2] == 3 and img.dtype == np.uint8
    assert meta["format"] == ".png"


def test_load_image_rejects_missing_file(tmp_path):
    with pytest.raises(ImageLoadError):
        load_image(tmp_path / "nope.png")


def test_load_image_rejects_unsupported_extension(tmp_path):
    p = tmp_path / "notes.txt"
    p.write_text("hello")
    with pytest.raises(ImageLoadError, match="Unsupported"):
        load_image(p)
