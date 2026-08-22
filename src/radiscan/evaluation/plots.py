"""Report-ready figures. Every plot writes a PNG *and* the CSV behind it."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import auc as _auc  # noqa: E402
from sklearn.metrics import precision_recall_curve, roc_curve

from radiscan.evaluation.metrics import expected_calibration_error  # noqa: E402

plt.rcParams.update(
    {
        "figure.dpi": 150,
        "savefig.bbox": "tight",
        "font.size": 9,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


def _save(fig, out_dir: Path, name: str) -> str:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.png"
    fig.savefig(path)
    plt.close(fig)
    return str(path)


def roc_curves(y_true, y_prob, labels, out_dir: str | Path, name: str = "roc_curves") -> str:
    out_dir = Path(out_dir)
    fig, ax = plt.subplots(figsize=(7, 6))
    cmap = plt.get_cmap("tab20")
    rows = []

    for i, label in enumerate(labels):
        yt = (y_true[:, i] >= 0.5).astype(int)
        if yt.sum() in (0, len(yt)):
            continue
        fpr, tpr, _ = roc_curve(yt, y_prob[:, i])
        auc = _auc(fpr, tpr)
        ax.plot(fpr, tpr, lw=1.4, color=cmap(i % 20), label=f"{label} ({auc:.3f})")
        rows.extend({"label": label, "fpr": f, "tpr": t} for f, t in zip(fpr, tpr))

    ax.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.5)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("Per-class ROC curves")
    ax.legend(fontsize=6, loc="lower right", ncol=2)
    pd.DataFrame(rows).to_csv(out_dir / f"{name}.csv", index=False)
    return _save(fig, out_dir, name)


def pr_curves(y_true, y_prob, labels, out_dir: str | Path, name: str = "pr_curves") -> str:
    out_dir = Path(out_dir)
    fig, ax = plt.subplots(figsize=(7, 6))
    cmap = plt.get_cmap("tab20")

    for i, label in enumerate(labels):
        yt = (y_true[:, i] >= 0.5).astype(int)
        if yt.sum() in (0, len(yt)):
            continue
        precision, recall, _ = precision_recall_curve(yt, y_prob[:, i])
        ax.plot(recall, precision, lw=1.4, color=cmap(i % 20), label=label)

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Per-class precision-recall curves")
    ax.legend(fontsize=6, ncol=2)
    return _save(fig, out_dir, name)


def reliability_diagram(
    y_true, y_prob, out_dir: str | Path, name: str = "calibration", n_bins: int = 15
) -> str:
    out_dir = Path(out_dir)
    ece, bins = expected_calibration_error(y_true, y_prob, n_bins)
    df = pd.DataFrame(bins)
    df.to_csv(out_dir / f"{name}.csv", index=False)

    fig, (ax, ax2) = plt.subplots(
        2, 1, figsize=(5, 6), gridspec_kw={"height_ratios": [3, 1]}, sharex=True
    )
    centers = (df["lo"] + df["hi"]) / 2
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect calibration")
    ax.plot(df["confidence"], df["accuracy"], "o-", lw=1.5, label=f"model (ECE={ece:.4f})")
    ax.set_ylabel("Observed frequency")
    ax.set_title("Reliability diagram")
    ax.legend()

    ax2.bar(centers, df["n"], width=1 / n_bins * 0.9)
    ax2.set_xlabel("Predicted probability")
    ax2.set_ylabel("Count")
    ax2.set_yscale("log")
    return _save(fig, out_dir, name)


def uncertainty_vs_accuracy(
    quartiles: list[dict], out_dir: str | Path, name: str = "uncertainty_quartiles"
) -> str:
    out_dir = Path(out_dir)
    df = pd.DataFrame(quartiles)
    df.to_csv(out_dir / f"{name}.csv", index=False)

    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.bar(range(len(df)), df["accuracy"], color="#4C78A8")
    ax.set_xticks(range(len(df)))
    ax.set_xticklabels([q.split(" ")[0] for q in df["quartile"]])
    ax.set_ylim(min(df["accuracy"]) * 0.95, 1.0)
    ax.set_xlabel("Uncertainty quartile (Q1 = most confident)")
    ax.set_ylabel("Accuracy")
    ax.set_title("Does uncertainty predict error?")
    for i, v in enumerate(df["accuracy"]):
        ax.text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    return _save(fig, out_dir, name)


def selective_prediction(
    curve: list[dict], out_dir: str | Path, name: str = "selective_prediction"
) -> str:
    out_dir = Path(out_dir)
    df = pd.DataFrame(curve)
    df.to_csv(out_dir / f"{name}.csv", index=False)

    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.plot(df["coverage"], df["accuracy"], "o-", lw=1.5)
    ax.set_xlabel("Coverage (fraction of cases the model keeps)")
    ax.set_ylabel("Accuracy on kept cases")
    ax.set_title("Accuracy under selective prediction")
    ax.invert_xaxis()
    return _save(fig, out_dir, name)


def training_curves(history: list[dict], out_dir: str | Path, name: str = "training_curves") -> str:
    out_dir = Path(out_dir)
    df = pd.DataFrame(history)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.5))

    if "train_total" in df:
        ax1.plot(df["epoch"], df["train_total"], "o-", label="train")
    if "val_loss" in df:
        ax1.plot(df["epoch"], df["val_loss"], "o-", label="val")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Loss")
    ax1.legend()

    if "val_mean_auc" in df:
        ax2.plot(df["epoch"], df["val_mean_auc"], "o-", color="#54A24B", label="mean AUC")
    if "val_mean_ap" in df:
        ax2.plot(df["epoch"], df["val_mean_ap"], "o-", color="#E45756", label="mean AP")
    ax2.set_xlabel("Epoch")
    ax2.set_title("Validation metrics")
    ax2.legend()
    return _save(fig, out_dir, name)


def per_class_bars(rows: list[dict], out_dir: str | Path, name: str = "per_class_auc") -> str:
    out_dir = Path(out_dir)
    df = pd.DataFrame(rows).dropna(subset=["auc"]).sort_values("auc")
    fig, ax = plt.subplots(figsize=(6, max(3, 0.32 * len(df))))
    ax.barh(df["label"], df["auc"], color="#4C78A8")
    ax.axvline(0.5, color="k", ls="--", lw=0.8)
    ax.set_xlim(0.4, 1.0)
    ax.set_xlabel("AUC-ROC")
    ax.set_title("Per-class discrimination")
    for i, (auc, sup) in enumerate(zip(df["auc"], df["support"])):
        ax.text(auc + 0.005, i, f"{auc:.3f} (n={sup})", va="center", fontsize=7)
    return _save(fig, out_dir, name)
