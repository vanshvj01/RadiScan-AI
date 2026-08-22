"""Metrics for imbalanced multi-label classification.

Mean AUC is the headline number, but it is not sufficient on its own: with a 2%
positive rate a class can have 0.85 AUC and a useless precision-recall curve.
Every result table produced here reports AUC *and* average precision (AP, the
area under the PR curve), because AP is the metric that actually degrades when
the rare classes are being guessed.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


def _valid(y_true: np.ndarray, y_prob: np.ndarray, mask: np.ndarray | None):
    if mask is not None:
        keep = mask.astype(bool)
        return y_true[keep], y_prob[keep]
    return y_true, y_prob


def per_class_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    labels: list[str] | tuple[str, ...],
    thresholds: dict[str, float] | None = None,
    mask: np.ndarray | None = None,
) -> list[dict]:
    """One row per class: AUC, AP, precision/recall/F1 at the chosen threshold."""
    rows = []
    for i, name in enumerate(labels):
        yt, yp = _valid(y_true[:, i], y_prob[:, i], None if mask is None else mask[:, i])
        yt_bin = (yt >= 0.5).astype(int)
        n_pos = int(yt_bin.sum())

        row: dict = {"label": name, "support": n_pos, "n": int(len(yt_bin))}
        if n_pos == 0 or n_pos == len(yt_bin):
            row.update(
                auc=float("nan"),
                ap=float("nan"),
                precision=float("nan"),
                recall=float("nan"),
                f1=float("nan"),
                threshold=float("nan"),
                brier=float("nan"),
            )
            rows.append(row)
            continue

        thr = (thresholds or {}).get(name, 0.5)
        yhat = (yp >= thr).astype(int)
        row.update(
            auc=float(roc_auc_score(yt_bin, yp)),
            ap=float(average_precision_score(yt_bin, yp)),
            precision=float(precision_score(yt_bin, yhat, zero_division=0)),
            recall=float(recall_score(yt_bin, yhat, zero_division=0)),
            f1=float(f1_score(yt_bin, yhat, zero_division=0)),
            accuracy=float(accuracy_score(yt_bin, yhat)),
            threshold=float(thr),
            brier=float(brier_score_loss(yt_bin, np.clip(yp, 0, 1))),
            prevalence=float(n_pos / len(yt_bin)),
        )
        rows.append(row)
    return rows


def summarize(rows: list[dict], subset: list[str] | tuple[str, ...] | None = None) -> dict:
    """Aggregate per-class rows into the numbers that go in the abstract."""
    sel = [r for r in rows if subset is None or r["label"] in subset]

    def _mean(key):
        vals = [r[key] for r in sel if not np.isnan(r.get(key, float("nan")))]
        return float(np.mean(vals)) if vals else float("nan")

    return {
        "n_classes": len(sel),
        "mean_auc": _mean("auc"),
        "mean_ap": _mean("ap"),
        "mean_f1": _mean("f1"),
        "mean_precision": _mean("precision"),
        "mean_recall": _mean("recall"),
        "mean_brier": _mean("brier"),
    }


def optimal_thresholds(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    labels: list[str] | tuple[str, ...],
    objective: str = "f1",
    min_recall: float = 0.0,
) -> dict[str, float]:
    """Per-class operating points, tuned on the validation set only.

    A single 0.5 threshold across 14 classes with prevalences from 1% to 40% is
    indefensible. Tuning per class typically doubles F1 on the rare classes.
    ``min_recall`` lets you encode the clinical asymmetry -- for a triage tool a
    false negative costs far more than a false positive, so constraining recall
    and then maximising precision is often the right objective.
    """
    out: dict[str, float] = {}
    for i, name in enumerate(labels):
        yt = (y_true[:, i] >= 0.5).astype(int)
        yp = y_prob[:, i]
        if yt.sum() == 0:
            out[name] = 0.5
            continue

        precision, recall, thr = precision_recall_curve(yt, yp)
        precision, recall = precision[:-1], recall[:-1]
        if len(thr) == 0:
            out[name] = 0.5
            continue

        feasible = recall >= min_recall
        if not feasible.any():
            feasible = np.ones_like(recall, dtype=bool)

        if objective == "f1":
            score = np.where(
                (precision + recall) > 0, 2 * precision * recall / (precision + recall + 1e-12), 0.0
            )
        elif objective == "youden":
            score = recall - (1 - precision)
        elif objective == "precision":
            score = precision
        else:
            raise ValueError(f"Unknown objective {objective!r}")

        score = np.where(feasible, score, -np.inf)
        out[name] = float(thr[int(np.argmax(score))])
    return out


def expected_calibration_error(
    y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 15
) -> tuple[float, list[dict]]:
    """ECE plus the per-bin data needed to draw a reliability diagram."""
    yt = (y_true.ravel() >= 0.5).astype(int)
    yp = np.clip(y_prob.ravel(), 0, 1)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece, bins = 0.0, []

    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (yp > lo) & (yp <= hi) if lo > 0 else (yp >= lo) & (yp <= hi)
        n = int(sel.sum())
        if n == 0:
            bins.append(
                {
                    "lo": float(lo),
                    "hi": float(hi),
                    "n": 0,
                    "confidence": float("nan"),
                    "accuracy": float("nan"),
                }
            )
            continue
        conf = float(yp[sel].mean())
        acc = float(yt[sel].mean())
        ece += (n / len(yp)) * abs(acc - conf)
        bins.append({"lo": float(lo), "hi": float(hi), "n": n, "confidence": conf, "accuracy": acc})
    return float(ece), bins


def uncertainty_quartiles(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    uncertainty: np.ndarray,
    thresholds: dict[str, float] | None = None,
    labels: list[str] | tuple[str, ...] | None = None,
    n_bins: int = 4,
) -> list[dict]:
    """The money table: does accuracy fall monotonically as uncertainty rises?

    If it does, the MC-Dropout estimates carry real information and the whole
    uncertainty component of the project is justified in one table. If it does
    not, say so -- an honest negative result reads far better in a viva than a
    fudged positive one.
    """
    thr_vec = np.full(y_prob.shape[1], 0.5)
    if thresholds and labels:
        thr_vec = np.array([thresholds.get(name, 0.5) for name in labels])

    correct = ((y_prob >= thr_vec) == (y_true >= 0.5)).ravel().astype(float)
    unc = uncertainty.ravel()
    order = np.argsort(unc)
    chunks = np.array_split(order, n_bins)

    rows = []
    for i, idx in enumerate(chunks, start=1):
        rows.append(
            {
                "quartile": f"Q{i}"
                + (" (most confident)" if i == 1 else " (least confident)" if i == n_bins else ""),
                "n": int(len(idx)),
                "accuracy": float(correct[idx].mean()),
                "mean_uncertainty": float(unc[idx].mean()),
                "uncertainty_range": [float(unc[idx].min()), float(unc[idx].max())],
            }
        )
    return rows


def selective_prediction_curve(
    y_true: np.ndarray, y_prob: np.ndarray, uncertainty: np.ndarray, steps: int = 20
) -> list[dict]:
    """Accuracy vs. coverage when the most uncertain cases are deferred.

    This is the deployment-relevant framing: "if the system refers its least
    confident 20% of predictions to a radiologist, what accuracy does it achieve
    on the 80% it keeps?"
    """
    correct = ((y_prob >= 0.5) == (y_true >= 0.5)).ravel().astype(float)
    unc = uncertainty.ravel()
    order = np.argsort(unc)
    rows = []
    for frac in np.linspace(1.0, 0.1, steps):
        k = max(int(len(order) * frac), 1)
        idx = order[:k]
        rows.append(
            {
                "coverage": float(frac),
                "n_kept": int(k),
                "accuracy": float(correct[idx].mean()),
            }
        )
    return rows
