"""Every experiment requested in the reviewer report, from one trained checkpoint.

    python -m paper.reviewer_experiments --config <cfg.yaml> \
        --checkpoint models/checkpoints/best_model.pth --out results/reviewer

Reviewer point -> what this script produces (all under ``--out``):

R1  Table I (per-class), Table II (aggregate)      table1_per_class.csv, summary in results.json
R2  Epistemic/aleatoric evidence                    uncertainty_error_detection.csv, quartiles_*.csv
R3  Deferral rates + performance vs coverage        deferral_policies.csv, coverage_table.csv, fig_risk_coverage.png
R4  ECE and Brier before/after temperature scaling  calibration.csv, fig_reliability.png
R6  Effect of the number of MC passes               mc_sweep.csv, fig_mc_sweep.png
R7  Confidence-head effectiveness                   uncertainty_error_detection.csv, aleatoric_proxy.json
R8  Grad-CAM: correct / incorrect / deferred        fig_gradcam.png (+ selection.csv)
R9  Patients and images per split, overlap check    splits.json, splits_per_class.csv
R10 Validation-only selection of the sigma tolerance deferral_policies.csv (tau chosen on val)
R11 Latency, throughput, memory                     latency.csv, throughput.csv, resources.json

(R5, the ablations, needs retraining and lives in ``paper/run_ablations.py``.)

Evaluation protocol. CheXpert test annotations that are *uncertain* (-1) are
masked out of every metric. The training-time uncertain-label policy (label
smoothing) must not leak into evaluation: smoothing turns an uncertain test
label into a random target, so the metric would depend on a random draw. For
comparability with the earlier run the script also recomputes Table I under that
earlier protocol (``table1_lsr_protocol.csv``) as a split-reproduction check.

Efficiency note that is also a finding. Every dropout layer in RadiScanNet sits
*after* the convolutional trunk (fusion block and heads); the DenseNet trunk has
no dropout. The trunk output is therefore identical across MC passes, so the
script computes it once per view and re-samples only the heads. This is exactly
equivalent to the deployed ``predict_with_uncertainty`` (asserted below) and is
~S times cheaper.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import platform
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from radiscan.cli.prepare_data import extract_patient_id
from radiscan.config import Config
from radiscan.constants import CHEXPERT_COMPETITION_LABELS
from radiscan.data.datasets import ChestXrayDataset
from radiscan.constants import IMAGENET_MEAN, IMAGENET_STD
from radiscan.data.transforms import eval_transforms
from radiscan.evaluation.metrics import (
    expected_calibration_error,
    per_class_metrics,
    summarize,
)
from radiscan.models.network import RadiScanNet
from radiscan.modules.explainability import GradCAM, overlay_heatmap
from radiscan.modules.uncertainty import enable_mc_dropout, predict_with_uncertainty

log = logging.getLogger("reviewer")

TAU_FIXED = 0.15  # the tolerance hard-coded in ReportGenerator before this revision
COVERAGE_TARGETS = (1.0, 0.95, 0.90, 0.80, 0.70, 0.60, 0.50)
TAU_TARGETS = (0.95, 0.90, 0.80)
S_GRID = (1, 2, 5, 10, 20, 30, 50)
CAM_FINDINGS = ("Cardiomegaly", "Pleural Effusion", "Pneumothorax", "Edema", "Support Devices")


# ---------------------------------------------------------------------------
# model
# ---------------------------------------------------------------------------
def load_checkpoint(path: str | Path, device: torch.device):
    """Load a RadiScan checkpoint, inferring which optional heads it carries.

    ``radiscan.models.network.load_model`` always builds both optional heads,
    which fails on an ablation checkpoint trained without the confidence head.
    """
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    sd = ckpt["state_dict"]
    model = RadiScanNet(
        labels=tuple(ckpt["labels"]),
        backbone=ckpt.get("backbone", "densenet121"),
        pretrained=False,
        use_transformer=ckpt.get("use_transformer", False),
        image_size=ckpt.get("image_size", 320),
        fracture_head=any(k.startswith("head.fracture") for k in sd),
        confidence_head=any(k.startswith("head.confidence") for k in sd),
    )
    model.load_state_dict(sd, strict=True)
    model.temperature.fill_(float(ckpt.get("temperature", 1.0)))
    model.to(device).eval()
    meta = {k: v for k, v in ckpt.items() if k != "state_dict"}
    return model, meta


def trunk_features(model: RadiScanNet, x: torch.Tensor) -> torch.Tensor:
    f = model.cnn(x)
    if model.vit is not None:
        f = torch.cat([f, model.vit(x)], dim=1)
    return f


def head_outputs(model: RadiScanNet, f: torch.Tensor):
    out = model.head(model.fuse(f))
    return out["disease_logits"], out.get("confidence")


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------
def frontal(df: pd.DataFrame) -> pd.DataFrame:
    for col in ("Frontal/Lateral", "ViewPosition", "View Position"):
        if col in df.columns:
            return df[df[col].astype(str).str.lower().str.startswith(("frontal", "pa", "ap"))]
    return df


def split_statistics(cfg: Config, labels, out: Path) -> dict:
    """R9: images, studies and patients per split, and a patient-overlap audit."""
    stats, pats, imgs, rows = {}, {}, {}, []
    for name in ("train", "val", "test"):
        csv = getattr(cfg.data, f"{name}_csv")
        if not csv or not Path(csv).exists():
            continue
        df = pd.read_csv(csv)
        if cfg.data.frontal_only:
            df = frontal(df)
        path_col = next(c for c in ("Path", "path", "image_path", "Image Index") if c in df.columns)
        p = df[path_col].map(extract_patient_id)
        studies = df[path_col].astype(str).str.extract(r"(patient\d+/study\d+)")[0]
        pats[name], imgs[name] = set(p), set(df[path_col].astype(str))
        stats[name] = {
            "images": int(len(df)),
            "studies": int(studies.nunique()),
            "patients": int(p.nunique()),
            "images_per_patient": round(len(df) / max(p.nunique(), 1), 2),
        }
        for lab in labels:
            col = df[lab]
            rows.append(
                {
                    "split": name,
                    "label": lab,
                    "positive": int((col == 1).sum()),
                    "negative": int((col == 0).sum()),
                    "uncertain": int((col == -1).sum()),
                    "blank": int(col.isna().sum()),
                }
            )
    overlap = {}
    names = list(pats)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            overlap[f"{a}-{b}"] = {
                "patients": len(pats[a] & pats[b]),
                "images": len(imgs[a] & imgs[b]),
            }
    stats["overlap"] = overlap
    stats["patient_disjoint"] = all(v["patients"] == 0 for v in overlap.values())
    pd.DataFrame(rows).to_csv(out / "splits_per_class.csv", index=False)
    (out / "splits.json").write_text(json.dumps(stats, indent=2))
    return stats


def make_dataset(cfg: Config, csv: str, labels, max_images=None) -> ChestXrayDataset:
    return ChestXrayDataset(
        csv,
        data_dir=cfg.data.data_dir,
        labels=labels,
        transform=eval_transforms(cfg.data.image_size),
        strict=True,
        uncertain_policy="ignore",  # labels only feed raw_labels below
        frontal_only=cfg.data.frontal_only,
        auto_crop=cfg.data.auto_crop,
        max_images=max_images,
    )


# ---------------------------------------------------------------------------
# inference cache
# ---------------------------------------------------------------------------
@torch.no_grad()
def collect(model, ds, device, s_max: int, batch_size: int, workers: int, check: bool = True):
    """Return per-sample logits (S, V, N, C), deterministic logits (V, N, C),
    MC-averaged confidence (N, C) and raw labels (N, C)."""
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=workers)
    mc, det, conf = [], [], []
    checked = not check
    for batch in tqdm(loader, desc="collect"):
        x = batch["image"].to(device)
        views = [x, torch.flip(x, dims=[3])]
        model.eval()
        feats = [trunk_features(model, v) for v in views]
        det.append(torch.stack([head_outputs(model, f)[0] for f in feats]).float().cpu())
        if not checked:  # the cached path must equal the full forward pass
            ref = model(x)["disease_logits"]
            assert torch.allclose(ref, det[-1][0].to(device), atol=1e-4), "trunk caching mismatch"
            checked = True
        enable_mc_dropout(model)
        per_s, conf_acc = [], 0.0
        for _ in range(s_max):
            outs = [head_outputs(model, f) for f in feats]
            per_s.append(torch.stack([o[0] for o in outs]))
            if outs[0][1] is not None:
                conf_acc = conf_acc + sum(o[1] for o in outs) / len(outs)
        model.eval()
        mc.append(torch.stack(per_s).float().cpu())
        if isinstance(conf_acc, torch.Tensor):
            conf.append((conf_acc / s_max).float().cpu())
    return {
        "mc": torch.cat(mc, dim=2).numpy(),  # (S, V, N, C)
        "det": torch.cat(det, dim=1).numpy(),  # (V, N, C)
        "conf": torch.cat(conf).numpy() if conf else None,
        "raw": ds.raw_labels.copy(),
        "paths": [str(ds.image_path(i)) for i in range(len(ds))],
    }


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def posterior(mc_logits: np.ndarray, temperature: float, idx=None, transform=None) -> dict:
    """Deployed estimator: per pass, average the two TTA views; then mean/std over passes."""
    z = mc_logits if idx is None else mc_logits[idx]
    if transform is not None:
        z = transform(z)
    else:
        z = z / temperature
    p = sigmoid(z).mean(axis=1)  # (S, N, C)
    mu = p.mean(0)
    sd = p.std(0) if p.shape[0] > 1 else np.zeros_like(mu)
    lo, hi = np.percentile(p, [2.5, 97.5], axis=0)
    eps = 1e-8
    m = np.clip(mu, eps, 1 - eps)
    ent = -(m * np.log(m) + (1 - m) * np.log(1 - m))
    ps = np.clip(p, eps, 1 - eps)
    exp_ent = (-(ps * np.log(ps) + (1 - ps) * np.log(1 - ps))).mean(0)
    return {"mu": mu, "sd": sd, "lo": lo, "hi": hi, "entropy": ent, "mi": np.clip(ent - exp_ent, 0, None)}


# ---------------------------------------------------------------------------
# metric helpers
# ---------------------------------------------------------------------------
def masks(raw):
    y = (raw == 1).astype(float)
    m = (raw != -1).astype(float)
    return y, m


def thr_vec(labels, thresholds):
    return np.array([thresholds.get(lab, 0.5) for lab in labels])


def macro_auc(y, p, m, cols=None):
    vals = []
    for k in cols if cols is not None else range(y.shape[1]):
        keep = m[:, k] > 0
        yk = y[keep, k]
        if 0 < yk.sum() < len(yk):
            vals.append(roc_auc_score(yk, p[keep, k]))
    return float(np.mean(vals)) if vals else float("nan")


def ece_brier(y, p, m):
    keep = m.ravel() > 0
    yt, pp = y.ravel()[keep], np.clip(p.ravel()[keep], 1e-7, 1 - 1e-7)
    ece, _ = expected_calibration_error(yt, pp, n_bins=15)
    brier = float(np.mean((pp - yt) ** 2))
    nll = float(-np.mean(yt * np.log(pp) + (1 - yt) * np.log(1 - pp)))
    cw = []
    for k in range(y.shape[1]):
        kk = m[:, k] > 0
        if kk.sum():
            cw.append(expected_calibration_error(y[kk, k], p[kk, k], n_bins=15)[0])
    return {"ece": float(ece), "brier": brier, "nll": nll, "classwise_ece": float(np.mean(cw))}


def aurc(error: np.ndarray, score: np.ndarray) -> float:
    """Area under the risk-coverage curve; ``score`` high = defer first."""
    order = np.argsort(score, kind="stable")  # most certain first
    e = error[order]
    risk = np.cumsum(e) / np.arange(1, len(e) + 1)
    return float(risk.mean())


def risk_at(error, score, coverage):
    order = np.argsort(score, kind="stable")
    k = max(int(round(coverage * len(order))), 1)
    return float(error[order[:k]].mean())


def binary_auroc(target, score):
    if 0 < target.sum() < len(target):
        return float(roc_auc_score(target, score))
    return float("nan")


def retained_metrics(y, mu, m, t, keep):
    """Element-level performance restricted to the retained (non-deferred) set."""
    use = (m > 0) & keep
    pred = mu >= t
    acc = float(((pred == (y > 0)) & use).sum() / max(use.sum(), 1))
    sens, spec, f1 = [], [], []
    for k in range(y.shape[1]):
        u = use[:, k]
        yk, pk = y[u, k] > 0, pred[u, k]
        if yk.sum():
            tp = (yk & pk).sum()
            sens.append(tp / yk.sum())
            prec = tp / max(pk.sum(), 1)
            r = tp / yk.sum()
            f1.append(0.0 if tp == 0 else 2 * prec * r / (prec + r))
        if (~yk).sum():
            spec.append(((~yk) & (~pk)).sum() / (~yk).sum())
    aucs = []
    for k in range(y.shape[1]):
        u = use[:, k]
        yk = y[u, k]
        if 0 < yk.sum() < len(yk):
            aucs.append(roc_auc_score(yk, mu[u, k]))
    nanmean = lambda v: float(np.mean(v)) if v else float("nan")  # noqa: E731
    return {
        "accuracy": acc,
        "sensitivity": nanmean(sens),
        "specificity": nanmean(spec),
        "macro_f1": nanmean(f1),
        "macro_auc": nanmean(aucs),
    }


# ---------------------------------------------------------------------------
# experiments
# ---------------------------------------------------------------------------
def table1(labels, post, raw, thresholds):
    y, m = masks(raw)
    rows = per_class_metrics(y, post["mu"], labels, thresholds=thresholds, mask=m)
    for r, k in zip(rows, range(len(labels))):
        keep = m[:, k] > 0
        r["uncertain_excluded"] = int((raw[:, k] == -1).sum())
        r["n"] = int(keep.sum())
    return rows


def lsr_targets(raw, low=0.05, high=0.85):
    """Reproduce the earlier evaluation, where the dataset's label-smoothing
    policy was also applied to the test split (row-seeded draw)."""
    out = raw.copy()
    for i in range(len(out)):
        u = out[i] == -1.0
        if u.any():
            out[i, u] = np.random.default_rng(i).uniform(low, high, size=u.sum())
    return np.clip(out, 0, 1)


def fit_temperature_np(logits, y, m):
    from radiscan.training.calibration import fit_temperature

    return fit_temperature(logits, y, m)


def fit_platt(logits, y, m):
    params = []
    for k in range(y.shape[1]):
        keep = m[:, k] > 0
        lr = LogisticRegression(C=1e4, max_iter=1000)
        lr.fit(logits[keep, k : k + 1], y[keep, k].astype(int))
        params.append((float(lr.coef_[0, 0]), float(lr.intercept_[0])))
    a = np.array([p[0] for p in params])
    b = np.array([p[1] for p in params])
    return a, b


def calibration_experiment(labels, val, test, T_ckpt, out: Path, S=20):
    """R4: ECE / Brier / NLL before and after post-hoc calibration."""
    yv, mv = masks(val["raw"])
    yt, mt = masks(test["raw"])
    det_val = val["det"][0]  # training-time protocol: single view, dropout off
    T_refit = float(fit_temperature_np(det_val, yv, mv))
    a, b = fit_platt(det_val, yv, mv)
    idx = np.arange(S)

    def det_probs(transform):
        return sigmoid(transform(test["det"])).mean(0)

    rows = []
    variants = {
        "uncalibrated (T=1)": lambda z: z,
        f"temperature, checkpoint (T={T_ckpt:.3f})": lambda z: z / T_ckpt,
        f"temperature, refit masked (T={T_refit:.3f})": lambda z: z / T_refit,
        "class-wise Platt (val-fitted)": lambda z: a * z + b,
    }
    for name, tf in variants.items():
        p_det = det_probs(tf)
        p_mc = posterior(test["mc"], 1.0, idx=idx, transform=tf)["mu"]
        for path, p in (("deterministic", p_det), (f"MC-{S}", p_mc)):
            r = {"calibration": name, "inference": path, **ece_brier(yt, p, mt)}
            r["mean_auc"] = macro_auc(yt, p, mt)
            rows.append(r)
    df = pd.DataFrame(rows)
    df.to_csv(out / "calibration.csv", index=False)

    # reliability diagrams (MC path)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(9, 3.1), sharey=True)
    for ax, (name, tf) in zip(axes, list(variants.items())[:2] + [list(variants.items())[3]]):
        p = posterior(test["mc"], 1.0, idx=idx, transform=tf)["mu"]
        keep = mt.ravel() > 0
        e, bins = expected_calibration_error(yt.ravel()[keep], p.ravel()[keep], 15)
        conf = [bb["confidence"] for bb in bins if bb["n"]]
        acc = [bb["accuracy"] for bb in bins if bb["n"]]
        ax.plot([0, 1], [0, 1], "--", color="0.6", lw=1)
        ax.plot(conf, acc, "o-", color="#1f5f6b", ms=3, lw=1.3)
        ax.set_title(f"{name.split(' (')[0]}\nECE={e:.3f}", fontsize=8)
        ax.set_xlabel("Mean predicted probability", fontsize=8)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Observed frequency", fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "fig_reliability.png", dpi=300)
    plt.close(fig)
    return df, {"T_ckpt": T_ckpt, "T_refit": T_refit}


def uncertainty_experiment(labels, post, test, thresholds, out: Path):
    """R2 + R7: do sigma / MI / entropy / (1-c) identify the model's errors?"""
    y, m = masks(test["raw"])
    t = thr_vec(labels, thresholds)
    mu = post["mu"]
    err = ((mu >= t) != (y > 0)).astype(float)
    keep = m > 0
    signals = {
        "posterior std (epistemic)": post["sd"],
        "mutual information (epistemic)": post["mi"],
        "predictive entropy (total)": post["entropy"],
        "1 - confidence head (aleatoric)": None if test["conf"] is None else 1 - test["conf"],
        "threshold margin -|mu-t| (baseline)": -np.abs(mu - t),
        "z-score -|mu-t|/sigma (deferral score)": -np.abs(mu - t) / np.maximum(post["sd"], 1e-6),
    }
    rows = []
    for name, s in signals.items():
        if s is None:
            continue
        macro = [
            binary_auroc(err[keep[:, k], k], s[keep[:, k], k])
            for k in range(len(labels))
        ]
        rows.append(
            {
                "signal": name,
                "error_auroc_pooled": binary_auroc(err[keep], s[keep]),
                "error_auroc_macro": float(np.nanmean(macro)),
                "aurc": aurc(err[keep], s[keep]),
                "risk_at_80": risk_at(err[keep], s[keep], 0.8),
                "risk_at_100": float(err[keep].mean()),
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(out / "uncertainty_error_detection.csv", index=False)

    def quartiles(s, name):
        order = np.argsort(s[keep], kind="stable")
        chunks = np.array_split(order, 4)
        acc = 1 - err[keep]
        q = [
            {
                "quartile": f"Q{i + 1}",
                "n": int(len(c)),
                "mean_signal": float(s[keep][c].mean()),
                "accuracy": float(acc[c].mean()),
            }
            for i, c in enumerate(chunks)
        ]
        pd.DataFrame(q).to_csv(out / f"quartiles_{name}.csv", index=False)
        return q

    q_sd = quartiles(post["sd"], "sigma")
    q_c = quartiles(1 - test["conf"], "confidence") if test["conf"] is not None else None
    extra = {}
    if test["conf"] is not None:
        extra["spearman_sigma_vs_1mc"] = float(spearmanr(post["sd"][keep], 1 - test["conf"][keep])[0])
        c = test["conf"]
        extra["confidence_mean"] = float(c.mean())
        extra["confidence_std"] = float(c.std())
        extra["confidence_p05_p95"] = [float(np.percentile(c, 5)), float(np.percentile(c, 95))]
        # aleatoric proxy: CheXpert marks a finding uncertain when the reporting
        # radiologist hedged, i.e. when the image itself is ambiguous.
        u = (test["raw"] == -1).astype(float)
        cols = [k for k in range(len(labels)) if 0 < u[:, k].sum() < len(u)]
        extra["uncertain_label_auroc_1mc"] = float(
            np.nanmean([binary_auroc(u[:, k], 1 - c[:, k]) for k in cols])
        )
        extra["uncertain_label_auroc_sigma"] = float(
            np.nanmean([binary_auroc(u[:, k], post["sd"][:, k]) for k in cols])
        )
        extra["confidence_mean_on_uncertain_labels"] = float(c[u > 0].mean())
        extra["confidence_mean_on_certain_labels"] = float(c[u == 0].mean())
    extra["sigma_quartiles_monotone"] = all(
        q_sd[i]["accuracy"] >= q_sd[i + 1]["accuracy"] for i in range(3)
    )
    if q_c:
        extra["confidence_quartiles_monotone"] = all(
            q_c[i]["accuracy"] >= q_c[i + 1]["accuracy"] for i in range(3)
        )
    extra["sigma_mean"] = float(post["sd"].mean())
    extra["sigma_p50_p95_p99_max"] = [
        float(np.percentile(post["sd"], q)) for q in (50, 95, 99, 100)
    ]
    (out / "aleatoric_proxy.json").write_text(json.dumps(extra, indent=2))
    return df, q_sd, q_c, extra


def deferral_mask(post, t, tau, cols):
    straddle = (post["lo"] < t) & (t <= post["hi"])
    d = straddle | (post["sd"] >= tau)
    d[:, [k for k in range(d.shape[1]) if k not in cols]] = False
    return d


def choose_tau(post_val, t, cols, target):
    """Smallest tau whose *validation* coverage reaches ``target``.

    Coverage is non-decreasing in tau, so this is a one-dimensional search on
    validation data only. If the interval rule alone already defers more than
    1 - target, no finite tau helps and the sigma rule is switched off (inf).
    """
    sd = post_val["sd"][:, cols]
    cand = np.unique(np.concatenate([np.quantile(sd, np.linspace(0.5, 1.0, 501)), [np.inf]]))
    for tau in cand:
        cov = 1 - deferral_mask(post_val, t, tau, cols)[:, cols].mean()
        if cov >= target:
            return float(tau), float(cov)
    return float("inf"), float(1 - deferral_mask(post_val, t, np.inf, cols)[:, cols].mean())


def deferral_experiment(labels, post_val, post_test, val, test, thresholds, out: Path):
    """R3 + R10: deferral rates, performance on retained vs deferred, tau from val."""
    t = thr_vec(labels, thresholds)
    cols = [k for k, lab in enumerate(labels) if lab != "No Finding"]
    y, m = masks(test["raw"])
    policies = {
        "no deferral": (np.inf, False),
        "interval rule only": (np.inf, True),
        f"sigma >= {TAU_FIXED} only (fixed)": (TAU_FIXED, False),
        f"interval OR sigma >= {TAU_FIXED} (initial)": (TAU_FIXED, True),
    }
    taus = {}
    for c in TAU_TARGETS:
        tau, cov_val = choose_tau(post_val, t, cols, c)
        taus[c] = {"tau": tau, "val_coverage": cov_val}
        policies[f"interval OR sigma >= tau_val({int(c * 100)}%)"] = (tau, True)

    rows = []
    for name, (tau, use_interval) in policies.items():
        if name == "no deferral":
            d = np.zeros_like(post_test["mu"], dtype=bool)
        elif use_interval:
            d = deferral_mask(post_test, t, tau, cols)
        else:
            d = post_test["sd"] >= tau
            d[:, [k for k in range(d.shape[1]) if k not in cols]] = False
        keep = ~d
        r = {
            "policy": name,
            "tau": tau,
            "deferred_findings_pct": 100 * float(d[:, cols].mean()),
            "studies_with_any_deferral_pct": 100 * float(d[:, cols].any(axis=1).mean()),
            "coverage": float(keep[:, cols].mean()),
        }
        sub = np.zeros_like(m)
        sub[:, cols] = m[:, cols]
        r.update({f"retained_{k}": v for k, v in retained_metrics(y, post_test["mu"], sub, t, keep).items()})
        dm = (sub > 0) & d
        r["deferred_accuracy"] = (
            float(((post_test["mu"] >= t) == (y > 0))[dm].mean()) if dm.any() else float("nan")
        )
        rows.append(r)
    pol = pd.DataFrame(rows)
    pol.to_csv(out / "deferral_policies.csv", index=False)

    # coverage table: rank by the policy score z = |mu - t| / sigma, cut-offs from val
    def z(post):
        return np.abs(post["mu"] - t) / np.maximum(post["sd"], 1e-6)

    cov_rows = []
    for score_name, fn in (
        ("z-score", lambda p: -z(p)),
        ("sigma", lambda p: p["sd"]),
        ("1-confidence", None),
    ):
        if score_name == "1-confidence":
            if test["conf"] is None:
                continue
            s_val, s_test = 1 - val["conf"], 1 - test["conf"]
        else:
            s_val, s_test = fn(post_val), fn(post_test)
        for c in COVERAGE_TARGETS:
            cut = np.inf if c >= 1 else float(np.quantile(s_val[:, cols], c))
            keep = s_test <= cut
            sub = np.zeros_like(m)
            sub[:, cols] = m[:, cols]
            cov_rows.append(
                {
                    "ranking": score_name,
                    "target_coverage": c,
                    "cutoff_from_val": cut,
                    "test_coverage": float(keep[:, cols].mean()),
                    **retained_metrics(y, post_test["mu"], sub, t, keep),
                }
            )
    cov = pd.DataFrame(cov_rows)
    cov.to_csv(out / "coverage_table.csv", index=False)

    # risk-coverage curves on test
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    mm = np.zeros_like(m, dtype=bool)
    mm[:, cols] = m[:, cols] > 0
    err = ((post_test["mu"] >= t) != (y > 0)).astype(float)[mm]
    curves = {
        "z-score (deferral score)": -z(post_test)[mm],
        "posterior std": post_test["sd"][mm],
        "threshold margin": -np.abs(post_test["mu"] - t)[mm],
    }
    if test["conf"] is not None:
        curves["1 - confidence head"] = (1 - test["conf"])[mm]
    fig, ax = plt.subplots(figsize=(3.4, 2.7))
    grid = np.linspace(0.1, 1.0, 46)
    for (name, s), style in zip(curves.items(), ("-", "--", ":", "-.")):
        ax.plot(grid * 100, [100 * (1 - risk_at(err, s, g)) for g in grid], style, lw=1.3, label=name)
    ax.axhline(100 * (1 - err.mean()), color="0.5", lw=0.8, label="random deferral")
    ax.set_xlabel("Coverage (%)", fontsize=8)
    ax.set_ylabel("Accuracy on retained (%)", fontsize=8)
    ax.legend(fontsize=6, frameon=False)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out / "fig_risk_coverage.png", dpi=300)
    plt.close(fig)
    return pol, cov, taus


def mc_sweep(labels, val, test, thresholds, T, out: Path, repeats: int = 5, seed: int = 0):
    """R6: how S changes discrimination, calibration, uncertainty quality, deferral."""
    s_max = test["mc"].shape[0]
    t = thr_vec(labels, thresholds)
    cols = [k for k, lab in enumerate(labels) if lab != "No Finding"]
    y, m = masks(test["raw"])
    keep = m > 0
    ref = posterior(test["mc"], T)
    rng = np.random.default_rng(seed)
    rows = []
    for S in [s for s in S_GRID if s <= s_max]:
        reps = 1 if S == s_max else repeats
        for r in range(reps):
            idx = np.sort(rng.choice(s_max, size=S, replace=False))
            pv, pt = posterior(val["mc"], T, idx=idx), posterior(test["mc"], T, idx=idx)
            err = ((pt["mu"] >= t) != (y > 0)).astype(float)
            tau, _ = choose_tau(pv, t, cols, 0.90)
            d = deferral_mask(pt, t, tau, cols)
            rows.append(
                {
                    "S": S,
                    "repeat": r,
                    "mean_auc": macro_auc(y, pt["mu"], m),
                    "ece": ece_brier(y, pt["mu"], m)["ece"],
                    "sigma_error_auroc_macro": float(
                        np.nanmean([binary_auroc(err[keep[:, k], k], pt["sd"][keep[:, k], k]) for k in range(len(labels))])
                    )
                    if S > 1
                    else float("nan"),
                    "sigma_rank_corr_vs_ref": float(spearmanr(pt["sd"].ravel(), ref["sd"].ravel())[0]) if S > 1 else float("nan"),
                    "mean_sigma": float(pt["sd"].mean()),
                    "mean_ci_width": float((pt["hi"] - pt["lo"]).mean()),
                    "deferred_pct_tau90": 100 * float(d[:, cols].mean()),
                }
            )
    df = pd.DataFrame(rows)
    agg = df.groupby("S").agg(["mean", "std"]).drop(columns="repeat")
    agg.columns = [f"{a}_{b}" for a, b in agg.columns]
    agg = agg.reset_index()
    agg.to_csv(out / "mc_sweep.csv", index=False)
    df.to_csv(out / "mc_sweep_raw.csv", index=False)
    return agg


# ---------------------------------------------------------------------------
# Grad-CAM panel
# ---------------------------------------------------------------------------
def denorm(tensor: torch.Tensor) -> np.ndarray:
    x = tensor.cpu().numpy().transpose(1, 2, 0) * np.array(IMAGENET_STD) + np.array(IMAGENET_MEAN)
    return np.uint8(np.clip(x, 0, 1) * 255)


def gradcam_panel(model, ds, labels, post, raw, thresholds, tau, device, out: Path):
    """R8: one row per finding; columns = TP, FP, FN, deferred.

    Selection is rule-based so the examples are not hand-picked: TP is the
    highest-probability retained true positive, FP the highest-probability
    retained false positive, FN the lowest-probability retained false negative,
    and the deferred case the highest-sigma deferred study for that finding.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = thr_vec(labels, thresholds)
    cols = [k for k, lab in enumerate(labels) if lab != "No Finding"]
    d = deferral_mask(post, t, tau, cols)
    mu, sd = post["mu"], post["sd"]
    chosen, rows = [], []
    for name in CAM_FINDINGS:
        if name not in labels:
            continue
        k = labels.index(name)
        pos, neg = raw[:, k] == 1, raw[:, k] == 0
        hi_ = mu[:, k] >= t[k]
        nd = ~d[:, k]
        picks = {}
        for cat, sel, key in (
            ("True positive", pos & hi_ & nd, -mu[:, k]),
            ("False positive", neg & hi_ & nd, -mu[:, k]),
            ("False negative", pos & ~hi_ & nd, mu[:, k]),
            ("Deferred", d[:, k], -sd[:, k]),
        ):
            ids = np.where(sel)[0]
            if len(ids):
                picks[cat] = int(ids[np.argmin(key[ids])])
        if picks:
            chosen.append((name, k, picks))
    # prefer findings with all four categories, keep the CAM_FINDINGS order otherwise
    chosen = sorted(chosen, key=lambda c: -len(c[2]))[:3]
    if not chosen:
        log.warning("no Grad-CAM candidates")
        return []

    cats = ["True positive", "False positive", "False negative", "Deferred"]
    fig, axes = plt.subplots(len(chosen), 4, figsize=(7.2, 1.95 * len(chosen) + 0.3))
    axes = np.atleast_2d(axes)
    with GradCAM(model) as cam:
        for r, (name, k, picks) in enumerate(chosen):
            for c, cat in enumerate(cats):
                if cat not in picks:
                    axes[r, c].axis("off")
                    axes[r, c].text(0.5, 0.5, "no case", ha="center", fontsize=7)
                    continue
                i = picks[cat]
                x = ds[i]["image"].unsqueeze(0).to(device)
                hm = cam.generate(x, [k])[k]
                img = denorm(ds[i]["image"])
                ax = axes[r, c]
                ax.imshow(overlay_heatmap(img, hm, alpha=0.38))
                ax.set_xticks([])
                ax.set_yticks([])
                lab = {1.0: "pos", 0.0: "neg", -1.0: "uncertain"}[float(raw[i, k])]
                ax.set_xlabel(f"label={lab}  p={mu[i, k]:.2f}  σ={sd[i, k]:.3f}", fontsize=6)
                if r == 0:
                    ax.set_title(cat, fontsize=8)
                if c == 0:
                    ax.set_ylabel(name, fontsize=8)
                rows.append(
                    {
                        "finding": name,
                        "category": cat,
                        "index": i,
                        "path": ds.df.iloc[i][ds.path_column],
                        "label": lab,
                        "p": float(mu[i, k]),
                        "sigma": float(sd[i, k]),
                        "threshold": float(t[k]),
                    }
                )
    fig.tight_layout()
    fig.savefig(out / "fig_gradcam.png", dpi=300)
    plt.close(fig)
    pd.DataFrame(rows).to_csv(out / "gradcam_selection.csv", index=False)
    return rows


# ---------------------------------------------------------------------------
# latency / throughput / resources
# ---------------------------------------------------------------------------
def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def hardware(device) -> dict:
    info = {
        "torch": torch.__version__,
        "python": platform.python_version(),
        "cpu": platform.processor() or platform.machine(),
        "cpu_count": os.cpu_count(),
    }
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                info["cpu"] = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    if device.type == "cuda":
        p = torch.cuda.get_device_properties(device)
        info["gpu"] = p.name
        info["gpu_memory_gb"] = round(p.total_memory / 1e9, 1)
    return info


def benchmark(ckpt_path, image_paths, device, out: Path, n_images: int, s_list, batch_sizes):
    """R11: per-stage latency of the deployed predictor, MC cost vs S, throughput."""
    from radiscan.inference.predictor import RadiScanPredictor

    pred = RadiScanPredictor(checkpoint=ckpt_path, device=str(device), mc_samples=20)
    model = pred.model
    timings = []
    paths = image_paths[: n_images + 3]
    for j, pth in enumerate(paths):
        _sync(device)
        t0 = time.perf_counter()
        view, tensor, meta, display = pred.prepare(pth)
        _sync(device)
        t1 = time.perf_counter()
        unc = predict_with_uncertainty(model, tensor, n_samples=20, tta_hflip=True, calibrated=True)
        _sync(device)
        t2 = time.perf_counter()
        ranked = np.argsort(-unc["mean"][0])[:3].tolist()
        with GradCAM(model) as cam:
            cam.generate(tensor, ranked, output_size=view.shape[:2])
        _sync(device)
        t3 = time.perf_counter()
        pred.reporter.generate("bench", pred.labels, {k: unc[k] for k in ("mean", "std", "ci_low", "ci_high", "entropy")})
        t4 = time.perf_counter()
        if j >= 3:  # warm-up
            timings.append(
                {
                    "preprocess_ms": 1e3 * (t1 - t0),
                    "mc_inference_ms": 1e3 * (t2 - t1),
                    "gradcam_top3_ms": 1e3 * (t3 - t2),
                    "report_ms": 1e3 * (t4 - t3),
                    "total_ms": 1e3 * (t4 - t0),
                }
            )
    tdf = pd.DataFrame(timings)
    stage = pd.DataFrame(
        {"median_ms": tdf.median(), "p95_ms": tdf.quantile(0.95), "mean_ms": tdf.mean()}
    )
    stage.index.name = "stage"
    stage.to_csv(out / f"latency_stages_{device.type}.csv")

    # MC cost vs S at batch 1: deployed (full forward per pass) vs trunk-cached
    _, tensor, _, _ = pred.prepare(paths[0])
    rows = []
    reps = 20 if device.type == "cuda" else 3
    for S in s_list:
        for mode in ("deployed", "cached"):
            ts = []
            for rep in range(reps + 2):
                _sync(device)
                t0 = time.perf_counter()
                with torch.no_grad():
                    if mode == "deployed":
                        predict_with_uncertainty(model, tensor, n_samples=S, tta_hflip=True)
                    else:
                        model.eval()
                        feats = [trunk_features(model, v) for v in (tensor, torch.flip(tensor, dims=[3]))]
                        enable_mc_dropout(model)
                        for _ in range(S):
                            [torch.sigmoid(head_outputs(model, f)[0] / model.temperature) for f in feats]
                        model.eval()
                _sync(device)
                if rep >= 2:
                    ts.append(1e3 * (time.perf_counter() - t0))
            rows.append({"S": S, "mode": mode, "median_ms": float(np.median(ts)), "p95_ms": float(np.percentile(ts, 95))})
    pd.DataFrame(rows).to_csv(out / f"latency_vs_S_{device.type}.csv", index=False)

    # throughput (deployed path, S=20, TTA)
    thr = []
    for bs in batch_sizes:
        x = tensor.repeat(bs, 1, 1, 1)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        n_iter = 6 if device.type == "cuda" else 2
        ts = []
        for it in range(n_iter + 1):
            _sync(device)
            t0 = time.perf_counter()
            predict_with_uncertainty(model, x, n_samples=20, tta_hflip=True)
            _sync(device)
            if it >= 1:
                ts.append(time.perf_counter() - t0)
        r = {"batch_size": bs, "images_per_s": bs / float(np.median(ts)), "ms_per_image": 1e3 * float(np.median(ts)) / bs}
        if device.type == "cuda":
            r["peak_gpu_mem_mb"] = torch.cuda.max_memory_allocated(device) / 2**20
        thr.append(r)
    pd.DataFrame(thr).to_csv(out / f"throughput_{device.type}.csv", index=False)

    res = {"hardware": hardware(device)}
    try:
        import psutil

        res["process_rss_mb"] = psutil.Process().memory_info().rss / 2**20
    except ImportError:
        pass
    res["checkpoint_mb"] = Path(ckpt_path).stat().st_size / 2**20
    res["parameters_m"] = sum(p.numel() for p in model.parameters()) / 1e6
    (out / f"resources_{device.type}.json").write_text(json.dumps(res, indent=2))
    del pred
    return stage, pd.DataFrame(rows), pd.DataFrame(thr), res


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", default="results/reviewer")
    ap.add_argument("--device", default=None)
    ap.add_argument("--s-max", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--max-images", type=int, default=None, help="debug only")
    ap.add_argument("--bench-images", type=int, default=30)
    ap.add_argument("--cpu-bench-images", type=int, default=10)
    ap.add_argument("--skip-bench", action="store_true")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = Config.from_yaml(args.config)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.manual_seed(0)
    np.random.seed(0)

    model, meta = load_checkpoint(args.checkpoint, device)
    labels = list(model.labels)
    thresholds = dict(meta.get("thresholds", {}))
    T = float(model.temperature.item())
    log.info("checkpoint %s | T=%.3f | device %s", args.checkpoint, T, device)

    splits = split_statistics(cfg, labels, out)
    log.info("splits: %s", json.dumps(splits))

    val_ds = make_dataset(cfg, cfg.data.val_csv, labels, args.max_images)
    test_ds = make_dataset(cfg, cfg.data.test_csv, labels, args.max_images)
    cache = out / "cache.npz"
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        val = {k[4:]: z[k] for k in z.files if k.startswith("val_")}
        test = {k[5:]: z[k] for k in z.files if k.startswith("test_")}
        for d in (val, test):
            d["conf"] = None if d["conf"].ndim == 0 else d["conf"]
    else:
        val = collect(model, val_ds, device, args.s_max, args.batch_size, args.workers)
        test = collect(model, test_ds, device, args.s_max, args.batch_size, args.workers)
        np.savez_compressed(
            cache,
            **{f"val_{k}": (np.array(None) if v is None else np.asarray(v)) for k, v in val.items()},
            **{f"test_{k}": (np.array(None) if v is None else np.asarray(v)) for k, v in test.items()},
        )

    # equivalence of the cached estimator with the deployed one, on a few images
    xb = torch.stack([test_ds[i]["image"] for i in range(min(4, len(test_ds)))]).to(device)
    torch.manual_seed(1)
    dep = predict_with_uncertainty(model, xb, n_samples=args.s_max, tta_hflip=True)
    mine = posterior(test["mc"][:, :, : len(xb)], T)
    equiv = {
        "max_abs_diff_mean": float(np.abs(dep["mean"] - mine["mu"]).max()),
        "spearman_std": float(spearmanr(dep["std"].ravel(), mine["sd"].ravel())[0]),
    }

    S = 20
    idx20 = np.arange(min(S, args.s_max))
    post_test = posterior(test["mc"], T, idx=idx20)
    post_val = posterior(val["mc"], T, idx=idx20)
    y, m = masks(test["raw"])

    # R1 ------------------------------------------------------------------
    rows = table1(labels, post_test, test["raw"], thresholds)
    pd.DataFrame(rows).to_csv(out / "table1_per_class.csv", index=False)
    overall = summarize(rows)
    comp = summarize(rows, subset=CHEXPERT_COMPETITION_LABELS)
    lsr_rows = per_class_metrics(lsr_targets(test["raw"]), post_test["mu"], labels, thresholds=thresholds)
    pd.DataFrame(lsr_rows).to_csv(out / "table1_lsr_protocol.csv", index=False)
    lsr_overall = summarize(lsr_rows)

    # R4 ------------------------------------------------------------------
    calib, temps = calibration_experiment(labels, val, test, T, out, S=len(idx20))
    # R2 / R7 ---------------------------------------------------------------
    unc_df, q_sd, q_c, extra = uncertainty_experiment(labels, post_test, test, thresholds, out)
    # R3 / R10 --------------------------------------------------------------
    pol, cov, taus = deferral_experiment(labels, post_val, post_test, val, test, thresholds, out)
    # R6 ------------------------------------------------------------------
    sweep = mc_sweep(labels, val, test, thresholds, T, out)
    # R8 ------------------------------------------------------------------
    tau90 = taus[0.90]["tau"]
    cam_rows = gradcam_panel(model, test_ds, labels, post_test, test["raw"], thresholds, tau90, device, out)

    results = {
        "checkpoint": str(args.checkpoint),
        "model_version": meta.get("model_version"),
        "epoch": meta.get("epoch"),
        "temperature": T,
        "thresholds": thresholds,
        "n_val": int(len(val["raw"])),
        "n_test": int(len(test["raw"])),
        "splits": splits,
        "overall": overall,
        "chexpert_competition_5": comp,
        "lsr_protocol_overall": lsr_overall,
        "calibration": calib.to_dict(orient="records"),
        "temperatures": temps,
        "uncertainty": unc_df.to_dict(orient="records"),
        "quartiles_sigma": q_sd,
        "quartiles_confidence": q_c,
        "uncertainty_extra": extra,
        "deferral_policies": pol.to_dict(orient="records"),
        "coverage": cov.to_dict(orient="records"),
        "tau": {str(k): v for k, v in taus.items()},
        "mc_sweep": sweep.to_dict(orient="records"),
        "gradcam": cam_rows,
        "estimator_equivalence": equiv,
        "train_meta": {k: meta.get(k) for k in ("epoch", "metrics", "model_version")},
    }

    # R11 -----------------------------------------------------------------
    if not args.skip_bench:
        paths = test["paths"].tolist() if hasattr(test["paths"], "tolist") else list(test["paths"])
        bench = {}
        devs = [device] + ([torch.device("cpu")] if device.type == "cuda" else [])
        for dv in devs:
            n = args.bench_images if dv.type == "cuda" else args.cpu_bench_images
            s_list = [1, 5, 10, 20, 30, 50] if dv.type == "cuda" else [1, 5, 20]
            bss = [1, 8, 32] if dv.type == "cuda" else [1]
            st, lat, thr, res = benchmark(args.checkpoint, paths, dv, out, n, s_list, bss)
            bench[dv.type] = {
                "stages": st.reset_index().to_dict(orient="records"),
                "latency_vs_S": lat.to_dict(orient="records"),
                "throughput": thr.to_dict(orient="records"),
                "resources": res,
            }
        results["benchmark"] = bench

    (out / "results.json").write_text(json.dumps(results, indent=2, default=float))
    log.info("done -> %s", out)
    print(json.dumps({"mean_auc": overall["mean_auc"], "lsr_protocol_mean_auc": lsr_overall["mean_auc"], "equivalence": equiv}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
