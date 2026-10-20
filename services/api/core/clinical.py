"""Clinical-workflow rules shared by the review, worklist, report and summary routes.

Kept in one module so that "is this finding deferred?" and "how urgent is this
study?" have exactly one answer everywhere -- the worklist, the review screen
and the agreement statistics must never disagree about what the model said.
"""

from __future__ import annotations

from typing import Any

from fastapi import HTTPException, status
from sqlalchemy.orm import Session

from radiscan.reporting.generator import CRITICAL_FINDINGS
from services.api.db.models import DoctorReport, Prediction, User

# Same rule as radiscan.reporting.generator.ReportGenerator: a finding whose
# credible interval straddles its own threshold, or whose posterior spread is
# too wide, is not a call the system makes alone.
UNCERTAINTY_DEFER = 0.15
NO_FINDING = "No Finding"

RISK_RANK = {"critical": 3, "high": 2, "moderate": 1, "low": 0}


def require_clinician(user: User) -> None:
    if not user.is_clinician:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Only doctors can do this")


def is_deferred(label: str, prob: float, std: float, lo: float, hi: float, thr: float) -> bool:
    if label == NO_FINDING:
        return False
    return (lo < thr <= hi) or std >= UNCERTAINTY_DEFER


def finding_rows(prediction: Prediction) -> list[dict[str, Any]]:
    """Every label with the model's call on it: positive, negative or deferred."""
    rows = []
    for label in prediction.label_set:
        prob = float(prediction.probabilities.get(label, 0.0))
        std = float(prediction.uncertainties.get(label, 0.0))
        lo, hi = prediction.confidence_intervals.get(label, [prob, prob])
        thr = float(prediction.thresholds.get(label, 0.5))
        deferred = is_deferred(label, prob, std, float(lo), float(hi), thr)
        rows.append(
            {
                "label": label,
                "probability": prob,
                "uncertainty": std,
                "ci_low": float(lo),
                "ci_high": float(hi),
                "threshold": thr,
                "positive": prob >= thr and not deferred,
                "deferred": deferred,
            }
        )
    return rows


def urgency(prediction: Prediction) -> dict[str, Any]:
    """What puts a study near the top of a reporting queue, and why."""
    rows = finding_rows(prediction)
    critical = [r["label"] for r in rows if r["positive"] and r["label"] in CRITICAL_FINDINGS]
    deferred = [r["label"] for r in rows if r["deferred"]]
    quality = (prediction.quality or {}).get("status")
    reasons = []
    if critical:
        reasons.append("possible " + ", ".join(critical))
    if deferred:
        reasons.append(f"{len(deferred)} indeterminate finding(s)")
    if quality == "poor":
        reasons.append("poor image quality")
    return {
        "risk_rank": RISK_RANK.get(prediction.risk_level, 0),
        "critical_findings": critical,
        "deferred_findings": deferred,
        "quality": quality,
        "reasons": reasons,
    }


def latest_report(db: Session, scan_id: int) -> DoctorReport | None:
    return (
        db.query(DoctorReport)
        .filter(DoctorReport.scan_id == scan_id)
        .order_by(DoctorReport.version.desc())
        .first()
    )
