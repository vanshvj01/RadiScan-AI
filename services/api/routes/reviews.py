"""Clinician review of the model's calls, finding by finding.

This is what turns the deferral policy from a displayed flag into a clinical
step: an indeterminate finding is shown to a doctor, who confirms it, rejects
it, or marks it unsure. Every verdict is stored with a snapshot of what the
model said, which makes two numbers measurable from real use:

* how often the doctor agreed with the calls the model made on its own, and
* how often deferred findings turned out to be present.

Only clinicians who can see a scan may review it. Patients cannot.
"""

from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from sqlalchemy.orm import Session

from services.api.core.access import viewable_scan
from services.api.core.clinical import finding_rows, require_clinician
from services.api.core.security import current_user
from services.api.db.models import (
    AuditLog,
    FindingReview,
    Prediction,
    ReviewDecision,
    Role,
    User,
)
from services.api.db.session import get_db
from services.api.schemas.schemas import (
    ReviewGroupStats,
    ReviewOut,
    ReviewStats,
    ReviewSubmit,
)

router = APIRouter(prefix="/reviews", tags=["reviews"])


def _out(r: FindingReview) -> ReviewOut:
    return ReviewOut(
        label=r.label,
        decision=r.decision,
        note=r.note,
        was_deferred=r.was_deferred,
        model_positive=r.model_positive,
        model_probability=r.model_probability,
        reviewer_username=r.reviewer.username,
        updated_at=r.updated_at,
    )


@router.get("/stats", response_model=ReviewStats)
def review_stats(
    scope: str = Query("mine", pattern="^(mine|all)$"),
    db: Session = Depends(get_db),
    user: User = Depends(current_user),
) -> ReviewStats:
    """Agreement between the model and clinician review.

    ``scope=all`` (every reviewer) is for admins; anyone else gets their own.
    """
    require_clinician(user)
    if scope == "all" and user.role != Role.admin:
        scope = "mine"
    query = db.query(FindingReview)
    if scope == "mine":
        query = query.filter(FindingReview.reviewer_id == user.id)
    rows = query.all()

    def group(items: list[FindingReview]) -> ReviewGroupStats:
        return ReviewGroupStats(
            reviewed=len(items),
            confirmed=sum(r.decision == ReviewDecision.confirmed for r in items),
            rejected=sum(r.decision == ReviewDecision.rejected for r in items),
            unsure=sum(r.decision == ReviewDecision.unsure for r in items),
        )

    calls = [r for r in rows if not r.was_deferred]
    deferred = [r for r in rows if r.was_deferred]

    decided_calls = [r for r in calls if r.decision != ReviewDecision.unsure]
    agreed = sum(
        (r.model_positive and r.decision == ReviewDecision.confirmed)
        or (not r.model_positive and r.decision == ReviewDecision.rejected)
        for r in decided_calls
    )
    decided_deferred = [r for r in deferred if r.decision != ReviewDecision.unsure]
    present = sum(r.decision == ReviewDecision.confirmed for r in decided_deferred)

    return ReviewStats(
        scope=scope,  # type: ignore[arg-type]
        scans_reviewed=len({r.scan_id for r in rows}),
        model_calls=group(calls),
        model_agreement=(agreed / len(decided_calls)) if decided_calls else None,
        deferred=group(deferred),
        deferred_present_rate=(present / len(decided_deferred)) if decided_deferred else None,
    )


@router.get("/{scan_id}", response_model=list[ReviewOut])
def get_reviews(
    scan_id: int, db: Session = Depends(get_db), user: User = Depends(current_user)
) -> list[ReviewOut]:
    """All clinician verdicts on this scan (every reviewer), for doctors only."""
    require_clinician(user)
    scan = viewable_scan(scan_id, db, user)
    rows = (
        db.query(FindingReview)
        .filter(FindingReview.scan_id == scan.id)
        .order_by(FindingReview.label)
        .all()
    )
    return [_out(r) for r in rows]


@router.put("/{scan_id}", response_model=list[ReviewOut])
def submit_reviews(
    scan_id: int,
    payload: ReviewSubmit,
    request: Request,
    db: Session = Depends(get_db),
    user: User = Depends(current_user),
) -> list[ReviewOut]:
    """Record or update my verdicts on some of this scan's findings."""
    require_clinician(user)
    scan = viewable_scan(scan_id, db, user)
    prediction = db.query(Prediction).filter(Prediction.scan_id == scan.id).first()
    if prediction is None:
        raise HTTPException(status.HTTP_425_TOO_EARLY, "This scan has no results to review yet")

    model = {r["label"]: r for r in finding_rows(prediction)}
    unknown = [i.label for i in payload.reviews if i.label not in model]
    if unknown:
        raise HTTPException(422, f"Not a label of this model: {unknown}")

    now = datetime.now(timezone.utc)
    for item in payload.reviews:
        m = model[item.label]
        row = (
            db.query(FindingReview)
            .filter(
                FindingReview.scan_id == scan.id,
                FindingReview.label == item.label,
                FindingReview.reviewer_id == user.id,
            )
            .first()
        )
        if row is None:
            row = FindingReview(
                scan_id=scan.id,
                reviewer_id=user.id,
                label=item.label,
                was_deferred=m["deferred"],
                model_positive=m["positive"],
                model_probability=m["probability"],
                created_at=now,
            )
            db.add(row)
        row.decision = item.decision
        row.note = (item.note or "").strip() or None
        row.updated_at = now
        db.add(
            AuditLog(
                user_id=user.id,
                action="review.finding",
                resource=f"scan:{scan.uuid}",
                detail=f"{item.label}={item.decision.value}"
                + (" (deferred)" if m["deferred"] else ""),
                ip_address=request.client.host if request.client else None,
            )
        )
    db.commit()
    return get_reviews(scan_id, db, user)
