"""The reporting queue: every study a doctor can see, most urgent first.

This is the use the paper's introduction argues for -- ordering a backlog by
clinical urgency so that a pneumothorax is read before a routine follow-up --
rather than autonomous diagnosis. Order, from the top:

1. not yet signed (signed studies are hidden unless asked for),
2. higher risk level,
3. a critical finding called positive (pneumothorax, oedema, ...),
4. more indeterminate findings, then poor image quality,
5. oldest first -- among equals, whoever has waited longest.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from services.api.core.access import visible_scans
from services.api.core.clinical import latest_report, require_clinician, urgency
from services.api.core.security import current_user
from services.api.db.models import FindingReview, Prediction, Scan, ScanStatus, User
from services.api.db.session import get_db
from services.api.schemas.schemas import Worklist, WorklistItem

router = APIRouter(prefix="/worklist", tags=["worklist"])

# A queue, not an archive: the newest few hundred is plenty to sort.
MAX_SCANS = 500


@router.get("", response_model=Worklist)
def worklist(
    include_signed: bool = Query(False),
    limit: int = Query(50, ge=1, le=200),
    db: Session = Depends(get_db),
    user: User = Depends(current_user),
) -> Worklist:
    require_clinician(user)
    scans = (
        visible_scans(db, user)
        .filter(Scan.status == ScanStatus.completed)
        .order_by(Scan.created_at.desc())
        .limit(MAX_SCANS)
        .all()
    )
    ids = [s.id for s in scans]
    preds = {p.scan_id: p for p in db.query(Prediction).filter(Prediction.scan_id.in_(ids)).all()}
    reviewed = {
        sid
        for (sid,) in db.query(FindingReview.scan_id)
        .filter(FindingReview.scan_id.in_(ids), FindingReview.reviewer_id == user.id)
        .distinct()
    }

    items: list[tuple[tuple, WorklistItem]] = []
    pending = 0
    for s in scans:
        p = preds.get(s.id)
        if p is None:
            continue
        report = latest_report(db, s.id)
        report_status = "none" if report is None else ("signed" if report.signed else "draft")
        if report_status != "signed":
            pending += 1
        elif not include_signed:
            continue
        u = urgency(p)
        item = WorklistItem(
            scan_id=s.id,
            created_at=s.created_at,
            filename=s.filename,
            patient_username=s.patient_username,
            patient_ref=s.patient_ref,
            uploaded_by=s.uploaded_by,
            risk_level=p.risk_level,
            top_finding=p.top_finding,
            top_probability=p.top_probability,
            critical_findings=u["critical_findings"],
            deferred_findings=u["deferred_findings"],
            quality=u["quality"],
            reasons=u["reasons"],
            reviewed=s.id in reviewed,
            report_status=report_status,
        )
        key = (
            report_status == "signed",
            -u["risk_rank"],
            -len(u["critical_findings"]),
            -len(u["deferred_findings"]),
            u["quality"] != "poor",
            s.created_at,
        )
        items.append((key, item))

    items.sort(key=lambda kv: kv[0])
    ordered = [i for _, i in items]
    return Worklist(items=ordered[:limit], total=len(ordered), pending=pending)
