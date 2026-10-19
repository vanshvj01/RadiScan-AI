"""Patient-controlled sharing of scan history with doctors.

A patient grants a named doctor access to their history and can revoke it at
any time. A doctor sees the list of patients who have shared with them and can
open each one's history for reference when reading a new scan. Every grant,
revocation and history view is written to the audit log.
"""

from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Request, status
from sqlalchemy.orm import Session

from services.api.core.access import active_grant, can_view_patient
from services.api.core.security import current_user
from services.api.db.models import AccessGrant, AuditLog, Prediction, Scan, User
from services.api.db.session import get_db
from services.api.schemas.schemas import GrantCreate, GrantOut, HistoryItem, PatientHistory

router = APIRouter(prefix="/sharing", tags=["sharing"])


def _audit(db: Session, user: User, action: str, resource: str, request: Request) -> None:
    db.add(
        AuditLog(
            user_id=user.id,
            action=action,
            resource=resource,
            ip_address=request.client.host if request.client else None,
        )
    )


def _grant_out(g: AccessGrant) -> GrantOut:
    return GrantOut(
        id=g.id,
        patient_id=g.patient_id,
        patient_username=g.patient.username,
        patient_name=g.patient.full_name,
        doctor_id=g.doctor_id,
        doctor_username=g.doctor.username,
        doctor_name=g.doctor.full_name,
        created_at=g.created_at,
    )


@router.post("/grants", response_model=GrantOut, status_code=status.HTTP_201_CREATED)
def grant_access(
    payload: GrantCreate,
    request: Request,
    db: Session = Depends(get_db),
    user: User = Depends(current_user),
) -> GrantOut:
    """Patient: share my history with a doctor."""
    if not user.is_patient:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Only patients can share their history")
    doctor = db.query(User).filter(User.username == payload.doctor_username.strip()).first()
    if doctor is None or not doctor.is_clinician or not doctor.is_active:
        raise HTTPException(
            status.HTTP_404_NOT_FOUND, f"No doctor with username {payload.doctor_username!r}"
        )

    existing = active_grant(db, user.id, doctor.id)
    if existing is not None:
        return _grant_out(existing)  # idempotent: sharing twice is not an error

    grant = AccessGrant(patient_id=user.id, doctor_id=doctor.id)
    db.add(grant)
    db.flush()
    _audit(db, user, "sharing.grant", f"doctor:{doctor.id}", request)
    db.commit()
    db.refresh(grant)
    return _grant_out(grant)


@router.get("/grants", response_model=list[GrantOut])
def list_grants(
    db: Session = Depends(get_db), user: User = Depends(current_user)
) -> list[GrantOut]:
    """Patient: doctors I share with. Doctor: patients who share with me."""
    query = db.query(AccessGrant).filter(AccessGrant.revoked_at.is_(None))
    if user.is_patient:
        query = query.filter(AccessGrant.patient_id == user.id)
    else:
        query = query.filter(AccessGrant.doctor_id == user.id)
    return [_grant_out(g) for g in query.order_by(AccessGrant.created_at.desc()).all()]


@router.delete("/grants/{grant_id}", status_code=status.HTTP_204_NO_CONTENT)
def revoke_access(
    grant_id: int,
    request: Request,
    db: Session = Depends(get_db),
    user: User = Depends(current_user),
) -> None:
    """Patient: stop sharing. Takes effect on the doctor's next request."""
    grant = db.get(AccessGrant, grant_id)
    if grant is None or grant.patient_id != user.id or not grant.active:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "No such active share")
    grant.revoked_at = datetime.now(timezone.utc)
    _audit(db, user, "sharing.revoke", f"doctor:{grant.doctor_id}", request)
    db.commit()


@router.get("/patients/{patient_id}/history", response_model=PatientHistory)
def patient_history(
    patient_id: int,
    request: Request,
    db: Session = Depends(get_db),
    user: User = Depends(current_user),
) -> PatientHistory:
    """A patient's scans, newest first, with each one's findings for comparison."""
    patient = db.get(User, patient_id)
    if patient is None or not patient.is_patient or not can_view_patient(db, user, patient_id):
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Patient not found")

    scans = (
        db.query(Scan)
        .filter(Scan.patient_id == patient_id)
        .order_by(Scan.created_at.desc())
        .limit(100)
        .all()
    )
    preds = {
        p.scan_id: p
        for p in db.query(Prediction).filter(Prediction.scan_id.in_([s.id for s in scans])).all()
    }
    items = []
    for s in scans:
        p = preds.get(s.id)
        items.append(
            HistoryItem(
                scan_id=s.id,
                created_at=s.created_at,
                filename=s.filename,
                uploaded_by=s.uploaded_by or "?",
                status=s.status,
                risk_level=p.risk_level if p else None,
                top_finding=p.top_finding if p else None,
                top_probability=p.top_probability if p else None,
                model_version=p.model_version if p else None,
                probabilities=(
                    {k: float(v) for k, v in (p.probabilities or {}).items()} if p else {}
                ),
            )
        )

    if user.id != patient_id:
        # A doctor reading someone's record is exactly what an audit trail is for.
        _audit(db, user, "history.view", f"patient:{patient_id}", request)
        db.commit()

    return PatientHistory(
        patient_id=patient.id,
        patient_username=patient.username,
        patient_name=patient.full_name,
        items=items,
    )
