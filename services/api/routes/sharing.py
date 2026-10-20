"""Patient-controlled sharing of scan history with doctors.

A patient grants a named doctor access to their history and can revoke it at
any time. A doctor sees the list of patients who have shared with them and can
open each one's history for reference when reading a new scan. Every grant,
revocation and history view is written to the audit log.
"""

from __future__ import annotations

import hashlib
import secrets
from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends, HTTPException, Request, status
from sqlalchemy.orm import Session

from services.api.core.access import active_grant, can_view_patient
from services.api.core.security import current_user
from services.api.db.models import AccessGrant, AuditLog, Prediction, Scan, ShareCode, User
from services.api.db.session import get_db
from services.api.schemas.schemas import (
    GrantCreate,
    GrantOut,
    HistoryItem,
    PatientHistory,
    ShareCodeOut,
    ShareCodeRedeem,
)

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


# --------------------------------------------------------------------------
# One-time share codes (typed, or scanned as a QR code)
# --------------------------------------------------------------------------
# No 0/O, 1/I/L: the code is read aloud and typed by hand in clinics.
CODE_ALPHABET = "ABCDEFGHJKMNPQRSTUVWXYZ23456789"
CODE_LENGTH = 8  # 31^8 ~ 8.5e11 combinations
CODE_TTL = timedelta(hours=24)
# Guessing: a doctor account that fails this many redemptions in an hour is
# refused further attempts until the hour passes.
MAX_FAILED_PER_HOUR = 10


def _normalise(code: str) -> str:
    return "".join(ch for ch in code.upper() if ch.isalnum())


def _hash(code: str) -> str:
    return hashlib.sha256(_normalise(code).encode()).hexdigest()


def _aware(dt: datetime) -> datetime:
    # SQLite hands back naive datetimes even for timezone=True columns.
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


@router.post("/codes", response_model=ShareCodeOut, status_code=status.HTTP_201_CREATED)
def create_share_code(
    request: Request, db: Session = Depends(get_db), user: User = Depends(current_user)
) -> ShareCodeOut:
    """Patient: a code to give a doctor in person. Shown once; valid 24 h, one use."""
    if not user.is_patient:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Only patients can create share codes")
    raw = "".join(secrets.choice(CODE_ALPHABET) for _ in range(CODE_LENGTH))
    expires = datetime.now(timezone.utc) + CODE_TTL
    db.add(ShareCode(patient_id=user.id, code_hash=_hash(raw), expires_at=expires))
    _audit(db, user, "sharing.code_create", f"patient:{user.id}", request)
    db.commit()
    return ShareCodeOut(code=f"{raw[:4]}-{raw[4:]}", expires_at=expires)


@router.post("/codes/redeem", response_model=GrantOut, status_code=status.HTTP_201_CREATED)
def redeem_share_code(
    payload: ShareCodeRedeem,
    request: Request,
    db: Session = Depends(get_db),
    user: User = Depends(current_user),
) -> GrantOut:
    """Doctor: redeem a patient's code to gain access to their history."""
    if not user.is_clinician:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Only doctors can redeem share codes")

    hour_ago = datetime.now(timezone.utc) - timedelta(hours=1)
    failures = (
        db.query(AuditLog)
        .filter(
            AuditLog.user_id == user.id,
            AuditLog.action == "sharing.code_failed",
            AuditLog.created_at >= hour_ago,
        )
        .count()
    )
    if failures >= MAX_FAILED_PER_HOUR:
        raise HTTPException(
            status.HTTP_429_TOO_MANY_REQUESTS, "Too many incorrect codes. Try again in an hour."
        )

    now = datetime.now(timezone.utc)
    code = db.query(ShareCode).filter(ShareCode.code_hash == _hash(payload.code)).first()
    if code is None or code.used_at is not None or _aware(code.expires_at) <= now:
        _audit(db, user, "sharing.code_failed", "code", request)
        db.commit()
        raise HTTPException(
            status.HTTP_404_NOT_FOUND, "That code is not valid. It may have expired or been used."
        )

    code.used_at = now
    code.used_by_id = user.id
    grant = active_grant(db, code.patient_id, user.id)
    if grant is None:
        grant = AccessGrant(patient_id=code.patient_id, doctor_id=user.id)
        db.add(grant)
        db.flush()
    _audit(db, user, "sharing.code_redeem", f"patient:{code.patient_id}", request)
    db.commit()
    db.refresh(grant)
    return _grant_out(grant)
