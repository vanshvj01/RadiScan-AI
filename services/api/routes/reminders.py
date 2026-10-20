"""Follow-up reminders: "repeat chest X-ray in 6 weeks".

A doctor creates one for a patient who shares their history with them, either
directly or by signing a report with a follow-up interval. The patient sees
their reminders, overdue ones first, and marks them done or dismissed.

In-app only. Email or SMS delivery needs a mail or SMS provider and the
patient's consent to be contacted; neither is part of this prototype.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

from fastapi import APIRouter, Depends, HTTPException, Request, status
from sqlalchemy.orm import Session

from services.api.core.access import can_view_patient, can_view_scan
from services.api.core.clinical import require_clinician
from services.api.core.security import current_user
from services.api.db.models import AuditLog, Reminder, ReminderStatus, Scan, User
from services.api.db.session import get_db
from services.api.schemas.schemas import ReminderIn, ReminderOut, ReminderUpdate

router = APIRouter(prefix="/reminders", tags=["reminders"])


def _out(r: Reminder) -> ReminderOut:
    days = (r.due_date - date.today()).days
    return ReminderOut(
        id=r.id,
        patient_id=r.patient_id,
        patient_username=r.patient.username,
        created_by_username=r.created_by.username,
        created_by_name=r.created_by.full_name,
        scan_id=r.scan_id,
        due_date=r.due_date,
        note=r.note,
        status=r.status,
        overdue=r.status == ReminderStatus.pending and days < 0,
        days_until=days,
        created_at=r.created_at,
    )


def _audit(db: Session, user: User, action: str, resource: str, request: Request) -> None:
    db.add(
        AuditLog(
            user_id=user.id,
            action=action,
            resource=resource,
            ip_address=request.client.host if request.client else None,
        )
    )


@router.get("", response_model=list[ReminderOut])
def list_reminders(
    db: Session = Depends(get_db), user: User = Depends(current_user)
) -> list[ReminderOut]:
    """Patient: my reminders. Doctor: reminders I set for patients I can still see."""
    query = db.query(Reminder)
    if user.is_patient:
        rows = query.filter(Reminder.patient_id == user.id).all()
    else:
        rows = [
            r
            for r in query.filter(Reminder.created_by_id == user.id).all()
            if can_view_patient(db, user, r.patient_id)
        ]
    # Pending first (soonest due on top), then completed ones, newest first.
    rows.sort(
        key=lambda r: (
            r.status != ReminderStatus.pending,
            r.due_date if r.status == ReminderStatus.pending else date.max - r.due_date,
        )
    )
    return [_out(r) for r in rows]


@router.post("", response_model=ReminderOut, status_code=status.HTTP_201_CREATED)
def create_reminder(
    payload: ReminderIn,
    request: Request,
    db: Session = Depends(get_db),
    user: User = Depends(current_user),
) -> ReminderOut:
    require_clinician(user)
    patient = db.get(User, payload.patient_id)
    if patient is None or not patient.is_patient or not can_view_patient(db, user, patient.id):
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Patient not found")
    if payload.scan_id is not None:
        scan = db.get(Scan, payload.scan_id)
        if scan is None or scan.patient_id != patient.id or not can_view_scan(db, user, scan):
            raise HTTPException(status.HTTP_404_NOT_FOUND, f"Scan {payload.scan_id} not found")
    if payload.due_date is None and payload.weeks is None:
        raise HTTPException(422, "Give a due_date or weeks")
    due = payload.due_date or (date.today() + timedelta(weeks=payload.weeks or 0))

    reminder = Reminder(
        patient_id=patient.id,
        created_by_id=user.id,
        scan_id=payload.scan_id,
        due_date=due,
        note=payload.note.strip(),
    )
    db.add(reminder)
    db.flush()
    _audit(db, user, "reminder.create", f"patient:{patient.id}", request)
    db.commit()
    db.refresh(reminder)
    return _out(reminder)


@router.patch("/{reminder_id}", response_model=ReminderOut)
def update_reminder(
    reminder_id: int,
    payload: ReminderUpdate,
    request: Request,
    db: Session = Depends(get_db),
    user: User = Depends(current_user),
) -> ReminderOut:
    """The patient, or the doctor who set it, marks a reminder done or dismissed."""
    r = db.get(Reminder, reminder_id)
    if r is None or user.id not in (r.patient_id, r.created_by_id):
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Reminder not found")
    r.status = payload.status
    r.completed_at = (
        None if payload.status == ReminderStatus.pending else datetime.now(timezone.utc)
    )
    _audit(db, user, f"reminder.{payload.status.value}", f"reminder:{r.id}", request)
    db.commit()
    db.refresh(r)
    return _out(r)
