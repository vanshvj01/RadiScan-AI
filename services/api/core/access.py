"""Who may see which scan. Every route asks this module; none decides alone.

The rules, in full:

* The **uploader** of a scan can always see it.
* A **patient** can see every scan in their own history, whoever uploaded it.
* A **doctor** can see a patient's history only while that patient has an
  active :class:`AccessGrant` for them. Revoking the grant ends access at once
  (scans the doctor uploaded themselves stay visible to them).
* An **admin** can see everything.

Note what is absent: no role, on its own, opens anyone's history. Anyone can
register as a doctor, so a rule like "doctors see all patients" would let
anyone read everything. Consent is the gate; the role only decides which side
of a grant a user can stand on.
"""

from __future__ import annotations

from fastapi import HTTPException, status
from sqlalchemy import or_, select
from sqlalchemy.orm import Query, Session

from services.api.db.models import AccessGrant, Role, Scan, User


def active_grant(db: Session, patient_id: int, doctor_id: int) -> AccessGrant | None:
    return (
        db.query(AccessGrant)
        .filter(
            AccessGrant.patient_id == patient_id,
            AccessGrant.doctor_id == doctor_id,
            AccessGrant.revoked_at.is_(None),
        )
        .first()
    )


def can_view_patient(db: Session, user: User, patient_id: int) -> bool:
    if user.role == Role.admin or user.id == patient_id:
        return True
    return user.is_clinician and active_grant(db, patient_id, user.id) is not None


def can_view_scan(db: Session, user: User, scan: Scan) -> bool:
    if user.role == Role.admin or scan.user_id == user.id:
        return True
    return scan.patient_id is not None and can_view_patient(db, user, scan.patient_id)


def viewable_scan(scan_id: int, db: Session, user: User) -> Scan:
    scan = db.get(Scan, scan_id)
    if scan is None or not can_view_scan(db, user, scan):
        # 404 rather than 403: never confirm that a scan exists to someone who
        # may not see it.
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"Scan {scan_id} not found")
    return scan


def visible_scans(db: Session, user: User) -> Query:
    """Every scan this user may see, as a query to filter and page further."""
    query = db.query(Scan)
    if user.role == Role.admin:
        return query
    shared_patients = select(AccessGrant.patient_id).where(
        AccessGrant.doctor_id == user.id, AccessGrant.revoked_at.is_(None)
    )
    return query.filter(
        or_(
            Scan.user_id == user.id,
            Scan.patient_id == user.id,
            Scan.patient_id.in_(shared_patients),
        )
    )
