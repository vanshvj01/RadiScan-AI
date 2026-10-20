"""SQLAlchemy 2.0 ORM models."""

from __future__ import annotations

import enum
from datetime import date, datetime, timezone

from sqlalchemy import (
    JSON,
    Boolean,
    Date,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class Role(str, enum.Enum):
    admin = "admin"
    radiologist = "radiologist"  # pre-dates the doctor/patient split; treated as a doctor
    viewer = "viewer"
    doctor = "doctor"
    patient = "patient"


# Roles that act as clinicians. What a clinician may see is still decided by
# consent (AccessGrant), never by the role alone -- see services/api/core/access.py.
CLINICIAN_ROLES = frozenset({Role.doctor, Role.radiologist, Role.admin})


class ScanStatus(str, enum.Enum):
    uploaded = "uploaded"
    processing = "processing"
    completed = "completed"
    failed = "failed"


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(primary_key=True)
    username: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    hashed_password: Mapped[str] = mapped_column(String(255))
    full_name: Mapped[str | None] = mapped_column(String(128), nullable=True)
    role: Mapped[Role] = mapped_column(Enum(Role), default=Role.radiologist)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    scans: Mapped[list[Scan]] = relationship(
        back_populates="user", cascade="all, delete-orphan", foreign_keys="Scan.user_id"
    )

    @property
    def is_clinician(self) -> bool:
        return self.role in CLINICIAN_ROLES

    @property
    def is_patient(self) -> bool:
        return self.role == Role.patient


class Scan(Base):
    __tablename__ = "scans"

    id: Mapped[int] = mapped_column(primary_key=True)
    uuid: Mapped[str] = mapped_column(String(36), unique=True, index=True)
    # Who uploaded the scan...
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    # ...and whose history it belongs to. Null for a doctor's unlinked upload.
    patient_id: Mapped[int | None] = mapped_column(
        ForeignKey("users.id"), nullable=True, index=True
    )

    filename: Mapped[str] = mapped_column(String(255))
    stored_path: Mapped[str] = mapped_column(String(512))
    file_size: Mapped[int] = mapped_column(Integer)
    file_format: Mapped[str] = mapped_column(String(16))

    patient_ref: Mapped[str | None] = mapped_column(String(64), nullable=True)
    patient_age: Mapped[int | None] = mapped_column(Integer, nullable=True)
    patient_sex: Mapped[str | None] = mapped_column(String(8), nullable=True)

    status: Mapped[ScanStatus] = mapped_column(
        Enum(ScanStatus), default=ScanStatus.uploaded, index=True
    )
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    heatmap_dir: Mapped[str | None] = mapped_column(String(512), nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    processed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    user: Mapped[User] = relationship(back_populates="scans", foreign_keys=[user_id])
    patient: Mapped[User | None] = relationship(foreign_keys=[patient_id])

    @property
    def patient_username(self) -> str | None:
        return self.patient.username if self.patient else None

    @property
    def uploaded_by(self) -> str | None:
        return self.user.username if self.user else None

    prediction: Mapped[Prediction | None] = relationship(
        back_populates="scan", cascade="all, delete-orphan", uselist=False
    )


class Prediction(Base):
    __tablename__ = "predictions"

    id: Mapped[int] = mapped_column(primary_key=True)
    scan_id: Mapped[int] = mapped_column(ForeignKey("scans.id"), index=True, unique=True)

    # The label set is stored with every prediction. Without it, a later model
    # upgrade silently re-interprets old rows against the wrong names.
    label_set: Mapped[list] = mapped_column(JSON)
    probabilities: Mapped[dict] = mapped_column(JSON)
    uncertainties: Mapped[dict] = mapped_column(JSON)
    confidence_intervals: Mapped[dict] = mapped_column(JSON)
    thresholds: Mapped[dict] = mapped_column(JSON)

    fracture_probability: Mapped[float] = mapped_column(Float, default=0.0)
    fracture_uncertainty: Mapped[float] = mapped_column(Float, default=0.0)

    risk_level: Mapped[str] = mapped_column(String(16), index=True)
    top_finding: Mapped[str | None] = mapped_column(String(64), nullable=True)
    top_probability: Mapped[float] = mapped_column(Float, default=0.0)

    model_version: Mapped[str] = mapped_column(String(64))
    temperature: Mapped[float] = mapped_column(Float, default=1.0)
    mc_samples: Mapped[int] = mapped_column(Integer, default=20)
    inference_ms: Mapped[float] = mapped_column(Float, default=0.0)
    # Image-quality gate result (radiscan.data.quality). Null for rows created
    # before the gate existed.
    quality: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    scan: Mapped[Scan] = relationship(back_populates="prediction")
    report: Mapped[Report | None] = relationship(
        back_populates="prediction", cascade="all, delete-orphan", uselist=False
    )


class Report(Base):
    __tablename__ = "reports"

    id: Mapped[int] = mapped_column(primary_key=True)
    prediction_id: Mapped[int] = mapped_column(
        ForeignKey("predictions.id"), index=True, unique=True
    )

    summary: Mapped[str] = mapped_column(Text)
    payload: Mapped[dict] = mapped_column(JSON)  # full ClinicalReport
    text: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    prediction: Mapped[Prediction] = relationship(back_populates="report")


class AccessGrant(Base):
    """A patient's consent for one doctor to see their scan history.

    Rows are never deleted: revoking sets ``revoked_at``, so the audit trail can
    always answer "who could see this patient's history, and when".
    """

    __tablename__ = "access_grants"

    id: Mapped[int] = mapped_column(primary_key=True)
    patient_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    doctor_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    revoked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    patient: Mapped[User] = relationship(foreign_keys=[patient_id])
    doctor: Mapped[User] = relationship(foreign_keys=[doctor_id])

    @property
    def active(self) -> bool:
        return self.revoked_at is None


class AuditLog(Base):
    """Append-only trail. Required by the HIPAA principles the design follows.

    Note the wording: this schema *supports* an audit requirement. It does not
    make the system HIPAA compliant -- that needs a BAA, a formal risk
    assessment, and an external audit. Say it that way in the report.
    """

    __tablename__ = "audit_logs"

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int | None] = mapped_column(ForeignKey("users.id"), nullable=True, index=True)
    action: Mapped[str] = mapped_column(String(64), index=True)
    resource: Mapped[str] = mapped_column(String(128))
    detail: Mapped[str | None] = mapped_column(Text, nullable=True)
    ip_address: Mapped[str | None] = mapped_column(String(64), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, index=True
    )


# --------------------------------------------------------------------------
# Clinical workflow: review of model calls, signed reports, follow-up
# --------------------------------------------------------------------------
class ReviewDecision(str, enum.Enum):
    confirmed = "confirmed"  # the finding is present
    rejected = "rejected"  # the finding is absent
    unsure = "unsure"  # the reader cannot decide from this film


class FindingReview(Base):
    """A clinician's verdict on one finding of one scan.

    One row per (scan, finding, reviewer); a changed mind updates the row and
    the audit log keeps the history. ``was_deferred`` and ``model_positive``
    snapshot what the model said at review time, so agreement statistics stay
    correct even if thresholds change later.
    """

    __tablename__ = "finding_reviews"
    __table_args__ = (UniqueConstraint("scan_id", "label", "reviewer_id", name="uq_review"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    scan_id: Mapped[int] = mapped_column(ForeignKey("scans.id", ondelete="CASCADE"), index=True)
    reviewer_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    label: Mapped[str] = mapped_column(String(64))
    decision: Mapped[ReviewDecision] = mapped_column(Enum(ReviewDecision))
    note: Mapped[str | None] = mapped_column(Text, nullable=True)
    was_deferred: Mapped[bool] = mapped_column(Boolean, default=False)
    model_positive: Mapped[bool] = mapped_column(Boolean, default=False)
    model_probability: Mapped[float] = mapped_column(Float, default=0.0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    reviewer: Mapped[User] = relationship(foreign_keys=[reviewer_id])


class FollowUp(str, enum.Enum):
    none = "none"
    routine = "routine"
    soon = "soon"
    urgent = "urgent"


class DoctorReport(Base):
    """One version of the clinician's report on a scan.

    Every save is a new row; nothing is overwritten, so the record can always
    show what the AI drafted, what the doctor changed, and what was signed.
    Only the latest version counts. A scan's report is *signed* when its latest
    version has ``signed_at`` set; editing after signing creates an unsigned
    amendment that must be signed again.
    """

    __tablename__ = "doctor_reports"
    __table_args__ = (UniqueConstraint("scan_id", "version", name="uq_report_version"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    scan_id: Mapped[int] = mapped_column(ForeignKey("scans.id", ondelete="CASCADE"), index=True)
    version: Mapped[int] = mapped_column(Integer)
    author_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)

    impression: Mapped[str] = mapped_column(Text)
    findings_text: Mapped[str] = mapped_column(Text, default="")
    recommendations: Mapped[str] = mapped_column(Text, default="")
    # What the patient-facing summary is built from: the findings the doctor
    # chose to tell the patient about, the urgency, and an optional plain note.
    patient_findings: Mapped[list] = mapped_column(JSON, default=list)
    follow_up: Mapped[FollowUp] = mapped_column(Enum(FollowUp), default=FollowUp.none)
    follow_up_weeks: Mapped[int | None] = mapped_column(Integer, nullable=True)
    patient_note: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Share of the AI draft the doctor rewrote (0 = accepted as is, 1 = replaced).
    edit_ratio: Mapped[float] = mapped_column(Float, default=0.0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    signed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    author: Mapped[User] = relationship(foreign_keys=[author_id])

    @property
    def signed(self) -> bool:
        return self.signed_at is not None


class ShareCode(Base):
    """A one-time code a patient hands to a doctor in person (typed or as a QR).

    Only a hash is stored. Redeeming it creates the same AccessGrant as sharing
    by username: creating the code is the patient's consent. Codes expire and
    work once, so a photographed code is useless after it has been used.
    """

    __tablename__ = "share_codes"

    id: Mapped[int] = mapped_column(primary_key=True)
    patient_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    code_hash: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    used_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    used_by_id: Mapped[int | None] = mapped_column(ForeignKey("users.id"), nullable=True)


class ReminderStatus(str, enum.Enum):
    pending = "pending"
    done = "done"
    dismissed = "dismissed"


class Reminder(Base):
    """A follow-up a doctor asked a patient to do, e.g. a repeat film in 6 weeks."""

    __tablename__ = "reminders"

    id: Mapped[int] = mapped_column(primary_key=True)
    patient_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    created_by_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    scan_id: Mapped[int | None] = mapped_column(
        ForeignKey("scans.id", ondelete="SET NULL"), nullable=True
    )
    due_date: Mapped[date] = mapped_column(Date, index=True)
    note: Mapped[str] = mapped_column(Text)
    status: Mapped[ReminderStatus] = mapped_column(
        Enum(ReminderStatus), default=ReminderStatus.pending
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    patient: Mapped[User] = relationship(foreign_keys=[patient_id])
    created_by: Mapped[User] = relationship(foreign_keys=[created_by_id])
