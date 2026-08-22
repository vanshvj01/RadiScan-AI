"""SQLAlchemy 2.0 ORM models."""

from __future__ import annotations

import enum
from datetime import datetime, timezone

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class Role(str, enum.Enum):
    admin = "admin"
    radiologist = "radiologist"
    viewer = "viewer"


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

    scans: Mapped[list[Scan]] = relationship(back_populates="user", cascade="all, delete-orphan")


class Scan(Base):
    __tablename__ = "scans"

    id: Mapped[int] = mapped_column(primary_key=True)
    uuid: Mapped[str] = mapped_column(String(36), unique=True, index=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)

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

    user: Mapped[User] = relationship(back_populates="scans")
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
