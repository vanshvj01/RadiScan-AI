"""Pydantic request/response models.

Every route is typed both ways. This is what makes ``/docs`` a usable contract
rather than decoration, and it is what lets the frontend be generated against a
real OpenAPI schema instead of guessed at.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, EmailStr, Field

from services.api.db.models import Role, ScanStatus


# --------------------------------------------------------------------------
# Auth
# --------------------------------------------------------------------------
class UserCreate(BaseModel):
    username: str = Field(min_length=3, max_length=64)
    email: EmailStr
    password: str = Field(min_length=8, description="Minimum 8 characters")
    full_name: str | None = None
    # Only these two can be chosen at sign-up. Admin is never self-assigned,
    # and choosing "doctor" grants nothing by itself: a doctor sees a patient's
    # history only after that patient shares it.
    role: Literal["doctor", "patient"] = "patient"


class UserOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    username: str
    email: EmailStr
    full_name: str | None
    role: Role
    is_active: bool
    created_at: datetime


class TokenPair(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int


class RefreshRequest(BaseModel):
    refresh_token: str


# --------------------------------------------------------------------------
# Scans and predictions
# --------------------------------------------------------------------------
class ScanOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    uuid: str
    filename: str
    file_size: int
    file_format: str
    patient_ref: str | None
    patient_age: int | None
    patient_sex: str | None
    patient_id: int | None = None
    patient_username: str | None = None
    uploaded_by: str | None = None
    status: ScanStatus
    error_message: str | None
    created_at: datetime
    processed_at: datetime | None


class ScanAccepted(BaseModel):
    scan_id: int
    uuid: str
    status: ScanStatus
    task_id: str | None = None
    message: str = "Scan accepted. Poll /predict/status/{scan_id} or subscribe to /ws/scan/{uuid}."


class ScanPage(BaseModel):
    items: list[ScanOut]
    total: int
    page: int
    page_size: int


class FindingOut(BaseModel):
    label: str
    probability: float
    uncertainty: float
    ci_low: float
    ci_high: float
    entropy: float
    reliability: float
    threshold: float
    positive: bool
    heatmap_url: str | None = None


class PredictionOut(BaseModel):
    scan_id: int
    uuid: str
    status: ScanStatus
    model_version: str
    label_set: list[str]
    findings: list[FindingOut]
    fracture_probability: float
    fracture_uncertainty: float
    risk_level: str
    top_finding: str | None
    top_probability: float
    temperature: float
    mc_samples: int
    inference_ms: float
    created_at: datetime
    # Cleaned-up view of the upload (film located, borders cropped, noise
    # reduced, contrast enhanced) and the steps that produced it.
    enhanced_url: str | None = None
    enhancement_steps: list[str] = []
    # Whose history this scan belongs to, so the client can fetch prior studies.
    patient_id: int | None = None
    patient_username: str | None = None


# --------------------------------------------------------------------------
# Sharing and history
# --------------------------------------------------------------------------
class GrantCreate(BaseModel):
    doctor_username: str = Field(min_length=3, max_length=64)


class GrantOut(BaseModel):
    id: int
    patient_id: int
    patient_username: str
    patient_name: str | None
    doctor_id: int
    doctor_username: str
    doctor_name: str | None
    created_at: datetime


class HistoryItem(BaseModel):
    scan_id: int
    created_at: datetime
    filename: str
    uploaded_by: str
    status: ScanStatus
    risk_level: str | None = None
    top_finding: str | None = None
    top_probability: float | None = None
    model_version: str | None = None
    # Per-finding probabilities, so the client can show change over time.
    probabilities: dict[str, float] = {}


class PatientHistory(BaseModel):
    patient_id: int
    patient_username: str
    patient_name: str | None
    items: list[HistoryItem]


class StatusOut(BaseModel):
    scan_id: int
    uuid: str
    status: ScanStatus
    progress: int = Field(ge=0, le=100)
    message: str | None = None
    error: str | None = None


class ReportOut(BaseModel):
    scan_id: int
    risk_level: str
    summary: str
    report: dict
    text: str
    created_at: datetime


class HealthOut(BaseModel):
    status: str
    version: str
    database: str
    model_loaded: bool
    model_version: str | None
    device: str
    labels: list[str] | None
    warnings: list[str] = []
