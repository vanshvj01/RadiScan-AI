"""Pydantic request/response models.

Every route is typed both ways. This is what makes ``/docs`` a usable contract
rather than decoration, and it is what lets the frontend be generated against a
real OpenAPI schema instead of guessed at.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, EmailStr, Field

from services.api.db.models import (
    FollowUp,
    ReminderStatus,
    ReviewDecision,
    Role,
    ScanStatus,
)


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
    # Credible interval straddles the threshold (or the spread is too wide):
    # the system declines to call this finding and refers it for review.
    deferred: bool = False
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
    # Image-quality gate: {"status": good|fair|poor, "issues": [...], ...}.
    quality: dict | None = None


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


# --------------------------------------------------------------------------
# Review of model calls
# --------------------------------------------------------------------------
class ReviewItemIn(BaseModel):
    label: str = Field(min_length=1, max_length=64)
    decision: ReviewDecision
    note: str | None = Field(None, max_length=2000)


class ReviewSubmit(BaseModel):
    reviews: list[ReviewItemIn] = Field(min_length=1, max_length=40)


class ReviewOut(BaseModel):
    label: str
    decision: ReviewDecision
    note: str | None
    was_deferred: bool
    model_positive: bool
    model_probability: float
    reviewer_username: str
    updated_at: datetime


class ReviewGroupStats(BaseModel):
    reviewed: int
    confirmed: int
    rejected: int
    unsure: int


class ReviewStats(BaseModel):
    """How the model's calls held up against clinician review.

    ``model_agreement`` covers only the calls the model made itself (positive or
    negative, not deferred): the share where the reviewer agreed. Deferred
    findings have no model call, so for them the useful number is how often the
    reviewer found the finding present -- if that is near 50%, deferral is
    picking out the genuinely hard cases.
    """

    scope: Literal["mine", "all"]
    scans_reviewed: int
    model_calls: ReviewGroupStats
    model_agreement: float | None
    deferred: ReviewGroupStats
    deferred_present_rate: float | None


# --------------------------------------------------------------------------
# Worklist
# --------------------------------------------------------------------------
class WorklistItem(BaseModel):
    scan_id: int
    created_at: datetime
    filename: str
    patient_username: str | None
    patient_ref: str | None
    uploaded_by: str | None
    risk_level: str
    top_finding: str | None
    top_probability: float
    critical_findings: list[str]
    deferred_findings: list[str]
    quality: str | None
    reasons: list[str]
    reviewed: bool
    report_status: Literal["none", "draft", "signed"]


class Worklist(BaseModel):
    items: list[WorklistItem]
    total: int
    pending: int


# --------------------------------------------------------------------------
# Doctor-signed report
# --------------------------------------------------------------------------
class DoctorReportIn(BaseModel):
    impression: str = Field(min_length=1, max_length=8000)
    findings_text: str = Field("", max_length=8000)
    recommendations: str = Field("", max_length=8000)
    patient_findings: list[str] = Field(default_factory=list, max_length=20)
    follow_up: FollowUp = FollowUp.none
    follow_up_weeks: int | None = Field(None, ge=1, le=104)
    patient_note: str | None = Field(None, max_length=2000)
    sign: bool = False


class DoctorReportVersion(BaseModel):
    version: int
    author_username: str
    author_name: str | None
    impression: str
    findings_text: str
    recommendations: str
    patient_findings: list[str]
    follow_up: FollowUp
    follow_up_weeks: int | None
    patient_note: str | None
    edit_ratio: float
    created_at: datetime
    signed_at: datetime | None


class DoctorReportState(BaseModel):
    scan_id: int
    status: Literal["none", "draft", "signed"]
    ai_draft: dict
    latest: DoctorReportVersion | None
    history: list[DoctorReportVersion]


# --------------------------------------------------------------------------
# Comparison with a prior study
# --------------------------------------------------------------------------
class CompareStudy(BaseModel):
    scan_id: int
    created_at: datetime
    risk_level: str
    enhanced_url: str | None


class CompareRow(BaseModel):
    label: str
    current: float
    prior: float
    delta: float
    current_call: Literal["positive", "negative", "deferred"]
    prior_call: Literal["positive", "negative", "deferred"]
    trend: Literal["new", "resolved", "worse", "better", "stable"]
    current_heatmap: str | None = None
    prior_heatmap: str | None = None


class Comparison(BaseModel):
    current: CompareStudy
    prior: CompareStudy
    days_between: int
    rows: list[CompareRow]


# --------------------------------------------------------------------------
# Patient-facing summary
# --------------------------------------------------------------------------
class PatientSummary(BaseModel):
    scan_id: int
    language: Literal["en", "hi", "mr"]
    title: str
    headline: str
    findings: list[dict]
    no_findings_text: str
    next_steps: str
    doctor_note: str | None
    doctor_note_label: str
    signed_by: str
    signed_at: datetime
    disclaimer: str


# --------------------------------------------------------------------------
# Share codes and reminders
# --------------------------------------------------------------------------
class ShareCodeOut(BaseModel):
    code: str
    expires_at: datetime


class ShareCodeRedeem(BaseModel):
    code: str = Field(min_length=6, max_length=20)


class ReminderIn(BaseModel):
    patient_id: int
    scan_id: int | None = None
    due_date: date | None = None
    weeks: int | None = Field(None, ge=1, le=104)
    note: str = Field(min_length=1, max_length=500)


class ReminderUpdate(BaseModel):
    status: ReminderStatus


class ReminderOut(BaseModel):
    id: int
    patient_id: int
    patient_username: str
    created_by_username: str
    created_by_name: str | None
    scan_id: int | None
    due_date: date
    note: str
    status: ReminderStatus
    overdue: bool
    days_until: int
    created_at: datetime
