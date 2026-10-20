"""Upload, status, results, heatmaps, scan history.

Every endpoint the frontend calls exists here. In v1 the UI called
``/predict/heatmap/{id}/{disease}`` and ``/predict/scans``, neither of which was
implemented, so two whole screens were dead.
"""

from __future__ import annotations

import json
import shutil
import uuid as uuidlib
from pathlib import Path

from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    HTTPException,
    Query,
    Request,
    UploadFile,
    status,
)
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from services.api.core.access import can_view_patient, can_view_scan, viewable_scan, visible_scans
from services.api.core.clinical import finding_rows, is_deferred
from services.api.core.security import current_user
from services.api.core.settings import get_settings
from services.api.core.tasks import process_scan
from services.api.db.models import (
    AuditLog,
    DoctorReport,
    FindingReview,
    Prediction,
    Reminder,
    Role,
    Scan,
    ScanStatus,
    User,
)
from services.api.db.session import get_db
from services.api.schemas.schemas import (
    CompareRow,
    CompareStudy,
    Comparison,
    FindingOut,
    PredictionOut,
    ScanAccepted,
    ScanOut,
    ScanPage,
    StatusOut,
)

# Starlette renamed HTTP_413_REQUEST_ENTITY_TOO_LARGE to HTTP_413_CONTENT_TOO_LARGE
# and warns on the old name. A getattr() fallback does not help -- its default is
# evaluated eagerly, so the deprecated attribute is touched either way. The bare
# integer is what both constants hold and works on every version.
HTTP_413 = 413

router = APIRouter(prefix="/predict", tags=["predictions"])
settings = get_settings()

PROGRESS = {
    ScanStatus.uploaded: 10,
    ScanStatus.processing: 55,
    ScanStatus.completed: 100,
    ScanStatus.failed: 100,
}


def _owned_scan(scan_id: int, db: Session, user: User) -> Scan:
    """A scan this user may *see*. Access rules live in core/access.py."""
    return viewable_scan(scan_id, db, user)


def _resolve_patient(db: Session, user: User, patient_username: str | None) -> int | None:
    """Whose history a new upload joins.

    A patient's upload always joins their own history. A doctor may attach a
    scan to a patient only while that patient shares their history with them;
    without consent, a doctor could write into a stranger's record.
    """
    if user.is_patient:
        return user.id
    if not patient_username:
        return None
    patient = (
        db.query(User)
        .filter(User.username == patient_username.strip(), User.role == Role.patient)
        .first()
    )
    if patient is None or not can_view_patient(db, user, patient.id):
        raise HTTPException(
            status.HTTP_403_FORBIDDEN,
            f"{patient_username!r} has not shared their history with you, so a scan "
            "cannot be added to it. Ask the patient to share access from their account.",
        )
    return patient.id


@router.post("", response_model=ScanAccepted, status_code=status.HTTP_202_ACCEPTED)
@router.post(
    "/", response_model=ScanAccepted, status_code=status.HTTP_202_ACCEPTED, include_in_schema=False
)
async def upload(
    request: Request,
    file: UploadFile = File(...),
    patient_ref: str | None = Form(None),
    patient_age: int | None = Form(None),
    patient_sex: str | None = Form(None),
    patient_username: str | None = Form(None),
    db: Session = Depends(get_db),
    user: User = Depends(current_user),
) -> ScanAccepted:
    patient_id = _resolve_patient(db, user, patient_username)
    ext = Path(file.filename or "").suffix.lower()
    if ext not in settings.allowed_extensions:
        raise HTTPException(
            status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            f"Unsupported file type {ext!r}. Allowed: {sorted(settings.allowed_extensions)}",
        )

    scan_uuid = str(uuidlib.uuid4())
    dest = Path(settings.upload_dir) / f"{scan_uuid}{ext}"
    dest.parent.mkdir(parents=True, exist_ok=True)

    size = 0
    limit = settings.max_upload_mb * 1024 * 1024
    with dest.open("wb") as out:
        while chunk := await file.read(1024 * 1024):
            size += len(chunk)
            if size > limit:
                out.close()
                dest.unlink(missing_ok=True)
                raise HTTPException(
                    HTTP_413,
                    f"File exceeds {settings.max_upload_mb} MB",
                )
            out.write(chunk)

    scan = Scan(
        uuid=scan_uuid,
        user_id=user.id,
        patient_id=patient_id,
        filename=file.filename or dest.name,
        stored_path=str(dest),
        file_size=size,
        file_format=ext,
        patient_ref=patient_ref,
        patient_age=patient_age,
        patient_sex=patient_sex,
        status=ScanStatus.uploaded,
    )
    db.add(scan)
    db.commit()
    db.refresh(scan)

    db.add(
        AuditLog(
            user_id=user.id,
            action="scan.upload",
            resource=f"scan:{scan.uuid}",
            detail=f"{file.filename} ({size} bytes)",
            ip_address=request.client.host if request.client else None,
        )
    )
    db.commit()

    task = process_scan.delay(scan.id)
    db.refresh(scan)
    return ScanAccepted(scan_id=scan.id, uuid=scan.uuid, status=scan.status, task_id=str(task.id))


@router.get("/scans", response_model=ScanPage)
def list_scans(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    scan_status: ScanStatus | None = Query(None, alias="status"),
    db: Session = Depends(get_db),
    user: User = Depends(current_user),
) -> ScanPage:
    query = visible_scans(db, user)
    if scan_status is not None:
        query = query.filter(Scan.status == scan_status)

    total = query.count()
    items = (
        query.order_by(Scan.created_at.desc()).offset((page - 1) * page_size).limit(page_size).all()
    )
    return ScanPage(
        items=[ScanOut.model_validate(s) for s in items],
        total=total,
        page=page,
        page_size=page_size,
    )


@router.get("/status/{scan_id}", response_model=StatusOut)
def scan_status(
    scan_id: int, db: Session = Depends(get_db), user: User = Depends(current_user)
) -> StatusOut:
    scan = _owned_scan(scan_id, db, user)
    return StatusOut(
        scan_id=scan.id,
        uuid=scan.uuid,
        status=scan.status,
        progress=PROGRESS[scan.status],
        message=scan.status.value,
        error=scan.error_message,
    )


@router.get("/results/{scan_id}", response_model=PredictionOut)
def results(
    scan_id: int, db: Session = Depends(get_db), user: User = Depends(current_user)
) -> PredictionOut:
    scan = _owned_scan(scan_id, db, user)
    if scan.status == ScanStatus.failed:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_ENTITY, scan.error_message or "Analysis failed"
        )
    prediction = db.query(Prediction).filter(Prediction.scan_id == scan.id).first()
    if prediction is None:
        raise HTTPException(
            status.HTTP_425_TOO_EARLY, f"Scan {scan_id} is still {scan.status.value}."
        )

    heatmap_dir = Path(scan.heatmap_dir) if scan.heatmap_dir else None
    findings: list[FindingOut] = []
    for label in prediction.label_set:
        prob = prediction.probabilities.get(label, 0.0)
        std = prediction.uncertainties.get(label, 0.0)
        lo, hi = prediction.confidence_intervals.get(label, [prob, prob])
        thr = prediction.thresholds.get(label, 0.5)
        safe = label.replace(" ", "_").replace("/", "-")
        has_map = bool(heatmap_dir and (heatmap_dir / f"{safe}.png").exists())
        findings.append(
            FindingOut(
                label=label,
                probability=prob,
                uncertainty=std,
                ci_low=lo,
                ci_high=hi,
                entropy=0.0,
                reliability=max(0.0, 1.0 - std / 0.25),
                threshold=thr,
                positive=prob >= thr,
                deferred=is_deferred(label, prob, std, lo, hi, thr),
                heatmap_url=(
                    f"{settings.api_prefix}/predict/heatmap/{scan.id}/{safe}" if has_map else None
                ),
            )
        )
    findings.sort(key=lambda f: f.probability, reverse=True)

    enhanced_url, steps = None, []
    if heatmap_dir and (heatmap_dir / "_enhanced.png").exists():
        enhanced_url = f"{settings.api_prefix}/predict/enhanced/{scan.id}"
        try:
            steps = json.loads((heatmap_dir / "_enhanced.json").read_text()).get("steps", [])
        except (OSError, ValueError):
            steps = []

    return PredictionOut(
        scan_id=scan.id,
        uuid=scan.uuid,
        status=scan.status,
        model_version=prediction.model_version,
        label_set=prediction.label_set,
        findings=findings,
        fracture_probability=prediction.fracture_probability,
        fracture_uncertainty=prediction.fracture_uncertainty,
        risk_level=prediction.risk_level,
        top_finding=prediction.top_finding,
        top_probability=prediction.top_probability,
        temperature=prediction.temperature,
        mc_samples=prediction.mc_samples,
        inference_ms=prediction.inference_ms,
        created_at=prediction.created_at,
        enhanced_url=enhanced_url,
        enhancement_steps=steps,
        patient_id=scan.patient_id,
        patient_username=scan.patient_username,
        quality=prediction.quality,
    )


def _heatmap_url(scan: Scan, label: str) -> str | None:
    if not scan.heatmap_dir:
        return None
    safe = label.replace(" ", "_").replace("/", "-")
    if (Path(scan.heatmap_dir) / f"{safe}.png").exists():
        return f"{settings.api_prefix}/predict/heatmap/{scan.id}/{safe}"
    return None


def _study(scan: Scan, prediction: Prediction) -> CompareStudy:
    has_enhanced = bool(scan.heatmap_dir and (Path(scan.heatmap_dir) / "_enhanced.png").exists())
    return CompareStudy(
        scan_id=scan.id,
        created_at=scan.created_at,
        risk_level=prediction.risk_level,
        enhanced_url=f"{settings.api_prefix}/predict/enhanced/{scan.id}" if has_enhanced else None,
    )


# A probability move smaller than this is reported as stable: MC-Dropout spread
# on the trained model is tiny, but calibration error is large (ECE 0.38), so
# small differences between two films are not evidence of change.
TREND_DELTA = 0.10


@router.get("/compare/{scan_id}", response_model=Comparison)
def compare(
    scan_id: int,
    prior_id: int | None = Query(None, description="Defaults to the patient's previous study"),
    db: Session = Depends(get_db),
    user: User = Depends(current_user),
) -> Comparison:
    """This study against an earlier one of the same patient, finding by finding."""
    scan = _owned_scan(scan_id, db, user)
    current = db.query(Prediction).filter(Prediction.scan_id == scan.id).first()
    if current is None or scan.patient_id is None:
        raise HTTPException(
            status.HTTP_404_NOT_FOUND, "This study has no patient history to compare"
        )

    if prior_id is not None:
        prior_scan = db.get(Scan, prior_id)
        if (
            prior_scan is None
            or prior_scan.patient_id != scan.patient_id
            or prior_scan.id == scan.id
            or not can_view_scan(db, user, prior_scan)
        ):
            raise HTTPException(status.HTTP_404_NOT_FOUND, f"Scan {prior_id} not found")
    else:
        prior_scan = (
            db.query(Scan)
            .filter(
                Scan.patient_id == scan.patient_id,
                Scan.id != scan.id,
                Scan.created_at <= scan.created_at,
                Scan.status == ScanStatus.completed,
            )
            .order_by(Scan.created_at.desc(), Scan.id.desc())
            .first()
        )
        if prior_scan is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "No earlier study for this patient")

    prior = db.query(Prediction).filter(Prediction.scan_id == prior_scan.id).first()
    if prior is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "The earlier study has no results")

    def call(row: dict) -> str:
        return "deferred" if row["deferred"] else ("positive" if row["positive"] else "negative")

    prior_rows = {r["label"]: r for r in finding_rows(prior)}
    rows: list[CompareRow] = []
    for r in finding_rows(current):
        p = prior_rows.get(r["label"])
        if p is None:  # label set changed between model versions
            continue
        delta = r["probability"] - p["probability"]
        now, before = call(r), call(p)
        # Crossing the threshold alone is not news: probabilities on this model
        # cluster near their thresholds, so a 4-point wobble can flip the call.
        # "new"/"resolved" need the same size of move as "worse"/"better".
        if now == "positive" and before == "negative" and delta >= TREND_DELTA:
            trend = "new"
        elif now == "negative" and before == "positive" and delta <= -TREND_DELTA:
            trend = "resolved"
        elif delta >= TREND_DELTA:
            trend = "worse"
        elif delta <= -TREND_DELTA:
            trend = "better"
        else:
            trend = "stable"
        if r["label"] == "No Finding" and trend in ("worse", "better", "new", "resolved"):
            # More "No Finding" is good news; flip the wording, not the numbers.
            trend = {"worse": "better", "better": "worse", "new": "better", "resolved": "worse"}[
                trend
            ]
        rows.append(
            CompareRow(
                label=r["label"],
                current=r["probability"],
                prior=p["probability"],
                delta=delta,
                current_call=now,
                prior_call=before,
                trend=trend,
                current_heatmap=_heatmap_url(scan, r["label"]),
                prior_heatmap=_heatmap_url(prior_scan, r["label"]),
            )
        )
    order = {"new": 0, "worse": 1, "resolved": 2, "better": 3, "stable": 4}
    rows.sort(key=lambda x: (order[x.trend], -abs(x.delta)))

    return Comparison(
        current=_study(scan, current),
        prior=_study(prior_scan, prior),
        days_between=abs((scan.created_at - prior_scan.created_at).days),
        rows=rows,
    )


@router.get("/enhanced/{scan_id}")
def enhanced_image(
    scan_id: int, db: Session = Depends(get_db), user: User = Depends(current_user)
) -> FileResponse:
    """The cleaned-up radiograph: film located, borders cropped, denoised, CLAHE."""
    scan = _owned_scan(scan_id, db, user)
    path = Path(scan.heatmap_dir or "") / "_enhanced.png"
    if not scan.heatmap_dir or not path.exists():
        raise HTTPException(status.HTTP_404_NOT_FOUND, "No enhanced image for this scan")
    return FileResponse(path, media_type="image/png")


@router.get("/heatmap/{scan_id}/{disease}")
def heatmap(
    scan_id: int,
    disease: str,
    db: Session = Depends(get_db),
    user: User = Depends(current_user),
) -> FileResponse:
    scan = _owned_scan(scan_id, db, user)
    if not scan.heatmap_dir:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "No heatmaps stored for this scan")

    safe = disease.replace(" ", "_").replace("/", "-")
    base = Path(scan.heatmap_dir).resolve()
    path = (base / f"{safe}.png").resolve()
    # Path-traversal guard: a crafted `disease` must not escape the scan folder.
    if not str(path).startswith(str(base)) or not path.exists():
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"No heatmap for {disease!r}")
    return FileResponse(path, media_type="image/png")


@router.get("/image/{scan_id}")
def original_image(
    scan_id: int, db: Session = Depends(get_db), user: User = Depends(current_user)
) -> FileResponse:
    scan = _owned_scan(scan_id, db, user)
    path = Path(scan.stored_path)
    if not path.exists():
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Stored image is missing")
    media = "application/dicom" if scan.file_format in (".dcm", ".dicom") else "image/png"
    return FileResponse(path, media_type=media)


@router.delete("/{scan_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_scan(
    scan_id: int,
    request: Request,
    db: Session = Depends(get_db),
    user: User = Depends(current_user),
) -> None:
    scan = _owned_scan(scan_id, db, user)
    # Seeing a scan is not owning it: a doctor with shared access must not be
    # able to delete a patient's record, nor a patient a doctor's upload.
    if scan.user_id != user.id and user.role != Role.admin:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Only the uploader can delete a scan")
    Path(scan.stored_path).unlink(missing_ok=True)
    if scan.heatmap_dir:
        shutil.rmtree(scan.heatmap_dir, ignore_errors=True)
    db.add(
        AuditLog(
            user_id=user.id,
            action="scan.delete",
            resource=f"scan:{scan.uuid}",
            ip_address=request.client.host if request.client else None,
        )
    )
    # SQLite does not enforce ON DELETE without a pragma, so clear dependants
    # explicitly rather than leave rows pointing at a deleted scan.
    db.query(FindingReview).filter(FindingReview.scan_id == scan.id).delete()
    db.query(DoctorReport).filter(DoctorReport.scan_id == scan.id).delete()
    db.query(Reminder).filter(Reminder.scan_id == scan.id).update({Reminder.scan_id: None})
    db.delete(scan)
    db.commit()
