"""Upload, status, results, heatmaps, scan history.

Every endpoint the frontend calls exists here. In v1 the UI called
``/predict/heatmap/{id}/{disease}`` and ``/predict/scans``, neither of which was
implemented, so two whole screens were dead.
"""

from __future__ import annotations

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

from services.api.core.security import current_user
from services.api.core.settings import get_settings
from services.api.core.tasks import process_scan
from services.api.db.models import AuditLog, Prediction, Scan, ScanStatus, User
from services.api.db.session import get_db
from services.api.schemas.schemas import (
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
    scan = db.get(Scan, scan_id)
    if scan is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"Scan {scan_id} not found")
    if scan.user_id != user.id and user.role.value != "admin":
        # 404 rather than 403: do not confirm that a scan id exists to someone
        # who is not allowed to see it.
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"Scan {scan_id} not found")
    return scan


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
    db: Session = Depends(get_db),
    user: User = Depends(current_user),
) -> ScanAccepted:
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
    query = db.query(Scan)
    if user.role.value != "admin":
        query = query.filter(Scan.user_id == user.id)
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
                heatmap_url=(
                    f"{settings.api_prefix}/predict/heatmap/{scan.id}/{safe}" if has_map else None
                ),
            )
        )
    findings.sort(key=lambda f: f.probability, reverse=True)

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
    )


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
    db.delete(scan)
    db.commit()
