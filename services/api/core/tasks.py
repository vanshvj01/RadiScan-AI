"""Inference job.

The same function body runs three ways:

* inline, when ``celery_eager`` is on (default dev mode -- no Redis needed)
* in a Celery worker, when a broker is configured
* directly, from tests

Because it is one function, an eager-mode demo exercises exactly the code path
that production runs.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

from celery import Celery
from sqlalchemy.orm import Session

from services.api.core.model_registry import get_predictor
from services.api.core.settings import get_settings
from services.api.db import session as db_session
from services.api.db.models import Prediction, Report, Scan, ScanStatus

log = logging.getLogger(__name__)
settings = get_settings()

celery_app = Celery(
    "radiscan",
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend,
)
celery_app.conf.update(
    task_always_eager=settings.celery_eager,
    task_eager_propagates=True,
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    task_track_started=True,
    worker_max_tasks_per_child=50,  # bound memory growth in long-running workers
)


def _run_inference(scan_id: int, db: Session) -> int:
    scan = db.get(Scan, scan_id)
    if scan is None:
        raise ValueError(f"Scan {scan_id} not found")

    scan.status = ScanStatus.processing
    db.commit()

    try:
        predictor = get_predictor()
        result = predictor.predict(
            scan.stored_path,
            explain=True,
            heatmap_dir=settings.heatmap_dir,
            top_k_heatmaps=settings.heatmap_top_k,
            scan_id=scan.uuid,
        )

        findings = result["predictions"]
        top = max(findings, key=lambda f: f["probability"]) if findings else None

        prediction = Prediction(
            scan_id=scan.id,
            label_set=result["labels"],
            probabilities={f["label"]: f["probability"] for f in findings},
            uncertainties={f["label"]: f["uncertainty"] for f in findings},
            confidence_intervals={f["label"]: [f["ci_low"], f["ci_high"]] for f in findings},
            thresholds={f["label"]: f["threshold"] for f in findings},
            fracture_probability=result["fracture"]["probability"],
            fracture_uncertainty=result["fracture"]["uncertainty"],
            risk_level=result["risk_level"],
            top_finding=top["label"] if top else None,
            top_probability=top["probability"] if top else 0.0,
            model_version=result["model_version"],
            temperature=float(predictor.model.temperature.item()),
            mc_samples=predictor.mc_samples,
            inference_ms=result["inference_ms"],
        )
        db.add(prediction)
        db.flush()

        report = result["report"]
        db.add(
            Report(
                prediction_id=prediction.id,
                summary=report.summary,
                payload=report.to_dict(),
                text=report.to_text(),
            )
        )

        scan.status = ScanStatus.completed
        scan.processed_at = datetime.now(timezone.utc)
        scan.heatmap_dir = str(Path(settings.heatmap_dir) / scan.uuid)
        scan.error_message = None
        db.commit()
        log.info(
            "Scan %s completed in %.0f ms (risk=%s)",
            scan.uuid,
            result["inference_ms"],
            result["risk_level"],
        )
        return prediction.id

    except Exception as exc:
        db.rollback()
        scan = db.get(Scan, scan_id)
        if scan is not None:
            scan.status = ScanStatus.failed
            # Store the message, not the traceback -- tracebacks can carry paths
            # and, in a hospital deployment, filenames containing patient refs.
            scan.error_message = f"{type(exc).__name__}: {exc}"
            db.commit()
        log.exception("Scan %s failed", scan_id)
        raise


@celery_app.task(name="radiscan.process_scan", bind=True, max_retries=2)
def process_scan(self, scan_id: int) -> dict:
    db = db_session.SessionLocal()
    try:
        prediction_id = _run_inference(scan_id, db)
        return {"scan_id": scan_id, "prediction_id": prediction_id, "status": "completed"}
    finally:
        db.close()
