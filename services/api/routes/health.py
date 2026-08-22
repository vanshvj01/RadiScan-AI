"""Liveness, readiness and model status."""

from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy import text
from sqlalchemy.orm import Session

from services.api.core import model_registry
from services.api.core.settings import get_settings
from services.api.db.session import get_db
from services.api.schemas.schemas import HealthOut

router = APIRouter(tags=["health"])
settings = get_settings()


@router.get("/health", response_model=HealthOut)
def health(db: Session = Depends(get_db)) -> HealthOut:
    """Honest health: reports *why* the system is not ready, not just that it isn't."""
    try:
        db.execute(text("SELECT 1"))
        database = "ok"
    except Exception as exc:
        database = f"error: {type(exc).__name__}"

    model = model_registry.status()
    warnings = list(model["warnings"])
    if database != "ok":
        warnings.append("Database unreachable.")

    return HealthOut(
        status="ok" if database == "ok" and model["loaded"] else "degraded",
        version=settings.version,
        database=database,
        model_loaded=model["loaded"],
        model_version=model["model_version"],
        device=model["device"],
        labels=model["labels"],
        warnings=warnings,
    )


@router.get("/health/live")
def live() -> dict:
    return {"status": "alive"}


@router.get("/health/ready")
def ready() -> dict:
    """Readiness forces the model to load, so orchestrators see the real state."""
    try:
        model_registry.get_predictor()
        return {"status": "ready"}
    except Exception as exc:
        return {"status": "not_ready", "reason": str(exc)}
