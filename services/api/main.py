"""FastAPI application factory.

Run it with no infrastructure at all:

    uvicorn services.api.main:app --reload

That is the whole dev setup -- SQLite file, Celery in eager mode, model loaded
lazily. Swap in Postgres and Redis through environment variables when you want
the production topology.
"""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from services.api.core.settings import get_settings
from services.api.db.session import create_tables
from services.api.routes import auth, health, predictions, reports, websocket

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("radiscan.api")
settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    create_tables()
    log.info("Database ready at %s", settings.database_url.split("://")[0])

    # Warm the model, but never let a missing checkpoint stop the API from
    # booting -- /health has to be reachable to tell you what is wrong.
    try:
        from services.api.core.model_registry import get_predictor

        get_predictor()
    except Exception as exc:
        log.warning("Model not loaded at startup: %s", exc)
        log.warning("The API is up; /health reports the details. Predictions will 503.")

    yield
    log.info("Shutting down.")


app = FastAPI(
    title=settings.app_name,
    version=settings.version,
    description=(
        "Uncertainty-aware, explainable chest radiograph analysis.\n\n"
        "**Decision support only. Not a diagnostic device. No regulatory clearance.**"
    ),
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def timing_header(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    response.headers["X-Process-Time-Ms"] = f"{(time.perf_counter() - start) * 1000:.1f}"
    return response


@app.exception_handler(FileNotFoundError)
async def missing_checkpoint_handler(request: Request, exc: FileNotFoundError):
    return JSONResponse(
        status_code=503,
        content={"detail": str(exc), "hint": "See /api/v1/health for model status."},
    )


for router in (auth.router, predictions.router, reports.router, health.router):
    app.include_router(router, prefix=settings.api_prefix)
app.include_router(websocket.router)


@app.get("/", tags=["meta"])
def root() -> dict:
    return {
        "name": settings.app_name,
        "version": settings.version,
        "docs": "/docs",
        "health": f"{settings.api_prefix}/health",
        "disclaimer": "Decision support prototype. Not for clinical use.",
    }
