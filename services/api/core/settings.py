"""API settings.

Zero-infrastructure by default: SQLite on disk and Celery in eager mode, so
``uvicorn services.api.main:app`` is the entire dev setup. Point
``DATABASE_URL`` at Postgres and ``CELERY_BROKER_URL`` at Redis to get the
production topology; no code changes.
"""

from __future__ import annotations

import secrets
from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # -- app ---------------------------------------------------------------
    app_name: str = "RadiScan API"
    version: str = "2.0.0"
    debug: bool = False
    api_prefix: str = "/api/v1"
    cors_origins: list[str] = ["http://localhost:3000", "http://127.0.0.1:3000"]

    # -- security ----------------------------------------------------------
    secret_key: str = ""
    algorithm: str = "HS256"
    access_token_minutes: int = 60
    refresh_token_days: int = 7

    # -- storage -----------------------------------------------------------
    database_url: str = f"sqlite:///{REPO_ROOT / 'radiscan.db'}"
    upload_dir: Path = REPO_ROOT / "data" / "uploads"
    heatmap_dir: Path = REPO_ROOT / "data" / "heatmaps"
    max_upload_mb: int = 50
    allowed_extensions: set[str] = {".dcm", ".dicom", ".png", ".jpg", ".jpeg"}

    # -- queue -------------------------------------------------------------
    celery_broker_url: str = "memory://"
    celery_result_backend: str = "cache+memory://"
    celery_eager: bool = True  # run tasks inline; flip off when Redis is up

    # -- model -------------------------------------------------------------
    checkpoint_path: Path = REPO_ROOT / "models" / "checkpoints" / "best_model.pth"
    device: str = "auto"
    mc_dropout_samples: int = 20
    allow_untrained: bool = False
    heatmap_top_k: int = 3

    def resolved_secret(self) -> str:
        if self.secret_key:
            return self.secret_key
        if not self.debug:
            raise RuntimeError(
                "SECRET_KEY is unset. Generate one with:\n"
                '  python -c "import secrets; print(secrets.token_urlsafe(48))"\n'
                "and put it in .env. Refusing to start with an ephemeral key outside DEBUG."
            )
        return secrets.token_urlsafe(48)


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    settings.upload_dir.mkdir(parents=True, exist_ok=True)
    settings.heatmap_dir.mkdir(parents=True, exist_ok=True)
    return settings
