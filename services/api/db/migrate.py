"""Additive schema upgrades for databases created by an earlier version.

``create_all`` creates missing *tables* but never alters existing ones. A
database from before the doctor/patient split has a ``scans`` table without
``patient_id``, and on Postgres a ``role`` enum without the new values. This
adds them in place, so upgrading never means deleting anyone's history.

Deliberately tiny and idempotent. A project that keeps growing should move to
Alembic; for one additive change, a migration framework is more moving parts
than the change itself.
"""

from __future__ import annotations

import logging

from sqlalchemy import inspect, text
from sqlalchemy.engine import Engine

log = logging.getLogger(__name__)


def upgrade(engine: Engine) -> list[str]:
    applied: list[str] = []
    insp = inspect(engine)

    if "scans" in insp.get_table_names():
        cols = {c["name"] for c in insp.get_columns("scans")}
        if "patient_id" not in cols:
            with engine.begin() as conn:
                conn.execute(
                    text("ALTER TABLE scans ADD COLUMN patient_id INTEGER REFERENCES users(id)")
                )
                conn.execute(
                    text("CREATE INDEX IF NOT EXISTS ix_scans_patient_id ON scans (patient_id)")
                )
            applied.append("scans.patient_id")

    if engine.dialect.name == "postgresql":
        # Native enum: new values must be added explicitly, outside a transaction.
        with engine.connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
            for value in ("doctor", "patient"):
                conn.execute(text(f"ALTER TYPE role ADD VALUE IF NOT EXISTS '{value}'"))
        applied.append("role enum values")

    for item in applied:
        log.info("Schema upgraded: %s", item)
    return applied
