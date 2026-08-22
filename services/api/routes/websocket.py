"""WebSocket progress streaming.

Inference plus 20 MC passes plus Grad-CAM is seconds, not milliseconds. Polling
works but wastes requests; this pushes status transitions instead.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from services.api.db import session as db_session
from services.api.db.models import Scan, ScanStatus

log = logging.getLogger(__name__)
router = APIRouter(tags=["websocket"])

PROGRESS = {
    ScanStatus.uploaded: 10,
    ScanStatus.processing: 55,
    ScanStatus.completed: 100,
    ScanStatus.failed: 100,
}
TERMINAL = {ScanStatus.completed, ScanStatus.failed}


@router.websocket("/ws/scan/{scan_uuid}")
async def scan_progress(websocket: WebSocket, scan_uuid: str) -> None:
    await websocket.accept()
    last_status: ScanStatus | None = None
    elapsed = 0.0
    timeout = 300.0

    try:
        while elapsed < timeout:
            db = db_session.SessionLocal()
            try:
                scan = db.query(Scan).filter(Scan.uuid == scan_uuid).first()
            finally:
                db.close()

            if scan is None:
                await websocket.send_json({"error": "scan_not_found", "uuid": scan_uuid})
                break

            if scan.status != last_status:
                await websocket.send_json(
                    {
                        "scan_id": scan.id,
                        "uuid": scan.uuid,
                        "status": scan.status.value,
                        "progress": PROGRESS[scan.status],
                        "error": scan.error_message,
                    }
                )
                last_status = scan.status

            if scan.status in TERMINAL:
                break

            await asyncio.sleep(1.0)
            elapsed += 1.0
        else:
            await websocket.send_json({"error": "timeout", "uuid": scan_uuid})
    except WebSocketDisconnect:
        log.debug("Client disconnected from scan %s", scan_uuid)
    finally:
        # Already-closed sockets raise; there is nothing left to do about it.
        with contextlib.suppress(RuntimeError):
            await websocket.close()
