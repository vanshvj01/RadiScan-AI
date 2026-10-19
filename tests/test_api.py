"""API tests against a real SQLite database and a real (untrained) model.

The whole stack runs in-process: no Postgres, no Redis, no checkpoint. That is
the point of the zero-infrastructure dev mode -- if CI can run it, so can you on
a laptop five minutes after cloning.
"""

from __future__ import annotations

import numpy as np
import pytest

fastapi = pytest.importorskip("fastapi")

PREFIX = "/api/v1"


@pytest.fixture
def auth(client):
    payload = {
        "username": "vansh",
        "email": "vansh@example.com",
        "password": "a-strong-password",
        "full_name": "Vansh V",
    }
    assert client.post(f"{PREFIX}/auth/register", json=payload).status_code == 201
    r = client.post(
        f"{PREFIX}/auth/login",
        data={"username": payload["username"], "password": payload["password"]},
    )
    assert r.status_code == 200
    return {"Authorization": f"Bearer {r.json()['access_token']}"}


# --- meta ------------------------------------------------------------------
def test_root_and_docs(client):
    assert client.get("/").status_code == 200
    assert client.get("/docs").status_code == 200


def test_health_reports_model_state(client):
    body = client.get(f"{PREFIX}/health").json()
    assert body["database"] == "ok"
    assert "model_loaded" in body and "warnings" in body


def test_openapi_schema_is_valid(client):
    schema = client.get("/openapi.json").json()
    assert f"{PREFIX}/predict/scans" in schema["paths"]
    assert f"{PREFIX}/predict/heatmap/{{scan_id}}/{{disease}}" in schema["paths"]


# --- auth ------------------------------------------------------------------
def test_duplicate_registration_conflicts(client, auth):
    r = client.post(
        f"{PREFIX}/auth/register",
        json={"username": "vansh", "email": "other@example.com", "password": "another-password"},
    )
    assert r.status_code == 409


def test_short_password_rejected(client):
    r = client.post(
        f"{PREFIX}/auth/register",
        json={"username": "bob", "email": "bob@example.com", "password": "short"},
    )
    assert r.status_code == 422


def test_bad_login_is_401(client, auth):
    r = client.post(f"{PREFIX}/auth/login", data={"username": "vansh", "password": "wrong"})
    assert r.status_code == 401


def test_me_returns_profile(client, auth):
    body = client.get(f"{PREFIX}/auth/me", headers=auth).json()
    assert body["username"] == "vansh"
    assert "hashed_password" not in body


def test_refresh_issues_new_access_token(client, auth):
    login = client.post(
        f"{PREFIX}/auth/login", data={"username": "vansh", "password": "a-strong-password"}
    ).json()
    r = client.post(f"{PREFIX}/auth/refresh", json={"refresh_token": login["refresh_token"]})
    assert r.status_code == 200 and r.json()["access_token"]


def test_access_token_rejected_as_refresh(client, auth):
    token = auth["Authorization"].split()[1]
    assert client.post(f"{PREFIX}/auth/refresh", json={"refresh_token": token}).status_code == 401


# --- authorisation ---------------------------------------------------------
def test_upload_requires_auth(client, png_bytes):
    """v1 injected a MockUser under DEBUG, so this endpoint was wide open."""
    r = client.post(f"{PREFIX}/predict", files={"file": ("x.png", png_bytes, "image/png")})
    assert r.status_code == 401


@pytest.mark.parametrize(
    "path", ["/predict/scans", "/predict/status/1", "/predict/results/1", "/reports/1"]
)
def test_protected_endpoints_reject_anonymous(client, path):
    assert client.get(f"{PREFIX}{path}").status_code == 401


def test_other_users_scan_is_not_visible(client, auth, png_bytes):
    scan_id = client.post(
        f"{PREFIX}/predict", headers=auth, files={"file": ("x.png", png_bytes, "image/png")}
    ).json()["scan_id"]

    client.post(
        f"{PREFIX}/auth/register",
        json={"username": "mallory", "email": "m@example.com", "password": "another-password"},
    )
    token = client.post(
        f"{PREFIX}/auth/login", data={"username": "mallory", "password": "another-password"}
    ).json()["access_token"]

    r = client.get(
        f"{PREFIX}/predict/status/{scan_id}", headers={"Authorization": f"Bearer {token}"}
    )
    assert r.status_code == 404  # not 403 -- do not confirm the id exists


# --- upload validation -----------------------------------------------------
def test_unsupported_extension_rejected(client, auth):
    r = client.post(
        f"{PREFIX}/predict",
        headers=auth,
        files={"file": ("notes.txt", b"hello", "text/plain")},
    )
    assert r.status_code == 415


def test_oversized_upload_rejected(client, auth, monkeypatch):
    from services.api.core.settings import get_settings

    monkeypatch.setattr(get_settings(), "max_upload_mb", 0)
    r = client.post(
        f"{PREFIX}/predict",
        headers=auth,
        files={"file": ("big.png", b"x" * (2 * 1024 * 1024), "image/png")},
    )
    assert r.status_code == 413


# --- full pipeline ---------------------------------------------------------
@pytest.mark.slow
def test_upload_to_report_end_to_end(client, auth, png_bytes):
    upload = client.post(
        f"{PREFIX}/predict",
        headers=auth,
        files={"file": ("chest.png", png_bytes, "image/png")},
        data={"patient_ref": "TEST001", "patient_age": "54", "patient_sex": "F"},
    )
    assert upload.status_code == 202
    scan_id = upload.json()["scan_id"]

    status = client.get(f"{PREFIX}/predict/status/{scan_id}", headers=auth).json()
    assert status["status"] == "completed"  # eager Celery -> already done

    results = client.get(f"{PREFIX}/predict/results/{scan_id}", headers=auth)
    assert results.status_code == 200
    body = results.json()
    assert len(body["findings"]) == len(body["label_set"])
    assert body["model_version"] == "untrained-demo"  # never mistakable for real
    assert all(0.0 <= f["probability"] <= 1.0 for f in body["findings"])
    assert body["findings"] == sorted(body["findings"], key=lambda f: -f["probability"])

    with_maps = [f for f in body["findings"] if f["heatmap_url"]]
    assert with_maps, "Grad-CAM overlays should have been persisted"
    img = client.get(with_maps[0]["heatmap_url"], headers=auth)
    assert img.status_code == 200 and img.headers["content-type"] == "image/png"

    report = client.get(f"{PREFIX}/reports/{scan_id}", headers=auth)
    assert report.status_code == 200
    assert "disclaimer" in report.json()["report"]

    pdf = client.get(f"{PREFIX}/reports/{scan_id}/pdf", headers=auth)
    assert pdf.status_code == 200 and pdf.content[:4] == b"%PDF"

    listing = client.get(f"{PREFIX}/predict/scans", headers=auth).json()
    assert listing["total"] == 1 and listing["items"][0]["id"] == scan_id

    assert client.delete(f"{PREFIX}/predict/{scan_id}", headers=auth).status_code == 204
    assert client.get(f"{PREFIX}/predict/scans", headers=auth).json()["total"] == 0


@pytest.mark.slow
def test_heatmap_path_traversal_blocked(client, auth, png_bytes):
    scan_id = client.post(
        f"{PREFIX}/predict", headers=auth, files={"file": ("x.png", png_bytes, "image/png")}
    ).json()["scan_id"]
    r = client.get(f"{PREFIX}/predict/heatmap/{scan_id}/..%2F..%2F..%2Fetc%2Fpasswd", headers=auth)
    assert r.status_code == 404


def test_results_before_completion_is_425(client, auth, monkeypatch):
    """A client polling too early gets a clear 'too early', not a 500."""
    from services.api.core import tasks

    monkeypatch.setattr(tasks.process_scan, "delay", lambda *a, **k: type("T", (), {"id": "x"})())
    import cv2

    arr = np.zeros((32, 32, 3), dtype=np.uint8)
    buf = cv2.imencode(".png", arr)[1].tobytes()
    scan_id = client.post(
        f"{PREFIX}/predict", headers=auth, files={"file": ("x.png", buf, "image/png")}
    ).json()["scan_id"]
    assert client.get(f"{PREFIX}/predict/results/{scan_id}", headers=auth).status_code == 425
