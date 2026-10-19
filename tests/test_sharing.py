"""Doctor and patient accounts, consent-based history sharing.

The property under test throughout: no role on its own opens anyone's history.
Only a patient's active grant does, and revoking it closes access at once.
"""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi")

PREFIX = "/api/v1"


def _account(client, username: str, role: str) -> dict:
    payload = {
        "username": username,
        "email": f"{username}@example.com",
        "password": "a-strong-password",
        "full_name": username.title(),
        "role": role,
    }
    r = client.post(f"{PREFIX}/auth/register", json=payload)
    assert r.status_code == 201, r.text
    uid = r.json()["id"]
    r = client.post(
        f"{PREFIX}/auth/login", data={"username": username, "password": "a-strong-password"}
    )
    return {"id": uid, "h": {"Authorization": f"Bearer {r.json()['access_token']}"}}


def _upload(client, who, png, **form):
    r = client.post(
        f"{PREFIX}/predict",
        headers=who["h"],
        files={"file": ("cxr.png", png, "image/png")},
        data=form,
    )
    return r


@pytest.fixture
def people(client):
    return {
        "pat": _account(client, "priya", "patient"),
        "doc": _account(client, "drmehta", "doctor"),
        "other": _account(client, "drother", "doctor"),
    }


def _share(client, people, doctor="drmehta"):
    r = client.post(
        f"{PREFIX}/sharing/grants", headers=people["pat"]["h"], json={"doctor_username": doctor}
    )
    assert r.status_code == 201, r.text
    return r.json()


def test_admin_cannot_be_self_assigned(client):
    r = client.post(
        f"{PREFIX}/auth/register",
        json={
            "username": "sneaky",
            "email": "s@example.com",
            "password": "a-strong-password",
            "role": "admin",
        },
    )
    assert r.status_code == 422


def test_role_is_reported_by_me(client, people):
    assert client.get(f"{PREFIX}/auth/me", headers=people["doc"]["h"]).json()["role"] == "doctor"
    assert client.get(f"{PREFIX}/auth/me", headers=people["pat"]["h"]).json()["role"] == "patient"


def test_patient_upload_joins_own_history(client, people, png_bytes):
    scan_id = _upload(client, people["pat"], png_bytes).json()["scan_id"]
    res = client.get(f"{PREFIX}/predict/results/{scan_id}", headers=people["pat"]["h"]).json()
    assert res["patient_id"] == people["pat"]["id"]
    assert res["patient_username"] == "priya"


def test_doctor_sees_nothing_without_consent(client, people, png_bytes):
    scan_id = _upload(client, people["pat"], png_bytes).json()["scan_id"]
    pid = people["pat"]["id"]
    assert (
        client.get(f"{PREFIX}/predict/results/{scan_id}", headers=people["doc"]["h"]).status_code
        == 404
    )
    assert (
        client.get(
            f"{PREFIX}/sharing/patients/{pid}/history", headers=people["doc"]["h"]
        ).status_code
        == 404
    )
    assert client.get(f"{PREFIX}/predict/scans", headers=people["doc"]["h"]).json()["total"] == 0


def test_grant_opens_history_to_that_doctor_only(client, people, png_bytes):
    scan_id = _upload(client, people["pat"], png_bytes).json()["scan_id"]
    pid = people["pat"]["id"]
    _share(client, people)

    hist = client.get(f"{PREFIX}/sharing/patients/{pid}/history", headers=people["doc"]["h"])
    assert hist.status_code == 200
    assert [i["scan_id"] for i in hist.json()["items"]] == [scan_id]
    assert hist.json()["items"][0]["probabilities"]  # findings available for comparison
    assert (
        client.get(f"{PREFIX}/predict/results/{scan_id}", headers=people["doc"]["h"]).status_code
        == 200
    )
    assert client.get(f"{PREFIX}/reports/{scan_id}", headers=people["doc"]["h"]).status_code == 200

    # A different doctor still sees nothing.
    assert (
        client.get(f"{PREFIX}/predict/results/{scan_id}", headers=people["other"]["h"]).status_code
        == 404
    )
    assert (
        client.get(
            f"{PREFIX}/sharing/patients/{pid}/history", headers=people["other"]["h"]
        ).status_code
        == 404
    )


def test_doctor_sees_shared_patients(client, people):
    _share(client, people)
    grants = client.get(f"{PREFIX}/sharing/grants", headers=people["doc"]["h"]).json()
    assert [g["patient_username"] for g in grants] == ["priya"]
    mine = client.get(f"{PREFIX}/sharing/grants", headers=people["pat"]["h"]).json()
    assert [g["doctor_username"] for g in mine] == ["drmehta"]


def test_sharing_twice_is_idempotent(client, people):
    a, b = _share(client, people), _share(client, people)
    assert a["id"] == b["id"]
    assert len(client.get(f"{PREFIX}/sharing/grants", headers=people["pat"]["h"]).json()) == 1


def test_doctor_upload_needs_consent_to_join_history(client, people, png_bytes):
    r = _upload(client, people["doc"], png_bytes, patient_username="priya")
    assert r.status_code == 403  # no write into a stranger's record

    _share(client, people)
    scan_id = _upload(client, people["doc"], png_bytes, patient_username="priya").json()["scan_id"]
    # ...and the patient sees the doctor's upload in their own history.
    hist = client.get(
        f"{PREFIX}/sharing/patients/{people['pat']['id']}/history", headers=people["pat"]["h"]
    ).json()
    assert scan_id in [i["scan_id"] for i in hist["items"]]
    assert hist["items"][0]["uploaded_by"] == "drmehta"


def test_doctor_unlinked_upload_still_works(client, people, png_bytes):
    r = _upload(client, people["doc"], png_bytes, patient_ref="anon-17")
    assert r.status_code == 202
    res = client.get(
        f"{PREFIX}/predict/results/{r.json()['scan_id']}", headers=people["doc"]["h"]
    ).json()
    assert res["patient_id"] is None


def test_revoke_closes_access_but_keeps_doctors_own_uploads(client, people, png_bytes):
    pat_scan = _upload(client, people["pat"], png_bytes).json()["scan_id"]
    grant = _share(client, people)
    doc_scan = _upload(client, people["doc"], png_bytes, patient_username="priya").json()["scan_id"]

    r = client.delete(f"{PREFIX}/sharing/grants/{grant['id']}", headers=people["pat"]["h"])
    assert r.status_code == 204

    assert (
        client.get(f"{PREFIX}/predict/results/{pat_scan}", headers=people["doc"]["h"]).status_code
        == 404
    )
    assert (
        client.get(
            f"{PREFIX}/sharing/patients/{people['pat']['id']}/history", headers=people["doc"]["h"]
        ).status_code
        == 404
    )
    assert (
        client.get(f"{PREFIX}/predict/results/{doc_scan}", headers=people["doc"]["h"]).status_code
        == 200
    )


def test_only_patients_grant_and_only_to_doctors(client, people):
    r = client.post(
        f"{PREFIX}/sharing/grants", headers=people["doc"]["h"], json={"doctor_username": "drother"}
    )
    assert r.status_code == 403
    _account(client, "rahul", "patient")
    r = client.post(
        f"{PREFIX}/sharing/grants", headers=people["pat"]["h"], json={"doctor_username": "rahul"}
    )
    assert r.status_code == 404


def test_patient_cannot_revoke_someone_elses_grant(client, people):
    grant = _share(client, people)
    other_patient = _account(client, "rahul", "patient")
    r = client.delete(f"{PREFIX}/sharing/grants/{grant['id']}", headers=other_patient["h"])
    assert r.status_code == 404


def test_shared_access_does_not_allow_deleting(client, people, png_bytes):
    scan_id = _upload(client, people["pat"], png_bytes).json()["scan_id"]
    _share(client, people)
    assert (
        client.delete(f"{PREFIX}/predict/{scan_id}", headers=people["doc"]["h"]).status_code == 403
    )
    assert (
        client.delete(f"{PREFIX}/predict/{scan_id}", headers=people["pat"]["h"]).status_code == 204
    )


def test_history_views_are_audited(client, people):
    from services.api.db import session as session_mod
    from services.api.db.models import AuditLog

    _share(client, people)
    client.get(
        f"{PREFIX}/sharing/patients/{people['pat']['id']}/history", headers=people["doc"]["h"]
    )
    db = session_mod.SessionLocal()
    try:
        actions = {a.action for a in db.query(AuditLog).all()}
    finally:
        db.close()
    assert {"sharing.grant", "history.view"} <= actions


def test_old_database_is_upgraded_in_place(tmp_path):
    """A scans table from before the split gains patient_id without data loss."""
    import sqlalchemy as sa

    from services.api.db.migrate import upgrade

    engine = sa.create_engine(f"sqlite:///{tmp_path / 'old.db'}")
    with engine.begin() as c:
        c.execute(sa.text("CREATE TABLE users (id INTEGER PRIMARY KEY)"))
        c.execute(sa.text("CREATE TABLE scans (id INTEGER PRIMARY KEY, user_id INTEGER)"))
        c.execute(sa.text("INSERT INTO scans (id, user_id) VALUES (1, 1)"))

    assert upgrade(engine) == ["scans.patient_id"]
    assert upgrade(engine) == []  # idempotent
    with engine.connect() as c:
        assert c.execute(sa.text("SELECT id, patient_id FROM scans")).all() == [(1, None)]
