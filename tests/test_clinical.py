"""Clinical workflow: quality gate, review of model calls, worklist, signed
reports, prior comparison, patient summaries, share codes and reminders.

The access property from test_sharing.py holds here too: nothing below lets a
role, on its own, reach a patient's data. Reviews, reports and reminders all go
through the same consent check.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import cv2
import numpy as np
import pytest

from radiscan.constants import CHEXPERT_LABELS
from radiscan.data.quality import assess_quality
from radiscan.reporting.patient_summary import FINDINGS, LANGUAGES, TEXT, build_summary

PREFIX = "/api/v1"


# --------------------------------------------------------------------------
# Quality gate (pure functions -- no API needed)
# --------------------------------------------------------------------------
def _film(size: int = 1024, seed: int = 0) -> np.ndarray:
    """A crude stand-in for a radiograph: smooth anatomy-like structure plus
    mild noise, with real contrast and edges."""
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:size, 0:size] / size
    img = 120 + 70 * np.sin(6 * x) * np.cos(5 * y) + rng.normal(0, 12, (size, size))
    img[:, size // 2 - 4 : size // 2 + 4] = 230  # a spine-like bright band
    return np.clip(img, 0, 255).astype(np.uint8)


def test_quality_good_film_passes():
    q = assess_quality(_film())
    assert q["status"] == "good", q
    assert q["issues"] == []


def test_quality_flags_blur_darkness_and_size():
    blurred = cv2.GaussianBlur(_film(), (0, 0), 12)
    assert any(i["code"] == "blurred" for i in assess_quality(blurred)["issues"])

    dark = (_film() * 0.12).astype(np.uint8)
    q = assess_quality(dark)
    assert q["status"] == "poor"
    assert any(i["code"] == "underexposed" for i in q["issues"])

    small = cv2.resize(_film(), (200, 200))
    q = assess_quality(small)
    assert any(i["code"] == "low_resolution" and i["level"] == "poor" for i in q["issues"])


def test_quality_flags_photographed_film():
    steps = ["converted colour photo to greyscale", "located the X-ray film in the photo"]
    q = assess_quality(_film(), steps)
    assert q["status"] == "fair"
    assert [i["code"] for i in q["issues"]] == ["photo_of_film"]


def test_quality_records_but_does_not_judge_model_confidence():
    q = assess_quality(_film(), model_confidence=np.full(14, 0.05))
    assert q["metrics"]["model_confidence"] == pytest.approx(0.05)
    assert q["status"] == "good"


# --------------------------------------------------------------------------
# Patient summary templates
# --------------------------------------------------------------------------
def test_every_label_and_string_is_written_in_every_language():
    assert set(FINDINGS) == set(CHEXPERT_LABELS)
    for label, langs in FINDINGS.items():
        for lang in LANGUAGES:
            name, explanation = langs[lang]
            assert name.strip() and explanation.strip(), (label, lang)
    keys = set(TEXT["en"])
    for lang in LANGUAGES:
        assert set(TEXT[lang]) == keys, lang


def test_summary_follows_doctor_choices():
    s = build_summary("hi", ["Pneumothorax", "Not A Label"], "soon", 6)
    assert [f["label"] for f in s["findings"]] == ["Pneumothorax"]
    assert "6" in s["next_steps"]
    urgent = build_summary("mr", [], "urgent", 6)
    assert urgent["next_steps"] == TEXT["mr"]["step_urgent"]  # urgency beats the interval
    with pytest.raises(ValueError):
        build_summary("fr", [], "none", None)


# --------------------------------------------------------------------------
# API helpers
# --------------------------------------------------------------------------
pytest.importorskip("fastapi")


def _account(client, username: str, role: str) -> dict:
    r = client.post(
        f"{PREFIX}/auth/register",
        json={
            "username": username,
            "email": f"{username}@example.com",
            "password": "a-strong-password",
            "full_name": username.title(),
            "role": role,
        },
    )
    assert r.status_code == 201, r.text
    uid = r.json()["id"]
    r = client.post(
        f"{PREFIX}/auth/login", data={"username": username, "password": "a-strong-password"}
    )
    return {
        "id": uid,
        "name": username,
        "h": {"Authorization": f"Bearer {r.json()['access_token']}"},
    }


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


def _upload(client, who, png) -> int:
    r = client.post(
        f"{PREFIX}/predict", headers=who["h"], files={"file": ("cxr.png", png, "image/png")}
    )
    assert r.status_code == 202, r.text
    return r.json()["scan_id"]


def _db():
    from services.api.db import session as session_mod

    return session_mod.SessionLocal()


@pytest.fixture
def shared_scan(client, people, png_bytes):
    """A patient's scan that drmehta (but not drother) may see."""
    _share(client, people)
    return _upload(client, people["pat"], png_bytes)


# --------------------------------------------------------------------------
# Results carry quality and deferral
# --------------------------------------------------------------------------
def test_results_include_quality_and_deferral(client, people, shared_scan):
    res = client.get(f"{PREFIX}/predict/results/{shared_scan}", headers=people["doc"]["h"]).json()
    assert res["quality"]["status"] in ("good", "fair", "poor")
    assert "metrics" in res["quality"]
    assert all("deferred" in f for f in res["findings"])
    report = client.get(f"{PREFIX}/reports/{shared_scan}", headers=people["doc"]["h"]).json()
    assert report["report"]["technical"]["quality"]["status"] == res["quality"]["status"]


# --------------------------------------------------------------------------
# Review workflow
# --------------------------------------------------------------------------
def test_review_round_trip_and_stats(client, people, shared_scan):
    from services.api.db.models import Prediction

    # Make the model's calls known: Cardiomegaly a confident positive, Edema a
    # confident negative, Pneumothorax deferred (interval straddles threshold).
    with _db() as db:
        p = db.query(Prediction).filter(Prediction.scan_id == shared_scan).one()
        p.probabilities = {
            **p.probabilities,
            "Cardiomegaly": 0.9,
            "Edema": 0.05,
            "Pneumothorax": 0.5,
        }
        p.uncertainties = {
            **p.uncertainties,
            "Cardiomegaly": 0.01,
            "Edema": 0.01,
            "Pneumothorax": 0.02,
        }
        p.confidence_intervals = {
            **p.confidence_intervals,
            "Cardiomegaly": [0.88, 0.92],
            "Edema": [0.04, 0.06],
            "Pneumothorax": [0.4, 0.6],
        }
        p.thresholds = {**p.thresholds, "Cardiomegaly": 0.5, "Edema": 0.5, "Pneumothorax": 0.5}
        db.commit()

    body = {
        "reviews": [
            {"label": "Cardiomegaly", "decision": "confirmed"},
            {"label": "Edema", "decision": "confirmed", "note": "subtle"},
            {"label": "Pneumothorax", "decision": "rejected"},
        ]
    }
    r = client.put(f"{PREFIX}/reviews/{shared_scan}", headers=people["doc"]["h"], json=body)
    assert r.status_code == 200, r.text
    by_label = {x["label"]: x for x in r.json()}
    assert by_label["Pneumothorax"]["was_deferred"] is True
    assert by_label["Cardiomegaly"]["model_positive"] is True
    assert by_label["Edema"]["note"] == "subtle"

    # Changing a verdict updates the row rather than adding one.
    client.put(
        f"{PREFIX}/reviews/{shared_scan}",
        headers=people["doc"]["h"],
        json={"reviews": [{"label": "Edema", "decision": "rejected"}]},
    )
    assert (
        len(client.get(f"{PREFIX}/reviews/{shared_scan}", headers=people["doc"]["h"]).json()) == 3
    )

    stats = client.get(f"{PREFIX}/reviews/stats", headers=people["doc"]["h"]).json()
    assert stats["scans_reviewed"] == 1
    assert stats["model_calls"]["reviewed"] == 2
    assert stats["model_agreement"] == pytest.approx(1.0)  # both confident calls upheld
    assert stats["deferred"]["reviewed"] == 1
    assert stats["deferred_present_rate"] == pytest.approx(0.0)


def test_review_is_for_doctors_with_access_only(client, people, shared_scan):
    body = {"reviews": [{"label": "Cardiomegaly", "decision": "confirmed"}]}
    assert (
        client.put(
            f"{PREFIX}/reviews/{shared_scan}", headers=people["pat"]["h"], json=body
        ).status_code
        == 403
    )
    assert (
        client.put(
            f"{PREFIX}/reviews/{shared_scan}", headers=people["other"]["h"], json=body
        ).status_code
        == 404
    )
    bad = {"reviews": [{"label": "Not A Label", "decision": "confirmed"}]}
    assert (
        client.put(
            f"{PREFIX}/reviews/{shared_scan}", headers=people["doc"]["h"], json=bad
        ).status_code
        == 422
    )


# --------------------------------------------------------------------------
# Worklist
# --------------------------------------------------------------------------
def test_worklist_orders_by_urgency_and_hides_signed(client, people, png_bytes):
    from services.api.db.models import Prediction

    _share(client, people)
    low = _upload(client, people["pat"], png_bytes)
    critical = _upload(client, people["pat"], png_bytes)
    high = _upload(client, people["pat"], png_bytes)
    with _db() as db:
        for sid, risk in ((low, "low"), (critical, "critical"), (high, "high")):
            p = db.query(Prediction).filter(Prediction.scan_id == sid).one()
            p.risk_level = risk
        db.commit()

    wl = client.get(f"{PREFIX}/worklist", headers=people["doc"]["h"]).json()
    order = [i["scan_id"] for i in wl["items"]]
    assert order.index(critical) < order.index(high) < order.index(low)
    assert wl["pending"] == 3

    client.post(
        f"{PREFIX}/reports/{critical}/doctor",
        headers=people["doc"]["h"],
        json={"impression": "Reviewed.", "sign": True},
    )
    wl = client.get(f"{PREFIX}/worklist", headers=people["doc"]["h"]).json()
    assert critical not in [i["scan_id"] for i in wl["items"]]
    assert wl["pending"] == 2
    signed = client.get(f"{PREFIX}/worklist?include_signed=true", headers=people["doc"]["h"]).json()
    assert signed["items"][-1]["scan_id"] == critical
    assert signed["items"][-1]["report_status"] == "signed"

    # Consent still gates the queue: an unrelated doctor sees none of it.
    assert client.get(f"{PREFIX}/worklist", headers=people["other"]["h"]).json()["total"] == 0
    assert client.get(f"{PREFIX}/worklist", headers=people["pat"]["h"]).status_code == 403


# --------------------------------------------------------------------------
# Doctor-signed report and patient summary
# --------------------------------------------------------------------------
def test_signed_report_versions_and_patient_visibility(client, people, shared_scan):
    doc, pat = people["doc"]["h"], people["pat"]["h"]
    state = client.get(f"{PREFIX}/reports/{shared_scan}/doctor", headers=doc).json()
    assert state["status"] == "none"
    draft = state["ai_draft"]

    # Accepting the AI draft unchanged is an edit ratio of zero.
    r = client.post(
        f"{PREFIX}/reports/{shared_scan}/doctor", headers=doc, json={**draft, "sign": False}
    )
    assert r.status_code == 200, r.text
    assert r.json()["status"] == "draft"
    assert r.json()["latest"]["edit_ratio"] == pytest.approx(0.0)

    # The patient sees nothing of a draft, and gets no summary yet.
    assert (
        client.get(f"{PREFIX}/reports/{shared_scan}/doctor", headers=pat).json()["status"] == "none"
    )
    assert (
        client.get(f"{PREFIX}/reports/{shared_scan}/patient-summary", headers=pat).status_code
        == 409
    )

    signed = client.post(
        f"{PREFIX}/reports/{shared_scan}/doctor",
        headers=doc,
        json={
            "impression": "Small right pleural effusion. Otherwise clear.",
            "findings_text": "Blunting of the right costophrenic angle.",
            "recommendations": "Repeat film in 6 weeks.",
            "patient_findings": ["Pleural Effusion"],
            "follow_up": "routine",
            "follow_up_weeks": 6,
            "patient_note": "Nothing to worry about, but let us check again.",
            "sign": True,
        },
    ).json()
    assert signed["status"] == "signed"
    assert signed["latest"]["version"] == 2
    assert [v["version"] for v in signed["history"]] == [2, 1]
    assert signed["latest"]["edit_ratio"] > 0.2

    seen = client.get(f"{PREFIX}/reports/{shared_scan}/doctor", headers=pat).json()
    assert seen["status"] == "signed" and seen["history"] == [] and seen["ai_draft"] == {}

    for lang in ("en", "hi", "mr"):
        s = client.get(f"{PREFIX}/reports/{shared_scan}/patient-summary?lang={lang}", headers=pat)
        assert s.status_code == 200, s.text
        body = s.json()
        assert body["findings"][0]["name"] == FINDINGS["Pleural Effusion"][lang][0]
        assert body["doctor_note"].startswith("Nothing to worry")
        assert "6" in body["next_steps"]

    # Signing with a follow-up interval put a reminder in the patient's account.
    reminders = client.get(f"{PREFIX}/reminders", headers=pat).json()
    assert len(reminders) == 1 and reminders[0]["scan_id"] == shared_scan
    assert 40 <= reminders[0]["days_until"] <= 43

    # The signed PDF still renders.
    pdf = client.get(f"{PREFIX}/reports/{shared_scan}/pdf", headers=doc)
    assert pdf.status_code == 200 and pdf.content[:4] == b"%PDF"


def test_only_doctors_write_reports(client, people, shared_scan):
    body = {"impression": "x", "sign": True}
    assert (
        client.post(
            f"{PREFIX}/reports/{shared_scan}/doctor", headers=people["pat"]["h"], json=body
        ).status_code
        == 403
    )
    assert (
        client.post(
            f"{PREFIX}/reports/{shared_scan}/doctor", headers=people["other"]["h"], json=body
        ).status_code
        == 404
    )
    bad = {"impression": "x", "patient_findings": ["Made Up"]}
    assert (
        client.post(
            f"{PREFIX}/reports/{shared_scan}/doctor", headers=people["doc"]["h"], json=bad
        ).status_code
        == 422
    )


# --------------------------------------------------------------------------
# Comparison with a prior study
# --------------------------------------------------------------------------
def test_compare_with_prior(client, people, png_bytes):
    from services.api.db.models import Prediction, Scan

    _share(client, people)
    first = _upload(client, people["pat"], png_bytes)
    second = _upload(client, people["pat"], png_bytes)
    with _db() as db:
        s1 = db.get(Scan, first)
        s1.created_at = datetime.now(timezone.utc) - timedelta(days=30)
        for sid, prob in ((first, 0.1), (second, 0.9)):
            p = db.query(Prediction).filter(Prediction.scan_id == sid).one()
            p.probabilities = {**p.probabilities, "Pleural Effusion": prob}
            p.uncertainties = {**p.uncertainties, "Pleural Effusion": 0.01}
            p.confidence_intervals = {
                **p.confidence_intervals,
                "Pleural Effusion": [prob - 0.01, prob + 0.01],
            }
            p.thresholds = {**p.thresholds, "Pleural Effusion": 0.5}
        db.commit()

    c = client.get(f"{PREFIX}/predict/compare/{second}", headers=people["doc"]["h"])
    assert c.status_code == 200, c.text
    body = c.json()
    assert body["prior"]["scan_id"] == first
    assert body["days_between"] in (29, 30)
    row = next(r for r in body["rows"] if r["label"] == "Pleural Effusion")
    assert row["trend"] == "new" and row["delta"] == pytest.approx(0.8)
    assert body["rows"][0]["trend"] == "new"  # biggest news first

    # The earliest study has nothing before it.
    assert (
        client.get(f"{PREFIX}/predict/compare/{first}", headers=people["doc"]["h"]).status_code
        == 404
    )
    # A doctor without consent cannot compare.
    assert (
        client.get(f"{PREFIX}/predict/compare/{second}", headers=people["other"]["h"]).status_code
        == 404
    )


def test_compare_rejects_another_patients_scan(client, people, png_bytes):
    other_pat = _account(client, "ravi", "patient")
    mine = _upload(client, people["pat"], png_bytes)
    theirs = _upload(client, other_pat, png_bytes)
    r = client.get(f"{PREFIX}/predict/compare/{mine}?prior_id={theirs}", headers=people["pat"]["h"])
    assert r.status_code == 404


# --------------------------------------------------------------------------
# Share codes
# --------------------------------------------------------------------------
def test_share_code_grants_access_once(client, people, png_bytes):
    scan = _upload(client, people["pat"], png_bytes)
    r = client.post(f"{PREFIX}/sharing/codes", headers=people["pat"]["h"])
    assert r.status_code == 201, r.text
    code = r.json()["code"]
    assert len(code) == 9 and code[4] == "-"

    assert (
        client.get(f"{PREFIX}/predict/results/{scan}", headers=people["doc"]["h"]).status_code
        == 404
    )
    typed = code.lower().replace("-", " ")  # read aloud, typed sloppily
    g = client.post(
        f"{PREFIX}/sharing/codes/redeem", headers=people["doc"]["h"], json={"code": typed}
    )
    assert g.status_code == 201, g.text
    assert g.json()["patient_username"] == "priya"
    assert (
        client.get(f"{PREFIX}/predict/results/{scan}", headers=people["doc"]["h"]).status_code
        == 200
    )

    # Single use.
    again = client.post(
        f"{PREFIX}/sharing/codes/redeem", headers=people["other"]["h"], json={"code": code}
    )
    assert again.status_code == 404

    # Only patients create, only doctors redeem.
    assert client.post(f"{PREFIX}/sharing/codes", headers=people["doc"]["h"]).status_code == 403
    fresh = client.post(f"{PREFIX}/sharing/codes", headers=people["pat"]["h"]).json()["code"]
    assert (
        client.post(
            f"{PREFIX}/sharing/codes/redeem", headers=people["pat"]["h"], json={"code": fresh}
        ).status_code
        == 403
    )


def test_share_code_expires_and_guessing_is_limited(client, people):
    from services.api.db.models import ShareCode

    code = client.post(f"{PREFIX}/sharing/codes", headers=people["pat"]["h"]).json()["code"]
    with _db() as db:
        row = db.query(ShareCode).one()
        row.expires_at = datetime.now(timezone.utc) - timedelta(minutes=1)
        db.commit()
    r = client.post(
        f"{PREFIX}/sharing/codes/redeem", headers=people["doc"]["h"], json={"code": code}
    )
    assert r.status_code == 404

    for _ in range(9):
        client.post(
            f"{PREFIX}/sharing/codes/redeem",
            headers=people["doc"]["h"],
            json={"code": "WRONG-CODE"},
        )
    r = client.post(
        f"{PREFIX}/sharing/codes/redeem", headers=people["doc"]["h"], json={"code": "WRONG-CODE"}
    )
    assert r.status_code == 429


# --------------------------------------------------------------------------
# Reminders
# --------------------------------------------------------------------------
def test_reminders_need_consent_and_can_be_completed(client, people):
    body = {"patient_id": people["pat"]["id"], "weeks": 2, "note": "Repeat chest X-ray"}
    assert (
        client.post(f"{PREFIX}/reminders", headers=people["doc"]["h"], json=body).status_code == 404
    )

    _share(client, people)
    r = client.post(f"{PREFIX}/reminders", headers=people["doc"]["h"], json=body)
    assert r.status_code == 201, r.text
    rid = r.json()["id"]
    assert r.json()["days_until"] == 14 and r.json()["overdue"] is False

    mine = client.get(f"{PREFIX}/reminders", headers=people["pat"]["h"]).json()
    assert [x["id"] for x in mine] == [rid]
    assert [
        x["id"] for x in client.get(f"{PREFIX}/reminders", headers=people["doc"]["h"]).json()
    ] == [rid]

    assert (
        client.patch(
            f"{PREFIX}/reminders/{rid}", headers=people["other"]["h"], json={"status": "done"}
        ).status_code
        == 404
    )
    done = client.patch(
        f"{PREFIX}/reminders/{rid}", headers=people["pat"]["h"], json={"status": "done"}
    )
    assert done.status_code == 200 and done.json()["status"] == "done"

    # After the patient stops sharing, the doctor no longer lists it.
    gid = client.get(f"{PREFIX}/sharing/grants", headers=people["pat"]["h"]).json()[0]["id"]
    client.delete(f"{PREFIX}/sharing/grants/{gid}", headers=people["pat"]["h"])
    assert client.get(f"{PREFIX}/reminders", headers=people["doc"]["h"]).json() == []


def test_deleting_a_scan_clears_its_reviews_and_reports(client, people, png_bytes):
    from services.api.db.models import DoctorReport, FindingReview

    doc_scan = _upload(client, people["doc"], png_bytes)
    client.put(
        f"{PREFIX}/reviews/{doc_scan}",
        headers=people["doc"]["h"],
        json={"reviews": [{"label": "Edema", "decision": "unsure"}]},
    )
    client.post(
        f"{PREFIX}/reports/{doc_scan}/doctor", headers=people["doc"]["h"], json={"impression": "x"}
    )
    assert (
        client.delete(f"{PREFIX}/predict/{doc_scan}", headers=people["doc"]["h"]).status_code == 204
    )
    with _db() as db:
        assert db.query(FindingReview).count() == 0
        assert db.query(DoctorReport).count() == 0


def test_small_move_across_threshold_is_stable(client, people, png_bytes):
    """57% -> 61% across a 59.5% threshold is noise, not a new finding."""
    from services.api.db.models import Prediction, Scan

    _share(client, people)
    first = _upload(client, people["pat"], png_bytes)
    second = _upload(client, people["pat"], png_bytes)
    with _db() as db:
        db.get(Scan, first).created_at = datetime.now(timezone.utc) - timedelta(days=7)
        for sid, prob in ((first, 0.57), (second, 0.61)):
            p = db.query(Prediction).filter(Prediction.scan_id == sid).one()
            p.probabilities = {**p.probabilities, "Pleural Effusion": prob}
            p.uncertainties = {**p.uncertainties, "Pleural Effusion": 0.005}
            p.confidence_intervals = {
                **p.confidence_intervals,
                "Pleural Effusion": [prob - 0.005, prob + 0.005],
            }
            p.thresholds = {**p.thresholds, "Pleural Effusion": 0.595}
        db.commit()
    rows = client.get(f"{PREFIX}/predict/compare/{second}", headers=people["doc"]["h"]).json()[
        "rows"
    ]
    row = next(r for r in rows if r["label"] == "Pleural Effusion")
    assert (row["prior_call"], row["current_call"]) == ("negative", "positive")
    assert row["trend"] == "stable"
