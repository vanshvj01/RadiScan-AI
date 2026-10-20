"""Clinical report retrieval and PDF export."""

from __future__ import annotations

import difflib
import io
from datetime import date, datetime, timedelta, timezone

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from radiscan.reporting.patient_summary import FINDINGS as PATIENT_LABELS
from radiscan.reporting.patient_summary import build_summary
from services.api.core.access import viewable_scan
from services.api.core.clinical import latest_report, require_clinician
from services.api.core.security import current_user
from services.api.db.models import (
    AuditLog,
    DoctorReport,
    FollowUp,
    Prediction,
    Reminder,
    Report,
    Scan,
    User,
)
from services.api.db.session import get_db
from services.api.schemas.schemas import (
    DoctorReportIn,
    DoctorReportState,
    DoctorReportVersion,
    PatientSummary,
    ReportOut,
)

router = APIRouter(prefix="/reports", tags=["reports"])


def _report_for(scan_id: int, db: Session, user: User) -> tuple[Scan, Report]:
    scan = viewable_scan(scan_id, db, user)
    prediction = db.query(Prediction).filter(Prediction.scan_id == scan.id).first()
    report = (
        db.query(Report).filter(Report.prediction_id == prediction.id).first()
        if prediction
        else None
    )
    if report is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "No report generated for this scan yet")
    return scan, report


@router.get("/{scan_id}", response_model=ReportOut)
def get_report(
    scan_id: int, db: Session = Depends(get_db), user: User = Depends(current_user)
) -> ReportOut:
    _, report = _report_for(scan_id, db, user)
    return ReportOut(
        scan_id=scan_id,
        risk_level=report.payload.get("risk_level", "unknown"),
        summary=report.summary,
        report=report.payload,
        text=report.text,
        created_at=report.created_at,
    )


# --------------------------------------------------------------------------
# Doctor-edited, signed report
# --------------------------------------------------------------------------
def _ai_draft(report: Report) -> dict[str, str]:
    """The AI report reduced to the three fields a doctor edits."""
    payload = report.payload
    findings = [
        f"{f['label']}: {f['probability'] * 100:.0f}% "
        f"(95% CI {f['ci_low'] * 100:.0f}-{f['ci_high'] * 100:.0f}%)"
        for f in payload.get("positive_findings", [])
    ]
    findings += [
        f"{f['label']}: indeterminate, {f['probability'] * 100:.0f}% "
        f"(95% CI {f['ci_low'] * 100:.0f}-{f['ci_high'] * 100:.0f}%) - needs review"
        for f in payload.get("deferred_findings", [])
    ]
    return {
        "impression": report.summary,
        "findings_text": "\n".join(findings),
        "recommendations": "\n".join(payload.get("recommendations", [])),
    }


def _edit_ratio(draft: dict[str, str], impression: str, findings: str, recs: str) -> float:
    before = "\n".join([draft["impression"], draft["findings_text"], draft["recommendations"]])
    after = "\n".join([impression, findings, recs])
    return round(1.0 - difflib.SequenceMatcher(None, before, after).ratio(), 3)


def _version_out(r: DoctorReport) -> DoctorReportVersion:
    return DoctorReportVersion(
        version=r.version,
        author_username=r.author.username,
        author_name=r.author.full_name,
        impression=r.impression,
        findings_text=r.findings_text,
        recommendations=r.recommendations,
        patient_findings=list(r.patient_findings or []),
        follow_up=r.follow_up,
        follow_up_weeks=r.follow_up_weeks,
        patient_note=r.patient_note,
        edit_ratio=r.edit_ratio,
        created_at=r.created_at,
        signed_at=r.signed_at,
    )


def _status(r: DoctorReport | None) -> str:
    return "none" if r is None else ("signed" if r.signed else "draft")


@router.get("/{scan_id}/doctor", response_model=DoctorReportState)
def get_doctor_report(
    scan_id: int, db: Session = Depends(get_db), user: User = Depends(current_user)
) -> DoctorReportState:
    """The doctor's report on this scan, with the AI draft and every version.

    A patient sees only a signed report, and only its latest version: drafts
    and the edit history are part of the clinical record, not the result.
    """
    scan, report = _report_for(scan_id, db, user)
    latest = latest_report(db, scan.id)
    if not user.is_clinician:
        if latest is None or not latest.signed:
            return DoctorReportState(
                scan_id=scan.id, status="none", ai_draft={}, latest=None, history=[]
            )
        return DoctorReportState(
            scan_id=scan.id,
            status="signed",
            ai_draft={},
            latest=_version_out(latest),
            history=[],
        )
    history = (
        db.query(DoctorReport)
        .filter(DoctorReport.scan_id == scan.id)
        .order_by(DoctorReport.version.desc())
        .all()
    )
    return DoctorReportState(
        scan_id=scan.id,
        status=_status(latest),  # type: ignore[arg-type]
        ai_draft=_ai_draft(report),
        latest=_version_out(latest) if latest else None,
        history=[_version_out(r) for r in history],
    )


@router.post("/{scan_id}/doctor", response_model=DoctorReportState)
def save_doctor_report(
    scan_id: int,
    payload: DoctorReportIn,
    request: Request,
    db: Session = Depends(get_db),
    user: User = Depends(current_user),
) -> DoctorReportState:
    """Save a new version of my report on this scan; optionally sign it.

    Signing with a follow-up interval on a scan that belongs to a patient also
    creates a reminder in that patient's account.
    """
    require_clinician(user)
    scan, report = _report_for(scan_id, db, user)
    unknown = [lb for lb in payload.patient_findings if lb not in PATIENT_LABELS]
    if unknown:
        raise HTTPException(422, f"Unknown findings: {unknown}")

    latest = latest_report(db, scan.id)
    now = datetime.now(timezone.utc)
    impression = payload.impression.strip()
    findings = payload.findings_text.strip()
    recs = payload.recommendations.strip()
    row = DoctorReport(
        scan_id=scan.id,
        version=(latest.version + 1) if latest else 1,
        author_id=user.id,
        impression=impression,
        findings_text=findings,
        recommendations=recs,
        patient_findings=list(dict.fromkeys(payload.patient_findings)),
        follow_up=payload.follow_up,
        follow_up_weeks=payload.follow_up_weeks,
        patient_note=(payload.patient_note or "").strip() or None,
        edit_ratio=_edit_ratio(_ai_draft(report), impression, findings, recs),
        created_at=now,
        signed_at=now if payload.sign else None,
    )
    db.add(row)
    action = "report.sign" if payload.sign else ("report.amend" if latest else "report.draft")
    db.add(
        AuditLog(
            user_id=user.id,
            action=action,
            resource=f"scan:{scan.uuid}",
            detail=f"version {row.version}, edited {row.edit_ratio:.0%} of AI draft",
            ip_address=request.client.host if request.client else None,
        )
    )
    if payload.sign and payload.follow_up_weeks and scan.patient_id is not None:
        db.add(
            Reminder(
                patient_id=scan.patient_id,
                created_by_id=user.id,
                scan_id=scan.id,
                due_date=date.today() + timedelta(weeks=payload.follow_up_weeks),
                note=f"Follow-up chest X-ray / review ({payload.follow_up.value})",
            )
        )
    db.commit()
    return get_doctor_report(scan_id, db, user)


@router.get("/{scan_id}/patient-summary", response_model=PatientSummary)
def patient_summary(
    scan_id: int,
    lang: str = Query("en", pattern="^(en|hi|mr)$"),
    db: Session = Depends(get_db),
    user: User = Depends(current_user),
) -> PatientSummary:
    """Plain-language summary for the patient, only once a doctor has signed."""
    scan, _ = _report_for(scan_id, db, user)
    latest = latest_report(db, scan.id)
    if latest is None or not latest.signed:
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            "Awaiting doctor review. A plain-language summary is available once a doctor "
            "has reviewed and signed this report.",
        )
    built = build_summary(
        lang, list(latest.patient_findings or []), latest.follow_up.value, latest.follow_up_weeks
    )
    return PatientSummary(
        scan_id=scan.id,
        language=lang,  # type: ignore[arg-type]
        title=built["title"],
        headline=built["headline"],
        findings=built["findings"],
        no_findings_text=built["no_findings_text"],
        next_steps=built["next_steps"],
        doctor_note=latest.patient_note,
        doctor_note_label=built["doctor_note_label"],
        signed_by=latest.author.full_name or latest.author.username,
        signed_at=latest.signed_at,  # type: ignore[arg-type]
        disclaimer=built["disclaimer"],
    )


@router.get("/{scan_id}/pdf")
def report_pdf(
    scan_id: int, db: Session = Depends(get_db), user: User = Depends(current_user)
) -> StreamingResponse:
    """Render the report as a PDF.

    Uses reportlab, not weasyprint: weasyprint needs a GTK3 runtime that fails
    to install on Windows for most students, and it was a hard dependency in v1
    purely for this one feature.
    """
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.platypus import (
        HRFlowable,
        Paragraph,
        SimpleDocTemplate,
        Spacer,
        Table,
        TableStyle,
    )

    scan, report = _report_for(scan_id, db, user)
    payload = report.payload

    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=A4,
        topMargin=18 * mm,
        bottomMargin=18 * mm,
        leftMargin=18 * mm,
        rightMargin=18 * mm,
        title=f"RadiScan Report {scan.uuid[:8]}",
    )
    styles = getSampleStyleSheet()
    h1 = ParagraphStyle("h1", parent=styles["Heading1"], fontSize=15, spaceAfter=4)
    h2 = ParagraphStyle(
        "h2",
        parent=styles["Heading2"],
        fontSize=10.5,
        spaceBefore=10,
        spaceAfter=4,
        textColor=colors.HexColor("#334155"),
    )
    body = ParagraphStyle("body", parent=styles["BodyText"], fontSize=9, leading=13)
    small = ParagraphStyle("small", parent=body, fontSize=7.5, textColor=colors.HexColor("#64748B"))

    story = [
        Paragraph("RadiScan AI &mdash; Chest Radiograph Analysis", h1),
        HRFlowable(width="100%", color=colors.HexColor("#CBD5E1"), spaceAfter=8),
    ]

    meta = [
        ["Scan ID", scan.uuid],
        ["Patient reference", scan.patient_ref or "-"],
        ["Generated", payload.get("generated_at", "-")],
        ["Model version", payload.get("model_version", "-")],
        ["Risk level", str(payload.get("risk_level", "-")).upper()],
    ]
    quality = (payload.get("technical") or {}).get("quality")
    if quality:
        meta.append(["Image quality", f"{quality['status'].upper()} - {quality['recommendation']}"])
    table = Table(meta, colWidths=[38 * mm, 130 * mm])
    table.setStyle(
        TableStyle(
            [
                ("FONTSIZE", (0, 0), (-1, -1), 8.5),
                ("TEXTCOLOR", (0, 0), (0, -1), colors.HexColor("#64748B")),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("LINEBELOW", (0, 0), (-1, -2), 0.25, colors.HexColor("#E2E8F0")),
            ]
        )
    )
    story += [table, Spacer(1, 6)]

    signed = latest_report(db, scan.id)
    if signed is not None and signed.signed:

        def para(text: str) -> Paragraph:
            safe = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            return Paragraph(safe.replace("\n", "<br/>"), body)

        who = signed.author.full_name or signed.author.username
        story += [
            Paragraph("Report &mdash; reviewed and signed", h2),
            Paragraph(
                f"Signed by {who} (@{signed.author.username}) on "
                f"{signed.signed_at:%d %b %Y %H:%M} UTC &middot; version {signed.version}",
                small,
            ),
            Paragraph("<b>Impression</b>", body),
            para(signed.impression),
        ]
        if signed.findings_text:
            story += [Paragraph("<b>Findings</b>", body), para(signed.findings_text)]
        if signed.recommendations:
            story += [Paragraph("<b>Recommendations</b>", body), para(signed.recommendations)]
        if signed.follow_up != FollowUp.none:
            weeks = f" in {signed.follow_up_weeks} weeks" if signed.follow_up_weeks else ""
            story.append(Paragraph(f"<b>Follow-up:</b> {signed.follow_up.value}{weeks}", body))
        story += [Spacer(1, 4), Paragraph("Automated analysis (AI draft)", h2)]
    else:
        story.append(Paragraph("NOT YET REVIEWED BY A DOCTOR &mdash; automated analysis only.", h2))

    story += [Paragraph("Impression", h2), Paragraph(payload.get("summary", ""), body)]

    positives = payload.get("positive_findings", [])
    if positives:
        story.append(Paragraph("Findings", h2))
        rows = [["Finding", "Probability", "95% CI", "Uncertainty", "Reliability"]]
        rows += [
            [
                f["label"],
                f"{f['probability'] * 100:.1f}%",
                f"{f['ci_low'] * 100:.0f}–{f['ci_high'] * 100:.0f}%",
                f"±{f['uncertainty']:.3f}",
                f["reliability"],
            ]
            for f in positives
        ]
        t = Table(rows, colWidths=[52 * mm, 26 * mm, 30 * mm, 28 * mm, 28 * mm])
        t.setStyle(
            TableStyle(
                [
                    ("FONTSIZE", (0, 0), (-1, -1), 8),
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#F1F5F9")),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#E2E8F0")),
                    ("ALIGN", (1, 1), (-1, -1), "RIGHT"),
                    ("TOPPADDING", (0, 0), (-1, -1), 4),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ]
            )
        )
        story += [t]

    deferred = payload.get("deferred_findings", [])
    if deferred:
        story.append(Paragraph("Indeterminate &mdash; referred for human review", h2))
        for f in deferred:
            story.append(
                Paragraph(
                    f"{f['label']}: {f['probability'] * 100:.1f}% "
                    f"(95% CI {f['ci_low'] * 100:.0f}–{f['ci_high'] * 100:.0f}%), "
                    f"uncertainty ±{f['uncertainty']:.3f}",
                    body,
                )
            )

    story.append(Paragraph("Recommendations", h2))
    for i, rec in enumerate(payload.get("recommendations", []), 1):
        story.append(Paragraph(f"{i}. {rec}", body))

    negatives = payload.get("negative_findings", [])
    if negatives:
        story += [Paragraph("Negative for", h2), Paragraph(", ".join(negatives), body)]

    story += [
        Spacer(1, 10),
        HRFlowable(width="100%", color=colors.HexColor("#CBD5E1")),
        Paragraph(payload.get("disclaimer", ""), small),
    ]

    doc.build(story)
    buf.seek(0)
    return StreamingResponse(
        buf,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="radiscan_{scan.uuid[:8]}.pdf"'},
    )
