"""Clinical report retrieval and PDF export."""

from __future__ import annotations

import io

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from services.api.core.security import current_user
from services.api.db.models import Prediction, Report, Scan, User
from services.api.db.session import get_db
from services.api.schemas.schemas import ReportOut

router = APIRouter(prefix="/reports", tags=["reports"])


def _report_for(scan_id: int, db: Session, user: User) -> tuple[Scan, Report]:
    scan = db.get(Scan, scan_id)
    if scan is None or (scan.user_id != user.id and user.role.value != "admin"):
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"Scan {scan_id} not found")
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
