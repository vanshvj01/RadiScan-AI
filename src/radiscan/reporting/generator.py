"""Structured clinical report generation.

This is deliberately rule-based, not generative. A language model that
hallucinates a finding into a radiology report is an unacceptable failure mode,
and a rule engine over calibrated probabilities is auditable line by line --
which is the property that matters for a decision-support tool. The roadmap
option of fine-tuning a ClinicalBERT/T5 report generator is a separate,
clearly-labelled experiment, not a replacement for this.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

# Findings whose presence changes management within hours.
CRITICAL_FINDINGS = {
    "Pneumothorax",
    "Pneumonia",
    "Edema",
    "Consolidation",
    "Fracture",
}

FOLLOW_UP = {
    "Pneumothorax": "Urgent clinical correlation; consider immediate chest tube evaluation if large or tension physiology is suspected.",
    "Pneumonia": "Correlate with fever, WBC and sputum culture; consider empiric antibiotics per local protocol.",
    "Edema": "Assess volume status, BNP and echocardiography; review fluid balance.",
    "Cardiomegaly": "Echocardiography recommended to assess chamber size and ejection fraction.",
    "Pleural Effusion": "Consider lateral decubitus film or ultrasound; diagnostic thoracentesis if clinically indicated.",
    "Lung Lesion": "Non-contrast chest CT recommended for characterisation.",
    "Mass": "Contrast-enhanced chest CT and multidisciplinary review recommended.",
    "Nodule": "Follow Fleischner Society surveillance guidance based on size and risk factors.",
    "Fracture": "Correlate with point tenderness and trauma history; dedicated rib views or CT if occult.",
    "Atelectasis": "Consider incentive spirometry; exclude obstructing lesion if persistent.",
    "Consolidation": "Correlate clinically for infection versus aspiration; follow-up film in 4-6 weeks to confirm resolution.",
    "Support Devices": "Verify line and tube positions against the intended anatomic landmarks.",
}


@dataclass
class Finding:
    label: str
    probability: float
    uncertainty: float
    ci_low: float
    ci_high: float
    entropy: float
    threshold: float
    present: bool
    severity: str
    reliability: str

    def sentence(self) -> str:
        pct = f"{self.probability * 100:.0f}%"
        band = f"{self.ci_low * 100:.0f}-{self.ci_high * 100:.0f}%"
        return (
            f"{self.label}: {self.severity} likelihood ({pct}, 95% CI {band}); "
            f"model reliability {self.reliability}."
        )


@dataclass
class ClinicalReport:
    scan_id: str
    generated_at: str
    model_version: str
    label_set: list[str]
    risk_level: str
    summary: str
    positive_findings: list[Finding] = field(default_factory=list)
    negative_findings: list[str] = field(default_factory=list)
    deferred_findings: list[Finding] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)
    technical: dict[str, Any] = field(default_factory=dict)
    disclaimer: str = (
        "AI-generated decision support. Not a diagnosis. This system has no "
        "regulatory clearance and has not been prospectively validated. All "
        "findings require confirmation by a qualified radiologist."
    )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    def to_text(self) -> str:
        lines = [
            "RADISCAN AI - AUTOMATED CHEST RADIOGRAPH ANALYSIS",
            "=" * 62,
            f"Scan ID      : {self.scan_id}",
            f"Generated    : {self.generated_at}",
            f"Model        : {self.model_version}",
            f"Risk level   : {self.risk_level.upper()}",
            "",
            "IMPRESSION",
            "-" * 62,
            self.summary,
            "",
            "FINDINGS",
            "-" * 62,
        ]
        if self.positive_findings:
            lines += [f"  * {f.sentence()}" for f in self.positive_findings]
        else:
            lines.append("  No findings exceeded their operating threshold.")

        if self.deferred_findings:
            lines += [
                "",
                "INDETERMINATE - REFERRED FOR HUMAN REVIEW",
                "-" * 62,
                *[f"  ? {f.sentence()}" for f in self.deferred_findings],
            ]

        if self.negative_findings:
            lines += ["", "NEGATIVE FOR", "-" * 62, "  " + ", ".join(self.negative_findings)]

        lines += ["", "RECOMMENDATIONS", "-" * 62]
        lines += [f"  {i}. {r}" for i, r in enumerate(self.recommendations, 1)] or ["  None."]
        lines += ["", "DISCLAIMER", "-" * 62, "  " + self.disclaimer]
        return "\n".join(lines)


class ReportGenerator:
    def __init__(
        self,
        thresholds: dict[str, float] | None = None,
        uncertainty_defer: float = 0.15,
        severity_bands: tuple[float, float, float] = (0.35, 0.60, 0.85),
    ):
        self.thresholds = thresholds or {}
        self.uncertainty_defer = uncertainty_defer
        self.low, self.moderate, self.high = severity_bands

    # -- helpers -------------------------------------------------------------
    def _severity(self, p: float) -> str:
        if p >= self.high:
            return "high"
        if p >= self.moderate:
            return "moderate"
        if p >= self.low:
            return "low"
        return "minimal"

    @staticmethod
    def _reliability(std: float) -> str:
        if std < 0.05:
            return "high"
        if std < 0.12:
            return "moderate"
        return "low"

    def _risk_level(self, positives: list[Finding]) -> str:
        if not positives:
            return "low"
        critical = [f for f in positives if f.label in CRITICAL_FINDINGS]
        top = max(f.probability for f in positives)
        if critical and max(f.probability for f in critical) >= self.high:
            return "critical"
        if top >= self.high or len(positives) >= 3:
            return "high"
        if top >= self.moderate:
            return "moderate"
        return "low"

    # -- main ----------------------------------------------------------------
    def generate(
        self,
        scan_id: str,
        labels: list[str] | tuple[str, ...],
        prediction: dict[str, Any],
        model_version: str = "unknown",
        technical: dict[str, Any] | None = None,
    ) -> ClinicalReport:
        mean = prediction["mean"].ravel()
        std = prediction["std"].ravel()
        ci_low = prediction["ci_low"].ravel()
        ci_high = prediction["ci_high"].ravel()
        entropy = prediction["entropy"].ravel()

        positives: list[Finding] = []
        deferred: list[Finding] = []
        negatives: list[str] = []

        for i, label in enumerate(labels):
            if label == "No Finding":
                continue
            thr = self.thresholds.get(label, 0.5)
            finding = Finding(
                label=label,
                probability=float(mean[i]),
                uncertainty=float(std[i]),
                ci_low=float(ci_low[i]),
                ci_high=float(ci_high[i]),
                entropy=float(entropy[i]),
                threshold=float(thr),
                present=bool(mean[i] >= thr),
                severity=self._severity(float(mean[i])),
                reliability=self._reliability(float(std[i])),
            )

            # A prediction whose confidence interval straddles its own operating
            # threshold is not a call the system should be making alone.
            straddles = finding.ci_low < thr <= finding.ci_high
            if straddles or finding.uncertainty >= self.uncertainty_defer:
                deferred.append(finding)
            elif finding.present:
                positives.append(finding)
            else:
                negatives.append(label)

        positives.sort(key=lambda f: f.probability, reverse=True)
        deferred.sort(key=lambda f: f.uncertainty, reverse=True)
        risk = self._risk_level(positives)

        if positives:
            names = ", ".join(f.label for f in positives[:3])
            summary = (
                f"Findings suggestive of {names}"
                + (" and others" if len(positives) > 3 else "")
                + f". Overall risk stratification: {risk}."
            )
        elif deferred:
            summary = (
                "No finding met its operating threshold with sufficient confidence. "
                f"{len(deferred)} finding(s) are indeterminate and referred for radiologist review."
            )
        else:
            summary = "No acute cardiopulmonary abnormality detected by the model."

        recs: list[str] = []
        for f in positives:
            if f.label in FOLLOW_UP:
                recs.append(FOLLOW_UP[f.label])
        if deferred:
            recs.append(
                "Radiologist review required for the indeterminate findings listed above "
                "before this study is used for clinical decision-making."
            )
        if risk in ("high", "critical"):
            recs.insert(0, "Expedite radiologist review; prioritise in the reporting queue.")
        if not recs:
            recs.append("Routine reporting workflow. Correlate with clinical presentation.")

        return ClinicalReport(
            scan_id=scan_id,
            generated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            model_version=model_version,
            label_set=list(labels),
            risk_level=risk,
            summary=summary,
            positive_findings=positives,
            negative_findings=negatives,
            deferred_findings=deferred,
            recommendations=list(dict.fromkeys(recs)),
            technical=technical or {},
        )
