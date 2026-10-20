"""The single entry point for making a prediction.

Everything downstream -- CLI, Celery worker, FastAPI route, notebook -- goes
through :class:`RadiScanPredictor`. One code path means the number shown in the
UI is provably the number the evaluation script measured.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from radiscan.data.enhance import enhance
from radiscan.data.loader import load_image
from radiscan.data.quality import assess_quality
from radiscan.data.transforms import apply_clahe, eval_transforms
from radiscan.models.network import RadiScanNet, load_model
from radiscan.modules.explainability import GradCAM, save_overlay
from radiscan.modules.uncertainty import predict_with_uncertainty, reliability_score
from radiscan.reporting.generator import ReportGenerator
from radiscan.utils.device import resolve_device

log = logging.getLogger(__name__)


class RadiScanPredictor:
    def __init__(
        self,
        checkpoint: str | Path | None = None,
        device: str | None = None,
        mc_samples: int = 20,
        tta_hflip: bool = True,
        model: RadiScanNet | None = None,
        meta: dict[str, Any] | None = None,
    ):
        self.device = resolve_device(device)
        self.mc_samples = mc_samples
        self.tta_hflip = tta_hflip

        if model is not None:
            self.model = model.to(self.device).eval()
            self.meta = meta or {}
        else:
            if checkpoint is None:
                raise ValueError("Provide either `checkpoint` or `model`.")
            self.model, self.meta = load_model(checkpoint, device=self.device)

        self.labels = tuple(self.model.labels)
        self.image_size = int(self.meta.get("image_size", self.model.image_size))
        self.thresholds: dict[str, float] = dict(self.meta.get("thresholds", {}))
        self.model_version = self.meta.get(
            "model_version",
            f"{self.model.backbone_name}-{self.meta.get('epoch', '?')}ep-v{self.meta.get('radiscan_version', '2')}",
        )
        self.transform = eval_transforms(self.image_size)
        # Apply to model input only what this checkpoint was trained with.
        # Checkpoints from before the clean-up step existed carry no flag and
        # keep their original behaviour.
        self.auto_crop = bool(
            (self.meta.get("config") or {}).get("data", {}).get("auto_crop", False)
        )
        self.reporter = ReportGenerator(thresholds=self.thresholds)

    # -- preprocessing -------------------------------------------------------
    def prepare(self, source: str | Path | bytes, filename: str | None = None):
        """Return (model_view, tensor, meta, display).

        ``model_view`` is the image exactly as the model saw it, before CLAHE and
        normalisation -- Grad-CAM must be overlaid on this, not on the upload,
        or a cropped scan's saliency lands in the wrong place.
        ``display`` is the cleaned-up image shown to the reader: cropped,
        denoised and contrast-enhanced, at display resolution.
        """
        raw, meta = load_image(source, filename=filename)
        model_view = (
            enhance(raw, reduce_noise=False, max_side=None).image if self.auto_crop else raw
        )
        tensor = self.transform(image=apply_clahe(model_view))["image"].unsqueeze(0).to(self.device)

        cleaned = enhance(raw, reduce_noise=True, max_side=1024)
        display = {
            "image": apply_clahe(cleaned.image),
            "steps": cleaned.steps + ["enhanced contrast (CLAHE)"],
            # Quality is judged on the upload itself, before any clean-up hides
            # its problems; the clean-up notes reveal a photographed film.
            "quality": assess_quality(raw, cleaned.steps),
        }
        return model_view, tensor, meta, display

    # -- prediction ----------------------------------------------------------
    def predict(
        self,
        source: str | Path | bytes,
        filename: str | None = None,
        explain: bool = True,
        heatmap_dir: str | Path | None = None,
        top_k_heatmaps: int = 3,
        scan_id: str = "unsaved",
    ) -> dict[str, Any]:
        t0 = time.perf_counter()
        raw_image, tensor, image_meta, display = self.prepare(source, filename)

        unc = predict_with_uncertainty(
            self.model,
            tensor,
            n_samples=self.mc_samples,
            tta_hflip=self.tta_hflip,
            calibrated=True,
        )
        inference_ms = (time.perf_counter() - t0) * 1000

        mean = unc["mean"][0]
        std = unc["std"][0]
        reliab = reliability_score(std, unc["entropy"][0])

        predictions = []
        for i, label in enumerate(self.labels):
            thr = self.thresholds.get(label, 0.5)
            predictions.append(
                {
                    "label": label,
                    "probability": float(mean[i]),
                    "uncertainty": float(std[i]),
                    "ci_low": float(unc["ci_low"][0][i]),
                    "ci_high": float(unc["ci_high"][0][i]),
                    "entropy": float(unc["entropy"][0][i]),
                    "mutual_information": float(unc["mutual_information"][0][i]),
                    "reliability": float(reliab[i]),
                    "threshold": float(thr),
                    "positive": bool(mean[i] >= thr),
                }
            )

        if heatmap_dir is not None:
            self._save_enhanced(display, Path(heatmap_dir) / str(scan_id))

        heatmaps: dict[str, str] = {}
        if explain:
            heatmaps = self._explain(tensor, raw_image, mean, heatmap_dir, top_k_heatmaps, scan_id)

        quality = display["quality"]
        if unc.get("confidence") is not None:
            quality["metrics"]["model_confidence"] = round(float(np.mean(unc["confidence"])), 3)

        report = self.reporter.generate(
            scan_id=scan_id,
            labels=self.labels,
            prediction={k: unc[k] for k in ("mean", "std", "ci_low", "ci_high", "entropy")},
            model_version=self.model_version,
            technical={
                "device": str(self.device),
                "image_size": self.image_size,
                "mc_samples": unc["n_samples"],
                "tta_hflip": self.tta_hflip,
                "temperature": float(self.model.temperature.item()),
                "inference_ms": round(inference_ms, 1),
                **image_meta,
                "quality": quality,
            },
        )
        if quality["status"] == "poor":
            # Above every clinical recommendation: nothing else on the report
            # should be acted on until the film is repeated.
            report.recommendations.insert(
                0,
                "Image quality is poor ("
                + "; ".join(
                    i["message"].rstrip(".").lower()
                    for i in quality["issues"]
                    if i["level"] == "poor"
                )
                + "). Repeat the radiograph before relying on this analysis.",
            )

        return {
            "scan_id": scan_id,
            "model_version": self.model_version,
            "labels": list(self.labels),
            "predictions": predictions,
            "fracture": {
                "probability": float(unc.get("fracture_mean", np.zeros(1))[0]),
                "uncertainty": float(unc.get("fracture_std", np.zeros(1))[0]),
            },
            "risk_level": report.risk_level,
            "quality": quality,
            "heatmaps": heatmaps,
            "preprocessing": {
                "steps": display["steps"],
                "applied_to_model_input": self.auto_crop,
            },
            "report": report,
            "inference_ms": round(inference_ms, 1),
        }

    # -- clean-up view -------------------------------------------------------
    @staticmethod
    def _save_enhanced(display: dict[str, Any], target_dir: Path) -> None:
        """Store the cleaned-up image and the list of steps beside the heatmaps.

        Leading underscore: label-derived file names never start with one, so
        these can never collide with a finding's heatmap.
        """
        try:
            import json

            import cv2

            target_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(
                str(target_dir / "_enhanced.png"), cv2.cvtColor(display["image"], cv2.COLOR_RGB2BGR)
            )
            (target_dir / "_enhanced.json").write_text(json.dumps({"steps": display["steps"]}))
        except Exception as exc:  # a display aid must never fail a prediction
            log.warning("Could not save enhanced image: %s", exc)

    # -- explainability ------------------------------------------------------
    def _explain(
        self,
        tensor: torch.Tensor,
        raw_image: np.ndarray,
        mean: np.ndarray,
        heatmap_dir: str | Path | None,
        top_k: int,
        scan_id: str,
    ) -> dict[str, str]:
        candidates = [i for i, label in enumerate(self.labels) if label != "No Finding"]
        ranked = sorted(candidates, key=lambda i: -mean[i])[: max(top_k, 1)]

        out: dict[str, str] = {}
        try:
            with GradCAM(self.model) as cam:
                maps = cam.generate(tensor, ranked, output_size=raw_image.shape[:2])
        except Exception as exc:  # explainability must never take down a prediction
            log.warning("Grad-CAM failed: %s", exc)
            return out

        if heatmap_dir is None:
            return {self.labels[i]: "in-memory" for i in maps}

        target_dir = Path(heatmap_dir) / str(scan_id)
        for idx, heatmap in maps.items():
            label = self.labels[idx]
            safe = label.replace(" ", "_").replace("/", "-")
            out[label] = save_overlay(raw_image, heatmap, target_dir / f"{safe}.png")
        return out


def load_predictor(
    checkpoint: str | Path,
    device: str | None = None,
    allow_untrained: bool = False,
    labels: tuple[str, ...] | None = None,
) -> RadiScanPredictor:
    """Load a predictor, optionally falling back to an untrained demo model.

    The untrained fallback exists so the UI can be demonstrated before a
    checkpoint is available. Its ``model_version`` is hard-coded to
    ``untrained-demo`` and every consumer surfaces that string, so a meaningless
    output can never be mistaken for a real one.
    """
    from radiscan.constants import CHEXPERT_LABELS

    path = Path(checkpoint)
    if path.exists():
        return RadiScanPredictor(checkpoint=path, device=device)
    if not allow_untrained:
        from radiscan.models.network import MissingCheckpointError

        raise MissingCheckpointError(
            f"No checkpoint at {path}. Train one, or set RADISCAN_ALLOW_UNTRAINED=1 "
            "to run a clearly-labelled demo with random weights."
        )

    log.warning("No checkpoint found -- serving an UNTRAINED demo model. Outputs are meaningless.")
    model = RadiScanNet(labels=labels or CHEXPERT_LABELS, pretrained=False, image_size=320)
    return RadiScanPredictor(model=model, meta={"model_version": "untrained-demo"}, device=device)
