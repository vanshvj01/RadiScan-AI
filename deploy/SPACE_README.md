---
title: RadiScan AI API
emoji: 🫁
colorFrom: blue
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# RadiScan AI — API

Uncertainty-aware, explainable chest radiograph analysis. This Space hosts the
FastAPI backend; the dashboard is deployed separately.

- **Interactive API docs:** [`/docs`](/docs)
- **Health and model status:** [`/api/v1/health`](/api/v1/health)

## Not a diagnostic device

This is a final-year research prototype. It has **no regulatory clearance**, has
never been prospectively validated, and has not been assessed in a reader study.
Do not use it for any clinical decision.

Check `/api/v1/health` before trusting any output. If `model_version` begins
with `SYNTHETIC-DEMO`, the model was trained on generated shapes rather than
radiographs and every number it returns is meaningless.

## Storage is ephemeral

Accounts, uploads and results are wiped whenever the Space restarts. Do not
upload anything you need to keep, and do not upload identifiable patient data —
use anonymised images only.

## Source

Full source, training pipeline and evaluation suite:
https://github.com/vanshvj01/RadiScan-AI
