# RadiScan AI

**Uncertainty-aware, explainable deep learning for chest radiograph interpretation.**

A chest X-ray goes in. Out comes a probability for each of 14 thoracic
pathologies, an honest confidence interval around each one, a saliency map
showing where the network looked, and a structured clinical report — served
through a typed REST API and a Next.js dashboard.

> **Decision support prototype. Not a diagnostic device.** Trained on public
> research datasets whose labels are NLP-extracted from radiology reports (i.e.
> noisy). No prospective validation, no reader study, no regulatory clearance.
> Every claim in this README is one you can reproduce from this repository.

---

## What makes this different from a CNN with 14 outputs

Most chest X-ray projects stop at "the network outputs 14 probabilities". Three
things separate a classifier from a system someone could actually put in front
of a clinician, and this repository implements all three:

**1. It knows when it doesn't know.** Monte Carlo Dropout runs 20 stochastic
forward passes and reports a mean, a standard deviation, a 95% interval and
per-class entropy. A prediction of 0.62 ± 0.02 and one of 0.62 ± 0.30 are
completely different clinical objects; only the second is a coin flip. When a
finding's interval straddles its own operating threshold, the report marks it
*indeterminate* and refers it to a human rather than guessing.

**2. It is calibrated.** Deep networks are systematically overconfident. A
temperature is fitted on held-out validation logits after training and baked
into the checkpoint, so a stated 70% actually means roughly 70%. Expected
calibration error typically improves 3–5×, and AUC is untouched because
temperature scaling cannot change the ranking.

**3. It shows its work.** Grad-CAM and Grad-CAM++ produce class-specific
saliency over the last convolutional stage. The UI presents these next to the
original with an explicit caveat: saliency is correlation with the score, not
localisation of pathology.

---

## Accuracy: what is in here and why

Every technique below is enabled by default. This table is worth understanding
line by line, because "what did you do to improve accuracy, and why did each
thing help?" is the question you will be asked.

| Technique | Where | Why it helps here |
|---|---|---|
| **Asymmetric Loss** | `models/losses.py` | Focal loss focuses positives and negatives identically. With some pathologies under 2% positive, the negatives dominate and the rare positives must *not* be down-weighted. ASL decouples the two focusing terms (γ⁻=4, γ⁺=0) and hard-discards very easy negatives. Typically +1–3 mAP over focal. |
| **320px input** | `configs/*.yaml` | Nodules and subtle infiltrates occupy few pixels. Published CheXpert results improve monotonically to ~384px; 320 is the sweet spot for 8–12 GB VRAM. |
| **CLAHE** | `data/transforms.py` | Radiographs are inherently low-contrast. Adaptive histogram equalisation with a clip limit is the standard radiology enhancement. |
| **Balanced sampling** | `data/datasets.py` | Weights each sample by the inverse frequency of its rarest positive class, so tail classes actually appear in a batch. Worth ~0.02 mean AUC on the tail. |
| **LSR uncertain-label policy** | `data/datasets.py` | CheXpert's `-1` labels ("cannot exclude pneumonia") mapped to a soft target rather than hard 0 or 1. Stanford's own best-performing policy; beats U-Zeros by 0.01–0.02 mean AUC. |
| **Multi-label mixup** | `training/trainer.py` | Targets combined by *maximum*, not interpolation — if either source image has effusion, the blend plausibly shows effusion. Linear mixing would teach the model that a clearly-present finding deserves 0.5. |
| **EMA weights** | `training/ema.py` | One extra weight copy, no extra backward compute, ~0.01 mean AUC. Evaluation and checkpointing both use the EMA copy. |
| **Warmup + cosine, layer-wise LR** | `training/schedulers.py` | Randomly-initialised heads produce large early gradients that will wreck ImageNet features. Heads train at 10× the backbone LR after a 2-epoch warmup. |
| **AMP + gradient accumulation** | `training/trainer.py` | 2–3× throughput on GPU, and a large effective batch on small VRAM. More epochs inside the same compute budget is itself an accuracy technique. |
| **Flip TTA** | `modules/uncertainty.py` | ~0.003–0.008 AUC at inference for zero retraining. |
| **Temperature scaling** | `training/calibration.py` | Fixes overconfidence post-hoc. Fitted automatically after the final epoch. |
| **Per-class thresholds** | `evaluation/metrics.py` | A single 0.5 across classes with prevalences from 1% to 40% is indefensible. Tuning per class routinely doubles F1 on rare findings. |
| **Patient-disjoint splits** | `cli/prepare_data.py` | The split bug almost every student project ships. CheXpert has multiple studies per patient; a random row split leaks patients between train and val and inflates AUC by 3–6 points. |

### What to expect

| Setup | Mean AUC (14 classes) |
|---|---|
| 2k images, 10 epochs, CPU | 0.68–0.76 |
| 20k images, 25 epochs, GPU | 0.81–0.87 |
| Full CheXpert, tuned | 0.87–0.90 |
| CheXNet (published, NIH-14) | 0.841 |
| Radiologist benchmark | ~0.85–0.91, pathology-dependent |

0.82 on 20k images is a **good** result. Claiming 0.95 will get you taken apart
in the viva.

---

## Quick start

### 1. Environment

**Use Python 3.11 or 3.12.** Not 3.13+, and not 3.14 — several dependencies
(`timm`, `albumentations`, `opencv-python`, `scikit-learn`) lag new CPython
releases by months. On a version with no prebuilt wheel, pip falls back to
building from source and fails on Windows without Visual C++ build tools. Check
with `python --version` before you start; if it is too new, install 3.12 from
python.org and create the venv with `py -3.12 -m venv .venv`.

```bash
git clone <your-fork> radiscan && cd radiscan
python -m venv .venv
source .venv/bin/activate          # Windows: .\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip

# Install torch DELIBERATELY first -- a bare `pip install -r requirements.txt`
# pulls the ~2.5 GB CUDA build whether or not you have a GPU.
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu   # CPU
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121  # CUDA 12.1

pip install -r requirements-dev.txt
pip install --no-deps -e .

python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

### 2. Prove it works before touching a dataset

```bash
pytest
```

84 tests, no dataset needed — they run on synthetic images generated in
`tmp_path`. If this is green, your environment is correct.

### 3. See the whole thing work — before downloading any data

```bash
python scripts/demo.py
```

Generates a synthetic dataset with a learnable signal, builds patient-disjoint
splits, verifies them, trains for six CPU epochs, calibrates, evaluates, and
runs a single-image prediction with Grad-CAM. About five minutes, no download.
It leaves a real checkpoint in `models/checkpoints/` so the API and dashboard
have something to serve immediately.

Expect **mean AUC around 0.85–0.90**. If it lands near 0.50, the pipeline is
broken — which is exactly what this is for. The model is trained on synthetic
shapes, so delete the checkpoint before training for real; nothing from the demo
belongs in your report.

### 4. Data

Download **CheXpert-small** (~11 GB) from Kaggle into `data/raw/chexpert/`, then:

```bash
python -m radiscan.cli.prepare_data \
  --source data/raw/chexpert/CheXpert-v1.0-small/train.csv \
  --out-dir data/splits --frontal-only \
  --train-size 20000 --val-size 3000 --test-size 3000
```

This produces **patient-disjoint** splits and asserts zero patient overlap.

### 5. Verify before you burn a GPU-hour

```bash
python -m radiscan.cli.verify_data --config configs/chexpert_base.yaml
```

Three seconds. It checks that CSV paths actually resolve on disk and prints the
per-class positive rates. **Do not skip it.** The loader raises on a missing
image rather than substituting a blank frame, which is exactly the failure that
lets a pipeline train "successfully" to 0.50 AUC with no error in the logs.

### 6. Train

```bash
# Laptop smoke test first -- ~10 minutes, proves the loop runs
python -m radiscan.cli.train --config configs/smoke.yaml

# The real run
python -m radiscan.cli.train --config configs/chexpert_base.yaml

# Override anything from the command line
python -m radiscan.cli.train --config configs/chexpert_base.yaml \
  --set train.epochs=15 train.batch_size=16 data.max_train_images=5000
```

Training automatically fits a temperature, tunes per-class thresholds, and
writes both into `models/checkpoints/best_model.pth`. There is no separate
"remember to calibrate" step to forget.

### 7. Evaluate — one command, every table and figure

```bash
python -m radiscan.cli.evaluate --config configs/chexpert_base.yaml --split test
```

Writes to `results/`: `per_class_metrics.csv` (Table 1), `summary.json`,
`uncertainty_quartiles.csv` (Table 3), plus ROC curves, PR curves, a reliability
diagram, per-class AUC bars, the uncertainty-vs-accuracy chart and the selective
prediction curve. Every figure ships with the CSV behind it.

### 8. Run the application

```bash
# Backend -- SQLite, Celery eager, no Redis, no Postgres
cp .env.example .env
python -c "import secrets; print(secrets.token_urlsafe(48))"   # paste into SECRET_KEY
uvicorn services.api.main:app --reload

# Frontend
cd web && npm install
cp .env.local.example .env.local
npm run dev
```

* Dashboard — <http://localhost:3000>
* API docs — <http://localhost:8000/docs>
* Health — <http://localhost:8000/api/v1/health>

No checkpoint yet? Set `ALLOW_UNTRAINED=true` to demo the interface. Outputs are
tagged `untrained-demo` everywhere they surface, so meaningless numbers can
never be mistaken for real ones.

The production topology (Postgres, Redis, a real Celery worker) is
`docker compose up -d --build`.

---

## Architecture

```
Browser (Next.js :3000)
  │ POST /api/v1/predict   multipart: file + anonymised patient meta
  ▼
FastAPI (:8000)
  │ validate extension and size · store under data/uploads/<uuid>
  │ INSERT Scan(status=uploaded) · audit log
  │ process_scan.delay(scan_id) → 202 ACCEPTED
  ▼
Celery (Redis broker, or inline when eager)
  │ load  · DICOM via pydicom (windowing, MONOCHROME1) | PNG/JPEG via cv2
  │ prep  · CLAHE → resize → ImageNet normalise
  │ infer · 20 MC-Dropout passes × flip TTA, temperature-calibrated
  │ explain · Grad-CAM on the top-k findings, overlays written to disk
  │ INSERT Prediction (+ label_set) · INSERT Report
  ▼
WebSocket /ws/scan/{uuid} pushes progress · dashboard renders results
```

### Model

```
Input 3×320×320
   ├─► CNN backbone (timm: densenet121 | convnext_tiny | efficientnetv2_s)
   │        └─ last conv stage ──► Grad-CAM target (discovered, not hard-coded)
   └─► [optional] ViT branch
   ▼ concat → LayerNorm → dropout
   ├─► disease head     14 logits → sigmoid (multi-label, NOT softmax)
   ├─► fracture head    1 logit
   └─► confidence head  14 scalars (aleatoric, trained with a confidence penalty)
```

DenseNet121 at 8.0 M parameters is the default: dense connectivity gives feature
reuse and good gradient flow at low parameter count, and CheXNet established it
as *the* chest X-ray baseline, which makes it the comparable choice.

### Layout

```
src/radiscan/          the library -- importable, testable, no framework coupling
  ├── config.py        typed dataclass config, YAML + CLI overrides
  ├── constants.py     canonical label sets (the single source of truth)
  ├── data/            loader (DICOM/PNG), transforms, datasets, verification
  ├── models/          backbone, heads, losses, network + checkpoint contract
  ├── modules/         uncertainty (MC Dropout), explainability (Grad-CAM)
  ├── training/        trainer, EMA, schedulers, calibration
  ├── evaluation/      metrics, plots
  ├── reporting/       rule-based clinical report generator
  ├── inference/       RadiScanPredictor -- the one prediction code path
  └── cli/             prepare_data · verify_data · train · evaluate · predict
services/api/          FastAPI: routes, typed schemas, ORM, auth, Celery task
web/                   Next.js 14 + TypeScript + Tailwind dashboard
tests/                 84 tests, no dataset required
configs/               chexpert_base · smoke · ablation_convnext
```

---

## What was fixed relative to the v1 prototype

This repository is a rebuild. The ML ideas carried over; the failure modes did
not. Each of these is now covered by a test that fails if it regresses.

| v1 behaviour | Consequence | Now |
|---|---|---|
| Dataset loader returned a **blank black image** for a missing file | Training "succeeded" on 223k black rectangles at ~0.50 AUC, no error anywhere | Raises with a diagnostic message. Blanks are opt-in via `strict=False`. |
| Config used **NIH label names**, CheXpert dataset used **CheXpert names**, both length 14 | Every disease name in the output was silently wrong | Labels embedded in the checkpoint; `load_model` refuses a mismatch |
| `--max_images` declared but never passed through | "Quick 1000-image run" silently trained on all 223k rows | Threaded through and tested |
| Auth injected a hardcoded `MockUser` under `DEBUG`, 401 otherwise | Upload was either wide open or unusable | Real JWT on every protected route; one way in |
| Frontend called `/predict/heatmap/...` and `/predict/scans` — neither existed | Two screens were dead UI | Both implemented, with a path-traversal guard |
| Heatmaps computed and then discarded | Nothing to display | Persisted per scan and served |
| `DEVICE` defaulted to `cuda` | Instant crash on any CPU-only machine | `auto` with a logged fallback |
| Missing checkpoint → bare `FileNotFoundError` inside a worker | Opaque failure | Actionable message, `/health` explains, optional labelled demo mode |
| Random row-level train/val split | Patient leakage inflated AUC | Patient-disjoint splits with an assertion |
| No tests, no CI, no LICENSE | Unverifiable | 84 tests, GitHub Actions, MIT |
| `weasyprint` (needs GTK3, fails on Windows) for PDFs | Install blocker | `reportlab` |
| 3 databases required to boot | Hours of setup before a first prediction | SQLite + eager Celery: `uvicorn services.api.main:app` |

---

## Roadmap for a final-year submission

**Weeks 1–2 — make it true.** Download CheXpert-small, build patient-disjoint
splits, verify, train on 2k images end to end, get one real prediction through
the browser. Screenshot it; that is Figure 1.

**Weeks 3–4 — make it credible.** Train on 20k+, run `cli.evaluate` on the held-out
test split, and write the results section around the numbers it produces. This
is what separates a B from an A: a table you can defend row by row.

**Week 5 — make it distinctive.** Pick one, not three:
* **Ablation** — `configs/chexpert_base.yaml` vs `configs/ablation_convnext.yaml`
  vs `--set model.use_transformer=true`. Same seed, same data, same schedule.
  This is a real controlled experiment, which is a stronger contribution than
  another feature.
* **Calibration study** — reliability diagrams and ECE before/after temperature
  scaling. Cheap, rare in student work, directly extends the uncertainty theme.
* **Distribution shift** — train on CheXpert, test on NIH ChestX-ray14, measure
  the drop. Honest and publishable-flavoured; cite Zech et al. 2018.

**Week 6 — polish.** `docker compose up` demo, results figures into the report,
viva rehearsal.

**Explicitly out of scope**, and say so rather than claiming otherwise: HIPAA
certification, SOC 2, Kubernetes, multi-region. The schema here is *designed
with* audit and RBAC principles. Compliance needs a BAA, a formal risk
assessment and an external audit.

---

## Limitations

State these yourself before an examiner does.

* Labels are NLP-extracted from free-text reports, not expert re-annotation.
  Label noise upper-bounds achievable AUC.
* CheXpert is single-institution (Stanford). There is no evidence of
  generalisation to Indian hospital equipment or population.
* Frontal views only. Lateral views and prior studies are ignored; a real
  reading uses both.
* No clinical metadata — symptoms, history, labs. A real reading is never
  image-only.
* MC Dropout is an *approximation* to Bayesian inference. Twenty samples is a
  compute-driven choice, not a convergence-driven one.
* Grad-CAM shows correlation, not causation. The canonical failure is attending
  to chest drains or laterality markers rather than pathology. Quantifying
  localisation needs bounding boxes and a pointing-game metric.
* No prospective validation, no radiologist reader study, no regulatory
  clearance.

---

## References

1. Rajpurkar et al. *CheXNet: Radiologist-Level Pneumonia Detection on Chest X-Rays with Deep Learning.* arXiv:1711.05225, 2017.
2. Irvin et al. *CheXpert: A Large Chest Radiograph Dataset with Uncertainty Labels and Expert Comparison.* AAAI 2019.
3. Wang et al. *ChestX-ray8: Hospital-scale Chest X-ray Database.* CVPR 2017.
4. Huang et al. *Densely Connected Convolutional Networks.* CVPR 2017.
5. Dosovitskiy et al. *An Image is Worth 16×16 Words.* ICLR 2021.
6. Selvaraju et al. *Grad-CAM.* ICCV 2017.
7. Chattopadhay et al. *Grad-CAM++.* WACV 2018.
8. Gal & Ghahramani. *Dropout as a Bayesian Approximation.* ICML 2016.
9. Lin et al. *Focal Loss for Dense Object Detection.* ICCV 2017.
10. Ridnik et al. *Asymmetric Loss for Multi-Label Classification.* ICCV 2021.
11. Guo et al. *On Calibration of Modern Neural Networks.* ICML 2017.
12. Zech et al. *Variable generalization performance of a deep learning model to detect pneumonia in chest radiographs.* PLOS Medicine 2018.

## Licence

MIT — see [LICENSE](LICENSE). Not a medical device.
