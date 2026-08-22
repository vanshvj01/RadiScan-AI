# RadiScan — 6-week execution plan

Written for a final-year mega project with a 4–6 week window. The repository is
built; what remains is turning it into *results* and a *submission*. Nothing
below asks you to write infrastructure — every command already exists.

The single rule for the whole plan: **never report a number you cannot
reproduce with a command in this repo.**

---

## Week 0 — before anything else (2 hours)

| Step | Command | Done when |
|---|---|---|
| Environment | see README §Quick start | `python -c "import torch; print(torch.cuda.is_available())"` runs |
| Prove the code works | `pytest` | 84 passed, no dataset downloaded |
| Pick your compute lane | `nvidia-smi` | You know whether you are Colab, local GPU, or CPU |

**Decide your lane now, because it sets every batch size downstream.**

| Lane | Train set | Epoch time | Realistic mean AUC |
|---|---|---|---|
| Colab / Kaggle free T4 ⭐ | 40k | ~8 min | 0.84–0.88 |
| Local NVIDIA GPU | 20k | ~6 min @ 8 GB | 0.81–0.86 |
| CPU only | 2k | ~25 min | 0.68–0.76 |

Recommended: **train on Colab, run the application locally.** Inference on CPU
is 2–5 s per image, which is perfectly demonstrable. `notebooks/01_train_on_colab.ipynb`
is the whole workflow.

---

## Week 1 — make it true

**Goal: one real prediction, end to end, in the browser.**

```bash
# 1. Data (CheXpert-small from Kaggle → data/raw/chexpert/)
python -m radiscan.cli.prepare_data \
  --source data/raw/chexpert/CheXpert-v1.0-small/train.csv \
  --out-dir data/splits --frontal-only \
  --train-size 20000 --val-size 3000 --test-size 3000

# 2. Verify. Three seconds. Never skip.
python -m radiscan.cli.verify_data --config configs/chexpert_base.yaml

# 3. Smoke test the loop before committing GPU time
python -m radiscan.cli.train --config configs/smoke.yaml

# 4. The real run
python -m radiscan.cli.train --config configs/chexpert_base.yaml
```

Then run the app, upload an X-ray, and **screenshot the result page**. That
screenshot is Figure 1 of your report and the proof that the system exists.

**Exit criterion:** `results/summary.json` has a mean AUC above 0.78, and you
have a browser screenshot of a real prediction with heatmaps.

**If mean AUC is stuck near 0.50**, it is a data problem, not a model problem.
Re-run `verify_data`. The strict loader means you will see an exception rather
than silently training on blanks, but check the positive rates it prints — if a
class shows 0%, your CSV columns are wrong.

---

## Week 2 — make it complete

**Goal: every screen in the UI does something, and the numbers are calibrated.**

- Confirm heatmaps render for the top findings and the scan history loads.
- Check `logs/final_summary.json`: the temperature should be > 1 (typically
  1.5–3.0) and ECE should have dropped several-fold. If T ≈ 1.0 the model was
  already calibrated, which is unusual — say so rather than hiding it.
- Inspect the tuned thresholds. Rare classes should land well below 0.5. This
  is a good slide.
- Export a PDF report and read it. Does the *indeterminate* section behave
  sensibly on a borderline case?

**Exit criterion:** you can demo upload → results → heatmap → PDF without
touching a terminal.

---

## Weeks 3–4 — make it credible

**This is the part that separates a B from an A.** The system existing is
table stakes; the evaluation is the contribution.

```bash
# Scale up (Colab), then evaluate on the HELD-OUT test split -- not val
python -m radiscan.cli.train --config configs/chexpert_base.yaml \
  --set data.max_train_images=40000 train.epochs=25
python -m radiscan.cli.evaluate --config configs/chexpert_base.yaml --split test
```

Everything you need lands in `results/`. Build the report's results section
directly from those files:

| Report artefact | File |
|---|---|
| Table 1 — per-class AUC / AP / P / R / F1 | `per_class_metrics.csv` |
| Table 3 — uncertainty stratification | `uncertainty_quartiles.csv` |
| Fig — ROC curves (14 overlaid) | `roc_curves.png` |
| Fig — precision-recall curves | `pr_curves.png` |
| Fig — reliability diagram + ECE | `calibration.png` |
| Fig — per-class AUC with support | `per_class_auc.png` |
| Fig — accuracy vs uncertainty quartile | `uncertainty_quartiles.png` |
| Fig — selective prediction curve | `selective_prediction.png` |
| Fig — training / validation curves | `training_curves.png` |

**Write the discussion around three questions:**

1. *Which classes work and which don't, and why?* Cardiomegaly and Effusion
   are large and high-contrast; Nodule and Pleural Other are small and rare.
   Correlate your per-class AUC against the `support` column and say what you
   see.
2. *Is the uncertainty informative?* `summary.json` has
   `uncertainty_is_informative`. If accuracy falls monotonically Q1→Q4, that
   one table justifies the entire uncertainty half of the project. **If it does
   not, report that.** A clean negative result with a hypothesis about why is a
   stronger piece of work than a fudged positive.
3. *What does deferral buy?* From `selective_prediction.csv`: "at 80% coverage
   the system reaches X% accuracy, referring the least confident 20% to a
   radiologist." That is the deployment-relevant framing and it is the sentence
   an examiner remembers.

**Exit criterion:** a results table you can defend row by row, and a paragraph
explaining every number in it.

---

## Week 5 — make it distinctive

**Pick exactly one.** Three half-done experiments read worse than one complete
one.

### Option A — Ablation study (recommended)

The strongest academic contribution here, because it is a real controlled
experiment rather than another feature.

```bash
python -m radiscan.cli.train --config configs/ablation_convnext.yaml
python -m radiscan.cli.evaluate --config configs/ablation_convnext.yaml \
  --split test --out results/convnext

python -m radiscan.cli.train --config configs/chexpert_base.yaml \
  --set model.use_transformer=true train.batch_size=16 \
        train.checkpoint_dir=models/checkpoints/hybrid
python -m radiscan.cli.evaluate --config configs/chexpert_base.yaml --split test \
  --checkpoint models/checkpoints/hybrid/best_model.pth --out results/hybrid
```

Same seed, same data, same schedule — change **only** the backbone, or the
comparison is not controlled. Then: does the ViT branch actually help at your
data scale? Most likely at 20–40k images it does not, and *explaining why*
(transformers are data-hungry; the inductive bias of convolution is worth more
than global attention at this scale) is a better answer than a win would be.

You can also ablate the loss — `--set loss.kind=focal` and `loss.kind=bce` —
which isolates the ASL contribution in one table.

### Option B — Calibration study

Reliability diagrams and ECE before/after temperature scaling, plus per-class
ECE. Cheap, rare in student projects, and it extends the uncertainty theme
directly.

### Option C — Distribution shift

Train on CheXpert, evaluate on NIH ChestX-ray14 (map the overlapping label
names), and measure the drop. Honest, publishable-flavoured, and it lets you
cite Zech et al. 2018 — the canonical "the model learned the scanner, not the
disease" paper — from your own data rather than as a footnote.

---

## Week 6 — submission

- [ ] `docker compose up -d --build` works from a clean clone
- [ ] README numbers updated to your actual results
- [ ] CI badge green
- [ ] Report: abstract with real N and AUC (not placeholders)
- [ ] Report: architecture diagram, request lifecycle diagram
- [ ] Report: all figures from `results/`
- [ ] Report: the limitations section — written by you, in your words
- [ ] Viva rehearsal (below)

---

## Viva preparation

Questions you will be asked, in rough order of likelihood.

| Question | Answer |
|---|---|
| Why multi-label and not multi-class? | A patient can have effusion *and* cardiomegaly. Independent sigmoids with per-class BCE, never a softmax. |
| Why DenseNet121? | Dense connectivity gives feature reuse and good gradient flow at low parameter count, and CheXNet established it as the chest X-ray baseline — which makes it the *comparable* choice, not just a good one. |
| How does MC Dropout give uncertainty? | Dropout stays active at inference; each pass samples a different sub-network, approximating a draw from the weight posterior. The variance across 20 passes approximates epistemic uncertainty. |
| Epistemic vs aleatoric? | Epistemic is the model's ignorance, reducible with more data — that is what MC Dropout captures. Aleatoric is noise inherent in the image, irreducible. The confidence head targets the latter. |
| Why Asymmetric Loss over Focal? | Focal applies the same focusing to positives and negatives. With classes under 2% positive, negatives dominate and the rare positives must not be down-weighted. ASL decouples γ⁻ and γ⁺ and clips very-easy negatives. |
| What does temperature scaling do to AUC? | Nothing — it is monotonic, so it cannot change ranking. It only fixes the *meaning* of the number, which matters because the whole premise is that a clinician can act on it. |
| Why not one 0.5 threshold? | Prevalences range from ~1% to ~40%. A single threshold guarantees the rare classes are never predicted. Tuning per class on validation roughly doubles F1 on the tail. |
| How do you know Grad-CAM is right? | Quantitatively, I don't — and that is stated as a limitation. Measuring it needs bounding boxes (VinBigData) and a pointing-game metric. Currently it is a qualitative sanity check. |
| Why is your AUC below CheXNet's? | Fewer images, fewer epochs, no ensembling, different label set. I report the gap rather than tuning on the test set. |
| Why Celery instead of predicting in the request? | 20 MC passes plus Grad-CAM is seconds, not milliseconds. Blocking HTTP workers would collapse throughput; an async queue with WebSocket progress keeps the API responsive and horizontally scalable. |
| Is this HIPAA compliant? | No. It is *designed with* HIPAA principles — an audit-log table, an RBAC schema, no PHI in logs. Actual compliance requires a BAA, a formal risk assessment and an external audit. Claiming otherwise would be overclaiming. |
| Would you deploy this? | Not as-is. As a triage-ordering aid under radiologist review, after prospective validation on local data — that is the realistic path. |
| **What was the hardest bug?** | **Lead with this one.** The v1 dataset loader silently substituted a blank black image whenever a file was missing, so training completed "successfully" on 223,000 black rectangles at ~0.50 AUC with no error anywhere in the logs. I made the loader raise, added a pre-flight verification command, and wrote a test that fails if the substitution ever comes back. It shows engineering judgment rather than just model knowledge. |

### Concepts to be able to explain in your own words

Multi-label vs multi-class · transfer learning · dense connectivity ·
self-attention and the CLS token · focal and asymmetric loss · MC Dropout ·
predictive entropy and mutual information (BALD) · Grad-CAM · CLAHE · AUC-ROC
vs average precision · calibration and ECE · CheXpert U-labels · DICOM
windowing and MONOCHROME1 · Celery brokers · WebSockets.

---

## Things that will cost you marks — avoid these

1. **Reporting validation AUC as your headline.** Use the held-out test split.
2. **A random train/val split.** `prepare_data` splits on patient and asserts
   no overlap. Do not bypass it.
3. **Claiming 0.95 AUC.** Nobody gets that on CheXpert with 40k images. It
   means leakage, and it will be found.
4. **Claiming HIPAA compliance, production-readiness, or 99.9% uptime.** The
   v1 docs did; do not repeat it.
5. **Placeholder numbers left in the abstract.** Fill N and X, or delete the
   sentence.
6. **Three half-finished experiments in week 5.** One complete one.
