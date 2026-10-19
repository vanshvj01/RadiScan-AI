# Paper and presentation

Everything here renders from **`content.py`** (paper text) and **`deck.js`**
(slides). Edit those, never the files in `out/`.

| Output | Command |
|---|---|
| `out/RadiScan-paper.docx` (IEEE, two-column) | `python -m paper.build_docx` |
| `out/RadiScan-paper.tex` + `refs.bib` | `python -m paper.build_tex` |
| `out/RadiScan-presentation.pptx` (17 slides, speaker notes) | `python -m paper.build_deck [--results results]` |
| Fill paper tables + abstract from results | `python -m paper.fill_results --results results` |

One-time setup for the Word and slide builders (Node.js 18+):

    cd paper
    npm init -y
    npm install docx pptxgenjs react react-dom react-icons sharp
    cd ..

On Windows set `NODE_PATH` so Python's subprocess finds them:

    $env:NODE_PATH = "$PWD\paper\node_modules"

---

## Finishing for the presentation — run in this order

### 1. Train on CheXpert (Kaggle, ~3-5 h, unattended)

Upload `notebooks/02_train_on_kaggle.ipynb`, attach the CheXpert dataset,
Accelerator = **GPU P100**, Persistence = Files, then **Save Version -> Save & Run All**.
You can close the browser. Download `best_model.pth` from the Output tab into
`models\checkpoints\`. Note the GPU and run time for Section VI-C.

### 2. Evaluate on the held-out test split

    python -m radiscan.cli.evaluate --config configs/chexpert_6gb.yaml `
        --checkpoint models/checkpoints/best_model.pth --split test --out results

### 3. Fill and rebuild everything

    python -m paper.fill_results --results results
    python -m paper.build_docx
    python -m paper.build_tex
    python -m paper.build_deck --results results

`fill_results` and `build_deck` **refuse** demo/synthetic output (stamped
`SYNTHETIC`, under a `demo` folder, or fewer than 1,000 test images). That is
deliberate: the demo model was trained on generated shapes and its numbers
must never appear as chest X-ray results.

### 4. Write the parts only you can write (~10 highlights left)

In `content.py`:

- `AUTHORS` — co-author names and emails; your guide conventionally last.
- Section VI-C — `[[GPU ...]]` and `[[hours]]`.
- Section VII — replace the `[[RESULTS PENDING ...]]` paragraph. Answer, in order:
  1. Which findings were strongest/weakest, and does that match how visible they are?
  2. Did ECE fall with temperature scaling while AUC stayed identical? (If AUC moved, it is a bug.)
  3. Is Table III monotone? `summary.json -> uncertainty_is_informative`. **If false, say so** — an honest negative result is defensible; a hidden one is not.
  4. Where does Table II put you vs MedViT 0.7511 / Hybrid CNN-ViT 0.7168? Do not claim you beat them: different splits.
- Section VII — the `[[Insert Fig. ...]]` note: insert `results/per_class_auc.png`,
  `calibration.png`, `selective_prediction.png` and a Grad-CAM example.

In `deck.js`, slide 1: co-author and guide names.

### 5. Check before you present or submit

- `python -m paper.build_docx` prints **0 placeholders**.
- PDF is 6-8 pages (trim Section II first if over).
- Rehearse with the speaker notes; slide 8 (deferral policy) is the one to nail.
