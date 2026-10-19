// RadiScan presentation deck. Driven by paper/build_deck.py, which supplies
// real results (or {real:false}) as JSON in argv[2]; output path in argv[3].
const pptxgen = require("pptxgenjs");
const fs = require("fs");
const path = require("path");
const React = require("react");
const RDS = require("react-dom/server");
const sharp = require("sharp");
const fa = require("react-icons/fa");

const DATA = JSON.parse(fs.readFileSync(process.argv[2], "utf8"));
const OUTFILE = process.argv[3];

const THEME = {
  name: "RadiScan",
  headFontFace: "Cambria",
  bodyFontFace: "Calibri",
  colors: {
    dk1: "14202B", lt1: "FFFFFF", dk2: "0E1A22", lt2: "EEF3F5",
    accent1: "1F9E98", accent2: "E3A13B", accent3: "D2504B",
    accent4: "4C6B7A", accent5: "7FC8C3", accent6: "9AA9B2",
    hlink: "1F9E98", folHlink: "4C6B7A",
  },
};
const HEX = THEME.colors;

async function icon(Comp, color, px = 256) {
  const svg = RDS.renderToStaticMarkup(React.createElement(Comp, { color: "#" + color, size: px }));
  const buf = await sharp(Buffer.from(svg)).png().toBuffer();
  return "image/png;base64," + buf.toString("base64");
}

(async () => {
  const pres = new pptxgen();
  pres.layout = "LAYOUT_WIDE"; // 13.333 x 7.5
  pres.title = "RadiScan: Uncertainty-Aware Chest Radiograph Classification";
  pres.author = "Vansh R. Vijayvargiya";
  pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
  // With the theme applier available (generation environment), use scheme
  // colours so the deck restyles from its theme. Without it (e.g. rebuilding
  // on a laptop), fall back to the same colours as literal hex so the deck
  // still looks right.
  const SKILL = process.env.PPTX_SKILL_DIR;
  const C = SKILL ? pres.SchemeColor : {
    text1: HEX.dk1, text2: HEX.dk2, background1: HEX.lt1, background2: HEX.lt2,
    accent1: HEX.accent1, accent2: HEX.accent2, accent3: HEX.accent3,
    accent4: HEX.accent4, accent5: HEX.accent5, accent6: HEX.accent6,
  };

  // ---------------------------------------------------------------- layouts
  pres.defineSlideMaster({
    title: "DARK",
    background: { color: C.text2 },
    objects: [
      { placeholder: { options: { name: "title", type: "title", x: 0.8, y: 2.1, w: 11.7, h: 1.9,
          fontSize: 40, bold: true, color: C.background1, valign: "bottom", align: "left" },
          text: "" } },
    ],
  });
  pres.defineSlideMaster({
    title: "CONTENT",
    background: { color: C.background1 },
    margin: [0.5, 0.6, 0.6, 0.6],
    objects: [
      { placeholder: { options: { name: "title", type: "title", x: 0.6, y: 0.35, w: 12.1, h: 0.9,
          fontSize: 32, bold: true, color: C.text1, valign: "middle", align: "left", margin: 0 },
          text: "" } },
      { text: { text: "RadiScan  ·  K. J. Somaiya School of Engineering",
          options: { x: 0.6, y: 7.0, w: 8, h: 0.3, fontSize: 10, color: C.accent6, margin: 0 } } },
    ],
    slideNumber: { x: 12.2, y: 7.0, w: 0.5, h: 0.3, fontSize: 10, color: C.accent6, align: "right" },
  });

  const content = (section) => pres.addSlide({ masterName: "CONTENT", sectionTitle: section });
  const T = (s, text) => s.addText(text, { placeholder: "title" });
  const txt = (s, text, o) => s.addText(text, { isTextBox: true, fontSize: 15, color: C.text1,
                                                 valign: "top", margin: 0, ...o });

  // Icon in a filled circle -- the deck's single visual motif.
  async function badge(s, Comp, x, y, fill = C.accent1, d = 0.62, name = "icon") {
    s.addShape(pres.shapes.OVAL, { x, y, w: d, h: d, fill: { color: fill }, line: { color: fill },
                                    objectName: name + "-circle" });
    s.addImage({ data: await icon(Comp, "FFFFFF"), x: x + d * 0.24, y: y + d * 0.24,
                 w: d * 0.52, h: d * 0.52, objectName: name });
  }
  const card = (s, x, y, w, h, fill = C.background2, name = "card") =>
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, rectRadius: 0.08,
      fill: { color: fill }, line: { color: fill }, objectName: name });

  // A clearly marked slot for numbers that only the training run can supply.
  const pending = (s, x, y, w, h, label) => {
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, rectRadius: 0.08,
      fill: { color: "FFF6E5" }, line: { color: HEX.accent2, width: 1.5, dashType: "dash" },
      objectName: "pending-slot" });
    s.addText([
      { text: "PENDING CheXpert RUN", options: { bold: true, fontSize: 14, color: "9A6412", breakLine: true } },
      { text: label, options: { fontSize: 13, color: "6B4A12" } },
    ], { isTextBox: true, x: x + 0.3, y: y + 0.25, w: w - 0.6, h: h - 0.5, valign: "middle",
         align: "center", margin: 0 });
  };
  const f4 = (v) => (v == null ? "--" : v.toFixed(4));

  // ================================================================ 1 title
  pres.addSection({ title: "Opening" });
  let s = pres.addSlide({ masterName: "DARK", sectionTitle: "Opening" });
  T(s, [{ text: "Uncertainty-Aware and Explainable", options: { breakLine: true } }, { text: "Chest X-ray Classification" }]);
  s.addText("with an explicit deferral policy", { isTextBox: true, x: 0.8, y: 4.05, w: 11.7, h: 0.6,
    fontSize: 22, italic: true, color: C.accent5, margin: 0 });
  s.addText([
    { text: "Vansh R. Vijayvargiya  ·  [Co-author 2]  ·  [Co-author 3]", options: { breakLine: true } },
    { text: "Guide: [Prof. Faculty Guide]", options: { breakLine: true } },
    { text: "Department of Information Technology, K. J. Somaiya School of Engineering, Mumbai" },
  ], { isTextBox: true, x: 0.8, y: 5.2, w: 11.7, h: 1.3, fontSize: 15, color: C.accent6, margin: 0,
       paraSpaceAfter: 4 });
  await badge(s, fa.FaLungs, 0.8, 0.9, C.accent1, 0.9, "logo");
  s.addText("RadiScan", { isTextBox: true, x: 1.9, y: 0.95, w: 5, h: 0.8, fontSize: 28, bold: true,
    color: C.background1, fontFace: THEME.headFontFace, margin: 0, valign: "middle" });
  s.addNotes("Introduce the team. One-sentence pitch: RadiScan reads chest X-rays for 14 findings, " +
    "and for each one it says how sure it is, shows where it looked, and refuses to answer when it " +
    "cannot tell. That refusal is the core idea of the project.");

  // ================================================================ 2 problem
  s = content("Opening");
  T(s, "A probability alone cannot be trusted");
  card(s, 0.6, 1.6, 5.8, 3.6, C.background2, "case-a");
  card(s, 6.9, 1.6, 5.8, 3.6, "FBEAE9", "case-b");
  s.addText("0.62", { isTextBox: true, x: 0.6, y: 1.9, w: 5.8, h: 1.3, fontSize: 66, bold: true,
    color: C.accent1, align: "center", fontFace: THEME.headFontFace, margin: 0 });
  s.addText("± 0.02", { isTextBox: true, x: 0.6, y: 3.2, w: 5.8, h: 0.6, fontSize: 26,
    color: C.text1, align: "center", margin: 0 });
  s.addText("Model is confident: actionable", { isTextBox: true, x: 0.6, y: 4.2, w: 5.8, h: 0.5,
    fontSize: 16, color: C.accent4, align: "center", margin: 0 });
  s.addText("0.62", { isTextBox: true, x: 6.9, y: 1.9, w: 5.8, h: 1.3, fontSize: 66, bold: true,
    color: C.accent3, align: "center", fontFace: THEME.headFontFace, margin: 0 });
  s.addText("± 0.30", { isTextBox: true, x: 6.9, y: 3.2, w: 5.8, h: 0.6, fontSize: 26,
    color: C.text1, align: "center", margin: 0 });
  s.addText("Model is guessing: should not be acted on", { isTextBox: true, x: 6.9, y: 4.2, w: 5.8,
    h: 0.5, fontSize: 16, color: C.accent3, align: "center", margin: 0 });
  txt(s, "Standard classifiers report only the 0.62. They are also overconfident [Guo 2017], and they " +
    "have no way to decline a case they cannot resolve.", { x: 0.6, y: 5.6, w: 12.1, h: 0.9, fontSize: 17 });
  s.addNotes("Both predictions are 0.62, but only the left one is usable. A conventional model shows the " +
    "clinician the same number in both cases. Three gaps follow: no uncertainty, poor calibration, " +
    "no ability to decline. RadiScan targets all three.");

  // ================================================================ 3 contributions
  s = content("Opening");
  T(s, "What this project contributes");
  const contrib = [
    [fa.FaChartArea, "Uncertainty per finding", "Posterior mean, σ, 90% credible interval, entropy and mutual information for all 14 findings"],
    [fa.FaHandPaper, "Explicit deferral policy", "A finding whose interval straddles its threshold is marked indeterminate and referred to a radiologist"],
    [fa.FaBalanceScale, "Built for extreme imbalance", "Asymmetric Loss, balanced sampling and per-class thresholds for prevalence from ~40% to <2%"],
    [fa.FaShieldAlt, "Reproducibility protocol", "Patient-disjoint splits with assertion and pre-flight data verification"],
    [fa.FaLaptopMedical, "Complete clinical workflow", "Authenticated web app, async inference, audit log and PDF reports"],
  ];
  for (let i = 0; i < contrib.length; i++) {
    const y = 1.55 + i * 1.05;
    await badge(s, contrib[i][0], 0.6, y, i === 1 ? C.accent2 : C.accent1, 0.62, "contrib-" + i);
    s.addText([
      { text: contrib[i][1], options: { bold: true, fontSize: 18, breakLine: true } },
      { text: contrib[i][2], options: { fontSize: 15, color: C.accent4 } },
    ], { isTextBox: true, x: 1.5, y: y - 0.08, w: 11.2, h: 0.85, valign: "middle", margin: 0, color: C.text1 });
  }
  s.addNotes("Five contributions. Emphasise number two, the deferral policy: it is what turns the " +
    "uncertainty estimate from something displayed into something that changes what the system does.");

  // ================================================================ 4 dataset
  pres.addSection({ title: "Data and Method" });
  s = content("Data and Method");
  T(s, "CheXpert: large, noisy and heavily imbalanced");
  const stats = [["224,316", "chest radiographs"], ["65,240", "patients"], ["14", "findings per image"]];
  stats.forEach(([n, l], i) => {
    const x = 0.6 + i * 2.6;
    s.addText(n, { isTextBox: true, x, y: 1.6, w: 2.4, h: 0.9, fontSize: 40, bold: true,
      color: C.accent1, fontFace: THEME.headFontFace, margin: 0 });
    s.addText(l, { isTextBox: true, x, y: 2.5, w: 2.4, h: 0.4, fontSize: 14, color: C.accent4, margin: 0 });
  });
  txt(s, [
    { text: "Labels are extracted automatically from free-text reports: positive, negative, uncertain or not mentioned.", options: { bullet: true, breakLine: true } },
    { text: "Uncertain labels → soft targets in [0.05, 0.85] (label-smoothing regularisation).", options: { bullet: true, breakLine: true } },
    { text: "Auto clean-up finds the film in phone photos and crops empty borders; then CLAHE at 320 × 320 px.", options: { bullet: true, breakLine: true } },
    { text: "Split by patient, not by image, so the model cannot memorise patients.", options: { bullet: true } },
  ], { x: 0.6, y: 3.4, w: 7.4, h: 3.2, paraSpaceAfter: 10 });
  card(s, 8.6, 1.6, 4.1, 5.0, "FBEAE9", "accuracy-card");
  await badge(s, fa.FaExclamationTriangle, 8.95, 1.9, C.accent3, 0.62, "warn");
  s.addText([
    { text: "Why we never report accuracy", options: { bold: true, fontSize: 18, breakLine: true } },
    { text: "Fracture appears in under 2% of images. A model that always answers “no fracture” scores over 98% accuracy and detects nothing.", options: { fontSize: 15, breakLine: true } },
    { text: " ", options: { fontSize: 8, breakLine: true } },
    { text: "We report AUC-ROC, average precision and calibration error instead.", options: { fontSize: 15, bold: true, color: C.accent3 } },
  ], { isTextBox: true, x: 8.95, y: 2.75, w: 3.45, h: 3.6, valign: "top", margin: 0, color: C.text1 });
  s.addNotes("Faculty will ask about accuracy. Answer: with prevalence below 2 percent, accuracy rewards " +
    "a model that never predicts the disease. AUC and average precision measure ranking and rare-class " +
    "detection instead.");

  // ================================================================ 5 architecture
  s = content("Data and Method");
  T(s, "Architecture: one trunk, three heads, four add-ons");
  const box = (x, y, w, h, title, sub, fill, name, dark = false) => {
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, rectRadius: 0.08, fill: { color: fill },
      line: { color: fill }, objectName: name });
    s.addText([
      { text: title, options: { bold: true, fontSize: 15, breakLine: !!sub } },
      ...(sub ? [{ text: sub, options: { fontSize: 12 } }] : []),
    ], { isTextBox: true, x: x + 0.1, y, w: w - 0.2, h, align: "center", valign: "middle", margin: 0,
         color: dark ? C.background1 : C.text1 });
  };
  const arrow = (x1, y1, x2, y2, name) => s.addShape(pres.shapes.LINE, {
    x: Math.min(x1, x2), y: Math.min(y1, y2), w: Math.abs(x2 - x1) || 0.001, h: Math.abs(y2 - y1) || 0.001,
    flipV: y2 < y1, line: { color: HEX.accent4, width: 1.5, endArrowType: "triangle" }, objectName: name });
  box(0.6, 3.0, 1.9, 1.3, "Chest X-ray", "DICOM / PNG", C.background2, "in");
  box(2.9, 3.0, 1.9, 1.3, "Preprocess", "CLAHE, 320 px", C.background2, "pre");
  box(5.2, 2.7, 2.2, 1.9, "DenseNet-121", "ImageNet-pretrained\n8.0M params\n1024-d feature", C.text2, "trunk", true);
  arrow(2.5, 3.65, 2.9, 3.65, "a1"); arrow(4.8, 3.65, 5.2, 3.65, "a2");
  const heads = [["Disease head", "14 sigmoids"], ["Fracture head", "auxiliary"], ["Confidence head", "aleatoric c∈(0,1)"]];
  heads.forEach(([h, d], i) => {
    const y = 1.55 + i * 1.75;
    box(7.9, y, 2.1, 1.25, h, d, i === 2 ? C.accent5 : C.background2, "head-" + i);
    arrow(7.4, 3.65, 7.9, y + 0.62, "ah-" + i);
  });
  const adds = [["MC Dropout ×20", "mean, σ, 90% CI"], ["Temperature scaling", "calibration"],
                ["Grad-CAM", "saliency map"], ["Deferral rule", "pos / neg / indeterminate"]];
  adds.forEach(([h, d], i) => {
    const y = 1.45 + i * 1.3;
    box(10.5, y, 2.2, 1.05, h, d, i === 3 ? C.accent2 : C.accent1, "add-" + i, i !== 3);
  });
  arrow(10.0, 2.18, 10.5, 2.18, "ad1");
  txt(s, "Dropout 0.2 before every head: a regulariser during training, and the source of randomness for uncertainty at inference.",
    { x: 0.6, y: 5.0, w: 6.8, h: 1.2, fontSize: 14, color: C.accent4 });
  s.addNotes("Walk left to right. Sigmoid, not softmax: findings co-occur, a patient can have effusion and " +
    "cardiomegaly together, and softmax would force them to compete. DenseNet-121 is chosen because it " +
    "is the standard reproducible baseline (CheXNet), which isolates our contribution: what sits on top.");

  // ================================================================ 6 imbalance
  s = content("Data and Method");
  T(s, "Learning rare findings: Asymmetric Loss");
  const cols = [
    [fa.FaFilter, "Asymmetric Loss", "γ⁻ = 4 on negatives, γ⁺ = 0 on positives, margin 0.05. Easy negatives stop drowning the gradient; scarce positives keep their full signal."],
    [fa.FaRandom, "Balanced sampling", "Studies containing rare positives are drawn more often, so every mini-batch carries signal for them."],
    [fa.FaSlidersH, "Per-class thresholds", "Tuned to maximise F1 on the validation split, then frozen before the test split is touched."],
  ];
  for (let i = 0; i < 3; i++) {
    const x = 0.6 + i * 4.1;
    card(s, x, 1.6, 3.8, 3.6, C.background2, "imb-" + i);
    await badge(s, cols[i][0], x + 0.3, 1.9, C.accent1, 0.62, "imb-icon-" + i);
    s.addText([
      { text: cols[i][1], options: { bold: true, fontSize: 19, breakLine: true } },
      { text: " ", options: { fontSize: 6, breakLine: true } },
      { text: cols[i][2], options: { fontSize: 15 } },
    ], { isTextBox: true, x: x + 0.3, y: 2.75, w: 3.2, h: 2.8, valign: "top", margin: 0, color: C.text1 });
  }
  txt(s, "Plus: EMA weights (0.999), mixup by element-wise maximum, warmup + cosine schedule, mixed precision.",
    { x: 0.6, y: 5.6, w: 12.1, h: 0.5, fontSize: 14, color: C.accent4 });
  s.addNotes("Binary cross-entropy treats every negative equally; with 2 percent prevalence, 98 percent of " +
    "the gradient comes from easy negatives. ASL down-weights them and leaves positives untouched. " +
    "Thresholds: a single 0.5 cutoff is wrong when prevalence varies 20-fold. Tuned on validation only, " +
    "never on test, otherwise the reported F1 is inflated.");

  // ================================================================ 7 uncertainty
  s = content("Data and Method");
  T(s, "Two kinds of uncertainty, measured separately");
  card(s, 0.6, 1.6, 5.9, 4.1, C.background2, "epi");
  card(s, 6.8, 1.6, 5.9, 4.1, "E5F4F3", "ale");
  await badge(s, fa.FaRedo, 0.9, 1.9, C.accent1, 0.62, "epi-icon");
  await badge(s, fa.FaCloudSunRain, 7.1, 1.9, C.accent4, 0.62, "ale-icon");
  s.addText([
    { text: "Epistemic: the model doesn't know", options: { bold: true, fontSize: 19, breakLine: true } },
    { text: "Monte Carlo Dropout", options: { italic: true, fontSize: 15, color: C.accent1, breakLine: true } },
    { text: " ", options: { fontSize: 6, breakLine: true } },
    { text: "Run the image 20 times with dropout switched on.", options: { bullet: true, fontSize: 15, breakLine: true } },
    { text: "Spread of the 20 answers = uncertainty.", options: { bullet: true, fontSize: 15, breakLine: true } },
    { text: "Only dropout layers enabled; batch-norm stays in eval mode.", options: { bullet: true, fontSize: 15, breakLine: true } },
    { text: "Shrinks with more training data.", options: { bullet: true, fontSize: 15 } },
  ], { isTextBox: true, x: 0.9, y: 2.75, w: 5.3, h: 3.6, valign: "top", margin: 0, color: C.text1, paraSpaceAfter: 6 });
  s.addText([
    { text: "Aleatoric: the image is ambiguous", options: { bold: true, fontSize: 19, breakLine: true } },
    { text: "Learned confidence head", options: { italic: true, fontSize: 15, color: C.accent4, breakLine: true } },
    { text: " ", options: { fontSize: 6, breakLine: true } },
    { text: "Network outputs its own confidence c per finding.", options: { bullet: true, fontSize: 15, breakLine: true } },
    { text: "Low c on rotated, under-exposed or cluttered films.", options: { bullet: true, fontSize: 15, breakLine: true } },
    { text: "Trained with a confidence penalty [DeVries 2018].", options: { bullet: true, fontSize: 15, breakLine: true } },
    { text: "Does not shrink with more data.", options: { bullet: true, fontSize: 15 } },
  ], { isTextBox: true, x: 7.1, y: 2.75, w: 5.3, h: 3.6, valign: "top", margin: 0, color: C.text1, paraSpaceAfter: 6 });
  s.addNotes("Likely viva question: why model.eval() plus dropout only, not model.train()? Because train " +
    "mode also switches batch-norm to per-batch statistics, and with a single image that corrupts the " +
    "prediction. We enable only the dropout modules.");

  // ================================================================ 8 deferral
  s = content("Data and Method");
  T(s, "The deferral policy: knowing when not to answer");
  const X0 = 0.9, X1 = 8.4, xs = (p) => X0 + (X1 - X0) * p, thr = 0.5;
  s.addShape(pres.shapes.LINE, { x: X0, y: 5.6, w: X1 - X0, h: 0, line: { color: HEX.accent6, width: 1 }, objectName: "axis" });
  [0, 0.25, 0.5, 0.75, 1].forEach((p) => s.addText(p.toFixed(2), { isTextBox: true, x: xs(p) - 0.35,
    y: 5.7, w: 0.7, h: 0.3, fontSize: 11, color: C.accent6, align: "center", margin: 0 }));
  s.addText("predicted probability", { isTextBox: true, x: X0, y: 6.05, w: X1 - X0, h: 0.3, fontSize: 12,
    color: C.accent6, align: "center", margin: 0 });
  s.addShape(pres.shapes.LINE, { x: xs(thr), y: 1.5, w: 0, h: 4.1,
    line: { color: HEX.accent2, width: 2, dashType: "dash" }, objectName: "threshold" });
  s.addText("tuned threshold t", { isTextBox: true, x: xs(thr) + 0.1, y: 1.45, w: 2, h: 0.3, fontSize: 12,
    color: "9A6412", margin: 0 });
  const rows = [[0.62, 0.81, C.accent3, "Positive", "whole interval above t"],
                [0.38, 0.71, C.accent2, "Indeterminate → refer", "interval straddles t"],
                [0.14, 0.33, C.accent1, "Negative", "whole interval below t"]];
  rows.forEach(([lo, hi, col, lab, why], i) => {
    const y = 2.1 + i * 1.15;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: xs(lo), y, w: xs(hi) - xs(lo), h: 0.36, rectRadius: 0.18,
      fill: { color: col }, line: { color: col }, objectName: "ci-" + i });
    s.addShape(pres.shapes.OVAL, { x: xs((lo + hi) / 2) - 0.09, y: y + 0.09, w: 0.18, h: 0.18,
      fill: { color: "FFFFFF" }, line: { color: "FFFFFF" }, objectName: "mean-" + i });
    s.addText([{ text: lab, options: { bold: true, fontSize: 16, color: col, breakLine: true } },
               { text: why, options: { fontSize: 12, color: C.accent4 } }],
      { isTextBox: true, x: 9.0, y: y - 0.15, w: 3.7, h: 0.75, margin: 0, valign: "middle" });
  });
  txt(s, "90% credible interval from 20 MC-Dropout passes; white dot = posterior mean.",
    { x: 0.6, y: 6.45, w: 8, h: 0.35, fontSize: 12, color: C.accent6 });
  s.addNotes("This is the central slide. Rule: positive if the lower bound is above the threshold, " +
    "negative if the upper bound is below, otherwise indeterminate and sent to a radiologist. No new " +
    "parameters, uses only quantities we already compute, explainable to a clinician in one sentence. " +
    "Its value is measured by the selective-prediction curve on the results slides.");

  // ================================================================ 9 calibration & explanation
  s = content("Data and Method");
  T(s, "Calibrated numbers and visible reasoning");
  card(s, 0.6, 1.6, 5.9, 4.1, C.background2, "cal");
  card(s, 6.8, 1.6, 5.9, 4.1, C.background2, "gc");
  await badge(s, fa.FaThermometerHalf, 0.9, 1.9, C.accent1, 0.62, "cal-icon");
  await badge(s, fa.FaEye, 7.1, 1.9, C.accent1, 0.62, "gc-icon");
  s.addText([
    { text: "Temperature scaling", options: { bold: true, fontSize: 19, breakLine: true } },
    { text: " ", options: { fontSize: 6, breakLine: true } },
    { text: "One scalar T fitted on validation: p = σ(z / T).", options: { bullet: true, fontSize: 15, breakLine: true } },
    { text: "Makes “0.9” mean right about 90% of the time.", options: { bullet: true, fontSize: 15, breakLine: true } },
    { text: "Monotonic, so ranking and AUC are unchanged.", options: { bullet: true, fontSize: 15, breakLine: true } },
    { text: "Built-in check: if AUC moves after scaling, there is a bug.", options: { bullet: true, fontSize: 15, bold: true } },
  ], { isTextBox: true, x: 0.9, y: 2.75, w: 5.3, h: 3.6, valign: "top", margin: 0, color: C.text1, paraSpaceAfter: 6 });
  s.addText([
    { text: "Grad-CAM saliency", options: { bold: true, fontSize: 19, breakLine: true } },
    { text: " ", options: { fontSize: 6, breakLine: true } },
    { text: "Heat map of the regions driving each finding's score.", options: { bullet: true, fontSize: 15, breakLine: true } },
    { text: "Used to audit failures, not to decorate successes.", options: { bullet: true, fontSize: 15, breakLine: true } },
    { text: "Catches shortcut learning: attention on tubes, markers or borders [Zech 2018].", options: { bullet: true, fontSize: 15, breakLine: true } },
    { text: "Shows correlation, not proof of localisation.", options: { bullet: true, fontSize: 15, bold: true } },
  ], { isTextBox: true, x: 7.1, y: 2.75, w: 5.3, h: 3.6, valign: "top", margin: 0, color: C.text1, paraSpaceAfter: 6 });
  s.addNotes("Be upfront that Grad-CAM shows correlation, not that the model truly looks at pathology. " +
    "Claiming localisation would need bounding-box annotations, which CheXpert does not have.");

  // ================================================================ 10 silent failure
  pres.addSection({ title: "Engineering" });
  s = content("Engineering");
  T(s, "Lesson learned: data failures must be loud");
  const steps = [
    [fa.FaFolderOpen, "Path mismatch", "Every image path was wrong", C.accent4],
    [fa.FaImage, "Silent fallback", "Loader returned a blank frame and only logged a warning", C.accent4],
    [fa.FaCheckCircle, "“Successful” run", "Training finished with no error at all", C.accent2],
    [fa.FaDiceSix, "Chance performance", "The model had learned nothing", C.accent3],
  ];
  for (let i = 0; i < steps.length; i++) {
    const x = 0.6 + i * 3.1;
    await badge(s, steps[i][0], x + 1.05, 1.65, steps[i][3], 0.75, "fail-" + i);
    s.addText([{ text: steps[i][1], options: { bold: true, fontSize: 17, breakLine: true } },
               { text: steps[i][2], options: { fontSize: 14, color: C.accent4 } }],
      { isTextBox: true, x, y: 2.55, w: 2.85, h: 1.2, align: "center", valign: "top", margin: 0, color: C.text1 });
    if (i < 3) s.addShape(pres.shapes.LINE, { x: x + 2.0, y: 2.02, w: 1.0, h: 0,
      line: { color: HEX.accent6, width: 1.5, endArrowType: "triangle" }, objectName: "flow-" + i });
  }
  card(s, 0.6, 4.2, 12.1, 2.3, "E5F4F3", "fix");
  await badge(s, fa.FaWrench, 0.9, 4.5, C.accent1, 0.62, "fix-icon");
  s.addText([
    { text: "The fix", options: { bold: true, fontSize: 19, breakLine: true } },
    { text: "The loader now raises an error instead of substituting a blank image.", options: { bullet: true, fontSize: 15, breakLine: true } },
    { text: "A verify command checks paths and per-class positive rates before training starts; a class at 0% means misaligned columns.", options: { bullet: true, fontSize: 15, breakLine: true } },
    { text: "Patient-disjoint splits are asserted, not assumed.", options: { bullet: true, fontSize: 15 } },
  ], { isTextBox: true, x: 1.8, y: 4.4, w: 10.6, h: 2.0, valign: "top", margin: 0, color: C.text1, paraSpaceAfter: 4 });
  s.addNotes("A pipeline that reports success while learning nothing is worse than one that crashes. " +
    "This came from a real failure in an earlier version, which is why it is in the paper.");

  // ================================================================ 11 stack
  s = content("Engineering");
  T(s, "From model to clinical workflow");
  const tiers = [
    [fa.FaBrain, "ML core", "PyTorch · timm · albumentations · OpenCV · pydicom · scikit-learn"],
    [fa.FaServer, "Backend", "FastAPI · Pydantic v2 · SQLAlchemy · JWT + bcrypt · Celery + Redis · ReportLab PDF"],
    [fa.FaDesktop, "Frontend", "Next.js 14 · React 18 · TypeScript · Tailwind: findings by urgency, deferred cases flagged"],
    [fa.FaCogs, "Engineering", "pytest · ruff · mypy · GitHub Actions CI · Docker Compose · append-only audit log"],
  ];
  for (let i = 0; i < tiers.length; i++) {
    const y = 1.6 + i * 1.2;
    card(s, 0.6, y, 8.4, 1.0, C.background2, "tier-" + i);
    await badge(s, tiers[i][0], 0.8, y + 0.19, C.accent1, 0.62, "tier-icon-" + i);
    s.addText([{ text: tiers[i][1], options: { bold: true, fontSize: 17, breakLine: true } },
               { text: tiers[i][2], options: { fontSize: 13, color: C.accent4 } }],
      { isTextBox: true, x: 1.65, y, w: 7.2, h: 1.0, valign: "middle", margin: 0, color: C.text1 });
  }
  s.addText("87", { isTextBox: true, x: 9.5, y: 1.9, w: 3.2, h: 1.3, fontSize: 72, bold: true,
    color: C.accent1, align: "center", fontFace: THEME.headFontFace, margin: 0 });
  s.addText("automated tests, all passing", { isTextBox: true, x: 9.5, y: 3.2, w: 3.2, h: 0.5, fontSize: 16,
    color: C.text1, align: "center", margin: 0 });
  s.addText("losses · uncertainty · calibration · data verification · API · paper safeguards",
    { isTextBox: true, x: 9.5, y: 3.8, w: 3.2, h: 1.2, fontSize: 13, color: C.accent4, align: "center", margin: 0 });
  s.addNotes("Inference is asynchronous because 20 Monte Carlo passes take seconds, not milliseconds, so it " +
    "runs on a Celery worker instead of blocking the request. Every prediction is audit-logged. " +
    "Offer a live demo here if time permits.");

  // ================================================================ 12 protocol
  pres.addSection({ title: "Evaluation" });
  s = content("Evaluation");
  T(s, "Evaluation protocol");
  const proto = [
    [fa.FaUserLock, "Patient-disjoint split", "Train / validation / test share no patient; asserted at load time."],
    [fa.FaLock, "Test set touched once", "Thresholds and temperature fitted on validation, then frozen."],
    [fa.FaChartLine, "Discrimination", "Mean AUC-ROC (14 and CheXpert-5), average precision, F1."],
    [fa.FaBullseye, "Calibration", "Expected calibration error (15 bins), Brier score, reliability diagram."],
    [fa.FaLayerGroup, "Does uncertainty work?", "Accuracy by σ quartile, and accuracy vs coverage under deferral."],
  ];
  for (let i = 0; i < proto.length; i++) {
    const col = i < 3 ? 0 : 1, row = i < 3 ? i : i - 3;
    const x = 0.6 + col * 6.2, y = 1.6 + row * 1.6;
    await badge(s, proto[i][0], x, y, i === 4 ? C.accent2 : C.accent1, 0.62, "proto-" + i);
    s.addText([{ text: proto[i][1], options: { bold: true, fontSize: 18, breakLine: true } },
               { text: proto[i][2], options: { fontSize: 15, color: C.accent4 } }],
      { isTextBox: true, x: x + 0.9, y: y - 0.05, w: 5.0, h: 1.2, valign: "top", margin: 0, color: C.text1 });
  }
  s.addNotes("The last item decides whether the uncertainty component is justified: if accuracy does not " +
    "fall as uncertainty rises, the estimates carry no information, and we would report that honestly.");

  // ================================================================ 13 results
  s = content("Evaluation");
  T(s, "Results: discrimination and calibration");
  if (DATA.real) {
    const pc = DATA.per_class.filter((r) => r.auc != null).sort((a, b) => a.auc - b.auc);
    s.addChart(pres.charts.BAR, [{ name: "AUC", labels: pc.map((r) => r.label), values: pc.map((r) => r.auc) }], {
      x: 0.6, y: 1.5, w: 7.6, h: 5.2, barDir: "bar", chartColors: [HEX.accent1],
      showTitle: true, title: "Per-class AUC-ROC (test split)", titleFontSize: 14, titleColor: HEX.dk1, titleFontFace: "+mj-lt",
      showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0.00", dataLabelFontSize: 10,
      dataLabelColor: HEX.dk1, dataLabelFontFace: "+mn-lt",
      valAxisMinVal: 0.5, valAxisMaxVal: 1.0, valAxisLabelFontSize: 10, catAxisLabelFontSize: 11,
      valAxisLabelColor: HEX.accent4, catAxisLabelColor: HEX.dk1, catAxisLabelFontFace: "+mn-lt", valAxisLabelFontFace: "+mn-lt",
      valGridLine: { color: "E3E8EB", size: 0.5 }, catGridLine: { style: "none" }, showLegend: false,
    });
    const kp = [[f4(DATA.mean_auc), "mean AUC, 14 findings"], [f4(DATA.comp_auc), "mean AUC, CheXpert-5"],
                [f4(DATA.ece), "calibration error (ECE)"]];
    kp.forEach(([v, l], i) => {
      s.addText(v, { isTextBox: true, x: 8.7, y: 1.6 + i * 1.65, w: 4, h: 0.9, fontSize: 44, bold: true,
        color: C.accent1, fontFace: THEME.headFontFace, margin: 0 });
      s.addText(l, { isTextBox: true, x: 8.7, y: 2.5 + i * 1.65, w: 4, h: 0.4, fontSize: 14, color: C.accent4, margin: 0 });
    });
    s.addText(`Held-out, patient-disjoint test split, n = ${DATA.n.toLocaleString("en-IN")} images.`,
      { isTextBox: true, x: 8.7, y: 6.3, w: 4, h: 0.4, fontSize: 12, color: C.accent6, margin: 0 });
  } else {
    pending(s, 0.6, 1.6, 7.6, 4.9, "Per-class AUC bar chart for all 14 findings\n(native chart, generated from results/per_class_metrics.csv)");
    ["mean AUC, 14 findings", "mean AUC, CheXpert-5", "calibration error (ECE)"].forEach((l, i) => {
      s.addText("—", { isTextBox: true, x: 8.7, y: 1.6 + i * 1.65, w: 4, h: 0.9, fontSize: 44, bold: true,
        color: C.accent2, fontFace: THEME.headFontFace, margin: 0 });
      s.addText(l, { isTextBox: true, x: 8.7, y: 2.5 + i * 1.65, w: 4, h: 0.4, fontSize: 14, color: C.accent4, margin: 0 });
    });
  }
  s.addNotes(DATA.real
    ? "Point out which findings are strongest and weakest and relate it to how visible each finding is. " +
      "Large high-contrast findings like cardiomegaly and effusion should lead; small or rare ones trail."
    : "PLACEHOLDER SLIDE. After the Kaggle run: python -m paper.build_deck --results results");

  // ================================================================ 14 comparison
  s = content("Evaluation");
  T(s, "Where this sits against published CheXpert work");
  const cmp = [["MedViT", 0.7511, "Agbo et al., Sci. Rep. 2026"], ["Hybrid CNN-ViT", 0.7168, "Agbo et al., Sci. Rep. 2026"]];
  if (DATA.real) {
    s.addChart(pres.charts.BAR, [{ name: "Mean AUC", labels: ["Hybrid CNN-ViT", "MedViT", "RadiScan (ours)"],
      values: [0.7168, 0.7511, DATA.mean_auc] }], {
      x: 0.6, y: 1.5, w: 7.6, h: 4.6, barDir: "col", chartColors: [HEX.accent6, HEX.accent6, HEX.accent1],
      showTitle: true, title: "Macro AUC-ROC on CheXpert", titleFontSize: 14, titleColor: HEX.dk1, titleFontFace: "+mj-lt",
      showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0.000", dataLabelFontSize: 12,
      dataLabelColor: HEX.dk1, dataLabelFontFace: "+mn-lt", valAxisMinVal: 0.5, valAxisMaxVal: 1.0,
      valAxisLabelColor: HEX.accent4, catAxisLabelColor: HEX.dk1, catAxisLabelFontFace: "+mn-lt", valAxisLabelFontFace: "+mn-lt",
      valGridLine: { color: "E3E8EB", size: 0.5 }, catGridLine: { style: "none" }, showLegend: false,
    });
  } else {
    cmp.forEach(([m, v, src], i) => {
      card(s, 0.6 + i * 3.9, 1.7, 3.6, 2.6, C.background2, "pub-" + i);
      s.addText(v.toFixed(4), { isTextBox: true, x: 0.6 + i * 3.9, y: 1.95, w: 3.6, h: 1.0, fontSize: 40, bold: true,
        color: C.accent4, align: "center", fontFace: THEME.headFontFace, margin: 0 });
      s.addText([{ text: m, options: { bold: true, fontSize: 16, breakLine: true } },
                 { text: src, options: { fontSize: 12, color: C.accent6 } }],
        { isTextBox: true, x: 0.6 + i * 3.9, y: 3.0, w: 3.6, h: 0.9, align: "center", margin: 0, color: C.text1 });
    });
    pending(s, 0.6, 4.6, 7.5, 1.6, "RadiScan mean AUC on the same corpus");
  }
  card(s, 8.6, 1.6, 4.1, 4.6, "FFF6E5", "caveat");
  await badge(s, fa.FaInfoCircle, 8.9, 1.9, C.accent2, 0.62, "caveat-icon");
  s.addText([
    { text: "Read this comparison carefully", options: { bold: true, fontSize: 17, breakLine: true } },
    { text: "The published models were evaluated on a different split of CheXpert. The comparison is indicative, not controlled.", options: { fontSize: 14, breakLine: true } },
    { text: " ", options: { fontSize: 6, breakLine: true } },
    { text: "Our claim is not higher AUC. It is that each prediction comes with calibrated uncertainty and a referral rule.", options: { fontSize: 14, bold: true } },
  ], { isTextBox: true, x: 8.9, y: 2.75, w: 3.5, h: 3.3, valign: "top", margin: 0, color: C.text1 });
  s.addNotes("Do not claim we beat these models; the splits differ. If asked: our contribution is the " +
    "uncertainty and deferral machinery, and a strong reproducible baseline trunk.");

  // ================================================================ 15 uncertainty validated
  s = content("Evaluation");
  T(s, "Does the uncertainty actually carry information?");
  if (DATA.real) {
    s.addChart(pres.charts.BAR, [{ name: "Accuracy", labels: DATA.quartiles.map((q) => q.q),
      values: DATA.quartiles.map((q) => q.acc) }], {
      x: 0.6, y: 1.5, w: 6.0, h: 4.6, barDir: "col", chartColors: [HEX.accent1],
      showTitle: true, title: "Accuracy by uncertainty quartile (Q1 = most certain)", titleFontSize: 13,
      titleColor: HEX.dk1, titleFontFace: "+mj-lt", showValue: true, dataLabelPosition: "outEnd",
      dataLabelFormatCode: "0.000", dataLabelFontSize: 11, dataLabelColor: HEX.dk1, dataLabelFontFace: "+mn-lt",
      valAxisLabelColor: HEX.accent4, catAxisLabelColor: HEX.dk1, catAxisLabelFontFace: "+mn-lt", valAxisLabelFontFace: "+mn-lt",
      valGridLine: { color: "E3E8EB", size: 0.5 }, catGridLine: { style: "none" }, showLegend: false,
    });
    if (DATA.selective) {
      s.addChart(pres.charts.LINE, [{ name: "Accuracy", labels: DATA.selective.map((r) => (r.coverage ?? 0).toFixed(2)),
        values: DATA.selective.map((r) => r.accuracy) }], {
        x: 6.9, y: 1.5, w: 5.8, h: 4.6, chartColors: [HEX.accent2], lineSize: 2.5, lineDataSymbol: "none",
        showTitle: true, title: "Accuracy vs coverage under deferral", titleFontSize: 13, titleColor: HEX.dk1,
        titleFontFace: "+mj-lt", valAxisLabelColor: HEX.accent4, catAxisLabelColor: HEX.accent4,
        catAxisLabelFontFace: "+mn-lt", valAxisLabelFontFace: "+mn-lt",
        valGridLine: { color: "E3E8EB", size: 0.5 }, catGridLine: { style: "none" }, showLegend: false,
      });
    }
    s.addText(DATA.monotone
      ? "Accuracy falls monotonically as uncertainty rises: the estimates are informative, and the deferral policy is justified."
      : "Accuracy does NOT fall monotonically. MC Dropout with 20 samples did not resolve error well at this scale; we report this as a negative result.",
      { isTextBox: true, x: 0.6, y: 6.25, w: 12.1, h: 0.55, fontSize: 15, bold: true,
        color: DATA.monotone ? C.accent1 : C.accent3, margin: 0 });
  } else {
    pending(s, 0.6, 1.6, 6.0, 4.6, "Accuracy per uncertainty quartile, Q1 (most certain) to Q4\n(from results/uncertainty_quartiles.csv)");
    pending(s, 6.9, 1.6, 5.8, 4.6, "Accuracy vs coverage when the least certain cases are deferred\n(from results/selective_prediction.csv)");
    txt(s, "If accuracy falls from Q1 to Q4, the uncertainty is informative. If not, we report it as a negative result.",
      { x: 0.6, y: 6.35, w: 12.1, h: 0.45, fontSize: 14, color: C.accent4 });
  }
  s.addNotes("This slide justifies the whole uncertainty component. If the line is not monotone, say so " +
    "plainly; an honest negative result is defensible, a hidden one is not.");

  // ================================================================ 16 limitations
  pres.addSection({ title: "Closing" });
  s = content("Closing");
  T(s, "Limitations we state openly");
  const lim = [
    [fa.FaTags, "Noisy labels", "Extracted from report text, not re-annotated by experts."],
    [fa.FaHospital, "One hospital", "Stanford only. No evidence yet for Indian hospitals or other scanners."],
    [fa.FaColumns, "Frontal view only", "No lateral views, prior studies or clinical history."],
    [fa.FaDice, "Approximate Bayes", "MC Dropout with S = 20 is chosen for speed; ensembles would be better."],
    [fa.FaSearch, "Saliency ≠ localisation", "Grad-CAM shows correlation; no bounding-box evaluation."],
    [fa.FaGavel, "Not a medical device", "No prospective study, reader study or regulatory approval."],
  ];
  for (let i = 0; i < lim.length; i++) {
    const col = i % 2, row = Math.floor(i / 2);
    const x = 0.6 + col * 6.2, y = 1.6 + row * 1.65;
    card(s, x, y, 5.9, 1.4, C.background2, "lim-" + i);
    await badge(s, lim[i][0], x + 0.25, y + 0.39, C.accent4, 0.62, "lim-icon-" + i);
    s.addText([{ text: lim[i][1], options: { bold: true, fontSize: 17, breakLine: true } },
               { text: lim[i][2], options: { fontSize: 14, color: C.accent4 } }],
      { isTextBox: true, x: x + 1.1, y, w: 4.6, h: 1.4, valign: "middle", margin: 0, color: C.text1 });
  }
  s.addNotes("Stating limitations before the panel asks shows you understand the work. The most important: " +
    "this is research software, not a diagnostic device.");

  // ================================================================ 17 conclusion
  s = pres.addSlide({ masterName: "DARK", sectionTitle: "Closing" });
  T(s, "A classifier that knows when to ask for help");
  s.addText([
    { text: "Quantifies, calibrates and explains its uncertainty for 14 findings", options: { bullet: true, breakLine: true } },
    { text: "Refers cases it cannot resolve instead of guessing", options: { bullet: true, breakLine: true } },
    { text: "Delivered as a tested, auditable clinical web application", options: { bullet: true, breakLine: true } },
    { text: "Next: deep ensembles · external validation on Indian data · reader study", options: { bullet: true, color: C.accent5 } },
  ], { isTextBox: true, x: 0.8, y: 4.25, w: 11.7, h: 2.2, fontSize: 18, color: C.background1, margin: 0, paraSpaceAfter: 8 });
  s.addText("Thank you  ·  Questions?", { isTextBox: true, x: 0.8, y: 6.55, w: 11.7, h: 0.5, fontSize: 16,
    italic: true, color: C.accent6, margin: 0 });
  s.addNotes("Close on the deferral idea: the value is not a higher number, it is a system that is honest " +
    "about what it does not know. Then invite questions.");

  await pres.writeFile({ fileName: OUTFILE });
  if (SKILL) {
    const { applyTheme } = require(path.join(SKILL, "scripts", "apply_theme.js"));
    await applyTheme(OUTFILE, THEME);
  }
  console.log("wrote " + OUTFILE);
})().catch((e) => { console.error(e); process.exit(1); });
