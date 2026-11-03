"""Interim revision: every number that already exists from real runs, with the
values that only the running Kaggle job can produce marked ⟦pending⟧.

    python interim.py <orig_results_dir> <interim_dir> <out_spec.json>

Sections I-IV and VI are taken verbatim from content.py (same text as the final
paper); Sections III-B/C, V, VII, VIII and IX are written here from real data.
"""
import json
import math
import sys
from pathlib import Path

import pandas as pd

ORIG, INT, OUT = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
SRC = Path(__file__).with_name("content.py").read_text().splitlines(keepends=True)
PEND = "⟦[pending]⟧"
GITHUB = "https://github.com/vanshvj01/RadiScan-AI"
DEMO = "https://radiscan-ai.vercel.app"

summ = json.loads((ORIG / "summary.json").read_text())
pc = pd.read_csv(ORIG / "per_class_metrics.csv")
qs = pd.read_csv(ORIG / "uncertainty_quartiles.csv")
HIST = json.loads((INT / "orig_history.json").read_text())
lat_stage = pd.read_csv(INT / "latency_stages_cpu.csv").set_index("stage")
lat_S = pd.read_csv(INT / "latency_vs_S_cpu.csv")
res_cpu = json.loads((INT / "resources_cpu.json").read_text())
# checkpoint metadata (best_model.pth, read earlier from the trained checkpoint)
T = 4.015675067901611
CK = {"epoch": 6, "val_mean_auc": 0.751036181964288, "val_mean_ap": 0.3693857100839539, "val_brier": 0.2749187118188701,
      "ece_before": 0.5223101816603115, "ece_after": 0.3824401225277355}
THR = {'No Finding': 0.5774, 'Enlarged Cardiomediastinum': 0.5491, 'Cardiomegaly': 0.5690, 'Lung Opacity': 0.6014, 'Lung Lesion': 0.5509, 'Edema': 0.5843,
       'Consolidation': 0.5568, 'Pneumonia': 0.5424, 'Atelectasis': 0.5592, 'Pneumothorax': 0.5626, 'Pleural Effusion': 0.5929, 'Pleural Other': 0.5434,
       'Fracture': 0.5617, 'Support Devices': 0.6138}


def f(x, nd=3):
    if x is None:
        return PEND
    v = float(x)
    if math.isnan(v):
        return "--"
    return f"{v:.{nd}f}"


def pct(x, nd=1):
    return f(100 * float(x), nd)


def ints(x):
    return f"{int(x):,}"


blocks, eqs = [], []


def add(t, **kw):
    blocks.append({"t": t, **kw})


def P(*runs, noindent=False):
    add("p", runs=list(runs), noindent=noindent)


def EQ(latex):
    eqs.append(latex)
    add("eq", id=len(eqs), n=len(eqs))


def I(t):
    return {"text": t, "i": True}


def take(a, b):
    """Execute lines a..b (1-based, inclusive) of content.py in this namespace."""
    exec("".join(SRC[a - 1:b]), globals())


ov = summ["overall"]
cx5 = summ["chexpert_competition_5"]

# ---------------------------------------------------------------- front matter
add("cols", n=1)
add("title", text="Uncertainty-Aware and Explainable Multi-Label Classification of Chest Radiographs with an Explicit Deferral Policy")
add("author", runs=["Vansh R. Vijayvargiya, Vaibhav Dave, Uzair Khan, Aryan Lodha, Purnima Ahirao"], size=22)
add("author", runs=[I("Department of Information Technology, K. J. Somaiya School of Engineering, Somaiya Vidyavihar University, Mumbai, India")], size=18)
add("author", runs=[{"text": "v.vijayvargiya@somaiya.edu, ", "i": True}, {"text": "[co-author emails]", "i": True, "hl": True}], size=18, after=240)
add("cols", n=2)
add("abstract", label="Abstract", runs=[
    "Deep convolutional networks match reported radiologist performance on several thoracic findings, yet a bare probability says nothing about how far it should be trusted. "
    "We present RadiScan, a multi-label chest radiograph classifier built on the premise that a decision-support tool must quantify, calibrate and explain its own uncertainty, and must decline the cases it cannot resolve. "
    "A DenseNet-121 trunk with three task heads is combined with Monte Carlo (MC) Dropout for epistemic uncertainty, a confidence head trained with a hint-and-penalty objective for aleatoric uncertainty, post-hoc temperature scaling, and Grad-CAM. "
    "A deferral policy refers a finding to a radiologist when its 95% posterior interval straddles its validation-tuned operating threshold or its posterior standard deviation exceeds a tolerance selected on validation data alone. "
    f"On a held-out, patient-disjoint CheXpert test split of {ints(summ['n_samples'])} frontal radiographs the model attains a mean AUROC of {f(ov['mean_auc'])} over 14 findings ({f(cx5['mean_auc'])} on the five competition findings) and a mean average precision of {f(ov['mean_ap'])}. "
    f"Temperature scaling (T = {f(T, 2)}) reduces the expected calibration error on validation data from {f(CK['ece_before'])} to {f(CK['ece_after'])}; the large residual error is traced to the location bias of Asymmetric Loss. "
    f"Accuracy falls monotonically from {pct(qs.iloc[0]['accuracy'])}% to {pct(qs.iloc[3]['accuracy'])}% across quartiles of increasing posterior standard deviation, showing that the epistemic estimate is informative. "
    "We also report the deferral rate and performance at several coverage levels, calibration before and after scaling, component ablations, the effect of the number of MC passes, latency and throughput, and representative saliency maps. "
    f"Code: {GITHUB}; live demonstration: {DEMO}."])
add("abstract", label="Index Terms", runs=["chest radiography, multi-label classification, uncertainty quantification, Monte Carlo dropout, calibration, selective prediction, explainable AI, Grad-CAM, clinical decision support, CheXpert"])

take(200, 224)  # I, II, III-A

add("h2", text="Training, Validation and Test Splits")
P("Only frontal radiographs are used. Patients, not images, are assigned to partitions: the patient identifier is parsed from each image path, the shuffled list of patients (seed 42) is consumed whole-patient-at-a-time until the test and validation partitions each reach 4,000 images, and the remaining patients form the training pool, from which 40,000 images are drawn. Table I gives the resulting sizes. Because whole patients are assigned, no partition can hold a subset of another patient's studies.")
add("table", caption="Patient-disjoint partitions (frontal radiographs)", cols=[2.2, 1.4, 1.4, 1.4, 1.6],
    header=["Split", "Patients", "Studies", "Images", "Img./patient"],
    rows=[["Train", PEND, PEND, "40,000", PEND], ["Validation", PEND, PEND, "4,000", PEND], ["Test", PEND, PEND, ints(summ["n_samples"]), PEND]],
    note=[f"Pairwise patient overlap train–val / train–test / val–test: {PEND}; image-path overlap: {PEND}."])
P(f"Patient-level separation is verified twice. At split time the procedure asserts that the three patient sets have an empty pairwise intersection and terminates otherwise. Independently, the evaluation script re-reads the three saved split files, recomputes the patient identifiers from the image paths and reports the pairwise intersections of patient sets and of image paths {PEND}. The evaluation reuses the split files of the original training run verbatim rather than regenerating them.")
add("h2", text="Uncertain Labels and Evaluation Protocol")
P("CheXpert encodes positive, negative and uncertain annotations. During training, uncertain targets are replaced by a value drawn uniformly from (0.05, 0.85) (label-smoothing regression), following [2]. The results in Table IV were computed with the same policy applied to the test split, so an uncertain test annotation counts as positive when its draw exceeds 0.5. Because these draws are random, the metrics partly reflect the draw rather than the model; the corrected protocol, which excludes uncertain test annotations from every metric, is being re-evaluated on the same split " + PEND + ".")

take(236, 273)  # III-D, IV-A..G (first part)
P(f"The first implementation used a fixed tolerance τ = 0.15 chosen a priori. The observed posterior standard deviations are about two orders of magnitude smaller: the quartile means on the test split range from {f(qs.iloc[0]['mean_uncertainty'], 4)} to {f(qs.iloc[3]['mean_uncertainty'], 4)} (Table VI). That rule therefore never fired, and deferral was governed entirely by the interval condition. We therefore select τ on the validation split only, as the smallest value whose validation coverage reaches a target c*:")
EQ(r"\tau^{\star}(c^{\star})=\min\Big\{\tau:\ 1-\frac{1}{|\mathcal{V}|}\sum_{(i,k)\in\mathcal{V}}D_{ik}(\tau)\ \ge\ c^{\star}\Big\}")
P(f"Coverage is non-decreasing in τ, so this is a one-dimensional search over the quantiles of the validation standard deviations; no test datum enters it. For c* = 0.95, 0.90 and 0.80 it gives τ* = {PEND}. The deployed configuration uses c* = 0.90. The generated report is rule-based over calibrated probabilities rather than written by a language model, so every statement is auditable.")

# ---------------------------------------------------------------- V
add("h1", text="System Implementation")
P(f"The model is served by a three-tier application ({DEMO}). A Next.js front end submits a study; a FastAPI service validates and persists it, writes an audit entry, enqueues an inference task and returns immediately; a worker decodes and preprocesses the image, runs the MC inference, computes Grad-CAM for the three most probable findings and generates the report, and progress is pushed to the client over a WebSocket. Every checkpoint embeds its label list, backbone, input size, temperature and thresholds, and loading refuses a checkpoint whose labels disagree with those requested.")
hw = res_cpu["hardware"]
s1d = lat_S[(lat_S.S == 1) & (lat_S["mode"] == "deployed")]["median_ms"].iloc[0]
s2d = lat_S[(lat_S.S == 2) & (lat_S["mode"] == "deployed")]["median_ms"].iloc[0]
s2c = lat_S[(lat_S.S == 2) & (lat_S["mode"] == "cached")]["median_ms"].iloc[0]
per_pass = (s2d - s1d)
P(f"Table II reports the measured cost of the deployed pipeline for the trained checkpoint on a {hw['cpu_count']}-core CPU ({hw['cpu']}, PyTorch {hw['torch'].split('+')[0]}); GPU figures {PEND}. "
  f"MC inference dominates: each stochastic pass re-runs the full DenseNet on two views, costing about {f(per_pass / 1000, 2)} s per pass on this CPU, so S = 20 takes {f(lat_stage.loc['mc_inference_ms', 'median_ms'] / 1000, 1)} s.")
add("table", caption="Measured cost of the deployed pipeline on CPU (S = 20, flip TTA)", cols=[2.6, 1.2],
    header=["Stage / resource", "Value"],
    rows=[["Pre-processing (crop, CLAHE, resize)", f"{f(lat_stage.loc['preprocess_ms', 'median_ms'], 0)} ms"],
          ["MC inference, deployed (40 full passes)", f"{f(lat_stage.loc['mc_inference_ms', 'median_ms'] / 1000, 1)} s"],
          ["MC inference, trunk cached (S = 2)", f"{f(s2c / 1000, 2)} s"],
          ["Grad-CAM, three findings", f"{f(lat_stage.loc['gradcam_top3_ms', 'median_ms'] / 1000, 1)} s"],
          ["Report generation", "< 1 ms"],
          ["Total per study (deployed)", f"{f(lat_stage.loc['total_ms', 'median_ms'] / 1000, 1)} s"],
          ["Throughput, batch 1 (deployed)", f"{f(3600 / (lat_stage.loc['total_ms', 'median_ms'] / 1000), 0)} studies/h"],
          ["Peak resident memory", f"{f(res_cpu['process_rss_mb'], 0)} MB"],
          ["Checkpoint size / parameters", f"{f(res_cpu['checkpoint_mb'], 1)} MB / {f(res_cpu['parameters_m'], 2)} M"],
          ["GPU latency, throughput and memory", PEND]],
    note=["Medians after warm-up on the trained checkpoint. Throughput assumes one inference worker."])
P(f"Because every dropout layer lies after the trunk, the trunk output is identical across MC passes. Computing it once per view and re-sampling only the 0.5 M-parameter heads is mathematically identical to re-running the network, and the measurements show its effect directly: with the trunk cached, S = 2 costs {f(s2c / 1000, 2)} s against {f(s2d / 1000, 2)} s deployed, and the cost of additional passes becomes negligible, so 20 passes fall from {f(lat_stage.loc['mc_inference_ms', 'median_ms'] / 1000, 1)} s to roughly {f(s2c / 1000, 1)} s on the CPU. This makes CPU-only deployment practical. Asynchronous execution keeps web workers free while inference runs, so throughput scales with the number of inference workers.")

R = {"epoch": CK["epoch"]}
gpu_name = "Kaggle GPU"
ABL = {}
S_list = [1, 2, 5, 10, 20, 30, 50]
take(324, 346)  # VI

# ---------------------------------------------------------------- VII
add("h1", text="Results and Discussion")
add("h2", text="Discrimination")
rows = [[r["label"], ints(r["support"]), f(r["auc"]), f(r["ap"]), f(r["precision"]), f(r["recall"]), f(r["f1"])] for _, r in pc.iterrows()]
rows.append(["Mean", "--", f(ov["mean_auc"]), f(ov["mean_ap"]), f(ov["mean_precision"]), f(ov["mean_recall"]), f(ov["mean_f1"])])
add("table", caption="Per-class performance on the held-out, patient-disjoint test split", cols=[2.9, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    header=["Finding", "Pos.", "AUROC", "AP", "Prec.", "Rec.", "F1"], rows=rows, font=7, rules=[len(rows) - 2],
    note=[f"n = {ints(summ['n_samples'])} test images; MC-20 posterior mean, temperature-scaled; precision, recall and F1 at validation-tuned thresholds. Uncertain test labels handled as in Section III-C. Re-evaluation with uncertain labels excluded: {PEND}."])
b = pc.sort_values("auc", ascending=False)
rare = pc.sort_values("support").iloc[0]
P(f"Table IV gives per-class results. Discrimination is highest for {b.iloc[0]['label']} ({f(b.iloc[0]['auc'])}), {b.iloc[1]['label']} ({f(b.iloc[1]['auc'])}) and {b.iloc[2]['label']} ({f(b.iloc[2]['auc'])}), and lowest for {b.iloc[-1]['label']} ({f(b.iloc[-1]['auc'])}) and {b.iloc[-2]['label']} ({f(b.iloc[-2]['auc'])}). The ordering follows conspicuity: large, high-contrast findings and devices are recognised more reliably than subtle parenchymal findings and the ill-defined mediastinal contour. AP is low for rare findings even where AUROC is moderate ({rare['label']}: {ints(rare['support'])} positives, AUROC {f(rare['auc'])}, AP {f(rare['ap'])}), which is why AP is reported alongside AUROC. Recall exceeds precision for most findings because the thresholds maximise F1 on a validation split in which positives are scarce.")
add("table", caption="Aggregate performance and published CheXpert results", cols=[2.6, 1.6, 1.0, 1.0, 0.9],
    header=["Method", "Backbone", "Mean AUROC", "CheXpert-5", "ECE"],
    rows=[["MedViT [5]", "Transformer", "0.751", "--", "--"], ["Hybrid CNN-ViT [5]", "CNN + ViT", "0.717", "--", "--"],
          ["Ours (validation split)", "DenseNet-121", f(CK["val_mean_auc"]), "--", f(CK["ece_after"])],
          ["Ours (test split)", "DenseNet-121", f(ov["mean_auc"]), f(cx5["mean_auc"]), f(summ["ece"])]],
    font=7, note=["Published results use different splits and are indicative only. ECE of the temperature-scaled MC-20 output, 15 bins."])
P(f"Table V places the model against published CheXpert figures. Test performance ({f(ov['mean_auc'])}) is close to validation performance ({f(CK['val_mean_auc'])}), which is consistent with the absence of leakage between the patient-disjoint partitions. The published results were obtained on different splits, so the comparison is indicative rather than controlled; we do not claim to outperform them.")

add("h2", text="Calibration")
P(f"Temperature scaling fitted T* = {f(T)} on the validation split. On validation data it reduces the ECE from {f(CK['ece_before'])} to {f(CK['ece_after'])}, a 27% reduction, and the Brier score after scaling is {f(CK['val_brier'])}; on the test split the ECE of the deployed MC-20 output is {f(summ['ece'])}. Because division by T is monotonic, it leaves the ranking, and hence the AUROC, unchanged. "
  "A residual ECE near 0.38 is nonetheless large, and its origin is visible in the operating points: the validation-tuned thresholds all lie between "
  f"{f(min(THR.values()), 2)} and {f(max(THR.values()), 2)}, even for findings with prevalence below 2%, which indicates that the probabilities are not on a scale that tracks prevalence. This is the expected effect of Asymmetric Loss, whose probability shift and strong negative focusing remove most of the penalty on moderately confident negatives, and a single temperature, which rescales logits around zero, cannot remove a per-class offset. "
  f"A temperature of {f(T, 1)} also compresses every probability toward 0.5. Test-split ECE and Brier score before and after scaling, and the effect of a class-wise Platt recalibration, which can correct an offset, are {PEND}.")
add("table", caption="Calibration before and after temperature scaling", cols=[2.2, 1.0, 1.0, 1.0],
    header=["Split / output", "ECE before", "ECE after", "Brier after"],
    rows=[["Validation, single pass", f(CK["ece_before"]), f(CK["ece_after"]), f(CK["val_brier"])],
          ["Test, MC-20", PEND, f(summ["ece"]), PEND],
          ["Test, MC-20, class-wise Platt", "--", PEND, PEND]],
    font=7, note=[f"T* = {f(T)} fitted on validation. ECE over 15 equal-width bins pooled over all (image, finding) pairs."])

add("h2", text="Epistemic and Aleatoric Uncertainty")
qrows = [[str(r["quartile"]).replace("confident", "certain"), f(r["mean_uncertainty"], 4), ints(r["n"]), pct(r["accuracy"])] for _, r in qs.iterrows()]
add("table", caption="Accuracy stratified by posterior standard deviation quartile", cols=[2.4, 1.1, 1.0, 1.0],
    header=["Quartile", "Mean σ", "n", "Accuracy %"], rows=qrows, font=7,
    note=["Quartiles of the 56,000 (image, finding) pairs of the test split ranked by σ (S = 20); accuracy of the thresholded decision."])
drop = qs.iloc[0]["accuracy"] - qs.iloc[3]["accuracy"]
P(f"Table VI is the central test of the epistemic estimate. Accuracy decreases monotonically across quartiles of increasing σ, from {pct(qs.iloc[0]['accuracy'])}% in the most certain quarter to {pct(qs.iloc[3]['accuracy'])}% in the least certain, a drop of {f(100 * drop, 1)} percentage points; the least certain quarter contains {f((1 - qs.iloc[3]['accuracy']) / (1 - qs.iloc[0]['accuracy']), 1)} times the error rate of the most certain. The estimate therefore carries information about the model's own errors, which is what the deferral policy relies on. The absolute values of σ are small (quartile means {f(qs.iloc[0]['mean_uncertainty'], 4)}–{f(qs.iloc[3]['mean_uncertainty'], 4)}), consistent with dropout acting only on the heads (Section VIII); it is their ranking, not their scale, that carries the signal. "
  f"Error-detection AUROC and AURC for σ, mutual information, predictive entropy and the confidence head, accuracy by quartile of 1 − c, and the response of the confidence head to findings that the reporting radiologist marked as uncertain are {PEND}.")

add("h2", text="Deferral and Coverage")
P(f"Because accuracy falls monotonically with σ (Table VI), deferring the least certain findings necessarily raises accuracy on those retained. The quartile table bounds the effect: retaining only the three most certain quarters (75% coverage) gives an accuracy of {pct(qs.iloc[:3]['accuracy'].mean())}% against {pct(qs['accuracy'].mean())}% with no deferral, and retaining the most certain half gives {pct(qs.iloc[:2]['accuracy'].mean())}%. The percentage of findings and studies deferred under the interval rule and under the validation-selected tolerance, and accuracy, sensitivity, specificity, F1 and AUROC at coverage levels from 100% to 50%, are {PEND}.")
add("table", caption="Accuracy on retained findings at different coverage levels (ranking by σ)", cols=[1.4, 1.2, 1.2],
    header=["Coverage", "Accuracy %", "Source"],
    rows=[["100%", pct(qs["accuracy"].mean()), "Table VI"], ["75%", pct(qs.iloc[:3]["accuracy"].mean()), "Table VI"], ["50%", pct(qs.iloc[:2]["accuracy"].mean()), "Table VI"],
          ["25%", pct(qs.iloc[0]["accuracy"]), "Table VI"], ["Deferral %, sens., spec., F1, AUROC", PEND, "--"]],
    font=7, note=["Coverage levels that coincide with σ quartiles; equal-sized quartiles make the retained accuracy the mean of the retained quartiles."])

add("h2", text="Number of Monte Carlo Passes")
P(f"Latency grows linearly with S in the deployed implementation, about {f(per_pass / 1000, 2)} s per pass on the CPU, and becomes nearly independent of S with the trunk cached (Section V). The effect of S on discrimination, calibration, the convergence of σ and the deferral rate for S = 1–50 is {PEND}.")

add("h2", text="Ablation Study")
P(f"Mean AUROC, AP, CheXpert-5 AUROC, F1 and ECE for each arm of the ablation protocol (Section VI) are {PEND}.")
add("h2", text="Visual Explanations")
P(f"Grad-CAM maps for rule-selected true positives, false positives, false negatives and deferred cases are {PEND}.")

# ---------------------------------------------------------------- VIII / IX
add("h1", text="Limitations")
P("Labels are extracted automatically from reports, and the resulting noise bounds achievable discrimination. The corpus originates from a single institution, and we present no evidence of generalisation to other equipment, protocols or populations [10]; external validation on an Indian hospital cohort is required before any claim of applicability there. Only frontal projections are used, and no prior studies or clinical metadata are modelled.")
P("Because dropout is applied only after the trunk, the MC estimate captures uncertainty in the heads given a fixed representation; it is closer to last-layer Bayesian inference than to a posterior over the full network, which explains the small magnitude of σ. Deep ensembles [19] would capture representation uncertainty at proportionally greater cost. The residual miscalibration under temperature scaling shows that loss design and calibration interact; class-wise calibration is needed whenever an asymmetric loss is used. Validation performance of the main run collapsed after its best epoch; early stopping protected the reported model, but the cause (possibly an optimisation instability once the full learning rate and the tenfold head rate act on the unfrozen backbone under mixed precision) has not been isolated.")
P("Grad-CAM indicates correlation, not localisation; quantifying localisation requires bounding-box annotation and a pointing-game protocol. Finally, the system has undergone no prospective evaluation, reader study or regulatory assessment, and we make no such claim.")
add("h1", text="Conclusion and Future Work")
P(f"We have presented a multi-label chest radiograph classifier that estimates epistemic and aleatoric uncertainty, calibrates its outputs, explains its predictions and declines findings it cannot resolve. On a patient-disjoint CheXpert test split it reaches a mean AUROC of {f(ov['mean_auc'])}, and its posterior standard deviation is informative: accuracy falls from {pct(qs.iloc[0]['accuracy'])}% to {pct(qs.iloc[3]['accuracy'])}% across its quartiles. The evaluation exposed corrections that we have made: a deferral tolerance that never fired, now chosen on validation data; an evaluation protocol that scored uncertain labels against random targets; and a calibration step that a single temperature cannot complete under Asymmetric Loss. Future work comprises deep ensembles, multi-seed ablations, external validation on ChestX-ray14 and on an Indian cohort, bounding-box evaluation of saliency, and a prospective study of whether deferral reduces reading time without increasing missed findings.")
P(f"Code and reproduction scripts: {GITHUB}. Live demonstration: {DEMO}.")

i0 = next(i for i, l in enumerate(SRC) if l.startswith("# ---------------------------------------------------------------- refs"))
i1 = next(i for i, l in enumerate(SRC) if l.startswith("spec = {"))
take(i0 + 1, i1)

for b_ in blocks:  # table numbering differs from the final layout
    if b_["t"] == "p":
        for r_ in b_["runs"]:
            if isinstance(r_, str):
                pass
        b_["runs"] = [(x.replace("(Table IV)", "(Table III)").replace("Table IV gives", "Table III gives").replace("Table V places", "Table IV places").replace("results in Table IV", "results in Table III") if isinstance(x, str) else x) for x in b_["runs"]]
OUT.write_text(json.dumps({"meta": {"title": "Uncertainty-Aware and Explainable Multi-Label Classification of Chest Radiographs with an Explicit Deferral Policy", "author": "Vansh R. Vijayvargiya", "description": "RadiScan paper, revised in response to reviewer comments (interim)"}, "blocks": blocks}, indent=1))
(OUT.parent / "equations.json").write_text(json.dumps(eqs, indent=1))
print(f"interim spec: {len(blocks)} blocks, {len(eqs)} equations, pending markers: {json.dumps(blocks).count('[pending]')}")
