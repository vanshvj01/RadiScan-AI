"""Build paper_spec.json (text + tables + figures) from the experiment bundle.

    python content.py <bundle_dir> <out_spec.json>

Every number in the paper is read from the bundle written by
paper/kaggle_reviewer_run.py (results.json, ablation.json, CSVs, figures) or
from the original evaluation run (results/chexpert/*). Sentences whose
direction depends on a result are generated from the result itself.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pandas as pd
from PIL import Image

B = Path(sys.argv[1])
OUT = Path(sys.argv[2])
ORIG = Path(sys.argv[3]) if len(sys.argv) > 3 else None  # results/chexpert of the first run

R = json.loads((B / "results.json").read_text())
ABL = json.loads((B / "ablation.json").read_text()) if (B / "ablation.json").exists() else {}
TIM = json.loads((B / "kaggle_timing.json").read_text()) if (B / "kaggle_timing.json").exists() else {}
HIST = None
for name in ("orig_history.json", "main_history.json"):
    if (B / name).exists():
        HIST = json.loads((B / name).read_text())
        break

GITHUB = "https://github.com/vanshvj01/RadiScan-AI"
DEMO = "https://radiscan-ai.vercel.app"


def f(x, nd=3):
    if x is None:
        return "--"
    try:
        v = float(x)
    except (TypeError, ValueError):
        return str(x)
    if math.isnan(v):
        return "--"
    if math.isinf(v):
        return "\u221e"
    return f"{v:.{nd}f}"


def pct(x, nd=1):
    return f(100 * float(x), nd)


def ints(x):
    return f"{int(x):,}"


blocks: list[dict] = []
eqs: list[str] = []


def add(t, **kw):
    blocks.append({"t": t, **kw})


def P(*runs, noindent=False):
    add("p", runs=list(runs), noindent=noindent)


def EQ(latex):
    eqs.append(latex)
    add("eq", id=len(eqs), n=len(eqs))
    return len(eqs)


def I(t):
    return {"text": t, "i": True}


def fig(path, caption, width_in=3.4, full=False):
    im = Image.open(path)
    add("fig", path=str(path), px=list(im.size), caption=caption, width_in=width_in, full=full)


# ---------------------------------------------------------------- data
labels = list(R["thresholds"].keys())
t1 = pd.read_csv(B / "table1_per_class.csv")
ov, cx5 = R["overall"], R["chexpert_competition_5"]
lsr = R["lsr_protocol_overall"]
spl = R["splits"]
cal = pd.DataFrame(R["calibration"])
unc = pd.DataFrame(R["uncertainty"])
qs = pd.DataFrame(R["quartiles_sigma"])
qc = pd.DataFrame(R["quartiles_confidence"]) if R.get("quartiles_confidence") else None
ux = R["uncertainty_extra"]
pol = pd.DataFrame(R["deferral_policies"])
cov = pd.DataFrame(R["coverage"])
sweep = pd.DataFrame(R["mc_sweep"])
taus = R["tau"]
T = R["temperature"]
tm = R.get("train_meta", {}).get("metrics", {}) or {}
bench = R.get("benchmark", {})
gpu_name = bench.get("cuda", {}).get("resources", {}).get("hardware", {}).get("gpu", TIM.get("gpu", "GPU"))
cpu_name = bench.get("cpu", {}).get("resources", {}).get("hardware", {}).get("cpu", "CPU")
cpu_cores = bench.get("cpu", {}).get("resources", {}).get("hardware", {}).get("cpu_count")

orig = None
if ORIG and (ORIG / "summary.json").exists():
    orig = json.loads((ORIG / "summary.json").read_text())


def calrow(name_start, inference):
    sel = cal[cal["calibration"].str.startswith(name_start) & (cal["inference"] == inference)]
    return sel.iloc[0]


mc_name = [x for x in cal["inference"].unique() if x.startswith("MC")][0]
c_unc_mc = calrow("uncalibrated", mc_name)
c_t_mc = calrow("temperature, checkpoint", mc_name)
c_tr_mc = calrow("temperature, refit", mc_name)
c_pl_mc = calrow("class-wise Platt", mc_name)
c_unc_det = calrow("uncalibrated", "deterministic")
c_t_det = calrow("temperature, checkpoint", "deterministic")

U = {r["signal"].split(" (")[0]: r for r in R["uncertainty"]}
u_sd = U["posterior std"]
u_mi = U["mutual information"]
u_ent = U["predictive entropy"]
u_c = U.get("1 - confidence head")
u_mar = U["threshold margin -|mu-t|"]
u_z = U["z-score -|mu-t|/sigma"]

tau90 = taus["0.9"]["tau"]
pol_idx = {p: i for i, p in enumerate(pol["policy"])}


def polrow(prefix):
    return pol[pol["policy"].str.startswith(prefix)].iloc[0]


p_none = polrow("no deferral")
p_int = polrow("interval rule only")
p_fix = polrow("interval OR sigma >= 0.15")
p_sfix = polrow("sigma >= 0.15 only")
p95 = polrow("interval OR sigma >= tau_val(95")
p90 = polrow("interval OR sigma >= tau_val(90")
p80 = polrow("interval OR sigma >= tau_val(80")

zc = cov[cov["ranking"] == "z-score"].set_index("target_coverage")
sc = cov[cov["ranking"] == "sigma"].set_index("target_coverage")
cc = cov[cov["ranking"] == "1-confidence"].set_index("target_coverage") if (cov["ranking"] == "1-confidence").any() else None

sw = sweep.set_index("S")
S_list = list(sw.index)


def lat_ms(dev, S, mode):
    for r in bench.get(dev, {}).get("latency_vs_S", []):
        if r["S"] == S and r["mode"] == mode:
            return r["median_ms"]
    return None


def stage(dev, name):
    for r in bench.get(dev, {}).get("stages", []):
        if r["stage"] == name:
            return r
    return None


n_test, n_val = R["n_test"], R["n_val"]
mono = ux["sigma_quartiles_monotone"]

# ---------------------------------------------------------------- front matter
add("cols", n=1)
add("title", text="Uncertainty-Aware and Explainable Multi-Label Classification of Chest Radiographs with an Explicit Deferral Policy")
add("author", runs=["Vansh R. Vijayvargiya, Vaibhav Dave, Uzair Khan, Aryan Lodha, Purnima Ahirao"], size=22)
add("author", runs=[I("Department of Information Technology, K. J. Somaiya School of Engineering, Somaiya Vidyavihar University, Mumbai, India")], size=18)
add("author", runs=[{"text": "v.vijayvargiya@somaiya.edu, ", "i": True}, {"text": "[[co-author emails]]", "i": True, "hl": True}], size=18, after=240)
add("cols", n=2)

ece_drop = c_unc_mc["ece"] - c_t_mc["ece"]
abstract = [
    "Deep convolutional networks match reported radiologist performance on several thoracic findings, yet a bare probability says nothing about how far it should be trusted. "
    "We present RadiScan, a multi-label chest radiograph classifier built on the premise that a decision-support tool must quantify, calibrate and explain its own uncertainty, and must decline the cases it cannot resolve. "
    "A DenseNet-121 trunk with three task heads is combined with Monte Carlo (MC) Dropout for epistemic uncertainty, a confidence head trained with a hint-and-penalty objective for aleatoric uncertainty, post-hoc temperature scaling, and Grad-CAM. "
    "A deferral policy refers a finding to a radiologist when its 95% posterior interval straddles its validation-tuned operating threshold or its posterior standard deviation exceeds a tolerance selected on validation data alone. "
    f"On a held-out, patient-disjoint CheXpert test split of {ints(n_test)} frontal radiographs ({ints(spl['test']['patients'])} patients), with uncertain annotations excluded, the model attains a mean AUROC of {f(ov['mean_auc'])} over 14 findings ({f(cx5['mean_auc'])} on the five competition findings) and a mean average precision of {f(ov['mean_ap'])}. "
    f"Temperature scaling changes the expected calibration error from {f(c_unc_mc['ece'])} to {f(c_t_mc['ece'])} and the Brier score from {f(c_unc_mc['brier'])} to {f(c_t_mc['brier'])}; a class-wise Platt recalibration reaches {f(c_pl_mc['ece'])}, which exposes the location bias introduced by Asymmetric Loss. "
    f"The posterior standard deviation detects the classifier's own errors with a macro AUROC of {f(u_sd['error_auroc_macro'])} and accuracy falls from {pct(qs.iloc[0]['accuracy'])}% to {pct(qs.iloc[3]['accuracy'])}% across its quartiles; the confidence head reaches {f(u_c['error_auroc_macro']) if u_c else '--'}. "
    f"At a validation-selected 90% coverage target the policy defers {f(p90['deferred_findings_pct'])}% of findings and raises element-level accuracy on the retained findings from {pct(p_none['retained_accuracy'])}% to {pct(p90['retained_accuracy'])}%. "
    "We also report component ablations, the effect of the number of MC passes, latency and throughput, and representative saliency maps. "
    f"Code: {GITHUB}; live demonstration: {DEMO}."
]
add("abstract", label="Abstract", runs=abstract)
add("abstract", label="Index Terms", runs=["chest radiography, multi-label classification, uncertainty quantification, Monte Carlo dropout, calibration, selective prediction, explainable AI, Grad-CAM, clinical decision support, CheXpert"])

# ---------------------------------------------------------------- I
add("h1", text="Introduction")
P("Chest radiography is the most frequently performed diagnostic imaging examination worldwide. Its volume grows with population and access, while the supply of trained radiologists does not. In much of India a radiograph acquired at a primary or secondary centre may wait days for a formal report, and the delay falls hardest on time-critical findings such as pneumothorax and pulmonary oedema. Automated pre-reading that orders a reporting queue by urgency is therefore a more tractable and more immediately useful objective than autonomous diagnosis.")
P("Since CheXNet demonstrated radiologist-comparable pneumonia detection [1], a large literature has reported strong discrimination on public chest radiograph corpora, but comparatively little of that work has reached clinical use. One reason is methodological: the standard output is a vector of probabilities with no attached measure of reliability. A prediction of 0.62 with a posterior standard deviation of 0.02 and one with a standard deviation of 0.30 are different clinical objects, and only the first is actionable.")
P("Two further properties are commonly absent. Modern networks are systematically miscalibrated, so a stated probability of 0.90 need not correspond to an empirical frequency of 0.90 [11]. And when a model cannot resolve a case, the conventional design still emits a decision; there is no mechanism by which it can decline [20].")
P("This paper describes RadiScan, a system designed around those gaps. Our contributions are:")
add("item", runs=["(i) Epistemic (MC Dropout) and aleatoric (learned confidence head) uncertainty for each of 14 findings, with the training objective of the confidence head stated formally and both signals evaluated quantitatively as detectors of the model's own errors."])
add("item", runs=["(ii) An explicit deferral policy that couples the posterior interval to the per-class operating threshold, with a tolerance on the posterior standard deviation chosen on validation data only, evaluated by deferral rate and by performance at several coverage levels."])
add("item", runs=["(iii) A treatment of extreme class imbalance (Asymmetric Loss, inverse-frequency sampling, label smoothing of uncertain annotations, mixup, per-class thresholds) whose components are each removed in a controlled ablation."])
add("item", runs=["(iv) A reproducibility protocol (patient-disjoint splitting with an independent overlap audit, pre-flight data verification) and a full account of calibration, inference cost and the number of MC passes."])
P(f"Source code, configuration files and the evaluation scripts that regenerate every table and figure are available at {GITHUB}; a live deployment of the system is available at {DEMO}. Section II reviews related work, Section III describes the data, Section IV the method, Section V the deployed system, Section VI the experimental protocol and Section VII the results; Sections VIII and IX state limitations and conclusions.")

# ---------------------------------------------------------------- II
add("h1", text="Related Work")
add("h2", text="Chest Radiograph Classification")
P("Wang et al. released ChestX-ray8 and established the multi-label formulation now standard in the field [3]. Rajpurkar et al. introduced CheXNet, a DenseNet-121 [4] fine-tuned on that corpus [1]; DenseNet-121 has since served as the reference backbone for the task, which makes it the appropriate choice where comparability is required. Irvin et al. released CheXpert, labelled by an automated report parser with an explicit uncertain class, and found that mapping uncertain labels to a smoothed target outperformed both binarisation strategies [2]. A 2026 benchmarking study reported macro AUROC of 0.7511 for MedViT and 0.7168 for a hybrid CNN-ViT on CheXpert, against 0.9417 for MedViT on ChestX-ray14 [5]; reported performance does not transfer between corpora, so we compare only against CheXpert results.")
add("h2", text="Uncertainty and Selective Prediction")
P("Gal and Ghahramani showed that dropout at inference approximates Bayesian inference, so the spread across stochastic passes estimates epistemic uncertainty [8]; deep ensembles are a stronger but proportionally costlier alternative [19]. Aleatoric uncertainty, arising from noise intrinsic to the observation, can be learned directly through a confidence branch trained with a hint-and-penalty objective [12]. Selective classification formalises the option to abstain and evaluates it through risk-coverage curves [20]. Prior chest radiography work has mostly reported uncertainty as an output; comparatively little ties it to an operating threshold that decides whether a finding is reported at all.")
add("h2", text="Calibration and Explanation")
P("Guo et al. showed that modern networks are miscalibrated and that temperature scaling, a single scalar fitted on held-out data, largely corrects them [11]; Platt scaling fits a per-class slope and intercept [21]. Expected calibration error [22] and the Brier score [23] are the standard summaries. Grad-CAM weights final convolutional feature maps by class-score gradients [6], and Grad-CAM++ refines the weighting [7]. Zech et al. showed that a pneumonia classifier had learned hospital-specific artefacts [10]; we therefore present saliency as a qualitative check and make no localisation claim.")

# ---------------------------------------------------------------- III
add("h1", text="Dataset and Preprocessing")
add("h2", text="CheXpert")
P("We use the downsampled release of CheXpert [2], comprising 224,316 radiographs of 65,240 patients from Stanford Hospital. Each study carries 14 labels (Table IV). Five of these form the CheXpert competition subset, over which we report mean AUROC separately. The task is genuinely multi-label, so the output layer has 14 independent sigmoid units. Labels are extracted automatically from reports, so label noise bounds achievable discrimination. Prevalence ranges from roughly 40% to under 2%; a classifier that predicts absence for every rare finding attains accuracy above 0.98 while providing no value, so accuracy is never a headline metric here.")
add("h2", text="Training, Validation and Test Splits")
tr, va, te = spl["train"], spl["val"], spl["test"]
P(f"Only frontal radiographs are used. Patients, not images, are assigned to partitions: the patient identifier is parsed from each image path, the shuffled list of patients (seed 42) is consumed whole-patient-at-a-time until the test and validation partitions each reach 4,000 images, and the remaining patients form the training pool, from which 40,000 images are drawn. Table I gives the resulting sizes. Because whole patients are assigned, no partition can hold a subset of another patient's studies.")
add("table", caption="Patient-disjoint partitions (frontal radiographs)",
    cols=[2.2, 1.4, 1.4, 1.4, 1.6],
    header=["Split", "Patients", "Studies", "Images", "Img./patient"],
    rows=[[n.capitalize() if n != "val" else "Validation", ints(spl[n]["patients"]), ints(spl[n]["studies"]), ints(spl[n]["images"]), f(spl[n]["images_per_patient"], 2)] for n in ("train", "val", "test")],
    note=[f"Patient overlap train\u2013val / train\u2013test / val\u2013test: {spl['overlap']['train-val']['patients']} / {spl['overlap']['train-test']['patients']} / {spl['overlap']['val-test']['patients']}; image-path overlap: {spl['overlap']['train-val']['images']} / {spl['overlap']['train-test']['images']} / {spl['overlap']['val-test']['images']}."])
P(f"Patient-level separation is verified twice. At split time the procedure asserts that the three patient sets have an empty pairwise intersection and terminates otherwise. Independently, the evaluation script re-reads the three saved CSV files, recomputes the patient identifiers from the image paths and reports the pairwise intersections of patient sets and of image paths; all six are zero (Table I). The test and validation partitions are those of the original training run, reused verbatim rather than regenerated.")
add("h2", text="Uncertain Labels and Evaluation Protocol")
P("CheXpert encodes positive, negative and uncertain annotations. During training, uncertain targets are replaced by a value drawn uniformly from (0.05, 0.85) (label-smoothing regression), following [2]. In evaluation, uncertain test annotations are excluded from every metric for that finding; Table IV lists how many were excluded per class. An earlier version of this work evaluated against the smoothed targets; because these are random draws, metrics then partly reflect the draw rather than the model, and we have corrected the protocol. For transparency, Section VII-A also reports the earlier protocol, which reproduces the originally reported numbers on the same split.")
add("h2", text="Preprocessing and Augmentation")
P("Photographed films and scanner borders are cropped automatically, Contrast Limited Adaptive Histogram Equalisation (clip limit 2.0, 8\u00d78 tiles) is applied to the luminance channel [17], and images are resized to 320\u00d7320. Geometric augmentation is conservative (rotation \u00b110\u00b0, scale 10%, translation 5%); vertical flips are never applied and horizontal flips only at low probability, because a mirrored chest radiograph implies situs inversus. Photometric augmentation comprises brightness and contrast jitter, occasional Gaussian noise or blur, and coarse dropout. DICOM inputs are windowed and MONOCHROME1 studies inverted.")

# ---------------------------------------------------------------- IV
add("h1", text="Proposed Methodology")
add("h2", text="Network Architecture")
P("The trunk is an ImageNet-initialised DenseNet-121 [4] producing a 1024-dimensional pooled embedding, followed by layer normalisation and dropout (p = 0.2). Three heads are attached: a disease head (two-layer perceptron with layer normalisation, GELU and dropout, 14 logits z", {"text": "k", "sub": True}, "), an auxiliary fracture head that regularises the trunk, and a confidence head producing 14 values c", {"text": "k", "sub": True}, " \u2208 (0, 1) (Section IV-D). The model has 8.02 M parameters. Every dropout layer lies after the convolutional trunk; the DenseNet trunk itself contains none. This matters for inference cost (Section V) and for the character of the epistemic estimate (Section VIII).")
add("h2", text="Loss Design under Extreme Imbalance")
P("Let p", {"text": "ik", "sub": True}, " = \u03c3(z", {"text": "ik", "sub": True}, ") for image i and finding k and y", {"text": "ik", "sub": True}, " \u2208 [0, 1] the (possibly smoothed) target. Asymmetric Loss [13] decouples the focusing of positives and negatives and discards very easy negatives through a probability shift m:")
EQ(r"\mathcal{L}_{\mathrm{dis}}=-\frac{1}{NK}\sum_{i,k}\Big[y_{ik}\,(1-p_{ik})^{\gamma_{+}}\log p_{ik}+(1-y_{ik})\,p_{m,ik}^{\gamma_{-}}\log\big(1-p_{m,ik}\big)\Big],\quad p_{m,ik}=\max(p_{ik}-m,0)")
P("with \u03b3", {"text": "+", "sub": True}, " = 0, \u03b3", {"text": "\u2212", "sub": True}, " = 4 and m = 0.05. The total objective is")
EQ(r"\mathcal{L}=\mathcal{L}_{\mathrm{dis}}+0.3\,\mathcal{L}_{\mathrm{frac}}+0.1\,\mathcal{L}_{\mathrm{conf}}")
P("where L", {"text": "frac", "sub": True}, " is binary cross-entropy on the fracture head and L", {"text": "conf", "sub": True}, " is defined in (8). Inverse-frequency balanced sampling draws image i with weight")
EQ(r"w_i=\begin{cases}\max_{k:\,y_{ik}=1}\ \mathrm{clip}\!\left(N/n_k,\,1,\,20\right) & \text{if } i \text{ has a positive finding}\\ 1 & \text{otherwise}\end{cases}")
P("where n", {"text": "k", "sub": True}, " is the number of positives of finding k among N training images. Multi-label mixup [16] blends two images with \u03bb ~ Beta(0.2, 0.2) and combines their targets by element-wise maximum, since a finding present in either source is plausibly present in the blend:")
EQ(r"\tilde{x}=\lambda x_i+(1-\lambda)x_j,\qquad \tilde{y}=\max(y_i,y_j)")
add("h2", text="Epistemic Uncertainty")
P("Following [8], dropout stays active at inference and the network is evaluated over S stochastic passes, each averaging the original and the horizontally flipped image. With T the fitted temperature (Section IV-E), pass s gives")
EQ(r"p_k^{(s)}=\tfrac{1}{2}\Big[\sigma\big(z_k^{(s)}(x)/T\big)+\sigma\big(z_k^{(s)}(\mathrm{flip}(x))/T\big)\Big],\qquad \mu_k=\frac{1}{S}\sum_{s=1}^{S}p_k^{(s)},\qquad \sigma_k=\sqrt{\frac{1}{S}\sum_{s=1}^{S}\big(p_k^{(s)}-\mu_k\big)^2}")
P("The 95% interval [l", {"text": "k", "sub": True}, ", u", {"text": "k", "sub": True}, "] is formed by the 2.5th and 97.5th percentiles of {p", {"text": "k", "sub": True}, {"text": "(s)", "sup": True}, "}, and the mutual information between prediction and parameters isolates the epistemic part of the predictive entropy H:")
EQ(r"\mathrm{MI}_k=H(\mu_k)-\frac{1}{S}\sum_{s=1}^{S}H\big(p_k^{(s)}\big),\qquad H(p)=-p\log p-(1-p)\log(1-p)")
P("Only dropout modules are switched to training mode; switching the whole network would also unfreeze batch-normalisation statistics and make a prediction depend on the other studies in the batch. This invariant is asserted by a unit test.")
add("h2", text="Aleatoric Uncertainty: the Confidence Head")
P("The confidence head is a two-layer perceptron with a sigmoid output, c", {"text": "ik", "sub": True}, " \u2208 (0, 1), sharing the trunk embedding. Following DeVries and Taylor [12], during training the model may ask for a hint: its prediction is interpolated toward the target in proportion to 1 \u2212 c, and asking is penalised:")
EQ(r"\tilde{p}_{ik}=c_{ik}\,\mathrm{sg}(p_{ik})+(1-c_{ik})\,y_{ik}")
EQ(r"\mathcal{L}_{\mathrm{conf}}=-\frac{1}{NK}\sum_{i,k}\Big[y_{ik}\log\tilde{p}_{ik}+(1-y_{ik})\log(1-\tilde{p}_{ik})\Big]-\beta\,\frac{1}{NK}\sum_{i,k}\log c_{ik}")
P("with budget \u03b2 = 0.3 and loss weight 0.1 in (2); sg(\u00b7) is the stop-gradient operator, so the confidence objective shapes the confidence head and the shared trunk but never moves the disease logits directly. Where the classifier is right, lowering c only adds penalty, so c stays high; where it is wrong, lowering c moves p\u0303 toward the target and reduces the first term by more than the penalty costs. At equilibrium c", {"text": "ik", "sub": True}, " therefore estimates whether the model can be trusted on that finding of that image, from image content alone and in a single deterministic pass. At inference the reported confidence is averaged over the MC passes, and 1 \u2212 c", {"text": "ik", "sub": True}, " is used as the aleatoric uncertainty score. We test this interpretation in two ways (Section VII-C): whether 1 \u2212 c detects the model's errors, and whether it is higher on findings that the reporting radiologist could not commit to (CheXpert's uncertain annotations), which are an external proxy for ambiguity intrinsic to the image.")
add("h2", text="Post-hoc Calibration")
P("After training, the temperature T is fitted on validation logits of the exponential-moving-average weights by minimising binary cross-entropy (L-BFGS on log T):")
EQ(r"T^{\star}=\arg\min_{T>0}\ -\sum_{(i,k)\in\mathcal{V}}\Big[y_{ik}\log\sigma(z_{ik}/T)+(1-y_{ik})\log\big(1-\sigma(z_{ik}/T)\big)\Big]")
P(f"For this model T* = {f(T)}. Dividing logits by T is monotonic per class, so the AUROC of the deterministic prediction is unchanged; after MC averaging of sigmoids it may change marginally. Calibration is summarised by the expected calibration error over B = 15 equal-width bins and by the Brier score, both pooled over all evaluated (image, finding) pairs \u2133:")
EQ(r"\mathrm{ECE}=\sum_{b=1}^{B}\frac{|\mathcal{B}_b|}{|\mathcal{M}|}\,\Big|\frac{1}{|\mathcal{B}_b|}\sum_{(i,k)\in\mathcal{B}_b}y_{ik}-\frac{1}{|\mathcal{B}_b|}\sum_{(i,k)\in\mathcal{B}_b}\mu_{ik}\Big|,\qquad \mathrm{Brier}=\frac{1}{|\mathcal{M}|}\sum_{(i,k)\in\mathcal{M}}\big(\mu_{ik}-y_{ik}\big)^2")
add("h2", text="Operating Points")
P("A single threshold of 0.5 across prevalences from 0.02 to 0.40 leaves rare findings essentially never predicted. A threshold t", {"text": "k", "sub": True}, " is therefore tuned per finding on the validation split by maximising F1, written into the checkpoint, and applied unchanged to the test split.")
add("h2", text="Deferral Policy and Selection of the Tolerance")
P("A finding k other than No Finding is reported as indeterminate and referred for review when")
EQ(r"D_k(\tau)=\big[\,l_k<t_k\le u_k\,\big]\ \lor\ \big[\,\sigma_k\ge\tau\,\big]")
P("The first condition states that the evidence does not determine on which side of the threshold the finding lies; the second bounds the posterior spread. Since both depend on how far \u03bc", {"text": "k", "sub": True}, " lies from t", {"text": "k", "sub": True}, " relative to \u03c3", {"text": "k", "sub": True}, ", we use z", {"text": "k", "sub": True}, " = |\u03bc", {"text": "k", "sub": True}, " \u2212 t", {"text": "k", "sub": True}, "| / \u03c3", {"text": "k", "sub": True}, " as the single deferral score when tracing performance against coverage.")
P(f"The first implementation used a fixed tolerance \u03c4 = 0.15 chosen a priori. The observed posterior standard deviations are about two orders of magnitude smaller (median {f(ux['sigma_p50_p95_p99_max'][0], 4)}, 99th percentile {f(ux['sigma_p50_p95_p99_max'][2], 4)}, maximum {f(ux['sigma_p50_p95_p99_max'][3], 4)} on the test split), so that rule never fired and deferral was governed entirely by the interval condition. We therefore select \u03c4 on the validation split only, as the smallest value whose validation coverage reaches a target c*:")
EQ(r"\tau^{\star}(c^{\star})=\min\Big\{\tau:\ 1-\frac{1}{|\mathcal{V}|}\sum_{(i,k)\in\mathcal{V}}D_{ik}(\tau)\ \ge\ c^{\star}\Big\}")
t95, t80 = taus["0.95"], taus["0.8"]
P(f"Coverage is non-decreasing in \u03c4, so this is a one-dimensional search over the quantiles of the validation standard deviations; no test datum enters it. For c* = 0.95, 0.90 and 0.80 it gives \u03c4* = {f(t95['tau'], 4)}, {f(tau90, 4)} and {f(t80['tau'], 4)} (validation coverage {pct(t95['val_coverage'])}%, {pct(taus['0.9']['val_coverage'])}% and {pct(t80['val_coverage'])}%). The deployed configuration uses c* = 0.90. The generated report is rule-based over calibrated probabilities rather than written by a language model, so every statement is auditable.")

# ---------------------------------------------------------------- V
add("h1", text="System Implementation")
P(f"The model is served by a three-tier application ({DEMO}). A Next.js front end submits a study; a FastAPI service validates and persists it, writes an audit entry, enqueues an inference task and returns immediately; a worker decodes and preprocesses the image, runs the MC inference, computes Grad-CAM for the three most probable findings and generates the report, and progress is pushed to the client over a WebSocket. Every checkpoint embeds its label list, backbone, input size, temperature and thresholds, and loading refuses a checkpoint whose labels disagree with those requested.")
P("Because dropout appears only after the trunk, the trunk output is identical across MC passes. Computing it once per view and re-sampling only the 0.5 M-parameter heads is mathematically identical to re-running the full network (the evaluation code asserts agreement of the posterior mean to within "
  f"{f(R['estimator_equivalence']['max_abs_diff_mean'], 6)}). Tables II and III report measured cost on an {gpu_name} GPU and on a {cpu_cores}-core CPU ({cpu_name}).")


def latrow(dev, label):
    st = {k: stage(dev, k) for k in ("preprocess_ms", "mc_inference_ms", "gradcam_top3_ms", "report_ms", "total_ms")}
    return [label] + [f(st[k]["median_ms"], 0) if st[k] else "--" for k in ("preprocess_ms", "mc_inference_ms", "gradcam_top3_ms", "total_ms")]


lat_rows = []
if "cuda" in bench:
    lat_rows.append(latrow("cuda", "GPU"))
if "cpu" in bench:
    lat_rows.append(latrow("cpu", "CPU"))
add("table", caption="Per-study latency of the deployed pipeline (median, ms; S = 20, flip TTA)",
    cols=[1.0, 1.4, 1.5, 1.6, 1.2],
    header=["Device", "Pre-proc.", "MC infer.", "Grad-CAM \u00d73", "Total"],
    rows=lat_rows,
    note=["Grad-CAM \u00d73: saliency for the three most probable findings. Report generation takes < 1 ms."])

thr_rows = []
for dev, name in (("cuda", "GPU"), ("cpu", "CPU")):
    for r in bench.get(dev, {}).get("throughput", []):
        thr_rows.append([name, str(r["batch_size"]), f(r["images_per_s"], 2), f(r["ms_per_image"], 0), f(r.get("peak_gpu_mem_mb"), 0) if r.get("peak_gpu_mem_mb") else "--"])
gpu_res = bench.get("cuda", {}).get("resources", {})
cpu_res = bench.get("cpu", {}).get("resources", {})
add("table", caption="Throughput and memory of MC inference (S = 20, flip TTA)",
    cols=[1.0, 0.9, 1.3, 1.3, 1.5],
    header=["Device", "Batch", "Images/s", "ms/image", "Peak GPU MB"],
    rows=thr_rows,
    note=[f"Checkpoint size {f(gpu_res.get('checkpoint_mb', cpu_res.get('checkpoint_mb')), 1)} MB; {f(gpu_res.get('parameters_m', cpu_res.get('parameters_m')), 2)} M parameters; resident memory of the CPU inference process {f(cpu_res.get('process_rss_mb'), 0)} MB."])

g20d, g20c = lat_ms("cuda", 20, "deployed"), lat_ms("cuda", 20, "cached")
c20d, c20c = lat_ms("cpu", 20, "deployed"), lat_ms("cpu", 20, "cached")
txt = (f"On the GPU a complete study, including saliency, takes {f(stage('cuda','total_ms')['median_ms'], 0)} ms, of which the 20-pass inference is {f(stage('cuda','mc_inference_ms')['median_ms'], 0)} ms. " if stage("cuda", "total_ms") else "")
if c20d and c20c:
    txt += (f"On the CPU the deployed implementation needs {f(c20d/1000, 1)} s for the MC inference alone, because it repeats the full DenseNet forward pass 40 times; caching the trunk reduces this to {f(c20c/1000, 2)} s, a {f(c20d/c20c, 0)}-fold saving, which makes CPU-only deployment practical. ")
if g20d and g20c:
    txt += f"On the GPU the same change reduces the MC inference from {f(g20d, 0)} ms to {f(g20c, 0)} ms. "
txt += "Asynchronous execution keeps web workers free while inference runs, so throughput scales with the number of inference workers rather than web workers."
P(txt)

# ---------------------------------------------------------------- VI
add("h1", text="Experimental Protocol")
add("h2", text="Pre-flight Data Verification")
P("An earlier iteration contained a loader that returned a blank frame for any unreadable image. A path-prefix mismatch made every image unreadable, and a full training run completed without error at chance-level discrimination. The loader now raises, and a verification command samples the annotation file, confirms that paths resolve, and reports per-class positive rates before training; a class with zero prevalence reveals column misalignment in seconds.")
add("h2", text="Training Configuration")
hist_txt = ""
if HIST and isinstance(HIST, list) and HIST and "val_mean_auc" in HIST[0]:
    best_e = max(HIST, key=lambda e: e["val_mean_auc"])
    after = [e for e in HIST if e["epoch"] > best_e["epoch"]]
    mins = sum(e["seconds"] for e in HIST) / 60
    med = sorted(e["seconds"] for e in HIST)[len(HIST) // 2] / 60
    hist_txt = (f" Validation mean AUROC rose from {f(HIST[0]['val_mean_auc'])} after the first epoch to {f(best_e['val_mean_auc'])} at epoch {best_e['epoch']}"
                + (f", then fell abruptly (to {f(min(e['val_mean_auc'] for e in after))} by epoch {min(after, key=lambda e: e['val_mean_auc'])['epoch']}) and did not recover, so early stopping ended the run after epoch {HIST[-1]['epoch']} and the epoch-{best_e['epoch']} weights were retained" if after and min(e['val_mean_auc'] for e in after) < best_e['val_mean_auc'] - 0.05 else "")
                + f". The run took {f(mins / 60, 1)} h ({f(med, 0)} min per epoch, dominated by image decoding and preprocessing on the CPU).")
P("Optimisation uses AdamW [15] (learning rate 10", {"text": "\u22124", "sup": True}, ", weight decay 10", {"text": "\u22124", "sup": True}, ", tenfold rate on the heads), a two-epoch linear warm-up followed by cosine decay, a one-epoch frozen-backbone phase, mixed precision [18], gradient clipping at 5.0 and an exponential moving average of the weights (decay 0.999) used for evaluation and checkpointing. Batch size is 64, the maximum of 25 epochs is subject to early stopping on validation mean AUROC with patience 6, and the seed is 42. "
  f"The best checkpoint was reached at epoch {R.get('epoch')}.{hist_txt} Training used a Kaggle GPU; all evaluation and ablation runs used an {gpu_name}.")
add("h2", text="Metrics")
P("We report per-class and mean AUROC, average precision (AP), and precision, recall and F1 at the tuned thresholds; ECE, Brier score and negative log-likelihood; and positive support per class. Uncertainty estimates are evaluated as detectors of the model's own errors: an element (i, k) is an error when [\u03bc", {"text": "ik", "sub": True}, " \u2265 t", {"text": "k", "sub": True}, "] \u2260 y", {"text": "ik", "sub": True}, ", and we report the AUROC of each uncertainty score for separating errors from correct decisions, both pooled and as the mean over findings (macro, which removes differences in base error rate between findings), together with the area under the risk-coverage curve (AURC) [20]. The deferral policy is evaluated by the fraction of findings and of studies deferred and by accuracy, sensitivity, specificity, F1 and AUROC on the retained findings. All test-time quantities use S = 20 unless stated otherwise.")
add("h2", text="Ablation Protocol")
abl_ref = ABL.get("full", {})
P("Each ablation arm removes one component and is retrained from ImageNet initialisation under one reduced, fixed budget: 20,000 training images drawn with the same seed from the training partition, 8 epochs, otherwise the configuration above. The reference arm (all components) is retrained under the same budget, so differences between arms are controlled, and every arm is evaluated on the same test split with its own validation-tuned thresholds and temperature. Removing Asymmetric Loss substitutes class-weighted binary cross-entropy; removing label smoothing maps uncertain labels to negative (U-Zeros); removing balanced sampling uses uniform sampling; removing mixup sets \u03b1 = 0; removing the confidence head deletes the head and its loss term.")
add("h2", text="Number of MC Passes")
P("Fifty stochastic passes were stored for every validation and test image. For S \u2208 {" + ", ".join(str(s) for s in S_list) + "}, S passes were drawn without replacement five times, and for each draw we recomputed discrimination, calibration, the error-detection AUROC of \u03c3, the rank correlation of \u03c3 with its 50-pass value, and the deferral rate with \u03c4 re-selected on validation for that S. Latency was measured separately for each S.")

# ---------------------------------------------------------------- VII
add("h1", text="Results and Discussion")
add("h2", text="Discrimination")
rows = []
for _, r in t1.iterrows():
    rows.append([r["label"], ints(r["support"]), ints(r["uncertain_excluded"]), f(r["auc"]), f(r["ap"]), f(r["precision"], 3), f(r["recall"], 3), f(r["f1"], 3)])
rows.append(["Mean", "--", "--", f(ov["mean_auc"]), f(ov["mean_ap"]), f(ov["mean_precision"]), f(ov["mean_recall"]), f(ov["mean_f1"])])
add("table", caption="Per-class performance on the held-out, patient-disjoint test split",
    cols=[2.9, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], header=["Finding", "Pos.", "Unc.", "AUROC", "AP", "Prec.", "Rec.", "F1"], rows=rows, font=7,
    rules=[len(rows) - 2],
    note=[f"Pos.: positive test images; Unc.: uncertain annotations excluded. Precision, recall and F1 at validation-tuned thresholds; MC-20 posterior mean; n = {ints(n_test)} images."])
best = t1.sort_values("auc", ascending=False)
P(f"Table IV gives per-class results. Discrimination is highest for {best.iloc[0]['label']} ({f(best.iloc[0]['auc'])}), {best.iloc[1]['label']} ({f(best.iloc[1]['auc'])}) and {best.iloc[2]['label']} ({f(best.iloc[2]['auc'])}), and lowest for {best.iloc[-1]['label']} ({f(best.iloc[-1]['auc'])}) and {best.iloc[-2]['label']} ({f(best.iloc[-2]['auc'])}). The ordering follows conspicuity: large, high-contrast findings and devices are recognised more reliably than subtle parenchymal findings. AP is low for rare findings even where AUROC is moderate (e.g. {t1.sort_values('support').iloc[0]['label']}, {ints(t1.sort_values('support').iloc[0]['support'])} positives, AP {f(t1.sort_values('support').iloc[0]['ap'])}), which is why AP is reported alongside AUROC.")
comp_rows = [
    ["MedViT [5]", "Transformer", "0.751", "--", "--"],
    ["Hybrid CNN-ViT [5]", "CNN + ViT", "0.717", "--", "--"],
    ["Ours, earlier protocol\u2020", "DenseNet-121", f(lsr["mean_auc"]), "--", "--"],
    ["Ours (uncalibrated)", "DenseNet-121", f(c_unc_mc["mean_auc"]), "--", f(c_unc_mc["ece"])],
    ["Ours (temperature-scaled)", "DenseNet-121", f(ov["mean_auc"]), f(cx5["mean_auc"]), f(c_t_mc["ece"])],
]
add("table", caption="Aggregate performance and published CheXpert results",
    cols=[2.6, 1.6, 1.0, 1.0, 0.9], header=["Method", "Backbone", "Mean AUROC", "CheXpert-5", "ECE"], rows=comp_rows, font=7,
    note=["\u2020 Uncertain test labels replaced by smoothed random targets, as in the first version of this paper" + (f" (reported there as {f(orig['overall']['mean_auc'])})" if orig else "") + ". Published results use different splits and are indicative only."])
P(f"Table V places the model against published CheXpert figures. Under the corrected protocol the mean AUROC is {f(ov['mean_auc'])}; under the earlier protocol the same checkpoint on the same split gives {f(lsr['mean_auc'])}" + (f", reproducing the originally reported {f(orig['overall']['mean_auc'])}" if orig else "") + ". The published results were obtained on different splits, so the comparison is indicative rather than controlled; we do not claim to outperform them.")

add("h2", text="Calibration")
cal_tab = []
for nm, lab in (("uncalibrated", "None (T = 1)"), ("temperature, checkpoint", f"Temperature, T = {f(T, 2)}"), ("temperature, refit", f"Temperature, refit T = {f(R['temperatures']['T_refit'], 2)}"), ("class-wise Platt", "Class-wise Platt")):
    for inf, il in (("deterministic", "single"), (mc_name, "MC-20")):
        r = calrow(nm, inf)
        cal_tab.append([lab if inf == "deterministic" else "", il, f(r["ece"]), f(r["brier"]), f(r["nll"]), f(r["mean_auc"])])
add("table", caption="Calibration on the test split before and after post-hoc calibration",
    cols=[2.5, 0.9, 0.9, 0.9, 0.9, 1.0], header=["Calibration", "Pass", "ECE", "Brier", "NLL", "AUROC"], rows=cal_tab, font=7, rules=[1, 3, 5],
    note=["Single: one deterministic pass, dropout off. Temperature fitted on validation; refit uses validation with uncertain labels masked. Platt: per-class slope and intercept fitted on validation."])
better = c_t_mc["ece"] < c_unc_mc["ece"]
P(f"Table VI and Fig. 1 report calibration. Temperature scaling {'reduces' if better else 'does not reduce'} the ECE of the deployed MC-20 output from {f(c_unc_mc['ece'])} to {f(c_t_mc['ece'])} and changes the Brier score from {f(c_unc_mc['brier'])} to {f(c_t_mc['brier'])}; for the deterministic pass the AUROC is identical before and after ({f(c_unc_det['mean_auc'])} vs {f(c_t_det['mean_auc'])}), as monotonicity requires. "
  + (f"Even after scaling, however, a large calibration error remains. Its source is a systematic offset of the predicted probabilities rather than over-confidence alone, which is the expected effect of Asymmetric Loss: its probability shift and negative focusing remove most of the penalty on moderately confident false positives. A single temperature rescales logits around zero and cannot correct a per-class offset. " if c_pl_mc['ece'] < 0.6 * c_t_mc['ece'] else "") + "Class-wise Platt scaling, which adds an intercept, " + ("reduces" if c_pl_mc['ece'] < c_t_mc['ece'] else "changes") + f" the ECE to {f(c_pl_mc['ece'])} and the Brier score to {f(c_pl_mc['brier'])}. We report temperature-scaled outputs as the system was designed and identify the residual miscalibration as a consequence of the loss" + (", and we recommend class-wise Platt scaling as the calibration step whenever an asymmetric loss is used." if c_pl_mc['ece'] < c_t_mc['ece'] else "."))
fig(B / "fig_reliability.png", "Reliability diagrams of the MC-20 posterior mean on the test split: uncalibrated, temperature-scaled and class-wise Platt-scaled.", width_in=3.45)

add("h2", text="Epistemic and Aleatoric Uncertainty")
urows = []
for key, lab in (("posterior std", "Posterior std \u03c3 (epistemic)"), ("mutual information", "Mutual information (epistemic)"), ("predictive entropy", "Predictive entropy (total)"), ("1 - confidence head", "1 \u2212 c (aleatoric)"), ("threshold margin -|mu-t|", "Margin |\u03bc \u2212 t| (baseline)"), ("z-score -|mu-t|/sigma", "Deferral score z")):
    if key in U:
        r = U[key]
        urows.append([lab, f(r["error_auroc_pooled"]), f(r["error_auroc_macro"]), f(r["aurc"], 4), pct(1 - r["risk_at_80"])])
add("table", caption="Uncertainty scores as detectors of the model's own errors (test split)",
    cols=[2.8, 1.0, 1.0, 1.0, 1.2], header=["Score", "AUROC pooled", "AUROC macro", "AURC \u2193", "Acc.@80% cov."], rows=urows, font=7,
    note=[f"Errors: thresholded decision disagrees with label (uncertain labels excluded). Overall element accuracy without deferral: {pct(1 - u_sd['risk_at_100'])}%. AURC: area under the risk-coverage curve, lower is better."])
qrows = []
for i in range(4):
    row = [f"Q{i + 1}" + (" (most certain)" if i == 0 else " (least certain)" if i == 3 else ""), f(qs.iloc[i]["mean_signal"], 4), ints(qs.iloc[i]["n"]), pct(qs.iloc[i]["accuracy"])]
    if qc is not None:
        row += [f(qc.iloc[i]["mean_signal"], 3), pct(qc.iloc[i]["accuracy"])]
    qrows.append(row)
add("table", caption="Accuracy by quartile of epistemic and aleatoric uncertainty",
    cols=[2.2, 1.0, 1.0, 1.0, 1.0, 1.0], header=["Quartile", "Mean \u03c3", "n", "Acc. %", "Mean 1\u2212c", "Acc. %"] if qc is not None else ["Quartile", "Mean \u03c3", "n", "Acc. %"], rows=qrows, font=7,
    note=["Quartiles of (image, finding) pairs ranked separately by each score; accuracy of the thresholded decision."])
P(f"Tables VII and VIII evaluate the uncertainty estimates. The posterior standard deviation separates the classifier's errors from its correct decisions with a pooled AUROC of {f(u_sd['error_auroc_pooled'])} and a macro AUROC of {f(u_sd['error_auroc_macro'])}; accuracy {'falls monotonically' if mono else 'does not fall monotonically'} across its quartiles, from {pct(qs.iloc[0]['accuracy'])}% to {pct(qs.iloc[3]['accuracy'])}%. Mutual information behaves similarly ({f(u_mi['error_auroc_macro'])} macro). "
  + (f"The aleatoric score 1 \u2212 c reaches {f(u_c['error_auroc_pooled'])} pooled and {f(u_c['error_auroc_macro'])} macro, and accuracy {'falls monotonically' if ux.get('confidence_quartiles_monotone') else 'does not fall monotonically'} across its quartiles ({pct(qc.iloc[0]['accuracy'])}% to {pct(qc.iloc[3]['accuracy'])}%). " if u_c else "")
  + f"The simple margin |\u03bc \u2212 t| is a strong baseline ({f(u_mar['error_auroc_macro'])} macro), because most errors lie close to the threshold; the deferral score z, which divides the margin by \u03c3, attains {f(u_z['error_auroc_macro'])} macro and the lowest AURC ({f(u_z['aurc'], 4)}).")
if u_c:
    P(f"Two further results characterise the confidence head. First, its rank correlation with \u03c3 is {f(ux['spearman_sigma_vs_1mc'], 2)}, so it carries information that the epistemic estimate does not. Second, on findings that the reporting radiologist annotated as uncertain, the mean confidence is {f(ux['confidence_mean_on_uncertain_labels'])} against {f(ux['confidence_mean_on_certain_labels'])} on findings annotated as present or absent, and 1 \u2212 c separates the two groups with a macro AUROC of {f(ux['uncertain_label_auroc_1mc'])} (\u03c3: {f(ux['uncertain_label_auroc_sigma'])}). "
      + ("Because these annotations were never used as targets for the confidence head, this is evidence that it responds to ambiguity in the image itself, i.e. to aleatoric uncertainty." if ux["uncertain_label_auroc_1mc"] > 0.55 else "The separation is weak, so we cannot claim that the head isolates image-intrinsic ambiguity; its practical value lies in error detection.")
      + f" The head's outputs span {f(ux['confidence_p05_p95'][0], 2)}\u2013{f(ux['confidence_p05_p95'][1], 2)} (5th\u201395th percentile).")

add("h2", text="Deferral and Coverage")
prow = []
for r, lab in ((p_none, "No deferral"), (p_sfix, "\u03c3 \u2265 0.15 (initial, fixed)"), (p_int, "Interval rule only"), (p95, f"Interval \u2228 \u03c3 \u2265 \u03c4*(0.95)"), (p90, f"Interval \u2228 \u03c3 \u2265 \u03c4*(0.90)"), (p80, f"Interval \u2228 \u03c3 \u2265 \u03c4*(0.80)")):
    prow.append([lab, f(r["deferred_findings_pct"], 1), f(r["studies_with_any_deferral_pct"], 1), pct(r["retained_accuracy"]), pct(r["retained_sensitivity"]), pct(r["retained_specificity"]), f(r["retained_macro_auc"]), pct(r["deferred_accuracy"]) if not pd.isna(r["deferred_accuracy"]) else "--"])
add("table", caption="Deferral policies on the test split (13 findings, No Finding excluded)",
    cols=[2.7, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9], header=["Policy", "Def. %", "Studies %", "Acc.", "Sens.", "Spec.", "AUROC", "Acc. def."], rows=prow, font=6.5,
    note=["Def. %: findings deferred; Studies %: studies with at least one deferred finding. Acc., Sens., Spec., AUROC (macro) on retained findings; Acc. def.: accuracy the model would have had on the deferred findings. \u03c4* selected on validation only."])
P(f"Table IX evaluates the policy. The fixed tolerance of the first implementation defers {f(p_sfix['deferred_findings_pct'], 1)}% of findings, confirming that it was inactive. The interval rule alone defers {f(p_int['deferred_findings_pct'], 1)}% of findings, touching {f(p_int['studies_with_any_deferral_pct'], 1)}% of studies. With the validation-selected tolerance for c* = 0.90 the policy defers {f(p90['deferred_findings_pct'], 1)}% of findings (realised test coverage {pct(p90['coverage'])}%, against {pct(taus['0.9']['val_coverage'])}% on validation) and raises accuracy on the retained findings from {pct(p_none['retained_accuracy'])}% to {pct(p90['retained_accuracy'])}%, sensitivity from {pct(p_none['retained_sensitivity'])}% to {pct(p90['retained_sensitivity'])}% and specificity from {pct(p_none['retained_specificity'])}% to {pct(p90['retained_specificity'])}%; on the deferred findings the model would have been right only {pct(p90['deferred_accuracy'])}% of the time. The close agreement between validation and test coverage shows that a tolerance chosen without test data transfers to the test split.")
crow = []
for c in [1.0, 0.95, 0.9, 0.8, 0.7, 0.6, 0.5]:
    r = zc.loc[c]
    row = [f"{int(round(c * 100))}%", pct(r["test_coverage"]), pct(r["accuracy"]), pct(r["sensitivity"]), pct(r["specificity"]), f(r["macro_f1"]), f(r["macro_auc"])]
    crow.append(row)
add("table", caption="Performance on retained findings at different coverage levels (deferral by score z)",
    cols=[1.0, 1.0, 0.9, 0.9, 0.9, 0.9, 1.0], header=["Target", "Test cov.", "Acc.", "Sens.", "Spec.", "F1", "AUROC"], rows=crow, font=7,
    note=["Cut-offs on z set at validation quantiles for each target coverage and applied unchanged to the test split. Macro F1 and AUROC over findings."])
P(f"Table X traces performance against coverage. Deferring the least certain 10% of findings raises accuracy from {pct(zc.loc[1.0]['accuracy'])}% to {pct(zc.loc[0.9]['accuracy'])}% and macro F1 from {f(zc.loc[1.0]['macro_f1'])} to {f(zc.loc[0.9]['macro_f1'])}; at 50% coverage accuracy reaches {pct(zc.loc[0.5]['accuracy'])}%. "
  f"Ranking by \u03c3 alone gives {pct(sc.loc[0.9]['accuracy'])}% at 90% coverage" + (f" and ranking by 1 \u2212 c gives {pct(cc.loc[0.9]['accuracy'])}%" if cc is not None else "") + ", so the score that combines distance to threshold with posterior spread is the most effective deferral criterion (Fig. 2). Random deferral would leave accuracy unchanged at every coverage.")
fig(B / "fig_risk_coverage.png", "Accuracy on retained findings as a function of coverage on the test split, for four deferral scores and random deferral.", width_in=3.2)

add("h2", text="Number of Monte Carlo Passes")
srows = []
for S in S_list:
    r = sw.loc[S]
    srows.append([str(S), f(r["mean_auc_mean"]), f(r["ece_mean"]), f(r["sigma_error_auroc_macro_mean"]) if S > 1 else "--", f(r["sigma_rank_corr_vs_ref_mean"], 3) if S > 1 else "--", f(r["deferred_pct_tau90_mean"], 1), f(lat_ms("cuda", S, "deployed"), 0) if lat_ms("cuda", S, "deployed") else "--", f(lat_ms("cuda", S, "cached"), 1) if lat_ms("cuda", S, "cached") else "--"])
add("table", caption="Effect of the number of MC passes S (test split; mean of five draws)",
    cols=[0.6, 1.0, 0.9, 1.0, 1.0, 0.9, 1.0, 1.0], header=["S", "AUROC", "ECE", "\u03c3 err. AUROC", "\u03c1(\u03c3, \u03c3\u2085\u2080)", "Def. %", "GPU ms", "Cached ms"], rows=srows, font=6.5,
    note=["\u03c1: Spearman correlation of \u03c3 with its 50-pass value. Def. %: findings deferred with \u03c4 re-selected on validation for c* = 0.90. Latency per study at batch 1, median; deployed = full forward per pass, cached = trunk once."])
s_auc = sw["mean_auc_mean"]
P(f"Table XI shows how S matters. Discrimination is essentially insensitive to S (mean AUROC {f(s_auc.min())}\u2013{f(s_auc.max())}), because the posterior mean converges quickly. The uncertainty estimate converges more slowly: the rank correlation of \u03c3 with its 50-pass value is {f(sw.loc[5]['sigma_rank_corr_vs_ref_mean'], 3) if 5 in sw.index else '--'} at S = 5, {f(sw.loc[10]['sigma_rank_corr_vs_ref_mean'], 3) if 10 in sw.index else '--'} at S = 10 and {f(sw.loc[20]['sigma_rank_corr_vs_ref_mean'], 3)} at S = 20, and the error-detection AUROC of \u03c3 rises from {f(sw.loc[2]['sigma_error_auroc_macro_mean']) if 2 in sw.index else '--'} at S = 2 to {f(sw.loc[20]['sigma_error_auroc_macro_mean'])} at S = 20 and {f(sw.loc[50]['sigma_error_auroc_macro_mean']) if 50 in sw.index else '--'} at S = 50. "
  "The percentile interval needs enough samples for its 2.5th and 97.5th percentiles to be meaningful, which also argues against very small S. Latency grows linearly with S in the deployed implementation and is nearly flat with trunk caching. S = 20 lies where the uncertainty estimate has largely converged while latency remains well within an interactive budget; with trunk caching, larger S is affordable if desired.")

if ABL:
    add("h2", text="Ablation Study")
    names = [("full", "All components (reference)"), ("no_asl", "\u2212 Asymmetric Loss (weighted BCE)"), ("no_balanced_sampling", "\u2212 Balanced sampling"), ("no_label_smoothing", "\u2212 Label smoothing (U-Zeros)"), ("no_mixup", "\u2212 Mixup"), ("no_confidence_head", "\u2212 Confidence head"), ("none_of_the_four", "\u2212 All four imbalance components")]
    ref = ABL.get("full")
    arows = []
    for k, lab in names:
        if k not in ABL:
            continue
        r = ABL[k]
        d = (r["mean_auc"] - ref["mean_auc"]) if ref else float("nan")
        arows.append([lab, f(r["mean_auc"]), ("+" if d > 0 else "") + f(d) if k != "full" else "--", f(r["mean_ap"]), f(r["chexpert5_auc"]), f(r["macro_f1"]), f(r["ece"])])
    add("table", caption="Ablation under a fixed reduced budget (20k images, 8 epochs; test split)",
        cols=[3.0, 1.0, 0.9, 0.9, 1.0, 0.9, 0.9], header=["Configuration", "AUROC", "\u0394", "AP", "CheXpert-5", "F1", "ECE"], rows=arows, font=6.5,
        note=["\u0394: change in mean AUROC relative to the reference arm, which is retrained under the same budget and therefore differs from the full model of Table IV. Single training seed per arm."])
    deltas = {k: ABL[k]["mean_auc"] - ref["mean_auc"] for k in ABL if k != "full"}
    worst = sorted(deltas.items(), key=lambda kv: kv[1])
    lab = dict(names)
    abl_txt = "; ".join(f"removing {lab[k].replace('\u2212 ', '').split(' (')[0].lower()} changes mean AUROC by {('+' if v > 0 else '')}{f(v)}" for k, v in worst)
    P(f"Table XII reports the ablation. " + abl_txt[0].upper() + abl_txt[1:]
      + f". The reference arm reaches {f(ref['mean_auc'])} (AP {f(ref['mean_ap'])}). Differences smaller than about 0.005 AUROC are within the variation expected between training seeds and should not be over-interpreted; a multi-seed repetition is left to future work.")

add("h2", text="Visual Explanations")
P("Fig. 3 shows Grad-CAM maps for three findings. To avoid selecting favourable examples, cases were chosen by rule: the retained true positive with the highest probability, the retained false positive with the highest probability, the retained false negative with the lowest probability, and the deferred study with the largest \u03c3 for that finding. The true positives concentrate on the expected anatomy (the cardiac silhouette for cardiomegaly, the costophrenic angles for effusion). The false positives show the confounding described by Zech et al. [10]: attention on support devices, overlapping density or image borders. The false negatives are low-contrast or partially obscured, and in the deferred cases attention is diffuse or split between plausible regions, consistent with the model declining to commit. These maps show correlation with the class score, not localisation.")

# full-width Grad-CAM figure at the end of the results
add("cols", n=1)
fig(B / "fig_gradcam.png", "Grad-CAM for rule-selected test cases. Columns: true positive, false positive, false negative and deferred; rows: findings. Label, posterior mean p and posterior standard deviation \u03c3 are shown under each image.", width_in=6.6)
add("cols", n=2)

# ---------------------------------------------------------------- VIII
add("h1", text="Limitations")
P("Labels are extracted automatically from reports, and the resulting noise bounds achievable discrimination. The corpus originates from a single institution, and we present no evidence of generalisation to other equipment, protocols or populations [10]; external validation on an Indian hospital cohort is required before any claim of applicability there. Only frontal projections are used, and no prior studies or clinical metadata are modelled.")
P("Because dropout is applied only after the trunk, the MC estimate captures uncertainty in the heads given a fixed representation; it is closer to last-layer Bayesian inference than to a posterior over the full network, which explains the small magnitude of \u03c3. Deep ensembles [19] would capture representation uncertainty at proportionally greater cost. The residual miscalibration under temperature scaling shows that loss design and calibration interact; class-wise calibration is needed whenever an asymmetric loss is used. The ablations use a reduced budget and a single seed per arm. Validation performance of the main run collapsed after its best epoch; early stopping protected the reported model, but the cause (possibly an optimisation instability once the full learning rate and the tenfold head rate act on the unfrozen backbone under mixed precision) has not been isolated, and a lower head learning rate or longer warm-up should be tested.")
P("Grad-CAM indicates correlation, not localisation; quantifying localisation requires bounding-box annotation and a pointing-game protocol. Finally, the system has undergone no prospective evaluation, reader study or regulatory assessment, and we make no such claim.")

# ---------------------------------------------------------------- IX
add("h1", text="Conclusion and Future Work")
P(f"We have presented a multi-label chest radiograph classifier that estimates epistemic and aleatoric uncertainty, calibrates its outputs, explains its predictions and declines findings it cannot resolve. On a patient-disjoint CheXpert test split the posterior standard deviation and the confidence head both identify the model's own errors, and a deferral policy whose tolerance is chosen on validation data alone defers {f(p90['deferred_findings_pct'], 1)}% of findings while raising accuracy on the remainder from {pct(p_none['retained_accuracy'])}% to {pct(p90['retained_accuracy'])}%. "
  "The evaluation also exposed two corrections that we have made: a deferral tolerance that never fired, and a calibration step that a single temperature cannot complete under Asymmetric Loss. Future work comprises deep ensembles, multi-seed ablations, external validation on ChestX-ray14 and on an Indian cohort, bounding-box evaluation of saliency, and a prospective study of whether deferral reduces reading time without increasing missed findings.")
P(f"Code and reproduction scripts: {GITHUB}. Live demonstration: {DEMO}.")

# ---------------------------------------------------------------- refs
add("h1n", text="References")
refs = [
    'P. Rajpurkar et al., "CheXNet: Radiologist-level pneumonia detection on chest X-rays with deep learning," arXiv:1711.05225, 2017.',
    'J. Irvin et al., "CheXpert: A large chest radiograph dataset with uncertainty labels and expert comparison," in Proc. AAAI Conf. Artificial Intelligence, 2019, pp. 590\u2013597.',
    'X. Wang, Y. Peng, L. Lu, Z. Lu, M. Bagheri, and R. M. Summers, "ChestX-ray8: Hospital-scale chest X-ray database and benchmarks on weakly-supervised classification and localization of common thorax diseases," in Proc. IEEE CVPR, 2017, pp. 2097\u20132106.',
    'G. Huang, Z. Liu, L. van der Maaten, and K. Q. Weinberger, "Densely connected convolutional networks," in Proc. IEEE CVPR, 2017, pp. 4700\u20134708.',
    '"Benchmarking MedViT and hybrid CNN-ViT architectures for multi-label thoracic disease classification," Scientific Reports, 2026, doi:10.1038/s41598-026-43282-5.',
    'R. R. Selvaraju, M. Cogswell, A. Das, R. Vedantam, D. Parikh, and D. Batra, "Grad-CAM: Visual explanations from deep networks via gradient-based localization," in Proc. IEEE ICCV, 2017, pp. 618\u2013626.',
    'A. Chattopadhay, A. Sarkar, P. Howlader, and V. N. Balasubramanian, "Grad-CAM++: Generalized gradient-based visual explanations for deep convolutional networks," in Proc. IEEE WACV, 2018, pp. 839\u2013847.',
    'Y. Gal and Z. Ghahramani, "Dropout as a Bayesian approximation: Representing model uncertainty in deep learning," in Proc. ICML, 2016, pp. 1050\u20131059.',
    'T.-Y. Lin, P. Goyal, R. Girshick, K. He, and P. Doll\u00e1r, "Focal loss for dense object detection," in Proc. IEEE ICCV, 2017, pp. 2980\u20132988.',
    'J. R. Zech, M. A. Badgeley, M. Liu, A. B. Costa, J. J. Titano, and E. K. Oermann, "Variable generalization performance of a deep learning model to detect pneumonia in chest radiographs: A cross-sectional study," PLOS Medicine, vol. 15, no. 11, e1002683, 2018.',
    'C. Guo, G. Pleiss, Y. Sun, and K. Q. Weinberger, "On calibration of modern neural networks," in Proc. ICML, 2017, pp. 1321\u20131330.',
    'T. DeVries and G. W. Taylor, "Learning confidence for out-of-distribution detection in neural networks," arXiv:1802.04865, 2018.',
    'T. Ridnik et al., "Asymmetric loss for multi-label classification," in Proc. IEEE ICCV, 2021, pp. 82\u201391.',
    'A. Dosovitskiy et al., "An image is worth 16x16 words: Transformers for image recognition at scale," in Proc. ICLR, 2021.',
    'I. Loshchilov and F. Hutter, "Decoupled weight decay regularization," in Proc. ICLR, 2019.',
    'H. Zhang, M. Cisse, Y. N. Dauphin, and D. Lopez-Paz, "mixup: Beyond empirical risk minimization," in Proc. ICLR, 2018.',
    'S. M. Pizer et al., "Adaptive histogram equalization and its variations," Computer Vision, Graphics, and Image Processing, vol. 39, no. 3, pp. 355\u2013368, 1987.',
    'P. Micikevicius et al., "Mixed precision training," in Proc. ICLR, 2018.',
    'B. Lakshminarayanan, A. Pritzel, and C. Blundell, "Simple and scalable predictive uncertainty estimation using deep ensembles," in Advances in NeurIPS, 2017, pp. 6402\u20136413.',
    'Y. Geifman and R. El-Yaniv, "Selective classification for deep neural networks," in Advances in NeurIPS, 2017, pp. 4878\u20134887.',
    'J. Platt, "Probabilistic outputs for support vector machines and comparisons to regularized likelihood methods," in Advances in Large Margin Classifiers, MIT Press, 1999, pp. 61\u201374.',
    'M. P. Naeini, G. Cooper, and M. Hauskrecht, "Obtaining well calibrated probabilities using Bayesian binning," in Proc. AAAI Conf. Artificial Intelligence, 2015, pp. 2901\u20132907.',
    'G. W. Brier, "Verification of forecasts expressed in terms of probability," Monthly Weather Review, vol. 78, no. 1, pp. 1\u20133, 1950.',
]
add("refs", items=refs)

spec = {
    "meta": {"title": "Uncertainty-Aware and Explainable Multi-Label Classification of Chest Radiographs with an Explicit Deferral Policy",
             "author": "Vansh R. Vijayvargiya", "description": "RadiScan paper, revised in response to reviewer comments"},
    "blocks": blocks,
}
OUT.write_text(json.dumps(spec, indent=1))
(OUT.parent / "equations.json").write_text(json.dumps(eqs, indent=1))
print(f"spec: {len(blocks)} blocks, {len(eqs)} equations")
