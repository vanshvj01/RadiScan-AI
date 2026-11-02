"""Build response_spec.json: point-by-point response to the reviewer."""
import json
import sys
from pathlib import Path

# reuse every value computed for the paper
sys.argv = [sys.argv[0], sys.argv[1], str(Path(sys.argv[2]).parent / "_paper_spec_tmp.json")] + sys.argv[3:]
ns: dict = {}
exec(Path(__file__).with_name("content.py").read_text(), ns)
f, pct, ints = ns["f"], ns["pct"], ns["ints"]
R, ABL, spl, ov, cx5 = ns["R"], ns["ABL"], ns["spl"], ns["ov"], ns["cx5"]
c_unc_mc, c_t_mc, c_pl_mc = ns["c_unc_mc"], ns["c_t_mc"], ns["c_pl_mc"]
u_sd, u_c, u_z, ux, qs = ns["u_sd"], ns["u_c"], ns["u_z"], ns["ux"], ns["qs"]
p_none, p90, p_sfix, p_int, taus, zc = ns["p_none"], ns["p90"], ns["p_sfix"], ns["p_int"], ns["taus"], ns["zc"]
sw, lat_ms, stage, gpu_name = ns["sw"], ns["lat_ms"], ns["stage"], ns["gpu_name"]
OUT = Path(sys.argv[2].replace("_paper_spec_tmp.json", "response_spec.json"))

blocks = []


def add(t, **kw):
    blocks.append({"t": t, **kw})


def P(*runs):
    add("p", runs=list(runs), noindent=True, after=80)


add("cols", n=1)
add("title", text="Response to Reviewer Comments")
add("author", runs=["Uncertainty-Aware and Explainable Multi-Label Classification of Chest Radiographs with an Explicit Deferral Policy"], size=20, after=200)
P("We thank the reviewer for a careful and constructive report. Every comment has been addressed with new experiments on the trained model, using the original patient-disjoint split. All numbers below are taken from the revised manuscript; table, figure and equation numbers refer to it. The scripts that regenerate every result are in the public repository (paper/reviewer_experiments.py and paper/run_ablations.py at https://github.com/vanshvj01/RadiScan-AI, branch paper-revision).")

items = []
items.append(("1. Results section incomplete; Tables I–III contain TBD values.",
    f"All tables are now populated from the held-out test split of {ints(R['n_test'])} images. Per-class results are in Table IV and aggregate results in Table V: mean AUROC {f(ov['mean_auc'])}, CheXpert-5 AUROC {f(cx5['mean_auc'])}, mean AP {f(ov['mean_ap'])}. While completing the evaluation we found that uncertain test annotations had been replaced by random smoothed targets; they are now excluded from the metrics (Section III-C), and Table V also reports the earlier protocol, which reproduces the originally reported value on the same split. Section VII now contains eight results subsections with twelve tables and three figures."))
items.append(("2. Effectiveness of epistemic and aleatoric uncertainty not demonstrated.",
    f"Section VII-C evaluates both signals as detectors of the model's own errors (Table VII) and by quartile (Table VIII). The posterior standard deviation reaches an error-detection AUROC of {f(u_sd['error_auroc_macro'])} (macro over findings) and accuracy falls from {pct(qs.iloc[0]['accuracy'])}% to {pct(qs.iloc[3]['accuracy'])}% across its quartiles; the confidence head reaches {f(u_c['error_auroc_macro']) if u_c else '--'}. We also report mutual information, predictive entropy, a threshold-margin baseline and the area under the risk-coverage curve."))
items.append(("3. Deferral policy not evaluated; report deferral percentage and performance at coverage levels.",
    f"Table IX reports, for six policies, the percentage of findings and of studies deferred and accuracy, sensitivity, specificity and AUROC on retained findings, plus the accuracy the model would have had on deferred findings. At the deployed setting the policy defers {f(p90['deferred_findings_pct'], 1)}% of findings and raises accuracy on retained findings from {pct(p_none['retained_accuracy'])}% to {pct(p90['retained_accuracy'])}%. Table X and Fig. 2 report performance at coverage levels from 100% to 50%."))
items.append(("4. Calibration results missing; report ECE and Brier before and after.",
    f"Table VI reports ECE, Brier score, NLL and AUROC with no calibration, with the fitted temperature, with a refitted temperature and with class-wise Platt scaling, for single-pass and MC-20 inference; Fig. 1 shows reliability diagrams. Temperature scaling changes the ECE from {f(c_unc_mc['ece'])} to {f(c_t_mc['ece'])} and the Brier score from {f(c_unc_mc['brier'])} to {f(c_t_mc['brier'])}. We report candidly that a substantial error remains, explain it as the location bias of Asymmetric Loss, and report class-wise Platt scaling as a remedy (ECE {f(c_pl_mc['ece'])})."))
items.append(("5. Ablation studies for ASL, balanced sampling, label smoothing and mixup.",
    "Section VI-D describes a controlled ablation and Table XII reports it: each component is removed in turn, plus the confidence head and all four imbalance components together, every arm retrained under one fixed budget with the same seed and data and evaluated identically. " + (f"The reference arm reaches mean AUROC {f(ABL['full']['mean_auc'])}." if ABL.get('full') else "")))
items.append(("6. Justification for 20 MC passes; effect on uncertainty and inference time.",
    f"Table XI evaluates S = 1 to 50 (five random draws each): AUROC, ECE, error-detection AUROC of σ, rank correlation of σ with its 50-pass value, deferral rate and measured latency. Discrimination is insensitive to S, whereas the uncertainty estimate converges more slowly (rank correlation {f(sw.loc[20]['sigma_rank_corr_vs_ref_mean'], 3)} at S = 20); S = 20 is where the estimate has largely converged within an interactive latency budget."))
items.append(("7. Confidence head: formulation and effectiveness.",
    "Section IV-D now gives the formulation, (7)–(8): hint interpolation with a stop-gradient on the disease probability, the budget penalty with β = 0.3 and loss weight 0.1, and the reasoning for why the equilibrium confidence estimates trustworthiness. Section VII-C evaluates it: error-detection AUROC, quartile accuracy, rank correlation with σ "
    + (f"({f(ux['spearman_sigma_vs_1mc'], 2)}), and its response to findings the reporting radiologist marked uncertain (AUROC {f(ux['uncertain_label_auroc_1mc'])}), an external proxy for image-intrinsic ambiguity. " if u_c else ". ")
    + "Table XII includes an arm trained without the head."))
items.append(("8. Grad-CAM examples for correct, incorrect and deferred cases.",
    "Fig. 3 shows Grad-CAM for true positives, false positives, false negatives and deferred cases for three findings. Cases are selected by a stated rule rather than by hand, to avoid choosing favourable examples, and Section VII-G discusses failure modes."))
items.append(("9. Split details and verification of patient-level separation.",
    f"Table I reports patients, studies and images per split (train {ints(spl['train']['patients'])} / validation {ints(spl['val']['patients'])} / test {ints(spl['test']['patients'])} patients; {ints(spl['train']['images'])} / {ints(spl['val']['images'])} / {ints(spl['test']['images'])} images). Section III-B describes the two verification steps: an assertion of empty patient intersection at split time, and an independent audit that recomputes patient identifiers from the saved split files. All pairwise patient and image overlaps are zero."))
items.append(("10. Threshold on posterior standard deviation: value and selection without test data.",
    f"Section IV-G now states that the initial fixed tolerance (0.15) was two orders of magnitude above the observed σ and never fired (Table IX: {f(p_sfix['deferred_findings_pct'], 1)}% deferred). The tolerance is now chosen on the validation split only, by (12), as the smallest value whose validation coverage reaches a target; for the deployed 90% target τ* = {f(taus['0.9']['tau'], 4)}. Validation coverage {pct(taus['0.9']['val_coverage'])}% transfers to {pct(p90['coverage'])}% on the test split."))
g = stage("cuda", "total_ms")
items.append(("11. Latency, throughput and resource usage.",
    "Section V now reports per-stage latency (Table II) and throughput and memory (Table III) on a GPU and on a CPU"
    + (f"; a complete study including saliency takes {f(g['median_ms'], 0)} ms on the {gpu_name}" if g else "")
    + ". Measuring cost revealed that the trunk is recomputed needlessly on every MC pass; computing it once is mathematically identical and makes CPU-only inference practical (Section V)."))

for head, body in items:
    add("p", runs=[{"text": head, "b": True}], noindent=True, after=40)
    P("Response. " + body)

P("Other changes: equations (1)–(12) formalise the loss, sampling, mixup, MC estimator, confidence objective, calibration, deferral rule and tolerance selection; references [21]–[23] were added; code and live-demonstration links were added to the abstract and Sections I, V and IX; the Limitations section now discusses last-layer MC Dropout and the interaction between loss design and calibration.")
OUT.write_text(json.dumps({"meta": {"title": "Response to Reviewer Comments", "author": "Vansh R. Vijayvargiya", "description": "Point-by-point response"}, "blocks": blocks}, indent=1))
print("response spec", OUT)
