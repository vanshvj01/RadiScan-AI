"""Single source of text for the RadiScan conference paper.

Both renderers (build_docx.py, build_tex.py) read this file, so the Word and
LaTeX versions cannot drift. Edit here, never in the output files.

Placeholder convention: any string containing [[...]] is unresolved. The docx
highlights it yellow; the LaTeX wraps it in \\todo{} and renders it red.
Tables I-III are filled automatically by `python -m paper.fill_results`.
"""

TITLE = (
    "Uncertainty-Aware and Explainable Multi-Label Classification of Chest "
    "Radiographs with an Explicit Deferral Policy"
)

AUTHORS = [
    {"name": "Vansh R. Vijayvargiya", "affil": 1, "email": "v.vijayvargiya@somaiya.edu"},
    {"name": "[[Co-author 2]]", "affil": 1, "email": "[[email]]"},
    {"name": "[[Co-author 3]]", "affil": 1, "email": "[[email]]"},
    {"name": "[[Prof. Faculty Guide]]", "affil": 1, "email": "[[email]]"},
]

AFFILIATION = (
    "Department of Information Technology, K. J. Somaiya School of Engineering, "
    "Somaiya Vidyavihar University, Mumbai, India"
)

ABSTRACT = (
    "Deep convolutional networks now match reported radiologist performance on "
    "several thoracic findings, yet they are rarely deployed as clinical decision "
    "support. A central obstacle is that a bare probability communicates nothing "
    "about how much that probability should be trusted: a network is equally "
    "fluent when it is confident and when it is guessing. This paper presents "
    "RadiScan, a multi-label chest radiograph classification system designed "
    "around the premise that a decision-support tool must quantify, calibrate and "
    "explain its own uncertainty, and must decline the cases it cannot resolve. "
    "The system combines a DenseNet-121 trunk with three task heads, and layers "
    "four mechanisms on top: Monte Carlo Dropout for epistemic uncertainty; a "
    "learned confidence head trained with a confidence-penalty objective for "
    "aleatoric noise; post-hoc temperature scaling for calibration; and "
    "gradient-weighted class activation mapping for visual explanation. "
    "Asymmetric Loss, inverse-frequency balanced sampling and per-class threshold "
    "tuning address prevalence that ranges from approximately 40 percent to under "
    "2 percent across the fourteen CheXpert labels. The distinguishing "
    "contribution is an explicit deferral policy: when a finding's posterior "
    "credible interval straddles its own tuned operating threshold, the system "
    "labels that finding indeterminate and refers it for human review rather than "
    "committing to either call. We further report a reproducibility protocol "
    "including patient-disjoint splitting with assertion, and a pre-flight data "
    "verification step motivated by a silent failure mode in which a "
    "missing-image fallback allowed a pipeline to train to chance performance "
    "without raising any error. On a held-out, patient-disjoint test split the "
    "system attains a mean AUC-ROC of [[mean AUC]] across fourteen findings and "
    "an expected calibration error of [[ECE]] after temperature scaling."
)

KEYWORDS = (
    "chest radiography, multi-label classification, uncertainty quantification, "
    "Monte Carlo dropout, model calibration, explainable AI, Grad-CAM, clinical "
    "decision support, CheXpert"
)

SECTIONS = [
    # ------------------------------------------------------------------ I
    {
        "heading": "Introduction",
        "body": [
            "Chest radiography is the most frequently performed diagnostic imaging "
            "examination worldwide. Its volume grows with population and access, while "
            "the supply of trained radiologists does not scale correspondingly. In much "
            "of India, a chest radiograph acquired at a primary or secondary centre may "
            "wait days for a formal report, and the delay falls hardest on exactly the "
            "findings for which time matters, such as pneumothorax and pulmonary oedema. "
            "Automated pre-reading that orders a reporting queue by clinical urgency is "
            "therefore a more tractable and more immediately useful objective than "
            "autonomous diagnosis.",
            "Since CheXNet demonstrated radiologist-comparable performance on pneumonia "
            "detection [1], a large literature has reported strong discriminative "
            "results on public chest radiograph corpora. Comparatively little of that "
            "work has reached clinical deployment. Among the reasons, one is "
            "methodological rather than organisational: the standard output of these "
            "systems is a vector of probabilities with no attached measure of "
            "reliability. A prediction of 0.62 accompanied by a posterior standard "
            "deviation of 0.02 and one accompanied by a standard deviation of 0.30 are "
            "different clinical objects, and only the first is actionable. A system that "
            "reports the point estimate alone provides the reader no means of "
            "distinguishing them.",
            "Two further properties are commonly absent. First, modern neural networks "
            "are systematically overconfident, so a stated probability of 0.90 "
            "corresponds to an empirical frequency materially below 0.90 [11]. For a "
            "system whose premise is that the clinician should act on the number, "
            "miscalibration is not a cosmetic defect. Second, when a model is genuinely "
            "unable to resolve a case, the conventional design still emits a decision. "
            "There is no mechanism by which the system can decline.",
            "This paper describes RadiScan, a system designed around those three gaps. "
            "Our contributions are as follows.",
            "(i) We integrate epistemic and aleatoric uncertainty estimation into a "
            "multi-label thoracic classifier, obtaining posterior means, standard "
            "deviations, credible intervals, predictive entropy and mutual information "
            "for each of fourteen findings, alongside a separately trained per-class "
            "aleatoric confidence.",
            "(ii) We define an explicit deferral policy grounded in the interaction "
            "between the posterior credible interval and the per-class operating "
            "threshold. A finding whose interval straddles its own threshold is reported "
            "as indeterminate and referred for human review. This converts the "
            "uncertainty estimate from a displayed quantity into a quantity that changes "
            "system behaviour.",
            "(iii) We combine Asymmetric Loss, inverse-frequency balanced sampling, "
            "label-smoothing treatment of uncertain annotations, and per-class threshold "
            "tuning to address prevalence spanning more than an order of magnitude, and "
            "we report average precision alongside AUC-ROC because the former degrades "
            "where the latter can remain deceptively high.",
            "(iv) We contribute a reproducibility protocol motivated by observed "
            "failure. In an earlier iteration of this work, a dataset loader that "
            "substituted a blank frame for any unreadable image allowed a complete "
            "training run to finish at chance performance with no error raised. We "
            "describe the verification step that now prevents this class of failure.",
            "(v) We deliver the model as a complete clinical workflow application with "
            "authenticated access, audit logging, asynchronous inference and "
            "structured report generation, so that the uncertainty and deferral outputs "
            "are evaluated in the form a reader would actually encounter them.",
        ],
    },
    # ------------------------------------------------------------------ II
    {
        "heading": "Related Work",
        "subsections": [
            {
                "heading": "Chest Radiograph Classification",
                "body": [
                    "The release of ChestX-ray14 [3] and CheXpert [2] made large-scale "
                    "supervised learning on chest radiographs practical. CheXNet [1] "
                    "applied a DenseNet-121 [4] to ChestX-ray14 and reported "
                    "radiologist-comparable F1 on pneumonia. Subsequent work has "
                    "explored attention mechanisms, higher input resolutions, and "
                    "transformer backbones; a recent benchmark on CheXpert reports macro "
                    "AUROC of 0.7511 for MedViT and 0.7168 for a hybrid CNN-ViT under a "
                    "common protocol [5]. DenseNet-121 remains a strong and widely "
                    "reproduced baseline, which is the reason it is adopted here: the "
                    "contribution of this work lies in what is layered on the trunk, and "
                    "a familiar trunk makes that contribution easier to isolate.",
                ],
            },
            {
                "heading": "Uncertainty Quantification",
                "body": [
                    "Gal and Ghahramani showed that dropout applied at inference "
                    "approximates variational inference in a deep Gaussian process [6], "
                    "making Monte Carlo Dropout the least invasive route to a posterior "
                    "over predictions. Kendall and Gal distinguished epistemic "
                    "uncertainty, reducible with more data, from aleatoric uncertainty, "
                    "inherent to the input [7]. Deep ensembles [8] provide stronger "
                    "estimates at proportionally greater training and inference cost. "
                    "DeVries and Taylor proposed learning a confidence estimate directly "
                    "through a hint-based penalty [9], which we adapt for the aleatoric "
                    "component. Within chest radiography, uncertainty has been used "
                    "mainly as a displayed quantity; its use to drive an explicit "
                    "referral decision is less common.",
                ],
            },
            {
                "heading": "Calibration and Explanation",
                "body": [
                    "Guo et al. demonstrated that modern networks are poorly calibrated "
                    "and that a single learned temperature restores calibration "
                    "effectively without affecting ranking [11]. Grad-CAM [12] and its "
                    "extension Grad-CAM++ [13] produce class-specific saliency from "
                    "gradient information at the final convolutional layer. Zech et al. "
                    "showed that chest radiograph classifiers can exploit "
                    "hospital-specific confounders such as scanner type and laterality "
                    "markers [10], a finding that motivates using saliency to audit "
                    "failure cases rather than to illustrate successes.",
                ],
            },
        ],
    },
    # ------------------------------------------------------------------ III
    {
        "heading": "Dataset and Preprocessing",
        "subsections": [
            {
                "heading": "CheXpert",
                "body": [
                    "We use CheXpert [2], comprising 224,316 chest radiographs from "
                    "65,240 patients at Stanford Hospital, labelled for fourteen "
                    "observations by an automated rule-based extractor applied to "
                    "free-text reports. Each label takes one of four states: positive, "
                    "negative, uncertain, or not mentioned. We restrict to frontal "
                    "projections and train at 320x320 pixels. Prevalence varies sharply: "
                    "Support Devices and Lung Opacity are present in a large fraction of "
                    "studies, while Pleural Other and Fracture fall below two percent. "
                    "Accuracy is deliberately excluded as a headline quantity: a "
                    "classifier that predicts every rare finding as absent attains "
                    "accuracy above 0.98 on those findings while detecting none of them.",
                ],
            },
            {
                "heading": "Uncertain Label Policy",
                "body": [
                    "Irvin et al. compared several policies for the uncertain label "
                    "state [2]. We adopt label-smoothing regularisation (LSR): an "
                    "uncertain annotation is mapped to a soft target drawn uniformly "
                    "from [0.05, 0.85] rather than to a hard 0 or 1. This expresses "
                    "that the radiologist's hedging carries information, while biasing "
                    "the target slightly towards presence, consistent with the "
                    "clinical convention that an equivocal mention is more often a true "
                    "finding than not. Not-mentioned labels are treated as negative.",
                ],
            },
            {
                "heading": "Preprocessing and Augmentation",
                "body": [
                    "Before enhancement, an automatic clean-up stage normalises the "
                    "input. Colour photographs are converted to greyscale; where the "
                    "input is a phone photograph of a film, the film is located and "
                    "the surrounding background removed; uniform padding and "
                    "collimator borders are cropped. The crop is deliberately to the "
                    "thorax rather than to the lung fields, since cardiomegaly, "
                    "effusion, fracture and support devices all lie partly outside "
                    "them. Every step declines rather than guesses when its detection "
                    "is implausible, the same stage is applied in training and "
                    "inference, and the checkpoint records whether it was enabled. "
                    "For display only, the image is additionally denoised by "
                    "non-local means with a strength matched to an estimate of its "
                    "noise and capped to preserve fine structures such as a "
                    "pneumothorax pleural line; denoising is not applied to model "
                    "input, as the downsampling to 320 pixels already averages pixel "
                    "noise.",
                    "Images are converted to grayscale, enhanced by contrast-limited "
                    "adaptive histogram equalisation (CLAHE), replicated to three "
                    "channels, and normalised with ImageNet statistics to match the "
                    "pretrained trunk. Training augmentation consists of random "
                    "resized crops, small rotations, brightness and contrast jitter, "
                    "and horizontal flips. Horizontal flipping is debatable for "
                    "radiographs because it reverses cardiac laterality; we retain it "
                    "because none of the fourteen labels encodes side, and we use it "
                    "again as test-time augmentation. Inverse-frequency balanced "
                    "sampling over-draws studies containing rare positives so that "
                    "each mini-batch carries gradient signal for them.",
                ],
            },
        ],
    },
    # ------------------------------------------------------------------ IV
    {
        "heading": "Proposed Methodology",
        "subsections": [
            {
                "heading": "Network Architecture",
                "body": [
                    "The trunk is an ImageNet-pretrained DenseNet-121 [4] with 8.0 "
                    "million parameters, producing a 1024-dimensional pooled feature. "
                    "Three heads share this feature: a fourteen-way disease head with "
                    "independent sigmoid outputs, since findings co-occur and a softmax "
                    "would force them to compete; an auxiliary fracture head; and a "
                    "confidence head emitting one scalar per finding. Each head is "
                    "preceded by dropout with rate 0.2, which serves both as "
                    "regularisation during training and as the stochastic element "
                    "sampled at inference. A scalar temperature is stored with the "
                    "model and fitted after training.",
                ],
            },
            {
                "heading": "Loss Design under Extreme Imbalance",
                "body": [
                    "Binary cross-entropy weights every negative equally, so with "
                    "prevalence below two percent the gradient is dominated by easy "
                    "negatives. Focal Loss down-weights well-classified examples "
                    "symmetrically [14]. Asymmetric Loss (ASL) [15] decouples the two "
                    "sides: it applies a focusing exponent gamma- = 4 to negatives and "
                    "gamma+ = 0 to positives, and additionally shifts negative "
                    "probabilities down by a margin of 0.05, discarding very easy "
                    "negatives entirely. Positive examples, which are scarce, therefore "
                    "contribute their full gradient. The total objective is the disease "
                    "ASL term plus 0.3 times the fracture term plus the confidence "
                    "penalty described below.",
                ],
            },
            {
                "heading": "Epistemic Uncertainty",
                "body": [
                    "At inference we enable only the dropout layers, leaving batch "
                    "normalisation in evaluation mode. Calling a framework's generic "
                    "training mode would also switch batch normalisation to "
                    "per-batch statistics, which corrupts single-image predictions. "
                    "We draw S = 20 stochastic forward passes, optionally doubled by "
                    "horizontal-flip augmentation, and report for each finding the "
                    "posterior mean, standard deviation, a 90 percent credible interval "
                    "from the empirical quantiles, predictive entropy, and mutual "
                    "information. Mutual information isolates the epistemic component: "
                    "it is high when individual samples are confident but disagree.",
                ],
            },
            {
                "heading": "Aleatoric Confidence",
                "body": [
                    "Following DeVries and Taylor [9], the confidence head outputs "
                    "c in (0, 1) per finding. During training the prediction is "
                    "interpolated towards the target in proportion to 1 - c, and a "
                    "penalty -log c discourages the network from requesting hints "
                    "indiscriminately. The network thereby learns to lower c on inputs "
                    "it cannot fit, such as rotated, under-penetrated or heavily "
                    "annotated films, independently of the epistemic estimate.",
                ],
            },
            {
                "heading": "Post-hoc Calibration",
                "body": [
                    "A single temperature T is fitted on the validation split by "
                    "minimising negative log-likelihood of the logits divided by T, "
                    "following Guo et al. [11]. Because division by a positive scalar "
                    "is monotonic, the ranking of examples is unchanged and AUC-ROC is "
                    "numerically identical before and after scaling. Any change in AUC "
                    "after calibration therefore indicates an implementation error, "
                    "which we use as a built-in check.",
                ],
            },
            {
                "heading": "Operating Points",
                "body": [
                    "A uniform threshold of 0.5 is inappropriate when prevalence "
                    "ranges over an order of magnitude. For each finding we select the "
                    "threshold that maximises F1 on the validation split, subject to a "
                    "minimum recall for findings where a miss is clinically costly. "
                    "Thresholds are frozen before the test split is evaluated; tuning "
                    "them on test data would inflate every reported precision, recall "
                    "and F1.",
                ],
            },
            {
                "heading": "Deferral Policy",
                "body": [
                    "For finding k with tuned threshold t_k and posterior credible "
                    "interval [l_k, u_k], the system emits positive if l_k > t_k, "
                    "negative if u_k < t_k, and indeterminate otherwise. An "
                    "indeterminate finding is shown to the reader with its interval "
                    "and saliency map and flagged for mandatory human review. The "
                    "rule is deliberately simple: it uses only quantities the system "
                    "already computes, it has no free parameter beyond the interval "
                    "width, and it can be explained to a clinician in one sentence. "
                    "Its value is measured empirically through accuracy as a function "
                    "of retained coverage (Section VI-D).",
                ],
            },
        ],
    },
    # ------------------------------------------------------------------ V
    {
        "heading": "System Implementation",
        "body": [
            "The model is delivered as a three-tier application. The machine-learning "
            "core is a Python library built on PyTorch and timm, with albumentations "
            "and OpenCV for preprocessing and pydicom for DICOM input. A FastAPI "
            "service exposes authenticated endpoints with JSON Web Token sessions and "
            "bcrypt password hashing; requests are validated by Pydantic schemas and "
            "persisted through SQLAlchemy. Inference runs asynchronously on a Celery "
            "worker backed by Redis, so a twenty-pass Monte Carlo prediction does not "
            "block the request thread. Every prediction, view and export is recorded "
            "in an append-only audit log. Accounts are either doctors or patients. "
            "Both may upload; a patient's scans form a longitudinal history, and a "
            "doctor can read that history only while the patient has granted them "
            "access, a consent that the patient can revoke at any time. No role on "
            "its own opens a patient's record, and every grant, revocation and "
            "history view is audited. When a doctor reads a new study, the "
            "patient's prior studies and the change in each leading finding are "
            "shown alongside it for reference. Reports are rendered to PDF with ReportLab "
            "and carry the posterior interval, the deferral state and the saliency "
            "overlay for each finding.",
            "The front end is a Next.js and TypeScript dashboard in which findings are "
            "ordered by urgency, indeterminate findings are visually distinguished, "
            "and saliency can be toggled per class. The repository carries 84 "
            "automated tests covering losses, uncertainty sampling, calibration, data "
            "verification and the API, run on every commit by continuous integration "
            "alongside static analysis; the full stack is containerised with Docker.",
        ],
    },
    # ------------------------------------------------------------------ VI
    {
        "heading": "Experimental Protocol",
        "subsections": [
            {
                "heading": "Patient-Disjoint Splitting",
                "body": [
                    "CheXpert contains multiple studies per patient. A split by image "
                    "would allow the same patient to appear in training and test sets, "
                    "and the model could recognise the patient rather than the "
                    "pathology. We split by patient identifier into training, "
                    "validation and test partitions and assert at load time that the "
                    "three identifier sets are pairwise disjoint; a violation aborts "
                    "the run rather than emitting a warning.",
                ],
            },
            {
                "heading": "Pre-flight Data Verification",
                "body": [
                    "An earlier iteration of this system contained a dataset loader "
                    "that logged a warning and returned a blank frame for any image it "
                    "could not read. Because a path-prefix mismatch rendered every "
                    "image unreadable, a full training run completed without error, "
                    "reporting approximately chance discrimination. No exception was "
                    "raised at any point.",
                    "We draw a general rule from this: in a machine learning pipeline, "
                    "data-integrity failures must be loud, because a pipeline that "
                    "reports success while learning nothing is more dangerous than one "
                    "that terminates. The loader now raises. A separate verification "
                    "command samples the annotation file, confirms that paths resolve "
                    "on disk, and reports per-class positive rates before any training "
                    "begins; a class reporting zero prevalence indicates column "
                    "misalignment and is caught in seconds rather than after hours of "
                    "computation.",
                ],
            },
            {
                "heading": "Training Configuration",
                "body": [
                    "Optimisation uses AdamW with base learning rate 1e-4, weight decay "
                    "1e-4, and a tenfold multiplier on head parameters, reflecting that "
                    "the backbone encodes transferable features while the heads are "
                    "randomly initialised. A two-epoch linear warmup precedes cosine "
                    "decay, and the backbone is frozen for the first epoch. "
                    "Mixed-precision arithmetic is enabled. An exponential moving "
                    "average of parameters with decay 0.999 is maintained, and "
                    "evaluation and checkpointing use the averaged weights.",
                    "Multi-label mixup is applied with alpha = 0.2. Targets are "
                    "combined by element-wise maximum rather than convex interpolation: "
                    "if either source image exhibits a finding, the blended image "
                    "plausibly exhibits it, whereas interpolation would assign a target "
                    "of 0.5 to a clearly present finding.",
                    "Training proceeds for up to 25 epochs with early stopping on "
                    "validation mean AUC, patience 6. Batch size was determined "
                    "empirically by measuring peak allocation across a real forward "
                    "and backward pass. We note that on the Windows display driver "
                    "model, an allocation exceeding physical device memory does not "
                    "raise an out-of-memory error but is transparently served from "
                    "host memory across the bus, so absence of failure is not evidence "
                    "that a configuration fits; peak allocation must be compared "
                    "against physical capacity directly. Training was performed on "
                    "[[GPU, e.g. NVIDIA Tesla P100 16 GB]] and took [[hours]] hours.",
                ],
            },
            {
                "heading": "Metrics",
                "body": [
                    "We report mean AUC-ROC over all fourteen labels and over the "
                    "five-label CheXpert competition subset; mean average precision, "
                    "which unlike AUC degrades visibly where rare classes are being "
                    "guessed; precision, recall and F1 at tuned thresholds; Brier score; "
                    "expected calibration error (ECE) with fifteen bins; and positive "
                    "support per class, so that a reader may judge which estimates rest "
                    "on few cases.",
                    "Two analyses specifically evaluate the uncertainty machinery. The "
                    "first stratifies predictions into quartiles by posterior standard "
                    "deviation and reports accuracy within each; a monotone decrease "
                    "indicates that the estimates carry information. The second reports "
                    "the selective prediction curve, accuracy as a function of retained "
                    "coverage under deferral of the least certain predictions.",
                ],
            },
        ],
    },
    # ------------------------------------------------------------------ VII
    {
        "heading": "Results and Discussion",
        "body": [
            "[[RESULTS PENDING: complete the training run described in Section VI "
            "and run paper.fill_results to populate Tables I-III. Then replace this "
            "paragraph with 3-5 paragraphs answering, in order: which findings were "
            "detected well or poorly and whether that matches lesion conspicuity; "
            "whether ECE fell under temperature scaling while AUC stayed identical; "
            "whether Table III is monotone; and where Table II places the system "
            "relative to [5].]]",
            "Table I reports per-class discrimination on the held-out, "
            "patient-disjoint test split. We expect performance to correlate with "
            "lesion conspicuity and positive support: findings that are large and "
            "high-contrast, such as cardiomegaly and pleural effusion, should exceed "
            "the mean, while small or rare findings such as lung lesion and pleural "
            "other should fall below it. The support column is reported so that "
            "estimates resting on few positive cases are identifiable as such.",
            "Table II reports aggregate discrimination and calibration, and situates "
            "the result against published CheXpert figures. We note that the "
            "comparison in Table II is between methods evaluated on different splits "
            "of the same corpus and is therefore indicative rather than a controlled "
            "comparison.",
            "Table III is the analysis on which the uncertainty component of this work "
            "depends. If accuracy decreases monotonically across quartiles of "
            "increasing posterior standard deviation, the estimates carry "
            "discriminative information about model error and the deferral policy of "
            "Section IV-G is justified. If it does not, the appropriate conclusion is "
            "that Monte Carlo Dropout with S = 20 provides insufficient posterior "
            "resolution at this data scale, and we would report that outcome rather "
            "than a favourable alternative.",
            "[[Insert Fig. 2: per-class AUC with support; Fig. 3: reliability diagram "
            "before and after temperature scaling; Fig. 4: accuracy against coverage "
            "under selective prediction; Fig. 5: representative Grad-CAM overlays for "
            "true positive, false positive and deferred cases.]]",
            "Calibration is expected to improve substantially under temperature "
            "scaling while mean AUC remains numerically identical, which serves as an "
            "arithmetic check on the implementation as well as a result: any observed "
            "change in AUC would indicate an error, since the transformation is "
            "monotonic.",
            "Qualitative inspection of saliency should be reported with explicit "
            "attention to failure cases. Where attention concentrates on support "
            "devices, laterality markers or image borders rather than parenchyma, that "
            "observation should be reported, since it is precisely the confounding "
            "described by Zech et al. [10] and is more informative than a selection of "
            "favourable examples.",
        ],
        "tables": ["table1", "table2", "table3"],
    },
    # ------------------------------------------------------------------ VIII
    {
        "heading": "Limitations",
        "body": [
            "Labels are derived by automated extraction from free-text reports rather "
            "than by expert re-annotation, and the resulting noise bounds achievable "
            "discrimination independently of modelling choices.",
            "The corpus originates from a single institution. We present no evidence "
            "of generalisation to different acquisition equipment, protocols or "
            "patient populations, and the findings of Zech et al. [10] indicate that "
            "such transfer cannot be assumed. External validation on an Indian "
            "hospital cohort is necessary before any claim of applicability in that "
            "setting.",
            "Only frontal projections are used. Lateral views and prior comparison "
            "studies, both routine in clinical reporting, are not modelled. No clinical "
            "metadata, such as presenting symptoms, history or laboratory values, is "
            "incorporated; a radiological reading is never image-only.",
            "Monte Carlo Dropout is an approximation to Bayesian inference, and the "
            "choice of twenty samples is dictated by inference latency rather than by "
            "a convergence criterion. Deep ensembles would likely yield better "
            "calibrated posteriors at proportionally greater cost.",
            "Grad-CAM indicates correlation between spatial features and class score. "
            "It does not establish that the model attends to pathology, and we make no "
            "localisation claim. Quantifying localisation requires bounding-box "
            "annotation and a pointing-game protocol, which the present corpus does "
            "not support.",
            "Finally, the system has undergone no prospective evaluation, no reader "
            "study and no regulatory assessment. The audit-logging and access-control "
            "structures follow established principles for handling clinical data, but "
            "the system is not certified against any regulatory framework, and we make "
            "no such claim.",
        ],
    },
    # ------------------------------------------------------------------ IX
    {
        "heading": "Conclusion and Future Work",
        "body": [
            "We have presented a multi-label chest radiograph classification system "
            "designed on the premise that clinical decision support requires more than "
            "discriminative accuracy. The system estimates epistemic uncertainty by "
            "Monte Carlo Dropout and aleatoric uncertainty by a learned confidence "
            "head, calibrates its outputs by temperature scaling, explains its "
            "predictions by gradient-weighted class activation mapping, and, "
            "distinctively, declines to report findings whose credible interval "
            "straddles their own operating threshold.",
            "We have additionally described a reproducibility protocol comprising "
            "patient-disjoint splitting with assertion and pre-flight data "
            "verification, motivated by an observed silent failure in which a "
            "missing-image fallback permitted training to complete at chance "
            "performance without error. We suggest that the underlying principle, "
            "that data-integrity failures must be loud, deserves wider attention in "
            "applied machine learning practice than it currently receives.",
            "Several directions follow. Deep ensembles would provide better posterior "
            "estimates than Monte Carlo Dropout, at proportional cost, and the "
            "resulting improvement in deferral quality is measurable within the "
            "framework presented here. Evaluating the trained model on ChestX-ray14 "
            "would quantify cross-institutional degradation directly. Bounding-box "
            "annotation on a subset would permit the pointing-game evaluation that "
            "would convert the saliency component from qualitative to quantitative. "
            "Finally, a prospective study measuring whether the deferral policy "
            "reduces radiologist reading time without increasing missed findings "
            "would address the question that ultimately determines clinical value.",
        ],
    },
]

_LABELS = [
    "No Finding", "Enlarged Cardiomed.", "Cardiomegaly", "Lung Opacity",
    "Lung Lesion", "Edema", "Consolidation", "Pneumonia", "Atelectasis",
    "Pneumothorax", "Pleural Effusion", "Pleural Other", "Fracture",
    "Support Devices",
]

TABLES = {
    "table1": {
        "caption": "Per-class performance on the held-out, patient-disjoint test split",
        "columns": ["Finding", "Support", "AUC", "AP", "Precision", "Recall", "F1"],
        "widths": [1900, 900, 900, 900, 1000, 900, 850],
        "rows": [
            [name, "[[n]]", "[[--]]", "[[--]]", "[[--]]", "[[--]]", "[[--]]"]
            for name in _LABELS
        ]
        + [["Mean", "--", "[[--]]", "[[--]]", "[[--]]", "[[--]]", "[[--]]"]],
    },
    "table2": {
        "caption": "Aggregate performance and comparison with published CheXpert results",
        "columns": ["Method", "Backbone", "Mean AUC", "CheXpert-5 AUC", "ECE"],
        "widths": [1700, 1300, 1000, 1100, 900],
        "rows": [
            ["MedViT [5]", "Transformer", "0.7511", "--", "--"],
            ["Hybrid CNN-ViT [5]", "CNN + ViT", "0.7168", "--", "--"],
            ["Ours (uncalibrated)", "DenseNet-121", "[[--]]", "[[--]]", "[[--]]"],
            ["Ours (temp. scaled)", "DenseNet-121", "[[--]]", "[[--]]", "[[--]]"],
        ],
    },
    "table3": {
        "caption": "Accuracy stratified by posterior standard deviation quartile",
        "columns": ["Quartile", "Mean sigma", "n", "Accuracy"],
        "widths": [1600, 1200, 1000, 1200],
        "rows": [
            ["Q1 (most certain)", "[[--]]", "[[n]]", "[[--]]"],
            ["Q2", "[[--]]", "[[n]]", "[[--]]"],
            ["Q3", "[[--]]", "[[n]]", "[[--]]"],
            ["Q4 (least certain)", "[[--]]", "[[n]]", "[[--]]"],
        ],
    },
}

REFERENCES = [
    'P. Rajpurkar et al., "CheXNet: Radiologist-level pneumonia detection on chest X-rays with deep learning," arXiv:1711.05225, 2017.',
    'J. Irvin et al., "CheXpert: A large chest radiograph dataset with uncertainty labels and expert comparison," in Proc. AAAI Conf. Artificial Intelligence, 2019, pp. 590-597.',
    'X. Wang, Y. Peng, L. Lu, Z. Lu, M. Bagheri, and R. M. Summers, "ChestX-ray8: Hospital-scale chest X-ray database and benchmarks on weakly-supervised classification and localization of common thorax diseases," in Proc. IEEE CVPR, 2017, pp. 2097-2106.',
    'G. Huang, Z. Liu, L. van der Maaten, and K. Q. Weinberger, "Densely connected convolutional networks," in Proc. IEEE CVPR, 2017, pp. 4700-4708.',
    'V. Agbo et al., "Benchmarking MedViT and hybrid CNN-ViT architectures for multi-label thoracic disease classification," Scientific Reports, vol. 16, Art. no. 16959, 2026, doi: 10.1038/s41598-026-43282-5.',
    'Y. Gal and Z. Ghahramani, "Dropout as a Bayesian approximation: Representing model uncertainty in deep learning," in Proc. ICML, 2016, pp. 1050-1059.',
    'A. Kendall and Y. Gal, "What uncertainties do we need in Bayesian deep learning for computer vision?" in Proc. NeurIPS, 2017, pp. 5574-5584.',
    'B. Lakshminarayanan, A. Pritzel, and C. Blundell, "Simple and scalable predictive uncertainty estimation using deep ensembles," in Proc. NeurIPS, 2017, pp. 6402-6413.',
    'T. DeVries and G. W. Taylor, "Learning confidence for out-of-distribution detection in neural networks," arXiv:1802.04865, 2018.',
    'J. R. Zech, M. A. Badgeley, M. Liu, A. B. Costa, J. J. Titano, and E. K. Oermann, "Variable generalization performance of a deep learning model to detect pneumonia in chest radiographs: A cross-sectional study," PLoS Medicine, vol. 15, no. 11, e1002683, 2018.',
    'C. Guo, G. Pleiss, Y. Sun, and K. Q. Weinberger, "On calibration of modern neural networks," in Proc. ICML, 2017, pp. 1321-1330.',
    'R. R. Selvaraju, M. Cogswell, A. Das, R. Vedantam, D. Parikh, and D. Batra, "Grad-CAM: Visual explanations from deep networks via gradient-based localization," in Proc. IEEE ICCV, 2017, pp. 618-626.',
    'A. Chattopadhay, A. Sarkar, P. Howlader, and V. N. Balasubramanian, "Grad-CAM++: Generalized gradient-based visual explanations for deep convolutional networks," in Proc. IEEE WACV, 2018, pp. 839-847.',
    'T.-Y. Lin, P. Goyal, R. Girshick, K. He, and P. Dollar, "Focal loss for dense object detection," in Proc. IEEE ICCV, 2017, pp. 2980-2988.',
    'T. Ridnik, E. Ben-Baruch, N. Zamir, A. Noy, I. Friedman, M. Protter, and L. Zelnik-Manor, "Asymmetric loss for multi-label classification," in Proc. IEEE ICCV, 2021, pp. 82-91.',
    'H. Zhang, M. Cisse, Y. N. Dauphin, and D. Lopez-Paz, "mixup: Beyond empirical risk minimization," in Proc. ICLR, 2018.',
    'I. Loshchilov and F. Hutter, "Decoupled weight decay regularization," in Proc. ICLR, 2019.',
    'S. M. Pizer et al., "Adaptive histogram equalization and its variations," Computer Vision, Graphics, and Image Processing, vol. 39, no. 3, pp. 355-368, 1987.',
    'Y. Geifman and R. El-Yaniv, "Selective classification for deep neural networks," in Proc. NeurIPS, 2017, pp. 4878-4887.',
    'E. J. Topol, "High-performance medicine: The convergence of human and artificial intelligence," Nature Medicine, vol. 25, no. 1, pp. 44-56, 2019.',
]
