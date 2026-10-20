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
    {"name": "Vansh R. Vijayvargiya", "affil": 1},
    {"name": "Vaibhav Dave", "affil": 1},
    {"name": "Uzair Khan", "affil": 1},
    {"name": "Aryan Lodha", "affil": 1},
    {"name": "Prof. Purnima Ahirao", "affil": 1},
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
    "system attains a mean AUC-ROC of 0.7423 across fourteen findings and "
    "an expected calibration error of 0.3809. Accuracy falls monotonically "
    "from 85.6 to 66.5 percent across quartiles of increasing posterior "
    "uncertainty, indicating that the uncertainty estimates carry information "
    "about model error."
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
            "No imaging study is ordered more often than the plain chest film, in "
            "India as elsewhere. Demand rises with every new district hospital and diagnostic camp, but"
            " the number of people qualified to report these films grows far more "
            "slowly. In much "
            "of India, a chest radiograph acquired at a primary or secondary centre may "
            "wait days for a formal report, and the delay falls hardest on exactly the "
            "findings for which time matters, such as pneumothorax and pulmonary oedema. "
            "Automated pre-reading that orders a reporting queue by clinical urgency is "
            "therefore a more tractable and more immediately useful objective than "
            "autonomous diagnosis.",
            "Since CheXNet [1] put a convolutional network level with radiologists "
            "on pneumonia, dozens of groups have posted strong ranking scores on "
            "the public chest film collections. Comparatively little of that "
            "work has reached clinical deployment. Among the reasons, one is "
            "methodological rather than organisational: the standard output of these "
            "systems is a vector of probabilities with no attached measure of "
            "reliability. A prediction of 0.62 accompanied by a posterior standard "
            "deviation of 0.02 and one accompanied by a standard deviation of 0.30 are "
            "different clinical objects, and only the first is actionable. A system that "
            "reports the point estimate alone provides the reader no means of "
            "distinguishing them.",
            "Two further properties are commonly absent. First, the probabilities such networks print are usually too high: "
            "among the cases a model scores at 0.90, noticeably fewer than nine in "
            "ten turn out positive [11]. For a "
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
                    "Two public corpora, ChestX-ray14 [3] and CheXpert [2], are what "
                    "turned chest radiograph classification into a problem that could be "
                    "attacked with supervised deep learning at scale. On the first of "
                    "these, CheXNet [1] used a densely connected network [4] and matched "
                    "the F1 of practising radiologists for pneumonia. Later systems have "
                    "added attention, raised the input resolution, or replaced the "
                    "convolutional trunk with a transformer; one recent CheXpert benchmark gives macro "
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
                    "Leaving the dropout masks active while predicting, and repeating the "
                    "forward pass, turns one trained network into a sampler over plausible "
                    "networks [6]; this is the attraction of the method for us, because the"
                    " trained model itself does not change. It helps to keep apart two reasons a prediction can be shaky [7]: the "
                    "model may simply not have seen enough similar films, or the film "
                    "itself may be too blurred, rotated or cluttered for anyone to be sure. Training several independent networks [8] gives better "
                    "estimates, but multiplies both training and inference cost by the "
                    "size of the ensemble. A network can also be taught to report its own "
                    "confidence by being offered hints at a price [9]; our aleatoric head "
                    "follows that idea. Within chest radiography, uncertainty has been used "
                    "mainly as a displayed quantity; its use to drive an explicit "
                    "referral decision is less common.",
                ],
            },
            {
                "heading": "Calibration and Explanation",
                "body": [
                    "When a network's stated confidence runs ahead of its hit rate, Guo and"
                    " colleagues [11] showed that one fitted divisor on the logits pulls "
                    "the two back together without reordering any case. To see where on the film a score came from, we use gradient-weighted "
                    "activation maps [12], [13], which colour the regions whose features "
                    "most raised that finding's score. Such maps matter here because chest radiograph models have "
                    "been shown to key on where an image came from, for instance the "
                    "scanner or a laterality marker, rather than on disease [10], a finding that motivates using saliency to audit "
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
                    "All experiments use CheXpert [2], released by Stanford with roughly "
                    "two hundred and twenty thousand films taken from some sixty-five "
                    "thousand people. Nobody hand-labelled those films; each of the fourteen findings was "
                    "read off the written report by a rule-based text parser. The parser can therefore say yes, no, maybe, or nothing at all about "
                    "any finding. We restrict to frontal "
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
                    "state [2]. We turn every \u201cmaybe\u201d into a soft target, a random value "
                    "between 0.05 and 0.85, instead of forcing it to 0 or 1. This expresses "
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
                    "Each film is reduced to one grey channel, given local contrast "
                    "enhancement (CLAHE), copied into the three channels the ImageNet-"
                    "trained trunk expects, and scaled to that trunk's input statistics. Training augmentation consists of random "
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
                    "Feature extraction is done by a DenseNet-121 [4] that starts from "
                    "ImageNet weights; its roughly eight million weights summarise each "
                    "film as a vector of 1024 numbers. "
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
                    "Plain cross-entropy treats every absent finding as equally "
                    "informative, so when a disease appears in fewer than two films in a "
                    "hundred, almost all of the learning signal comes from obvious "
                    "negatives. Focal Loss [14] quietens examples the model already gets right, but it"
                    " does so for positives and negatives alike. Asymmetric Loss [15] lets the two sides differ, and we set them far "
                    "apart: negatives are damped with an exponent of 4, positives not at "
                    "all, and any negative already scored below 0.05 contributes nothing. Positive examples, which are scarce, therefore "
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
                    "After training we fit one number, the temperature T, on the validation"
                    " films: logits are divided by T, and T is chosen to make the observed "
                    "labels as likely as possible [11]. Dividing by a positive constant cannot change which case scores higher"
                    " than which, so the AUC must come out exactly the same before and "
                    "after. Any change in AUC "
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
                    "CheXpert contains multiple studies per patient. If films were split at random, one person's follow-up X-rays could sit"
                    " on both sides of the split, and a high test score might reflect "
                    "remembering a ribcage rather than spotting a disease. We split by patient identifier into training, "
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
                    "Weights are updated with AdamW at a learning rate of 1e-4 and weight "
                    "decay of 1e-4; the freshly created output heads learn ten times faster"
                    " than the pretrained trunk, which already knows useful image features. The rate ramps up over the first two epochs and then falls along a "
                    "cosine curve, and the trunk stays fixed for epoch one so the new heads"
                    " can settle first. "
                    "Mixed-precision arithmetic is enabled. A slowly updated running average of the weights (decay 0.999) is kept "
                    "alongside them, and it is this averaged copy that we validate and "
                    "save.",
                    "Multi-label mixup is applied with alpha = 0.2. Targets are "
                    "combined by element-wise maximum rather than convex interpolation: "
                    "if either source image exhibits a finding, the blended image "
                    "plausibly exhibits it, whereas interpolation would assign a target "
                    "of 0.5 to a clearly present finding.",
                    "From the frontal CheXpert images we draw patient-disjoint subsets of "
                    "40,000 training, 4,000 validation and 4,000 test images, a scale chosen "
                    "to fit a free cloud GPU allocation. "
                    "Training is capped at 25 epochs and would stop early if validation AUC"
                    " failed to improve for six epochs in a row. Batch size was determined "
                    "empirically by measuring peak allocation across a real forward "
                    "and backward pass. We note that on the Windows display driver "
                    "model, an allocation exceeding physical device memory does not "
                    "raise an out-of-memory error but is transparently served from "
                    "host memory across the bus, so absence of failure is not evidence "
                    "that a configuration fits; peak allocation must be compared "
                    "against physical capacity directly. Training was performed on "
                    "an NVIDIA Tesla T4 (16 GB) on Kaggle and took approximately 8.8 "
                    "hours for 25 epochs.",
                ],
            },
            {
                "heading": "Metrics",
                "body": [
                    "Results are given as AUC averaged over all fourteen findings and over "
                    "the five used in the CheXpert competition; average precision, which "
                    "drops sharply when a rare finding is being guessed; precision, recall "
                    "and F1 at the tuned cut-offs; the Brier score; calibration error over "
                    "fifteen bins; and, for every finding, how many positive films the test"
                    " set held.",
                    "Two analyses specifically evaluate the uncertainty machinery. The "
                    "first stratifies predictions into quartiles by posterior standard "
                    "deviation and reports accuracy within each; a monotone decrease "
                    "indicates that the estimates carry information. The second asks how accuracy changes as the system hands back more and"
                    " more of its shakiest calls to a human reader.",
                ],
            },
        ],
    },
    # ------------------------------------------------------------------ VII
    {
        "heading": "Results and Discussion",
        "body": [
            "Table I and Fig. 1 report per-class discrimination on the held-out, "
            "patient-disjoint test split of 4,000 images. Performance follows lesion "
            "conspicuity closely. The highest AUCs are for No Finding (0.863), "
            "pleural effusion (0.828), pneumothorax (0.820), support devices (0.806) "
            "and cardiomegaly (0.802): large or high-contrast findings. The lowest "
            "are enlarged cardiomediastinum (0.607), atelectasis (0.647) and "
            "consolidation (0.650), which are also among the CheXpert labels with "
            "the highest rates of uncertain annotation, so label noise as well as "
            "visual subtlety plausibly limits them. We treat the pneumothorax figure with caution: a collapsed lung that "
            "has already been treated has a drain in it, and a network can earn "
            "credit for finding the tube instead of the air [10].",
            "The gap between AUC and average precision is largest for the rare "
            "findings, which is the reason AP is reported. Pleural other reaches an "
            "AUC of 0.734 with only 71 positives, but its AP is 0.062 and its F1 "
            "0.066: the model ranks these cases somewhat above negatives, yet almost "
            "none of its positive calls at the tuned threshold are correct. A "
            "single-number AUC summary would have hidden this.",
            "Table II places the system against published CheXpert results. The mean "
            "AUC over fourteen findings is 0.7423 and over the five competition "
            "findings 0.7435, slightly below MedViT (0.7511) and above the hybrid "
            "CNN-ViT (0.7168) reported in [5]. The comparison is indicative only: "
            "the published figures come from a different split, and our model was "
            "trained on a 40,000-image subset rather than the full training set. We "
            "therefore do not claim a discriminative improvement; the contribution "
            "of this work is the uncertainty and deferral machinery built around a "
            "competitive baseline.",
            "Calibration is the weakest result. The expected calibration error after "
            "temperature scaling is 0.381, far from well calibrated. A likely "
            "explanation is that two design choices made to detect rare findings, "
            "Asymmetric Loss and inverse-frequency balanced sampling, both shift "
            "predicted probabilities upwards relative to true prevalence, and a "
            "single global temperature can rescale confidence but cannot correct a "
            "prior shift that differs from class to class. The probabilities "
            "should therefore be read as rankings and inputs to the per-class "
            "thresholds, not as calibrated risks. Per-class calibration, for "
            "example a temperature or isotonic mapping fitted for each finding, is "
            "the direct remedy and is left to future work.",
            "Table III and Fig. 2 evaluate the uncertainty machinery directly. Accuracy falls "
            "monotonically with posterior standard deviation, from 85.6 percent in "
            "the most certain quartile to 81.9, 77.1 and finally 66.5 percent in the "
            "least certain, a gap of 19.1 percentage points. The Monte Carlo Dropout "
            "estimates therefore carry real information about which predictions "
            "are wrong, which is the condition under which the deferral policy of "
            "Section IV-G is justified: referring the least certain predictions for "
            "review removes a disproportionate share of the errors. The absolute "
            "spread of the posterior is small (mean standard deviation 0.002 to "
            "0.0065), so it is the ordering of predictions by uncertainty, rather "
            "than the raw width of an interval, that carries the signal.",
        ],
        "tables": ["table1", "table2", "table3"],
        "figures": ["fig_auc", "fig_unc"],
    },
    # ------------------------------------------------------------------ VIII
    {
        "heading": "Limitations",
        "body": [
            "The model was trained on 40,000 of the available frontal images and its "
            "probabilities are poorly calibrated (ECE 0.381); both bound how far the "
            "reported numbers generalise and how literally a probability can be read.",
            "Because the labels were parsed from report text and never checked by a"
            " radiologist, some are simply wrong, and no choice of model can score "
            "above what that noise allows.",
            "Every film came from one hospital system in California. We present no evidence "
            "of generalisation to different acquisition equipment, protocols or "
            "patient populations, and the findings of Zech et al. [10] indicate that "
            "such transfer cannot be assumed. External validation on an Indian "
            "hospital cohort is necessary before any claim of applicability in that "
            "setting.",
            "Only frontal projections are used. Lateral views and prior comparison "
            "studies, both routine in clinical reporting, are not modelled. No clinical "
            "metadata, such as presenting symptoms, history or laboratory values, is "
            "incorporated; a radiological reading is never image-only.",
            "Dropout sampling only approximates a true posterior, and we stopped at"
            " twenty passes because more would slow each prediction, not because "
            "twenty was shown to be enough. Training several separate networks would probably estimate uncertainty"
            " better, at several times the cost.",
            "A heatmap shows where the score was sensitive, which is not the same "
            "as showing where the disease is. "
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
            "RadiScan was built on one conviction: a tool meant to help a doctor "
            "read chest films must say how sure it is, not just what it thinks. The system estimates epistemic uncertainty by "
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
                    ['No Finding', '375', '0.8631', '0.4218', '0.437', '0.491', '0.462'],
                    ['Enlarged Cardiomed.', '270', '0.6073', '0.1197', '0.097', '0.478', '0.162'],
                    ['Cardiomegaly', '490', '0.8018', '0.4464', '0.385', '0.543', '0.450'],
                    ['Lung Opacity', '2053', '0.6934', '0.6675', '0.587', '0.924', '0.718'],
                    ['Lung Lesion', '191', '0.7610', '0.1386', '0.137', '0.277', '0.183'],
                    ['Edema', '1052', '0.7916', '0.5486', '0.485', '0.667', '0.562'],
                    ['Consolidation', '513', '0.6497', '0.1875', '0.188', '0.517', '0.276'],
                    ['Pneumonia', '246', '0.6933', '0.1492', '0.173', '0.358', '0.233'],
                    ['Atelectasis', '888', '0.6468', '0.3153', '0.286', '0.771', '0.417'],
                    ['Pneumothorax', '420', '0.8200', '0.3939', '0.350', '0.602', '0.443'],
                    ['Pleural Effusion', '1765', '0.8278', '0.7785', '0.652', '0.860', '0.741'],
                    ['Pleural Other', '71', '0.7341', '0.0617', '0.150', '0.042', '0.066'],
                    ['Fracture', '174', '0.6962', '0.1571', '0.203', '0.172', '0.186'],
                    ['Support Devices', '2267', '0.8060', '0.8257', '0.718', '0.886', '0.793'],
                    ['Mean', '--', '0.7423', '0.3723', '0.346', '0.542', '0.407'],
                ],
    },
    "table2": {
        "caption": "Aggregate performance and comparison with published CheXpert results",
        "columns": ["Method", "Backbone", "Mean AUC", "CheXpert-5 AUC", "ECE"],
        "widths": [1700, 1300, 1000, 1100, 900],
        "rows": [
                    ['MedViT [5]', 'Transformer', '0.7511', '--', '--'],
                    ['Hybrid CNN-ViT [5]', 'CNN + ViT', '0.7168', '--', '--'],
                    ['Ours (temp. scaled)', 'DenseNet-121', '0.7423', '0.7435', '0.3809'],
                ],
    },
    "table3": {
        "caption": "Accuracy stratified by posterior standard deviation quartile",
        "columns": ["Quartile", "Mean sigma", "n", "Accuracy"],
        "widths": [1600, 1200, 1000, 1200],
        "rows": [
                    ['Q1 (most confident)', '0.0020', '14000', '0.8559'],
                    ['Q2', '0.0026', '14000', '0.8194'],
                    ['Q3', '0.0034', '14000', '0.7706'],
                    ['Q4 (least confident)', '0.0065', '14000', '0.6651'],
                ],
    },
}

# Figures live in paper/figures/, generated from results/ by paper/make_figures.py.
FIGURES = {
    "fig_auc": {
        "path": "figures/fig_per_class_auc.png",
        "caption": "Per-class AUC-ROC on the test split, with positive support n. "
        "The dashed line marks the mean over fourteen findings (0.742).",
    },
    "fig_unc": {
        "path": "figures/fig_uncertainty_quartiles.png",
        "caption": "Accuracy by quartile of posterior standard deviation. Accuracy "
        "falls monotonically as uncertainty rises, from 85.6% to 66.5%.",
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
