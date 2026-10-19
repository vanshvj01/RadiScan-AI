"""Fill the paper's results from a completed evaluation run.

    python -m radiscan.cli.evaluate --config configs/chexpert_6gb.yaml ^
        --checkpoint models/checkpoints/best_model.pth --split test
    python -m paper.fill_results --results results

Reads results/per_class_metrics.csv, results/uncertainty_quartiles.csv and
results/summary.json, then rewrites paper/content.py in place:

* Tables I-III
* the [[mean AUC]] and [[ECE]] tokens in the abstract

Still yours to write afterwards: co-authors, GPU/hours, the Section VII
discussion paragraph and the figure note.

Refuses demo/synthetic results. Those numbers describe generated shapes, not
chest radiographs; putting them in a paper would be fabrication.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
CONTENT = HERE / "content.py"

# Scientific Reports 2026 CheXpert benchmark, cited as [5] (abstract figures).
PUBLISHED = [
    ["MedViT [5]", "Transformer", "0.7511", "--", "--"],
    ["Hybrid CNN-ViT [5]", "CNN + ViT", "0.7168", "--", "--"],
]
MIN_TEST_SAMPLES = 1000  # a real CheXpert test split is far larger than this


def fmt(x, nd: int = 4) -> str:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return "--"
    return "--" if v != v else f"{v:.{nd}f}"  # NaN: no positives in split


def refuse_if_not_real(results: Path, summary: dict) -> None:
    ckpt = str(summary.get("checkpoint", ""))
    n = int(summary.get("n_samples", 0))
    reasons = []
    if "SYNTHETIC" in ckpt.upper():
        reasons.append(f"checkpoint {ckpt!r} is stamped SYNTHETIC")
    if "demo" in results.resolve().parts:
        reasons.append(f"results directory {results} is the demo output")
    if n < MIN_TEST_SAMPLES:
        reasons.append(f"only {n} test samples (a CheXpert test split has thousands)")
    if reasons:
        raise SystemExit(
            "Refusing to fill the paper:\n  - " + "\n  - ".join(reasons) + "\n"
            "These look like synthetic-demo results. Their metrics do not describe "
            "chest radiographs and must not appear in a paper. Train on CheXpert "
            "(notebooks/02_train_on_kaggle.ipynb) and evaluate that checkpoint."
        )


def build_tables(results: Path):
    summary = json.loads((results / "summary.json").read_text())
    refuse_if_not_real(results, summary)
    per_class = pd.read_csv(results / "per_class_metrics.csv")
    quart = pd.read_csv(results / "uncertainty_quartiles.csv")

    rows1 = [
        [str(r["label"]), str(int(r["support"])), fmt(r["auc"]), fmt(r["ap"]),
         fmt(r["precision"], 3), fmt(r["recall"], 3), fmt(r["f1"], 3)]
        for _, r in per_class.iterrows()
    ]
    ov = summary["overall"]
    rows1.append(["Mean", "--", fmt(ov["mean_auc"]), fmt(ov["mean_ap"]),
                  fmt(ov["mean_precision"], 3), fmt(ov["mean_recall"], 3),
                  fmt(ov["mean_f1"], 3)])

    comp = summary.get("chexpert_competition_5", {})
    rows2 = list(PUBLISHED) + [
        ["Ours (temp. scaled)", "DenseNet-121", fmt(ov["mean_auc"]),
         fmt(comp.get("mean_auc")), fmt(summary.get("ece"))]
    ]

    rows3 = [
        [str(r["quartile"]), fmt(r["mean_uncertainty"]), str(int(r["n"])), fmt(r["accuracy"])]
        for _, r in quart.iterrows()
    ]
    return {"table1": rows1, "table2": rows2, "table3": rows3}, summary


def _replace_rows(src: str, key: str, rows: list) -> str | None:
    """Replace one table's rows expression, found by bracket balancing.

    Table I builds its rows as a concatenation ([...] + [[...]]), so a regex
    that stops at the first ']' would truncate it into a syntax error. And the
    table keys also appear in SECTIONS, so the search starts at TABLES.
    """
    base = src.find("TABLES = {")
    anchor = src.find(f'"{key}"', base)
    rk = src.find('"rows"', anchor)
    if base == -1 or anchor == -1 or rk == -1:
        return None
    start = src.index("[", src.index(":", rk))
    i, depth = start, 0
    while i < len(src):
        ch = src[i]
        if ch in "[{(":
            depth += 1
        elif ch in "]})":
            depth -= 1
            if depth == 0:
                j = i + 1
                while j < len(src) and src[j] in " \t\n":
                    j += 1
                if j < len(src) and src[j] == "+":
                    i = j  # concatenation continues
                else:
                    break
        i += 1
    indent = " " * len(src[:start].rsplit("\n", 1)[-1])
    body = "[\n" + "".join(
        indent + "    [" + ", ".join(repr(c) for c in row) + "],\n" for row in rows
    ) + indent + "]"
    return src[:start] + body + src[i + 1:]


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Fill paper results from an eval run")
    p.add_argument("--results", default="results", type=Path)
    args = p.parse_args(argv)

    tables, summary = build_tables(args.results)
    src = CONTENT.read_text(encoding="utf-8")
    for key, rows in tables.items():
        new = _replace_rows(src, key, rows)
        if new is None:
            print(f"  ! could not locate {key}; fill it by hand")
        else:
            src = new
    src = src.replace("[[mean AUC]]", fmt(summary["overall"]["mean_auc"]))
    src = src.replace("[[ECE]]", fmt(summary.get("ece")))
    compile(src, str(CONTENT), "exec")  # never write a broken content.py
    CONTENT.write_text(src, encoding="utf-8")

    print(f"Filled Tables I-III and the abstract in {CONTENT}")
    print(f"  checkpoint : {summary.get('checkpoint')}")
    print(f"  test images: {summary.get('n_samples')}")
    print(f"  mean AUC   : {summary['overall']['mean_auc']:.4f}")
    print(f"  ECE        : {summary.get('ece'):.4f}")
    print(f"  uncertainty informative (Table III monotone): {summary.get('uncertainty_is_informative')}")
    print("\nRebuild:  python -m paper.build_docx  &&  python -m paper.build_tex")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
