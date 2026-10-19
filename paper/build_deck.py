"""Build the presentation deck.

    python -m paper.build_deck                       # placeholders on result slides
    python -m paper.build_deck --results results     # real numbers + native charts

Result slides are filled only from a real CheXpert evaluation; the same
refusal rules as fill_results apply, so demo numbers can never reach a slide.
Needs Node.js with pptxgenjs, react, react-dom, react-icons and sharp.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"


def gather(results: Path | None) -> dict:
    if results is None:
        return {"real": False}
    import pandas as pd

    from .fill_results import refuse_if_not_real

    summary = json.loads((results / "summary.json").read_text())
    refuse_if_not_real(results, summary)
    pc = pd.read_csv(results / "per_class_metrics.csv")
    q = pd.read_csv(results / "uncertainty_quartiles.csv")
    sel = None
    if (results / "selective_prediction.csv").exists():
        sel = pd.read_csv(results / "selective_prediction.csv").to_dict("records")
    return {
        "real": True,
        "n": int(summary["n_samples"]),
        "mean_auc": summary["overall"]["mean_auc"],
        "mean_ap": summary["overall"]["mean_ap"],
        "comp_auc": summary["chexpert_competition_5"]["mean_auc"],
        "ece": summary["ece"],
        "monotone": bool(summary.get("uncertainty_is_informative")),
        "per_class": [
            {"label": r["label"], "auc": None if r["auc"] != r["auc"] else float(r["auc"]),
             "support": int(r["support"])}
            for _, r in pc.iterrows()
        ],
        "quartiles": [{"q": str(r["quartile"]).split(" ")[0], "acc": float(r["accuracy"])}
                      for _, r in q.iterrows()],
        "selective": sel,
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--results", type=Path, default=None)
    args = p.parse_args(argv)
    OUT.mkdir(exist_ok=True)
    data = OUT / "_deck.json"
    data.write_text(json.dumps(gather(args.results)), encoding="utf-8")
    env = dict(os.environ)
    subprocess.run(["node", str(HERE / "deck.js"), str(data), str(OUT / "RadiScan-presentation.pptx")],
                   check=True, env=env)
    data.unlink()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
