"""The paper must never be filled from synthetic-demo results."""

import json

import pandas as pd
import pytest

from paper import fill_results


def _write(tmp, n_samples, ckpt="models/checkpoints/best_model.pth"):
    pd.DataFrame(
        [
            {
                "label": "Edema",
                "support": 5,
                "n": n_samples,
                "auc": 0.8,
                "ap": 0.5,
                "precision": 0.5,
                "recall": 0.5,
                "f1": 0.5,
            }
        ]
    ).to_csv(tmp / "per_class_metrics.csv", index=False)
    pd.DataFrame([{"quartile": "Q1", "n": 10, "accuracy": 0.9, "mean_uncertainty": 0.01}]).to_csv(
        tmp / "uncertainty_quartiles.csv", index=False
    )
    (tmp / "summary.json").write_text(
        json.dumps(
            {
                "checkpoint": ckpt,
                "n_samples": n_samples,
                "ece": 0.05,
                "overall": {
                    k: 0.8
                    for k in ("mean_auc", "mean_ap", "mean_precision", "mean_recall", "mean_f1")
                },
                "chexpert_competition_5": {"mean_auc": 0.85},
            }
        )
    )


def test_refuses_small_demo_split(tmp_path):
    _write(tmp_path, n_samples=60)
    with pytest.raises(SystemExit, match="Refusing"):
        fill_results.build_tables(tmp_path)


def test_refuses_synthetic_checkpoint(tmp_path):
    _write(tmp_path, n_samples=5000, ckpt="SYNTHETIC-DEMO-best.pth")
    with pytest.raises(SystemExit, match="SYNTHETIC"):
        fill_results.build_tables(tmp_path)


def test_rows_replacement_keeps_content_valid(tmp_path):
    _write(tmp_path, n_samples=5000)
    tables, _ = fill_results.build_tables(tmp_path)
    src = fill_results.CONTENT.read_text(encoding="utf-8")
    for key, rows in tables.items():
        src = fill_results._replace_rows(src, key, rows)
        assert src is not None
    ns: dict = {}
    exec(compile(src, "content.py", "exec"), ns)
    assert ns["TABLES"]["table1"]["rows"][0][0] == "Edema"
    assert ns["TABLES"]["table2"]["rows"][-1][2] == "0.8000"
    assert ns["TABLES"]["table3"]["rows"][0][0] == "Q1"
