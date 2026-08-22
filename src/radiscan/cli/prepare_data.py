"""Build train/val/test CSVs with a *patient-level* split.

This is the split bug almost every student project ships: CheXpert contains
multiple studies per patient, so a random row-level split puts the same
patient's images in both train and validation. The model memorises the patient
and the validation AUC comes out 3-6 points too high. Splitting on patient ID
is the difference between a number you can defend and one you cannot.

    python -m radiscan.cli.prepare_data \
        --source data/raw/chexpert/CheXpert-v1.0-small/train.csv \
        --out-dir data/splits --train-size 20000 --val-size 3000
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd


def extract_patient_id(path: str) -> str:
    m = re.search(r"(patient\d+)", str(path))
    if m:
        return m.group(1)
    m = re.match(r"^(\d{8})_", Path(str(path)).name)  # NIH style
    if m:
        return m.group(1)
    return str(path)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Create patient-disjoint splits")
    p.add_argument("--source", required=True, help="master CSV (e.g. CheXpert train.csv)")
    p.add_argument("--out-dir", default="data/splits")
    p.add_argument("--train-size", type=int, default=None, help="rows (None = all remaining)")
    p.add_argument("--val-size", type=int, default=3000)
    p.add_argument("--test-size", type=int, default=3000)
    p.add_argument("--path-column", default=None)
    p.add_argument("--frontal-only", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args(argv)

    df = pd.read_csv(args.source)
    path_col = args.path_column or next(
        c for c in ("Path", "path", "image_path", "Image Index") if c in df.columns
    )

    if args.frontal_only:
        for col in ("Frontal/Lateral", "View Position", "ViewPosition"):
            if col in df.columns:
                df = df[df[col].astype(str).str.lower().str.startswith(("frontal", "pa", "ap"))]
                break

    df["_patient"] = df[path_col].map(extract_patient_id)
    patients = df["_patient"].unique()
    rng = np.random.default_rng(args.seed)
    rng.shuffle(patients)

    def take(pool: np.ndarray, target_rows: int) -> tuple[list[str], np.ndarray]:
        chosen, count, i = [], 0, 0
        while i < len(pool) and count < target_rows:
            chosen.append(pool[i])
            count += int((df["_patient"] == pool[i]).sum())
            i += 1
        return chosen, pool[i:]

    test_p, rest = take(patients, args.test_size)
    val_p, rest = take(rest, args.val_size)
    train_p = list(rest)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = {}

    for name, pats, cap in (
        ("train", train_p, args.train_size),
        ("val", val_p, None),
        ("test", test_p, None),
    ):
        subset = df[df["_patient"].isin(pats)].drop(columns=["_patient"])
        if cap is not None and cap < len(subset):
            subset = subset.sample(n=cap, random_state=args.seed)
        path = out_dir / f"{name}.csv"
        subset.to_csv(path, index=False)
        written[name] = (path, len(subset), len(pats))

    overlap = (
        (set(train_p) & set(val_p)) | (set(train_p) & set(test_p)) | (set(val_p) & set(test_p))
    )
    assert not overlap, f"Patient leakage detected: {list(overlap)[:5]}"

    print("Patient-disjoint splits written:\n")
    for name, (path, rows, pats) in written.items():
        print(f"  {name:<6} {rows:>8,} rows  {pats:>7,} patients  -> {path}")
    print("\n  Verified: zero patient overlap between splits.")
    print("\nPoint your config at these files:")
    print(f"  data.train_csv: {written['train'][0]}")
    print(f"  data.val_csv:   {written['val'][0]}")
    print(f"  data.test_csv:  {written['test'][0]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
