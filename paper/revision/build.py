"""Rebuild the revised paper and the response to reviewers from the Kaggle bundle.

    python paper/revision/build.py --bundle reviewer_bundle.zip [--orig results/chexpert]

Needs Python (pandas, Pillow) and Node.js with the ``docx`` package
(``cd paper && npm install docx``; on Windows set NODE_PATH to paper/node_modules).
Writes paper/out/RadiScan-paper-revised.docx and
paper/out/RadiScan-response-to-reviewers.docx.
"""
import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", required=True, help="reviewer_bundle.zip or its extracted folder")
    ap.add_argument("--orig", default=None, help="folder with the first run's summary.json (optional)")
    ap.add_argument("--out", default=str(HERE.parent / "out"))
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="radiscan_paper_"))
    bundle = Path(args.bundle)
    if bundle.suffix == ".zip":
        zipfile.ZipFile(bundle).extractall(work / "bundle")
        bundle = work / "bundle"
    py = sys.executable
    extra = [args.orig] if args.orig else []
    subprocess.run([py, str(HERE / "content.py"), str(bundle), str(work / "paper_spec.json"), *extra], check=True)
    subprocess.run(["node", str(HERE / "render.js"), str(work / "paper_spec.json"), str(work / "paper_raw.docx")], check=True)
    subprocess.run([py, str(HERE / "inject.py"), str(work / "paper_raw.docx"), str(work / "equations.json"), str(out / "RadiScan-paper-revised.docx")], check=True)
    subprocess.run([py, str(HERE / "response.py"), str(bundle), str(work / "x.json"), *extra], check=True)
    subprocess.run(["node", str(HERE / "render.js"), str(work / "response_spec.json"), str(out / "RadiScan-response-to-reviewers.docx")], check=True)
    shutil.rmtree(work, ignore_errors=True)
    print("written:", out / "RadiScan-paper-revised.docx", "and", out / "RadiScan-response-to-reviewers.docx")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
