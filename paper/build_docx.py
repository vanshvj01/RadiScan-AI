"""Render the conference paper to IEEE two-column .docx.

    python -m paper.build_docx       -> paper/out/RadiScan-paper.docx

Everything renders from paper/content.py so the .docx and the LaTeX build
cannot drift. Any string containing '[[' is an unresolved placeholder and is
highlighted yellow, so a reviewer can see at a glance what is still missing.
Needs Node.js with the `docx` package (npm install docx).
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

from . import content as C

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
ROMAN = ["I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X", "XI", "XII"]
ALPHA = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
COL_DXA = 4700  # usable width of one body column
PLACEHOLDER = re.compile(r"\[\[[^\]]*\]\]")


def runs(text: str, **style) -> list[dict]:
    """Split text into runs, marking placeholder spans for yellow highlight."""
    out, last = [], 0
    for m in PLACEHOLDER.finditer(text):
        if m.start() > last:
            out.append({"text": text[last : m.start()], **style})
        out.append({"text": m.group(0), "highlight": "yellow", **style})
        last = m.end()
    if last < len(text) or not out:
        out.append({"text": text[last:], **style})
    return out


def build_spec() -> dict:
    body: list[dict] = []
    body.append({"kind": "title", "runs": runs(C.TITLE)})
    body.append({"kind": "authors", "runs": runs(", ".join(a["name"] for a in C.AUTHORS))})
    body.append({"kind": "affil", "runs": runs(C.AFFILIATION)})
    body.append({"kind": "affil", "runs": runs(", ".join(a["email"] for a in C.AUTHORS))})
    body.append({"kind": "abstract", "label": "Abstract—", "runs": runs(C.ABSTRACT)})
    body.append({"kind": "abstract", "label": "Index Terms—", "runs": runs(C.KEYWORDS)})

    for i, sec in enumerate(C.SECTIONS):
        body.append({"kind": "h1", "runs": runs(f"{ROMAN[i]}. {sec['heading'].upper()}")})
        for para in sec.get("body", []):
            body.append({"kind": "p", "runs": runs(para)})
        for j, sub in enumerate(sec.get("subsections", [])):
            body.append({"kind": "h2", "runs": runs(f"{ALPHA[j]}. {sub['heading']}")})
            for para in sub.get("body", []):
                body.append({"kind": "p", "runs": runs(para)})
        for tkey in sec.get("tables", []):
            t = C.TABLES[tkey]
            n = ROMAN[list(C.TABLES).index(tkey)]
            # Scale authored widths to one IEEE body column so cells never run
            # off the column edge.
            scale = COL_DXA / sum(t["widths"])
            widths = [max(int(w * scale), 300) for w in t["widths"]]
            body.append(
                {
                    "kind": "table",
                    "caption_runs": runs(f"TABLE {n}: {t['caption']}"),
                    "columns": t["columns"],
                    "widths": widths,
                    "font": 13 if len(t["columns"]) > 5 else 15,
                    "rows": [[runs(c) for c in row] for row in t["rows"]],
                }
            )

    body.append({"kind": "h1", "runs": runs("REFERENCES")})
    for i, ref in enumerate(C.REFERENCES, 1):
        body.append({"kind": "ref", "runs": runs(f"[{i}] {ref}")})
    return {"body": body}


JS = r"""
const d = require('docx');
const fs = require('fs');
const spec = JSON.parse(fs.readFileSync(process.argv[2], 'utf8'));
const OUT = process.argv[3];
const FONT = 'Times New Roman';
const mkRuns = (rs, o = {}) => rs.map(r => new d.TextRun({
  text: r.text, font: FONT, size: o.size || 20,
  bold: r.bold || o.bold || false, italics: r.italics || o.italics || false,
  allCaps: o.allCaps || false, highlight: r.highlight || undefined,
}));
const front = [], children = [];
for (const b of spec.body) {
  if (b.kind === 'title') {
    front.push(new d.Paragraph({ alignment: d.AlignmentType.CENTER, spacing: { after: 120 },
      children: mkRuns(b.runs, { size: 48 }) }));
  } else if (b.kind === 'authors') {
    front.push(new d.Paragraph({ alignment: d.AlignmentType.CENTER, spacing: { after: 40 },
      children: mkRuns(b.runs, { size: 24 }) }));
  } else if (b.kind === 'affil') {
    front.push(new d.Paragraph({ alignment: d.AlignmentType.CENTER, spacing: { after: 40 },
      children: mkRuns(b.runs, { size: 18, italics: true }) }));
  } else if (b.kind === 'abstract') {
    children.push(new d.Paragraph({ alignment: d.AlignmentType.JUSTIFIED, spacing: { before: 160, after: 80 },
      children: [new d.TextRun({ text: b.label, font: FONT, size: 18, bold: true, italics: true }),
                 ...mkRuns(b.runs, { size: 18, bold: true })] }));
  } else if (b.kind === 'h1') {
    children.push(new d.Paragraph({ alignment: d.AlignmentType.CENTER, spacing: { before: 200, after: 100 },
      children: mkRuns(b.runs, { size: 20, allCaps: true }) }));
  } else if (b.kind === 'h2') {
    children.push(new d.Paragraph({ spacing: { before: 120, after: 60 },
      children: mkRuns(b.runs, { size: 20, italics: true }) }));
  } else if (b.kind === 'p') {
    children.push(new d.Paragraph({ alignment: d.AlignmentType.JUSTIFIED,
      indent: { firstLine: 200 }, spacing: { after: 60 }, children: mkRuns(b.runs) }));
  } else if (b.kind === 'ref') {
    children.push(new d.Paragraph({ alignment: d.AlignmentType.JUSTIFIED, spacing: { after: 20 },
      indent: { left: 260, hanging: 260 }, children: mkRuns(b.runs, { size: 16 }) }));
  } else if (b.kind === 'table') {
    children.push(new d.Paragraph({ alignment: d.AlignmentType.CENTER, spacing: { before: 160, after: 60 },
      keepNext: true, children: mkRuns(b.caption_runs, { size: 16 }) }));
    const total = b.widths.reduce((a, c) => a + c, 0);
    const cell = (rs, bold, w) => new d.TableCell({
      width: { size: w, type: d.WidthType.DXA },
      margins: { top: 20, bottom: 20, left: 60, right: 60 },
      shading: bold ? { type: d.ShadingType.CLEAR, fill: 'E8E8E8' } : undefined,
      children: [new d.Paragraph({ keepNext: true, children: mkRuns(rs, { size: b.font, bold }) })],
    });
    const header = new d.TableRow({ tableHeader: true,
      children: b.columns.map((c, i) => cell([{ text: c }], true, b.widths[i])) });
    const rows = b.rows.map(r => new d.TableRow({ cantSplit: true, children: r.map((rs, i) => cell(rs, false, b.widths[i])) }));
    children.push(new d.Table({ columnWidths: b.widths, width: { size: total, type: d.WidthType.DXA },
      rows: [header, ...rows] }));
    children.push(new d.Paragraph({ text: '', spacing: { after: 80 } }));
  }
}
const PAGE = { size: { width: 12240, height: 15840 },
               margin: { top: 1080, bottom: 1440, left: 1080, right: 1080 } };
const doc = new d.Document({ sections: [
  // IEEE title block spans the full width, so it is its own one-column section.
  { properties: { page: PAGE, column: { count: 1 } }, children: front },
  // CONTINUOUS so the two-column body starts on the same page.
  { properties: { type: d.SectionType.CONTINUOUS, page: PAGE,
      column: { count: 2, space: 360, equalWidth: true } }, children },
]});
d.Packer.toBuffer(doc).then(buf => { fs.writeFileSync(OUT, buf); console.log('wrote ' + OUT); });
"""


def main() -> int:
    OUT.mkdir(exist_ok=True)
    spec = build_spec()
    spec_path, js_path = OUT / "_spec.json", OUT / "_build.js"
    spec_path.write_text(json.dumps(spec), encoding="utf-8")
    js_path.write_text(JS, encoding="utf-8")
    docx = OUT / "RadiScan-paper.docx"
    subprocess.run(["node", str(js_path), str(spec_path), str(docx)], check=True)
    spec_path.unlink()
    js_path.unlink()
    n = len(PLACEHOLDER.findall(json.dumps(spec)))
    print(f"{n} placeholders highlighted yellow")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
