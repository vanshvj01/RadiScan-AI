"""Render the conference paper to IEEEtran LaTeX.

    python -m paper.build_tex        -> paper/out/RadiScan-paper.tex + refs.bib

Same content source as build_docx.py, so the two outputs cannot drift.
Unresolved placeholders ('[[...]]') are wrapped in \\todo{} and render red.

Compile on Overleaf (IEEE conference template, pdfLaTeX), or locally:
    pdflatex RadiScan-paper && bibtex RadiScan-paper && pdflatex RadiScan-paper && pdflatex RadiScan-paper
"""

from __future__ import annotations

import re
from pathlib import Path

from . import content as C

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
PLACEHOLDER = re.compile(r"\[\[([^\]]*)\]\]")

# Backslash first: everything after it introduces backslashes of its own.
ESCAPES = [
    ("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"), ("$", r"\$"),
    ("#", r"\#"), ("_", r"\_"), ("{", r"\{"), ("}", r"\}"),
    ("~", r"\textasciitilde{}"), ("^", r"\textasciicircum{}"),
]


def esc(text: str) -> str:
    for a, b in ESCAPES:
        text = text.replace(a, b)
    return text.replace('"', "''")


def tex(text: str) -> str:
    """Escape, re-wrap placeholders in \\todo{}, turn [n] into \\cite."""
    marked = PLACEHOLDER.sub(lambda m: "\x00" + m.group(1) + "\x01", text)
    out = esc(marked).replace("\x00", r"\todo{").replace("\x01", "}")
    return re.sub(
        r"\[(\d+(?:,\s*\d+)*)\]",
        lambda m: r"\cite{ref" + m.group(1).replace(", ", ",ref") + "}",
        out,
    )


def table_tex(key: str) -> str:
    t = C.TABLES[key]
    lines = [
        r"\begin{table}[!t]", r"\centering",
        r"\caption{" + tex(t["caption"]) + "}", r"\label{tab:" + key + "}",
        r"\footnotesize", r"\setlength{\tabcolsep}{3pt}",
        r"\begin{tabular}{l" + "c" * (len(t["columns"]) - 1) + "}", r"\toprule",
        " & ".join(r"\textbf{" + esc(c) + "}" for c in t["columns"]) + r" \\",
        r"\midrule",
    ]
    lines += [" & ".join(tex(c) for c in row) + r" \\" for row in t["rows"]]
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(lines)


def build() -> tuple[str, str]:
    L = [
        r"\documentclass[conference]{IEEEtran}",
        r"\IEEEoverridecommandlockouts",
        r"\usepackage{cite}", r"\usepackage{amsmath,amssymb,amsfonts}",
        r"\usepackage{graphicx}", r"\usepackage{booktabs}", r"\usepackage{xcolor}",
        r"\usepackage[hidelinks]{hyperref}", r"\usepackage{textcomp}", "",
        r"% Unresolved placeholders. Delete this macro once nothing uses it.",
        r"\newcommand{\todo}[1]{\textcolor{red}{\textbf{[#1]}}}", "",
        r"\begin{document}", "",
        r"\title{" + tex(C.TITLE) + "}", "", r"\author{",
    ]
    blocks = [
        r"\IEEEauthorblockN{" + tex(a["name"]) + "}\n"
        + r"\IEEEauthorblockA{\textit{" + tex(C.AFFILIATION) + "} \\\\ " + tex(a["email"]) + "}"
        for a in C.AUTHORS
    ]
    L += ["\n\\and\n".join(blocks), "}", "", r"\maketitle", "",
          r"\begin{abstract}", tex(C.ABSTRACT), r"\end{abstract}", "",
          r"\begin{IEEEkeywords}", tex(C.KEYWORDS), r"\end{IEEEkeywords}", ""]
    for sec in C.SECTIONS:
        L.append(r"\section{" + tex(sec["heading"]) + "}")
        for para in sec.get("body", []):
            L += [tex(para), ""]
        for sub in sec.get("subsections", []):
            L.append(r"\subsection{" + tex(sub["heading"]) + "}")
            for para in sub.get("body", []):
                L += [tex(para), ""]
        for tkey in sec.get("tables", []):
            L.append(table_tex(tkey))
    L += [r"\bibliographystyle{IEEEtran}", r"\bibliography{refs}", r"\end{document}"]

    # @misc with a verbatim note reproduces each hand-formatted reference
    # exactly; re-parsing them into fields would only introduce errors.
    bib = "\n".join(
        "@misc{ref%d,\n  note = {%s}\n}\n" % (i, esc(PLACEHOLDER.sub(r"\1", r)))
        for i, r in enumerate(C.REFERENCES, 1)
    )
    return "\n".join(L) + "\n", bib


def main() -> int:
    OUT.mkdir(exist_ok=True)
    doc, bib = build()
    (OUT / "RadiScan-paper.tex").write_text(doc, encoding="utf-8")
    (OUT / "refs.bib").write_text(bib, encoding="utf-8")
    print(f"wrote {OUT / 'RadiScan-paper.tex'} and refs.bib")
    print(f"{doc.count(chr(92) + 'todo{')} placeholders marked \\todo{{}} (rendered red)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
