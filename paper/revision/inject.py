"""Replace EQN#### placeholders in a docx with native Word (OMML) equations.

    python inject.py in.docx equations.json out.docx
"""
import json
import re
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

src, eq_json, dst = sys.argv[1:4]
eqs = json.loads(Path(eq_json).read_text())

cache = Path(__file__).with_name("equations_omml.json")
cached = json.loads(cache.read_text()) if cache.exists() else {}
if cached.get("latex") == eqs:  # pre-generated: no pandoc needed
    maths = list(cached["omml"])
else:
    with tempfile.TemporaryDirectory() as td:
        md = Path(td) / "eq.md"
        md.write_text("\n\n".join(f"$${e}$$" for e in eqs))
        out = Path(td) / "eq.docx"
        subprocess.run(["pandoc", str(md), "-o", str(out)], check=True)
        xml = zipfile.ZipFile(out).read("word/document.xml").decode()
    maths = re.findall(r"<m:oMathPara>.*?(<m:oMath>.*?</m:oMath>)</m:oMathPara>", xml, re.S)
assert len(maths) == len(eqs), (len(maths), len(eqs))


def schema_order(x: str) -> str:
    """pandoc writes some OMML children out of schema order; Word tolerates it,
    strict validators do not."""
    x = re.sub(r"(<m:mcJc [^>]*/>)(<m:count [^>]*/>)", r"\2\1", x)
    x = re.sub(r"(<m:begChr [^>]*/>)(<m:endChr [^>]*/>)(<m:sepChr [^>]*/>)", r"\1\3\2", x)
    x = re.sub(r"(<m:sty [^>]*/>)(<m:scr [^>]*/>)", r"\2\1", x)
    x = re.sub(r"<m:nor />\s*<m:sty [^>]*/>", "<m:nor />", x)  # nor and sty are alternatives
    return x


maths = [schema_order(m) for m in maths]

zin = zipfile.ZipFile(src)
doc = zin.read("word/document.xml").decode()
if "xmlns:m=" not in doc[:3000]:
    doc = doc.replace("<w:document ", '<w:document xmlns:m="http://schemas.openxmlformats.org/officeDocument/2006/math" ', 1)
n = 0
for i, m in enumerate(maths, start=1):
    tag = f"EQN{i:04d}"
    pat = re.compile(r"<w:r>(?:(?!<w:r>).)*?<w:t[^>]*>" + tag + r"</w:t></w:r>", re.S)
    doc, k = pat.subn(lambda _, m=m: m, doc, count=1)
    n += k
assert n == len(eqs), f"injected {n}/{len(eqs)}"
assert "EQN0" not in doc
zout = zipfile.ZipFile(dst, "w", zipfile.ZIP_DEFLATED)
for item in zin.infolist():
    data = doc.encode() if item.filename == "word/document.xml" else zin.read(item.filename)
    zout.writestr(item, data)
zout.close()
print(f"injected {n} equations -> {dst}")
