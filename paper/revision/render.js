// Render paper_spec.json -> IEEE-style two-column .docx (equations injected afterwards).
const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, AlignmentType, Table, TableRow, TableCell,
  WidthType, BorderStyle, ImageRun, SectionType, TabStopType, ShadingType,
  VerticalAlign, LineRuleType, HeightRule,
} = require("docx");

const spec = JSON.parse(fs.readFileSync(process.argv[2], "utf8"));
const out = process.argv[3];

const FONT = "Times New Roman";
const PAGE_W = 12240, MARG_LR = 900, COL_GAP = 360;
const BODY_W = PAGE_W - 2 * MARG_LR;           // 10440
const COL_W = (BODY_W - COL_GAP) / 2;          // 5040
const ROMAN = ["I","II","III","IV","V","VI","VII","VIII","IX","X","XI","XII"];
const TROMAN = ROMAN;

function runs(items, base = {}) {
  if (typeof items === "string") items = [items];
  const res = [];
  for (const it of items) {
    const r = typeof it === "string" ? { text: it } : it;
    const o = {
      text: r.text, font: FONT, size: r.size || base.size || 20,
      bold: r.b || base.b, italics: r.i || base.i,
      superScript: r.sup, subScript: r.sub,
      smallCaps: r.sc || base.sc,
    };
    if (r.hl) o.highlight = "yellow";
    if (r.color) o.color = r.color;
    res.push(new TextRun(o));
  }
  return res;
}

let h1 = 0, h2 = 0, tableNo = 0, figNo = 0;
const sections = [];
let cur = { cols: 1, children: [] };
function flush() { if (cur.children.length) sections.push(cur); }

function para(children, opts = {}) {
  return new Paragraph({
    children,
    alignment: opts.align || AlignmentType.JUSTIFIED,
    indent: opts.indent,
    spacing: Object.assign({ before: 0, after: 0, line: 240, lineRule: LineRuleType.AUTO }, opts.spacing || {}),
    tabStops: opts.tabs,
    keepNext: opts.keepNext,
    keepLines: opts.keepLines,
  });
}

function cellBorders(top, bottom) {
  const none = { style: BorderStyle.NONE, size: 0, color: "FFFFFF" };
  const line = (sz) => ({ style: BorderStyle.SINGLE, size: sz, color: "000000" });
  return { top: top ? line(top) : none, bottom: bottom ? line(bottom) : none, left: none, right: none };
}

function makeTable(b) {
  tableNo += 1;
  const width = b.full ? BODY_W : COL_W;
  const totalRel = b.cols.reduce((a, c) => a + c, 0);
  const widths = b.cols.map((c) => Math.floor((c / totalRel) * width));
  widths[widths.length - 1] += width - widths.reduce((a, c) => a + c, 0);
  const fs_ = b.font ? Math.round(b.font * 2) : 15;
  const mk = (cells, isHead, idx, n) => new TableRow({
    tableHeader: isHead,
    cantSplit: true,
    children: cells.map((c, j) => new TableCell({
      width: { size: widths[j], type: WidthType.DXA },
      borders: cellBorders(isHead ? 8 : (idx === 0 ? 4 : 0), isHead ? 4 : (idx === n - 1 ? 8 : (b.rules && b.rules.includes(idx) ? 2 : 0))),
      margins: { top: 15, bottom: 15, left: 50, right: 50 },
      verticalAlign: VerticalAlign.CENTER,
      shading: (b.shade && b.shade.includes(idx) && !isHead) ? { fill: "F2F2F2", type: ShadingType.CLEAR, color: "auto" } : undefined,
      children: [para(runs(c, { size: fs_, b: isHead }), {
        align: j === 0 ? AlignmentType.LEFT : AlignmentType.CENTER,
      })],
    })),
  });
  const rows = [mk(b.header, true, -1, b.rows.length)];
  b.rows.forEach((r, i) => rows.push(mk(r, false, i, b.rows.length)));
  const capLabel = para(runs([{ text: `TABLE ${TROMAN[tableNo - 1]}`, sc: true }], { size: 16 }), {
    align: AlignmentType.CENTER, spacing: { before: 160, after: 0 }, keepNext: true,
  });
  const cap = para(runs([{ text: b.caption, sc: true }], { size: 16 }), {
    align: AlignmentType.CENTER, spacing: { after: 60 }, keepNext: true,
  });
  const out = [capLabel, cap, new Table({ width: { size: width, type: WidthType.DXA }, columnWidths: widths, rows, alignment: AlignmentType.CENTER })];
  if (b.note) out.push(para(runs(b.note, { size: 14 }), { spacing: { before: 40, after: 120 } }));
  else out.push(para(runs(""), { spacing: { after: 100 } }));
  return out;
}

function makeFig(b) {
  figNo += 1;
  const img = fs.readFileSync(b.path);
  const wIn = b.width_in || 3.4;
  const [pw, ph] = b.px; // pixel size for aspect
  const wPx = Math.round(wIn * 96), hPx = Math.round(wPx * ph / pw);
  return [
    para([new ImageRun({ type: "png", data: img, transformation: { width: wPx, height: hPx } })], {
      align: AlignmentType.CENTER, spacing: { before: 120, after: 40 }, keepNext: true,
    }),
    para(runs([{ text: `Fig. ${figNo}. `, size: 16 }, ...(Array.isArray(b.caption) ? b.caption : [b.caption]).map((c) => typeof c === "string" ? { text: c, size: 16 } : Object.assign({ size: 16 }, c))]), {
      spacing: { after: 140 },
    }),
  ];
}

for (const b of spec.blocks) {
  switch (b.t) {
    case "cols":
      flush(); cur = { cols: b.n, children: [] }; break;
    case "title":
      cur.children.push(para(runs(b.text, { size: 48 }), { align: AlignmentType.CENTER, spacing: { after: 160, line: 264 } }));
      break;
    case "author":
      cur.children.push(para(runs(b.runs, { size: b.size || 22 }), { align: AlignmentType.CENTER, spacing: { after: b.after || 0 } }));
      break;
    case "abstract":
      cur.children.push(para(runs([{ text: b.label + "—", b: true, i: true, size: 18 }, ...b.runs.map((r) => typeof r === "string" ? { text: r, b: true, size: 18 } : Object.assign({ b: true, size: 18 }, r))]), { spacing: { after: 120 } }));
      break;
    case "h1":
      h1 += 1; h2 = 0;
      cur.children.push(para(runs([{ text: `${ROMAN[h1 - 1]}. ${b.text}`, sc: true }]), {
        align: AlignmentType.CENTER, spacing: { before: 200, after: 100 }, keepNext: true,
      }));
      break;
    case "h1n": // unnumbered (references)
      cur.children.push(para(runs([{ text: b.text, sc: true }]), { align: AlignmentType.CENTER, spacing: { before: 200, after: 100 }, keepNext: true }));
      break;
    case "h2":
      h2 += 1;
      cur.children.push(para(runs([{ text: `${String.fromCharCode(64 + h2)}. ${b.text}`, i: true }]), {
        align: AlignmentType.LEFT, spacing: { before: 120, after: 60 }, keepNext: true,
      }));
      break;
    case "p":
      cur.children.push(para(runs(b.runs), { indent: b.noindent ? undefined : { firstLine: 202 }, spacing: { after: b.after || 0 } }));
      break;
    case "item":
      cur.children.push(para(runs(b.runs), { indent: { left: 360, hanging: 300 }, spacing: { after: 20 } }));
      break;
    case "eq": {
      const w = b.full ? BODY_W : COL_W;
      cur.children.push(para(runs(["\t", { text: `EQN${String(b.id).padStart(4, "0")}` }, `\t(${b.n})`]), {
        align: AlignmentType.LEFT,
        tabs: [{ type: TabStopType.CENTER, position: Math.round(w / 2) }, { type: TabStopType.RIGHT, position: w }],
        spacing: { before: 80, after: 80 }, keepLines: true,
      }));
      break;
    }
    case "table":
      cur.children.push(...makeTable(b)); break;
    case "fig":
      cur.children.push(...makeFig(b)); break;
    case "refs":
      b.items.forEach((r, i) => cur.children.push(para(runs([{ text: `[${i + 1}]\t`, size: 16 }, ...(Array.isArray(r) ? r : [r]).map((x) => typeof x === "string" ? { text: x, size: 16 } : Object.assign({ size: 16 }, x))]), {
        indent: { left: 360, hanging: 360 }, tabs: [{ type: TabStopType.LEFT, position: 360 }], spacing: { after: 20 },
      })));
      break;
    default:
      throw new Error("unknown block " + b.t);
  }
}
flush();

const doc = new Document({
  creator: spec.meta.author, title: spec.meta.title, description: spec.meta.description,
  styles: { default: { document: { run: { font: FONT, size: 20 } } } },
  sections: sections.map((s, i) => ({
    properties: {
      type: i === 0 ? undefined : SectionType.CONTINUOUS,
      page: { size: { width: PAGE_W, height: 15840 }, margin: { top: 1080, bottom: 1440, left: MARG_LR, right: MARG_LR } },
      column: s.cols === 2 ? { count: 2, space: COL_GAP, equalWidth: true } : { count: 1 },
    },
    children: s.children,
  })),
});
Packer.toBuffer(doc).then((buf) => { fs.writeFileSync(out, buf); console.log("wrote", out, "tables", tableNo, "figs", figNo); });
