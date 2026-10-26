import { useEffect, useRef } from "react";
import { T, mono } from "./theme";

// Minimal Python highlighter + a code listing whose lines light up by tag.
// `focus` maps tag -> color; a line takes the color of its first active tag.

const KW = new Set(["def", "return", "if", "elif", "else", "for", "while", "break", "continue", "del", "in", "not", "and", "or", "is", "None", "True", "False", "class", "raise", "assert", "import", "from", "lambda"]);
const BUILTIN = new Set(["range", "len", "slice", "self", "min", "max", "sum", "hash", "tuple", "print", "reversed", "enumerate", "zip"]);
const TOKEN_RE = /(#.*$)|("(?:[^"\\]|\\.)*"|'(?:[^'\\]|\\.)*')|(@[\w.]+)|(\b\d+(?:\.\d+)?\b)|([A-Za-z_]\w*)|(\s+)|(.)/g;

const esc = (s) => s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
const span = (c, s) => `<span style="color:${c}">${esc(s)}</span>`;

function pyHtml(line) {
  let html = "";
  let m;
  TOKEN_RE.lastIndex = 0;
  while ((m = TOKEN_RE.exec(line))) {
    const [tok, cmt, str, deco, num, ident] = m;
    if (cmt) html += span("#6B7A8F", cmt);
    else if (str) html += span("#A5D6A7", str);
    else if (deco) html += span("#C49CFF", deco);
    else if (num) html += span("#FFB86B", num);
    else if (ident) {
      const next = line.slice(TOKEN_RE.lastIndex).trimStart()[0];
      if (KW.has(ident)) html += span("#FF8AC6", ident);
      else if (BUILTIN.has(ident)) html += span("#5FD3F3", ident);
      else if (next === "(") html += span("#8AB4FF", ident);
      else if (/^[A-Z]/.test(ident)) html += span("#F2E366", ident);
      else html += esc(ident);
    } else html += esc(tok);
  }
  return html;
}

// Live comments: notes = [{ m | tag, t, after? }]. Each note attaches to the
// first line carrying `tag` (or containing `m`, after the first line
// containing `after`, if given) and is drawn as a "# ..." comment under it.
function attachNotes(lines, notes) {
  const out = lines.map(() => []);
  for (const n of notes || []) {
    let from = 0;
    if (n.after) {
      const a = lines.findIndex((l) => l.text.includes(n.after));
      if (a >= 0) from = a + 1;
    }
    const i = lines.findIndex((l, k) => k >= from && (n.tag ? l.tags.includes(n.tag) : l.text.includes(n.m)));
    if (i >= 0) out[i].push(n.t);
  }
  return out;
}

export function CodeBlock({ title, lines, focus, maxHeight, note, notes }) {
  const box = useRef(null);
  const colorOf = (tags) => {
    // a lit tag wins over a dimmed (calling-line) tag
    let dim = null;
    for (const t of tags) {
      const c = focus[t];
      if (c && !c.startsWith("dim:")) return c;
      if (c) dim = dim || c;
    }
    return dim;
  };
  const anyLit = lines.some((l) => colorOf(l.tags));
  const lineNotes = attachNotes(lines, notes);
  const key = JSON.stringify(focus) + JSON.stringify(notes || []);

  // Keep the first lit line in view as the phase changes.
  useEffect(() => {
    if (!box.current) return;
    let lit = box.current.querySelectorAll("[data-lit='1']");
    if (!lit.length) lit = box.current.querySelectorAll("[data-lit='2']");
    if (!lit.length) return;
    const first = lit[0], lastEl = lit[lit.length - 1];
    const mid = (first.offsetTop + lastEl.offsetTop + lastEl.offsetHeight) / 2;
    const span = lastEl.offsetTop + lastEl.offsetHeight - first.offsetTop;
    const h = box.current.clientHeight;
    // center the lit region, but never push its first line out of view
    box.current.scrollTop = Math.max(0, span > h - 40 ? first.offsetTop - 30 : mid - h / 2);
  }, [key]);

  return (
    <div style={{ background: T.panel, border: `1px solid ${T.rule}`, borderRadius: 10, overflow: "hidden" }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", gap: 10, padding: "9px 14px", borderBottom: `1px solid ${T.rule}`, background: T.well }}>
        <span style={{ fontFamily: mono, fontSize: 12, color: T.soft }}>{title}</span>
        {note && <span style={{ fontFamily: mono, fontSize: 10.5, color: T.dim, textAlign: "right" }}>{note}</span>}
      </div>
      <div ref={box} style={{ maxHeight, overflow: "auto", position: "relative", padding: "8px 0" }}>
        {lines.map((l, i) => {
          const c = colorOf(l.tags);
          const dim = c && c.startsWith("dim:");
          const col = dim ? c.slice(4) : c;
          return (
            <div
              key={i}
              className="code-line"
              data-lit={c ? (dim ? "2" : "1") : "0"}
              style={{
                display: "flex", fontFamily: mono, fontSize: 12, lineHeight: "19px", whiteSpace: "pre",
                borderLeft: `3px solid ${col ? col : "transparent"}`,
                background: col ? `${col}${dim ? "10" : "22"}` : "transparent",
                opacity: anyLit && !col ? 0.5 : 1,
                minWidth: "max-content",
              }}
            >
              <span style={{ width: 34, textAlign: "right", paddingRight: 10, color: col || T.rule, userSelect: "none", flexShrink: 0 }}>{i + 1}</span>
              <span style={{ color: T.ink, paddingRight: 16, flexShrink: 0 }} dangerouslySetInnerHTML={{ __html: pyHtml(l.text) || " " }} />
            </div>
          );
        }).flatMap((row, i) => {
          // live comment: an extra comment line under its code line, same indent
          const ln = lineNotes[i];
          if (!ln.length) return [row];
          const c = colorOf(lines[i].tags);
          const col = c ? (c.startsWith("dim:") ? c.slice(4) : c) : null;
          const indent = lines[i].text.match(/^\s*/)[0];
          return [row, ...ln.map((t, k) => (
            <div key={`n${i}-${k}`} style={{ display: "flex", fontFamily: mono, fontSize: 11.5, lineHeight: "17px",
              borderLeft: `3px solid ${col || "transparent"}`, background: col ? `${col}12` : "transparent", paddingBottom: 2 }}>
              <span style={{ width: 34, flexShrink: 0, paddingRight: 10, textAlign: "right", color: "#9FE8D9", userSelect: "none" }}>↳</span>
              <span style={{ whiteSpace: "pre", flexShrink: 0 }}>{indent}</span>
              <span style={{ color: "#9FE8D9", fontStyle: "italic", whiteSpace: "pre-wrap", paddingRight: 14 }}># {t}</span>
            </div>
          ))];
        })}
      </div>
    </div>
  );
}
