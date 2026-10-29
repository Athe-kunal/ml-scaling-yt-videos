// Minimal Python highlighter for the code inside flowchart nodes.

const KW = new Set(["def", "async", "await", "yield", "return", "if", "elif", "else", "for", "while", "break", "continue", "del", "in", "not", "and", "or", "is", "None", "True", "False", "class", "raise", "assert", "import", "from", "lambda", "with"]);
const BUILTIN = new Set(["range", "len", "slice", "self", "min", "max", "sum", "hash", "tuple", "print", "reversed", "enumerate", "zip", "all", "any", "next", "list", "isinstance", "getattr"]);
const TOKEN_RE = /(#.*$)|("(?:[^"\\]|\\.)*"|'(?:[^'\\]|\\.)*')|(@[\w.]+)|(\b\d+(?:\.\d+)?\b)|([A-Za-z_]\w*)|(\s+)|(.)/g;

const esc = (s) => s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
const span = (c, s) => `<span style="color:${c}">${esc(s)}</span>`;

export function pyHtml(line) {
  let html = "";
  let m;
  TOKEN_RE.lastIndex = 0;
  while ((m = TOKEN_RE.exec(line))) {
    const [tok, cmt, str, deco, num, ident] = m;
    if (cmt) html += span("#7C8BA0", cmt);
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
