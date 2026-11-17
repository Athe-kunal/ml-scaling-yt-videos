import { useEffect, useState } from "react";
import { T, mono, btn } from "./theme";
import { Row, Label } from "./draw";
import { loadSpm, EOS, CTX_S } from "./siglipSpm";

// Three sub-steps of SigLIP's real SentencePiece (unigram) tokenizer:
//   SpmClean    — lowercase, drop punctuation, "▁" marks each word start
//   SpmViterbi  — every vocab piece that fits, best total log-prob wins
//   SpmIds      — pieces → ids, append </s>, pad with </s> to 64

export function useSpm() {
  const [tk, setTk] = useState(null);
  useEffect(() => { let on = true; loadSpm().then((t) => on && setTk(t)); return () => { on = false; }; }, []);
  return tk;
}

export const S_EXAMPLES = ["A Goldendoodle, doing Photosynthesis!", "a xylophonist on stage", "naïve café 🐶"];
const Loading = () => <div style={{ fontFamily: mono, fontSize: 15, color: T.dim, padding: 20 }}>loading SigLIP's 32,000 pieces…</div>;

function Piece({ s, hot, dim, big, byte }) {
  const lead = s.startsWith("▁"), core = lead ? s.slice(1) : s;
  return (
    <span style={{
      fontFamily: mono, fontSize: big ? 18 : 16, padding: big ? "4px 9px" : "3px 7px", borderRadius: 5, whiteSpace: "nowrap",
      border: `1.5px solid ${hot ? T.accent2 : byte ? T.bad : T.rule}`, background: hot ? `${T.accent2}30` : T.well, color: dim ? T.dim : byte ? T.bad : T.ink,
    }}>
      {lead && <span style={{ color: T.wire }}>▁</span>}{core}
    </span>
  );
}

export function SpmPicker({ n, captions, text, setText }) {
  const opts = [...captions.slice(0, n), ...S_EXAMPLES];
  return (
    <div style={{ display: "grid", gap: 6 }}>
      <Row gap={5}>
        {opts.map((c) => (
          <button key={c} onClick={() => setText(c)} style={{ ...btn(c === text, false), height: 26, fontSize: 14, padding: "0 8px" }}>{c}</button>
        ))}
      </Row>
      <input value={text} onChange={(e) => setText(e.target.value)} spellCheck={false} placeholder="type any caption…"
        style={{ fontFamily: mono, fontSize: 17, padding: "7px 10px", borderRadius: 7, border: `1px solid ${T.rule}`, background: T.bg, color: T.ink, maxWidth: 560 }} />
    </div>
  );
}

const Big = ({ children, color = T.ink }) => <span style={{ fontFamily: mono, fontSize: 20, color, wordBreak: "break-all" }}>{children}</span>;

export function SpmClean({ tk, text }) {
  if (!tk) return <Loading />;
  const clean = tk.clean(text), marked = tk.marked(text);
  return (
    <div style={{ display: "grid", gap: 20 }}>
      <div style={{ display: "grid", gap: 6 }}>
        <Label>1 · lowercase, drop punctuation</Label>
        <Big>"{clean}"</Big>
      </div>
      <div style={{ display: "grid", gap: 6 }}>
        <Label>2 · ▁ = start of a word</Label>
        <Big>{[...marked].map((c, i) => <span key={i} style={{ color: c === "▁" ? T.wire : T.ink, fontWeight: c === "▁" ? 700 : 400 }}>{c}</span>)}</Big>
      </div>
      <div style={{ display: "grid", gap: 8 }}>
        <Label>3 · one string, no word list</Label>
        <Row gap={6}>{marked.split(/(?=▁)/).map((w, i) => <Piece key={i} s={w} big />)}</Row>
      </div>
      <div style={{ fontFamily: mono, fontSize: 15, color: T.soft, lineHeight: 1.6 }}>
        CLIP: words end with {"</w>"} · SigLIP: words start with ▁
      </div>
    </div>
  );
}

// the lattice for one word: every vocab piece that matches a span of its
// characters, packed into rows; the best cut lit up
function Lattice({ tk, word }) {
  const ch = [...word], edges = tk.lattice(word), best = tk.viterbi(word).path;
  const onBest = (e) => best.some((b) => b.from === e.from && b.piece === e.piece);
  const cw = ch.length > 16 ? 26 : 34;
  // pack: longest first so the row count stays small, best-path pieces in row 0
  const rows = [[]];
  const sorted = [...edges].sort((a, b) => (onBest(b) - onBest(a)) || (b.to - b.from) - (a.to - a.from) || a.from - b.from);
  for (const e of sorted) {
    let r = onBest(e) ? 0 : 1;
    for (; ; r++) {
      rows[r] ??= [];
      if (!rows[r].some((o) => o.from < e.to && e.from < o.to)) { rows[r].push(e); break; }
    }
  }
  const MAXR = 9, shown = rows.slice(0, MAXR), hidden = rows.slice(MAXR).flat().length;
  const W = ch.length * cw;
  return (
    <div style={{ display: "grid", gap: 4, overflowX: "auto" }}>
      <div style={{ display: "flex" }}>
        {ch.map((c, i) => <span key={i} style={{ width: cw, textAlign: "center", fontFamily: mono, fontSize: 19, fontWeight: 700, color: c === "▁" ? T.wire : T.ink }}>{c}</span>)}
      </div>
      {shown.map((r, k) => (
        <div key={k} style={{ position: "relative", height: 30, width: W, borderTop: k === 1 ? `1px dashed ${T.rule}` : "none", paddingTop: k === 1 ? 4 : 0 }}>
          {r.map((e, j) => {
            const hot = onBest(e);
            return (
              <div key={j} title={`${e.piece}  log-prob ${e.score.toFixed(2)}`} style={{
                position: "absolute", left: e.from * cw + 1, width: (e.to - e.from) * cw - 2, top: k === 1 ? 4 : 0, height: 26, borderRadius: 5,
                display: "grid", placeItems: "center", overflow: "hidden", fontFamily: mono, fontSize: 13, whiteSpace: "nowrap",
                border: `1.5px solid ${hot ? T.accent : T.rule}`, background: hot ? `${T.accent}33` : T.well, color: hot ? T.ink : T.dim,
              }}>{(e.to - e.from) * cw >= 44 ? e.score.toFixed(1) : ""}</div>
            );
          })}
        </div>
      ))}
      {hidden > 0 && <span style={{ fontFamily: mono, fontSize: 13, color: T.dim }}>+ {hidden} more pieces</span>}
    </div>
  );
}

export function SpmViterbi({ tk, text, wi, setWi }) {
  if (!tk) return <Loading />;
  const words = tk.marked(text).split(/(?=▁)/).filter((w) => w.length);
  const longest = words.reduce((a, w, i) => ([...w].length > [...words[a]].length ? i : a), 0);
  const i = wi == null || wi >= words.length ? longest : wi;
  const word = words[i] ?? "";
  const cuts = tk.wordCuts(word, 4);
  return (
    <div style={{ display: "grid", gap: 16 }}>
      <Row gap={5}>
        <Label>word</Label>
        {words.map((w, j) => <button key={j} onClick={() => setWi(j)} style={{ ...btn(j === i, false), height: 26, fontSize: 15, padding: "0 9px" }}>{w}</button>)}
      </Row>
      <Row gap={30} style={{ alignItems: "flex-start" }}>
        <div style={{ display: "grid", gap: 8 }}>
          <Label>every vocab piece that fits · number = log-prob</Label>
          <Lattice tk={tk} word={word} />
        </div>
        <div style={{ display: "grid", gap: 8, minWidth: 300 }}>
          <Label>best cuts · total = Σ log-prob</Label>
          {cuts.map((c, k) => (
            <div key={k} style={{ display: "flex", alignItems: "center", gap: 10, opacity: k === 0 ? 1 : 0.55 }}>
              <span style={{ fontFamily: mono, fontSize: 16, fontWeight: 700, color: k === 0 ? T.accent : T.soft, minWidth: 64, textAlign: "right" }}>{c.total.toFixed(1)}</span>
              <Row gap={3} wrap={false}>{c.pieces.map((p, j) => <Piece key={j} s={p} hot={k === 0} />)}</Row>
            </div>
          ))}
          {!cuts.length && <span style={{ fontFamily: mono, fontSize: 15, color: T.bad }}>no cut: unknown characters → bytes</span>}
        </div>
      </Row>
      <div style={{ fontFamily: mono, fontSize: 16, color: T.soft, lineHeight: 1.6, borderTop: `1px solid ${T.rule}`, paddingTop: 10 }}>
        highest total wins (Viterbi) · no merge rules
      </div>
    </div>
  );
}

export function SpmIds({ tk, text, captions, n }) {
  if (!tk) return <Loading />;
  const { toks } = tk.encode(text);
  const { ids, len, rawLen, truncated } = tk.tokenize(text);
  const shown = [...toks.slice(0, len - 1), { piece: "</s>", id: EOS }];
  return (
    <div style={{ display: "grid", gap: 16 }}>
      <div style={{ display: "grid", gap: 6 }}>
        <Label>pieces → row index in the vocab ({tk.vocabSize.toLocaleString()} entries)</Label>
        <Row gap={6}>
          {shown.map((p, j) => (
            <div key={j} style={{ display: "grid", gap: 3, justifyItems: "center" }}>
              <Piece s={p.piece} hot={p.id === EOS} byte={p.byte} />
              <span style={{ fontFamily: mono, fontSize: 14, color: p.id === EOS ? T.wire : T.accent }}>{p.id}</span>
            </div>
          ))}
        </Row>
      </div>
      <div style={{ display: "grid", gap: 6 }}>
        <Label>pad with &lt;/s&gt; (id 1) to {CTX_S}</Label>
        <div style={{ display: "flex", gap: 1 }}>
          {ids.map((v, j) => <span key={j} title={`${v}`} style={{ flex: 1, height: 14, borderRadius: 1, background: j >= len - 1 ? T.wire : T.accent, opacity: j < len ? 1 : 0.45 }} />)}
        </div>
        <span style={{ fontFamily: mono, fontSize: 14, color: truncated ? T.bad : T.dim }}>{truncated
          ? `${rawLen} tokens: cut to 63 + </s>`
          : `${len - 1} pieces + </s> + ${CTX_S - len} × </s> padding`}</span>
      </div>
      <div style={{ display: "grid", gap: 4 }}>
        <Label>the whole batch · ids [{n}, {CTX_S}] (first 12 columns)</Label>
        {captions.slice(0, n).map((c) => {
          const tc = tk.tokenize(c), r = tc.ids.slice(0, 12);
          return (
            <Row key={c} gap={3} wrap={false}>
              {r.map((v, j) => <span key={j} style={{ fontFamily: mono, fontSize: 14, minWidth: 46, textAlign: "right", color: j >= tc.len - 1 ? T.wire : T.ink }}>{v}</span>)}
              <span style={{ fontFamily: mono, fontSize: 14, color: T.wire }}> … 1</span>
            </Row>
          );
        })}
      </div>
      <div style={{ fontFamily: mono, fontSize: 15, color: T.soft, lineHeight: 1.6 }}>
        no start token · the last slot is always &lt;/s&gt;
      </div>
    </div>
  );
}
