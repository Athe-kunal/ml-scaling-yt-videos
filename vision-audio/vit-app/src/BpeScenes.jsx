import { useEffect, useState } from "react";
import { T, mono, btn } from "./theme";
import { Row, Label } from "./draw";
import { loadTokenizer, SOT, EOT } from "./clipBpe";

// Three sub-steps of CLIP's real BPE tokenizer:
//   BpeSplit  — clean, regex-split into words, mark the last char with </w>
//   BpeMerge  — replay the learned merges for one word, lowest rank first
//   BpeIds    — pieces -> vocab ids, wrap in <sot>/<eot>, pad to 77

export function useTokenizer() {
  const [tk, setTk] = useState(null);
  useEffect(() => { let on = true; loadTokenizer().then((t) => on && setTk(t)); return () => { on = false; }; }, []);
  return tk;
}

export const EXAMPLES = ["a goldendoodle doing photosynthesis", "a xylophonist on stage"];
const short = (s) => (s === SOT ? "<sot>" : s === EOT ? "<eot>" : s);

function Sym({ s, hot, end, dim, big }) {
  const isEnd = s.endsWith("</w>"), core = isEnd ? s.slice(0, -4) : s;
  return (
    <span style={{
      fontFamily: mono, fontSize: big ? 15 : 13, padding: big ? "4px 9px" : "3px 7px", borderRadius: 5, whiteSpace: "nowrap", transition: "all .3s",
      border: `1.5px solid ${hot ? T.accent2 : end ? T.wire : T.rule}`, background: hot ? `${T.accent2}30` : T.well, color: dim ? T.dim : T.ink,
    }}>
      {core}{isEnd && <span style={{ color: T.wire, fontSize: big ? 11 : 10 }}>{"</w>"}</span>}
    </span>
  );
}

// caption picker shared by the three sub-steps
export function BpePicker({ n, captions, text, setText }) {
  const opts = [...captions.slice(0, n), ...EXAMPLES];
  return (
    <div style={{ display: "grid", gap: 6 }}>
      <Row gap={5}>
        {opts.map((c) => (
          <button key={c} onClick={() => setText(c)} style={{ ...btn(c === text, false), height: 26, fontSize: 11, padding: "0 8px" }}>{c}</button>
        ))}
      </Row>
      <input value={text} onChange={(e) => setText(e.target.value)} spellCheck={false} placeholder="type any caption…"
        style={{ fontFamily: mono, fontSize: 14, padding: "7px 10px", borderRadius: 7, border: `1px solid ${T.rule}`, background: T.bg, color: T.ink, maxWidth: 560 }} />
    </div>
  );
}

const Loading = () => <div style={{ fontFamily: mono, fontSize: 12, color: T.dim, padding: 20 }}>loading CLIP's 48,894 merges…</div>;

export function BpeSplit({ tk, text }) {
  if (!tk) return <Loading />;
  const det = tk.encodeDetailed(text);
  const clean = text.replace(/\s+/g, " ").trim().toLowerCase();
  return (
    <div style={{ display: "grid", gap: 18 }}>
      <div style={{ display: "grid", gap: 6 }}>
        <Label>1 · clean + lowercase</Label>
        <span style={{ fontFamily: mono, fontSize: 15, color: T.ink }}>"{clean}"</span>
      </div>
      <div style={{ display: "grid", gap: 6 }}>
        <Label>2 · regex split into words / punctuation</Label>
        <Row gap={6}>{det.map((d, i) => <Sym key={i} s={d.word} big />)}</Row>
      </div>
      <div style={{ display: "grid", gap: 8 }}>
        <Label>3 · each word → characters, the last one marked {"</w>"}</Label>
        {det.map((d, i) => (
          <Row key={i} gap={4} wrap={false}>
            <span style={{ fontFamily: mono, fontSize: 11, color: T.dim, minWidth: 110 }}>{d.word}</span>
            {(d.trace[0]?.word ?? d.pieces).map((c, j, a) => <Sym key={j} s={c} end={j === a.length - 1} />)}
          </Row>
        ))}
      </div>
      <div style={{ fontFamily: mono, fontSize: 12, color: T.soft, lineHeight: 1.6 }}>
        {"</w>"} = "end of word": <b style={{ color: T.ink }}>photo{"</w>"}</b> (the word) and <b style={{ color: T.ink }}>photo</b> (start of "photosynthesis") are different tokens<br />
        words never merge across a space, so BPE works on one word at a time
      </div>
    </div>
  );
}

export function BpeMerge({ tk, text, wi, setWi }) {
  const det = tk ? tk.encodeDetailed(text) : [];
  const i = Math.min(wi, Math.max(0, det.length - 1));
  const d = det[i];
  const tr = d?.trace ?? [];
  const [k, setK] = useState(0);
  const [run, setRun] = useState(0);
  useEffect(() => {
    setK(0);
    if (!tr.length) return;
    let c = 0;
    const t = setInterval(() => { c += 1; setK(c); if (c >= tr.length - 1) clearInterval(t); }, 700);
    return () => clearInterval(t);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [text, i, run, tk]);
  if (!tk) return <Loading />;
  if (!d) return null;
  return (
    <div style={{ display: "grid", gap: 14 }}>
      <Row gap={5}>
        <Label>word</Label>
        {det.map((x, j) => <button key={j} onClick={() => setWi(j)} style={{ ...btn(j === i, false), height: 26, fontSize: 12, padding: "0 9px" }}>{x.word}</button>)}
        <button onClick={() => setRun(run + 1)} style={{ ...btn(false, false), height: 26, fontSize: 11, padding: "0 9px" }}>↻ replay</button>
      </Row>
      <div style={{ display: "grid", gap: 6 }}>
        {tr.slice(0, k + 1).map((s, r) => {
          const merged = s.pair ? s.pair.join("") : null;
          return (
            <Row key={r} gap={4} wrap={false}>
              <span style={{ fontFamily: mono, fontSize: 11, color: s.pair ? T.accent2 : T.dim, minWidth: 190, whiteSpace: "nowrap" }}>
                {s.pair ? <>merge <b>{short(s.pair[0])}</b> + <b>{short(s.pair[1])}</b> <span style={{ color: T.dim }}>· rank {s.rank.toLocaleString()}</span></> : "start: characters"}
              </span>
              {s.word.map((c, j, a) => <Sym key={j} s={c} hot={r === k && r > 0 && c === merged} end={j === a.length - 1} dim={r < k} />)}
            </Row>
          );
        })}
      </div>
      {k >= tr.length - 1 && (
        <div style={{ fontFamily: mono, fontSize: 12, color: T.soft, lineHeight: 1.6 }}>
          no adjacent pair left in the merge table → done: <b style={{ color: T.accent }}>{d.pieces.length} token{d.pieces.length > 1 ? "s" : ""}</b>
          {d.pieces.length > 1 ? " (rare word: it stays split into known pieces)" : " (common word: one token)"}
        </div>
      )}
      <div style={{ fontFamily: mono, fontSize: 12, color: T.soft, lineHeight: 1.6, borderTop: `1px solid ${T.rule}`, paddingTop: 10 }}>
        each round: of all adjacent pairs, merge the one with the <b style={{ color: T.ink }}>lowest rank</b> (= learned earliest, most frequent in training text)<br />
        rank = line number in CLIP's merge file · {tk.nMerges.toLocaleString()} merges learned from the training captions
      </div>
    </div>
  );
}

export function BpeIds({ tk, text, captions, n }) {
  if (!tk) return <Loading />;
  const det = tk.encodeDetailed(text);
  const { ids, len, truncated } = tk.tokenize(text);
  const pieces = [SOT, ...det.flatMap((d) => d.pieces), EOT];
  return (
    <div style={{ display: "grid", gap: 16 }}>
      <div style={{ display: "grid", gap: 6 }}>
        <Label>pieces → row index in the vocab ({tk.vocabSize.toLocaleString()} entries)</Label>
        <Row gap={6}>
          {pieces.slice(0, 76).map((p, j) => (
            <div key={j} style={{ display: "grid", gap: 3, justifyItems: "center" }}>
              <Sym s={short(p)} end={false} hot={p === EOT} />
              <span style={{ fontFamily: mono, fontSize: 11, color: p === SOT || p === EOT ? T.wire : T.accent }}>{ids[j]}</span>
            </div>
          ))}
        </Row>
      </div>
      <div style={{ display: "grid", gap: 6 }}>
        <Label>pad with 0 to 77 · this row of T [n, 77]</Label>
        <div style={{ display: "flex", gap: 1 }}>
          {ids.map((v, j) => <span key={j} title={`${v}`} style={{ flex: 1, height: 14, borderRadius: 1, background: v === 49407 ? T.wire : v === 49406 ? `${T.wire}99` : v ? T.accent : T.rule, opacity: v ? 1 : 0.45 }} />)}
        </div>
        <span style={{ fontFamily: mono, fontSize: 11, color: truncated ? T.bad : T.dim }}>{len} real tokens + {77 - len} zeros{truncated ? " · truncated to 77!" : ""}</span>
      </div>
      <div style={{ display: "grid", gap: 4 }}>
        <Label>the whole batch · T [{n}, 77] (first 12 columns)</Label>
        {captions.slice(0, n).map((c) => {
          const r = tk.tokenize(c).ids.slice(0, 12);
          return (
            <Row key={c} gap={3} wrap={false}>
              {r.map((v, j) => <span key={j} style={{ fontFamily: mono, fontSize: 11, minWidth: 46, textAlign: "right", color: v === 49407 || v === 49406 ? T.wire : v ? T.ink : T.rule }}>{v}</span>)}
              <span style={{ fontFamily: mono, fontSize: 11, color: T.dim }}> … 0</span>
            </Row>
          );
        })}
      </div>
      <div style={{ fontFamily: mono, fontSize: 12, color: T.soft, lineHeight: 1.6 }}>
        &lt;eot&gt; = 49407 is the largest id, so <b style={{ color: T.ink }}>T.argmax(-1)</b> finds its position — that's where the text feature is read out
      </div>
    </div>
  );
}
