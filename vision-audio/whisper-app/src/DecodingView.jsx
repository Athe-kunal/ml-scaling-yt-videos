import { useState } from "react";
import { T, mono } from "./theme";
import StepShell, { Stat } from "./StepShell";
import { DimCtx, Row, Op, Mat, Label } from "./draw";
import { PREFIX, GEN, TOK, WORDS, tokText, tsId, tsOf, vocabOf } from "./model";
import { Tape, TokChip, VocabStrip, MaskLegend, KIND_COL } from "./tokens";

// decoding.py on the example clip. The loop iterations replay one greedy run;
// the masks come from the same rules as the code (SuppressBlank, SuppressTokens,
// ApplyTimestampRules), computed from the tokens so far. Probabilities are toy.

const CODE = {
  detect: ["x = torch.tensor([[tokenizer.sot]])   # one token", "logits = model.logits(x, audio_features)[:, 0]", "mask = ones(n_vocab, bool); mask[all_language_tokens] = False", "logits[:, mask] = -inf; lang = logits.argmax(-1)   # → tokens[sot_index + 1]"],
  init: ["initial_tokens = sot_sequence   # (+ <|notimestamps|> if without_timestamps)", "if prompt: initial_tokens = [sot_prev] + prompt[-223:] + initial_tokens", "sample_begin = len(initial_tokens); sample_len = 448 // 2   # max 224 new tokens"],
  fwd: ["logits = self.inference.logits(tokens, audio_features)   # [1, n, V]", "if i == 0: no_speech_prob = logits[:, sot_index].softmax(-1)[:, no_speech]", "logits = logits[:, -1]   # only the last position"],
  kv: ["if tokens.shape[-1] > initial_token_length:", "    tokens = tokens[:, -1:]   # after the first call: only the new token", "# hook on key/value Linear: cache[module] = cat([cache[module], output], dim=1)", "# cross-attn: output.shape[1] = 1500 > 448 → saved once, reused"],
  filters: ["SuppressBlank: at sample_begin, logits[[' ', eot]] = -inf", "SuppressTokens: logits[non_speech_tokens + sot, sot_prev, sot_lm, task, nospeech] = -inf"],
  rules: ["logits[no_timestamps] = -inf", "if last_was_ts: (penultimate_was_ts ? no timestamps : no text)", "logits[ts_begin : timestamp_last] = -inf   # never go back in time", "if at sample_begin: logits[:ts_begin] = -inf; logits[ts_begin + 50 + 1:] = -inf   # start ≤ 1.00 s", "if logsumexp(logprobs[ts_begin:]) > logprobs[:ts_begin].max(): logits[:ts_begin] = -inf"],
  pick: ["temperature == 0: next = logits.argmax(-1)", "temperature > 0: next = Categorical(logits=logits / temperature).sample()", "beam_size = 5: keep the 5 best by Σ log p, finish at <|endoftext|>", "tokens = cat([tokens, next]); completed = (tokens[:, -1] == eot).all()"],
  result: ["tokens = tokens[sample_begin : first eot]", "text = tokenizer.decode(tokens).strip()   # timestamp ids are dropped", "avg_logprob = sum_logprob / (len(tokens) + 1)", "compression_ratio = len(text_bytes) / len(zlib.compress(text_bytes))"],
};

const FLOW = [
  { id: "detect", label: "detect_language", sub: "1 step after <|sot|>", code: CODE.detect },
  { id: "init", label: "initial tokens", sub: "sot · lang · task", code: CODE.init },
  {
    group: "_main_loop",
    loop: "≤ 224 × · until eot",
    items: [
      { id: "fwd", label: "decoder → last logits", sub: "[1, V]", code: CODE.fwd },
      { id: "kv", label: "KV cache", sub: "self grows · cross fixed", code: CODE.kv },
      { id: "filters", label: "suppress tokens", sub: "blank · symbols · specials", code: CODE.filters },
      { id: "rules", label: "timestamp rules", sub: "pairs · forward only", code: CODE.rules },
      { id: "pick", label: "pick next", sub: "argmax · sample · beam", code: CODE.pick },
    ],
  },
  { id: "result", label: "DecodingResult", sub: "text · avg_logprob · ratio", code: CODE.result },
];

// ---------------------------------------------------------------- the rules, as in decoding.py
const SYMBOLS = [1, 2, 7, 8, 9, 10, 14, 25, 26, 27, 28, 29, 31, 58, 59, 60, 61, 62, 63, 90, 91, 92, 93];
function masksAt(i) {
  const seq = GEN.slice(0, i), ts0 = TOK.ts0, out = [];
  out.push({ a: TOK.sot, b: TOK.sot, rule: "SuppressTokens", label: "<|startoftranscript|>", color: "#C79BFF" });
  out.push({ a: TOK.translate, b: TOK.nospeech, rule: "SuppressTokens", label: "task · <|startoflm|> · <|startofprev|> · <|nospeech|>", color: "#C79BFF" });
  out.push({ a: SYMBOLS[0], b: SYMBOLS[SYMBOLS.length - 1], rule: "SuppressTokens", label: 'symbols  " # ( ) * [ ] { } ♪ …  (only these ids)', color: "#C79BFF", thin: true });
  out.push({ a: TOK.notimestamps, b: TOK.notimestamps, rule: "timestamp rules", label: "<|notimestamps|>", color: T.wire });
  if (i === 0) out.push({ a: 220, b: 220, rule: "SuppressBlank", label: "' ' and <|endoftext|> at the first step", color: "#8DB4FF" }, { a: TOK.eot, b: TOK.eot, rule: "SuppressBlank", label: "", color: "#8DB4FF", hide: true });
  const last = seq.length >= 1 && seq[seq.length - 1] >= ts0;
  const pen = seq.length < 2 || seq[seq.length - 2] >= ts0;
  if (last) {
    if (pen) out.push({ a: ts0, b: 51864, rule: "pairs", label: "a pair just closed → must be text", color: T.bad });
    else out.push({ a: 0, b: TOK.eot - 1, rule: "pairs", label: "text then <|t|> → next is <|t|> or <|endoftext|>", color: T.bad });
  }
  const tss = seq.filter((t) => t >= ts0);
  if (tss.length) {
    const tl = last && !pen ? tss[tss.length - 1] : tss[tss.length - 1] + 1;
    if (tl > ts0) out.push({ a: ts0, b: tl - 1, rule: "forward only", label: `no timestamp before <|${tsOf(tl)}|>`, color: T.accent2 });
  }
  if (i === 0) {
    out.push({ a: 0, b: ts0 - 1, rule: "start", label: "first token must be a timestamp", color: T.bad });
    out.push({ a: ts0 + 51, b: 51864, rule: "start", label: "max_initial_timestamp = 1.00 s", color: T.bad });
  }
  if (GEN[i] >= ts0 && i > 0 && !last) out.push({ a: 0, b: ts0 - 1, rule: "Σ p(time)", label: "Σ p(timestamps) > max p(text) → force a timestamp", color: T.wire });
  return out;
}
const masked = (id, ms) => ms.some((m) => id >= m.a && id <= m.b && !(m.thin && !SYMBOLS.includes(id)));

// toy candidate lists (before the filters)
const ALT = [400, 370, 452, 1029, 437, 291, 393, 13];
function candsAt(i) {
  const next = GEN[i], prev = GEN[i - 1], ts0 = TOK.ts0;
  const w = WORDS.find((x) => x.id === next);
  const tNow = w ? w.a : next >= ts0 ? (next - ts0) * 0.02 : 0;
  if (i === 0) return [[ts0, 0.5], [400, 0.2], [tsId(0.02), 0.14], [tsId(1.4), 0.08], [TOK.eot, 0.03], [TOK.nospeech, 0.02]];
  if (next === TOK.eot) return [[TOK.eot, 0.82], [tsId(10.9), 0.06], [13, 0.05], [tsId(10.5), 0.03], [1029, 0.02]];
  if (next >= ts0 && prev >= ts0) return [[next, 0.12], [tsId(+tsOf(next) + 0.02), 0.06], [1029, 0.7], [tsId(+tsOf(next) - 0.4), 0.02], [TOK.eot, 0.02]];
  if (next >= ts0) return [[next, 0.34], [next - 1, 0.17], [next + 1, 0.13], [1029, 0.28], [TOK.eot, 0.02]];
  const alts = ALT.filter((a) => a !== next).slice(i % 3, (i % 3) + 2);
  return [[next, 0.84], [alts[0], 0.06], [alts[1], 0.04], [tsId(Math.max(0, tNow)), 0.03], [TOK.translate, 0.02]];
}

const LOOP = ["fwd", "kv", "filters", "rules", "pick"];

function Cands({ i, ms }) {
  const cs = candsAt(i);
  const kept = cs.filter(([id]) => !masked(id, ms)), z = kept.reduce((a, [, p]) => a + p, 0);
  const best = kept.reduce((a, c) => (c[1] > a[1] ? c : a), [null, -1])[0];
  return (
    <div style={{ display: "grid", gridTemplateColumns: "auto 1fr 64px 1fr 64px", gap: "6px 12px", alignItems: "center", fontFamily: mono, fontSize: 15, maxWidth: 760 }}>
      <span /><span style={{ fontSize: 12, color: T.dim }}>p before</span><span /><span style={{ fontSize: 12, color: T.accent }}>p after the filters</span><span />
      {cs.map(([id, p]) => {
        const x = masked(id, ms), q = x ? 0 : p / z;
        return [
          <span key={`a${id}`} style={{ justifySelf: "end" }}><TokChip id={id} showId={false} state={x ? "struck" : id === best ? "active" : undefined} /></span>,
          <div key={`b${id}`} style={{ height: 14, width: `${p * 100}%`, background: x ? `${T.bad}55` : `${T.soft}55`, borderRadius: 3 }} />,
          <span key={`c${id}`} style={{ color: x ? T.bad : T.soft }}>{p.toFixed(2)}</span>,
          <div key={`d${id}`} style={{ height: 14, width: `${q * 100}%`, background: id === best ? T.accent : `${T.accent}66`, borderRadius: 3, transition: "width .4s" }} />,
          <span key={`e${id}`} style={{ color: x ? T.bad : id === best ? T.accent : T.soft }}>{x ? "−∞" : q.toFixed(2)}</span>,
        ];
      })}
    </div>
  );
}

function KvCache({ i }) {
  const all = [...PREFIX, ...GEN.slice(0, i)];
  const inp = i === 0 ? PREFIX : [GEN[i - 1]];
  return (
    <div style={{ display: "grid", gap: 14 }}>
      <Row gap={14} style={{ alignItems: "flex-start" }}>
        <div style={{ display: "grid", gap: 6 }}>
          <Label>input this call · [1, {inp.length}]</Label>
          <Row gap={5}>{inp.map((id, k) => <TokChip key={k} id={id} state="new" showId={false} />)}</Row>
          <span style={{ fontSize: 14, color: T.dim }}>position offset = {all.length - inp.length}</span>
        </div>
      </Row>
      <Row gap={20} style={{ alignItems: "flex-start" }} wrap>
        <div style={{ display: "grid", gap: 6, flex: 1, minWidth: 320 }}>
          <Label color={T.accent}>self-attn K, V cache · per layer · [1, {all.length}, D]</Label>
          <div style={{ display: "grid", gap: 3 }}>
            {all.map((id, k) => (
              <div key={k} style={{ display: "flex", alignItems: "center", gap: 8, fontSize: 13, fontFamily: mono }}>
                <span style={{ width: 120, textAlign: "right", color: KIND_COL[id >= TOK.ts0 ? "ts" : id >= TOK.eot ? "special" : "text"], whiteSpace: "pre" }}>{tokText(id).replace(/^ /, "·").replace("<|startoftranscript|>", "<|sot|>")}</span>
                <div style={{ height: 12, flex: 1, maxWidth: 200, borderRadius: 2, background: k >= all.length - inp.length ? T.accent : `${T.accent}44`, transition: "all .3s" }} />
                {k >= all.length - inp.length && <span style={{ color: T.accent }}>new</span>}
              </div>
            ))}
          </div>
        </div>
        <div style={{ display: "grid", gap: 6, justifyItems: "start" }}>
          <Label color={T.accent2}>cross-attn K, V · per layer</Label>
          <div style={{ width: 120, height: 200, borderRadius: 6, border: `2px solid ${T.accent2}`, background: `${T.accent2}22`, display: "grid", placeItems: "center", textAlign: "center", fontSize: 14, color: T.accent2 }}>
            [1, 1500, D]<br />from audio
          </div>
          <span style={{ fontSize: 14, color: i === 0 ? T.wire : T.dim }}>{i === 0 ? "computed now" : "reused from call 0"}</span>
        </div>
      </Row>
    </div>
  );
}

export default function DecodingView({ A }) {
  const V = vocabOf(A.size);
  const [idx, setIdx] = useState(0);
  const [itOver, setIt] = useState(null);
  const onIndex = (k) => { setIdx(k); setIt(null); };

  const steps = [
    { id: "detect", block: "detect", caption: "which language? one forward step", vars: () => [["<|en|>", "0.962"], ["write to", "tokens[1]"]] },
    { id: "init", block: "init", caption: "the decoder starts from 3 tokens", vars: () => [["sample_begin", 3], ["sample_len", 224]] },
    { id: "fwd", block: "fwd", it: 0, caption: "only the last row of logits matters", vars: () => [["call", 0], ["logits", `[1, 3, ${V}] → [1, ${V}]`], ["no_speech_prob", "0.004"]] },
    { id: "kv", block: "kv", it: 2, caption: "feed 1 new token, reuse the rest", vars: () => [["input", "[1, 1]"]] },
    { id: "filters", block: "filters", it: 0, caption: "some ids are never allowed", vars: () => [["−∞", "set before softmax"]] },
    { id: "rules", block: "rules", it: 0, caption: "start with a time ≤ 1.00 s", vars: () => [["i", 0]] },
    { id: "rules1", block: "rules", it: 1, caption: "after the opening time: text", vars: () => [["i", 1]] },
    { id: "rules17", block: "rules", it: 17, caption: "time mass wins → close the segment", vars: () => [["i", 17]] },
    { id: "rules18", block: "rules", it: 18, caption: "after text + time: time or end", vars: () => [["i", 18]] },
    { id: "rules19", block: "rules", it: 19, caption: "a pair closed: text only", vars: () => [["i", 19]] },
    { id: "rules29", block: "rules", it: 29, caption: "<|endoftext|> ends the loop", vars: () => [["i", 29]] },
    { id: "pick", block: "pick", it: 5, caption: "greedy, sample, or beam", vars: () => [["default", "T = 0 · beam 5"]] },
    { id: "result", block: "result", caption: "tokens → segments and scores", vars: () => [["avg_logprob", "−0.09"], ["compression", "1.35"]] },
  ];
  const step = steps[idx];
  const it = itOver ?? step.it ?? 0;
  const ms = masksAt(it);
  const inLoop = LOOP.includes(step.block);

  const tags = (s) => ({
    detect: [{ name: "tokens", dims: "1 1", io: "in" }, { name: "audio_features", dims: "1 A D", io: "in" }, { name: "lang probs", dims: "99", io: "out" }],
    init: [{ name: "tokens", dims: "1 3", io: "out" }],
    fwd: [{ name: "tokens", dims: it === 0 ? "1 3" : "1 1", io: "in" }, { name: "logits", dims: it === 0 ? "1 3 V" : "1 1 V" }, { name: "logits[:, -1]", dims: "1 V", io: "out" }],
    kv: [{ name: "tokens", dims: "1 1", io: "in" }, { name: "self k, v", dims: `1 ${3 + it} D` }, { name: "cross k, v", dims: "1 A D" }],
    filters: [{ name: "logits", dims: "1 V", io: "in" }, { name: "logits", dims: "1 V", io: "out" }],
    rules: [{ name: "logits", dims: "1 V", io: "in" }, { name: "logits", dims: "1 V", io: "out" }],
    pick: [{ name: "logits", dims: "1 V", io: "in" }, { name: "tokens", dims: `1 ${4 + it}`, io: "out" }],
    result: [{ name: "tokens", dims: `${GEN.length - 1}`, io: "in" }, { name: "text", dims: [], io: "out" }],
  }[s.block]);

  const tapeNow = (
    <Tape ids={[...PREFIX, ...GEN.slice(0, it + 1)]} newAt={PREFIX.length + it} label={`tokens · iteration ${it}`} showId={false} />
  );

  const scene = (s) => {
    switch (s.block) {
      case "detect": {
        const L = [["en", 0.962], ["de", 0.009], ["nl", 0.006], ["es", 0.005], ["fr", 0.004], ["94 others", 0.014]];
        return (
          <div style={{ display: "grid", gap: 14 }}>
            <Row gap={10}><TokChip id={TOK.sot} big /><Op>+ audio_features → decoder →</Op><Mat name="logits" shape="1 V" rows={1} cols={V} /></Row>
            <VocabStrip masks={[{ a: 0, b: TOK.sot }, { a: TOK.translate, b: 51864 }]} />
            <Label>softmax over the 99 language tokens only</Label>
            <div style={{ display: "grid", gap: 6, maxWidth: 560 }}>
              {L.map(([l, p], i) => (
                <div key={l} style={{ display: "grid", gridTemplateColumns: "110px 1fr 60px", gap: 10, alignItems: "center", fontSize: 15 }}>
                  <span style={{ textAlign: "right", color: KIND_COL.special }}>{i < 5 ? `<|${l}|>` : l}</span>
                  <div style={{ height: 16, width: `${Math.max(1, p * 100)}%`, background: i === 0 ? T.accent : `${T.accent}66`, borderRadius: 3 }} />
                  <span>{p.toFixed(3)}</span>
                </div>
              ))}
            </div>
          </div>
        );
      }
      case "init":
        return (
          <div style={{ display: "grid", gap: 16 }}>
            <Tape ids={PREFIX} label="sot_sequence · timestamps on" />
            <Tape ids={[...PREFIX, TOK.notimestamps]} label="without_timestamps=True" />
            <Tape ids={[TOK.sot_prev, 1029, 406, ...PREFIX]} label="with a prompt (previous text)" />
            <Row gap={24}><Stat k="sample_begin" v="len(initial_tokens)" tone={T.accent} /><Stat k="sample_len" v="448 // 2 = 224" tone={T.wire} /><Stat k="prompt" v="≤ 223 tokens" tone={T.accent2} /></Row>
          </div>
        );
      case "fwd":
        return (
          <div style={{ display: "grid", gap: 14 }}>
            {tapeNow}
            <Row gap={10}>
              <Mat name="tokens" shape={it === 0 ? "1 3" : "1 1"} rows={it === 0 ? 3 : 1} cols={1} color={T.soft} />
              <Op>+ audio_features →</Op>
              <Mat name="logits" shape={it === 0 ? "1 3 V" : "1 1 V"} rows={it === 0 ? 3 : 1} cols={V} />
              <Op>[:, -1] →</Op>
              <Mat name="logits" shape="1 V" rows={1} cols={V} glow />
            </Row>
            {it === 0 && (
              <Row gap={10}>
                <Stat k="call 0 only" v="softmax(logits at <|sot|>)" tone={T.dim} />
                <TokChip id={TOK.nospeech} sub="p = 0.004" />
                <Stat k="→" v="no_speech_prob" tone={T.wire} />
              </Row>
            )}
          </div>
        );
      case "kv":
        return <KvCache i={it} />;
      case "filters":
      case "rules": {
        const show = s.block === "filters" ? ms.filter((m) => ["SuppressTokens", "SuppressBlank"].includes(m.rule)) : ms;
        return (
          <div style={{ display: "grid", gap: 14 }}>
            {tapeNow}
            <VocabStrip masks={show} pick={GEN[it]} />
            <MaskLegend masks={show.filter((m) => !m.hide)} />
            <Cands i={it} ms={show} />
          </div>
        );
      }
      case "pick": {
        const cs = candsAt(it), lg = cs.map(([, p]) => Math.log(p));
        const soft = (tmp) => { const e = lg.map((v) => Math.exp(v / tmp)), z = e.reduce((a, b) => a + b, 0); return e.map((v) => v / z); };
        const beams = [[" Americans", -0.31], [" American", -1.92], [" citizens", -2.6], [" Americans,", -2.9], [" friends", -3.4]];
        return (
          <div style={{ display: "grid", gap: 16 }}>
            {tapeNow}
            <Row gap={22} style={{ alignItems: "flex-start" }}>
              {[[0, "T = 0 · argmax"], [0.5, "T = 0.5 · sample"], [1.0, "T = 1.0 · sample"]].map(([tmp, l]) => {
                const p = tmp === 0 ? cs.map((_, k) => (k === 0 ? 1 : 0)) : soft(tmp);
                return (
                  <div key={l} style={{ display: "grid", gap: 4, minWidth: 200 }}>
                    <Label color={tmp === 0 ? T.accent : T.wire}>{l}</Label>
                    {cs.map(([id], k) => (
                      <div key={id} style={{ display: "grid", gridTemplateColumns: "100px 1fr 44px", gap: 6, alignItems: "center", fontSize: 13 }}>
                        <span style={{ textAlign: "right", whiteSpace: "pre", color: KIND_COL[id >= TOK.ts0 ? "ts" : id >= TOK.eot ? "special" : "text"] }}>{tokText(id).replace(/^ /, "·").slice(0, 12)}</span>
                        <div style={{ height: 12, width: `${p[k] * 100}%`, background: tmp === 0 ? T.accent : T.wire, borderRadius: 2, transition: "width .4s" }} />
                        <span style={{ color: T.soft }}>{p[k].toFixed(2)}</span>
                      </div>
                    ))}
                  </div>
                );
              })}
            </Row>
            <div style={{ display: "grid", gap: 4 }}>
              <Label color={T.accent2}>beam_size = 5 · keep the 5 best Σ log p</Label>
              {beams.map(([t, lp], k) => (
                <div key={t} style={{ display: "grid", gridTemplateColumns: "300px 70px", gap: 10, fontSize: 14, color: k === 0 ? T.accent2 : T.soft }}>
                  <span style={{ whiteSpace: "pre" }}>… my fellow{t}</span><span>{lp.toFixed(2)}</span>
                </div>
              ))}
            </div>
          </div>
        );
      }
      case "result": {
        const segs = [[0, 7.68, WORDS.slice(0, 16)], [7.68, 10.88, WORDS.slice(16)]];
        return (
          <div style={{ display: "grid", gap: 16 }}>
            <Tape ids={GEN} showId={false} label="sampled tokens" />
            <div style={{ display: "grid", gap: 8 }}>
              {segs.map(([a, b, ws]) => (
                <div key={a} style={{ display: "flex", gap: 12, alignItems: "baseline", fontSize: 17 }}>
                  <span style={{ color: KIND_COL.ts, minWidth: 150 }}>[{a.toFixed(2)} → {b.toFixed(2)}]</span>
                  <span style={{ color: T.ink }}>{ws.map((w) => w.s).join("")}</span>
                </div>
              ))}
            </div>
            <Row gap={24}>
              <Stat k="sum_logprob" v="−2.71" />
              <Stat k="÷ (29 + 1)" v="avg −0.09" tone={T.accent} />
              <Stat k="compression_ratio" v="1.35" tone={T.wire} />
              <Stat k="no_speech_prob" v="0.004" tone={T.accent2} />
            </Row>
            <span style={{ fontSize: 15, color: T.dim }}>these three numbers decide retries in transcribe.py →</span>
          </div>
        );
      }
      default:
        return null;
    }
  };

  const controls = inLoop && step.it != null ? (
    <div style={{ display: "flex", gap: 10, alignItems: "center", fontSize: 14, color: T.dim }}>
      iteration
      <input type="range" min={0} max={GEN.length - 1} value={it} onChange={(e) => setIt(+e.target.value)} style={{ width: 200 }} />
      <span style={{ color: T.accent, width: 120, whiteSpace: "pre" }}>{it} → {tokText(GEN[it]).replace(/^ /, "·")}</span>
    </div>
  ) : null;

  return (
    <DimCtx.Provider value={{ V: ["vocabulary", V], A: ["audio tokens", 1500], D: ["width", A.m.D] }}>
      <StepShell title="decoding.py · DecodingTask.run" subtitle="example · jfk.flac · greedy, timestamps on" blocks={FLOW} steps={steps} scene={scene} tags={tags} controls={controls} index={idx} onIndex={onIndex} />
    </DimCtx.Provider>
  );
}
