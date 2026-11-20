import { useMemo } from "react";
import { T, mono } from "./theme";
import StepShell, { Stat } from "./StepShell";
import { DimCtx, Row, Op, Mat, Label, Canvas, heat } from "./draw";
import { TOK, WORDS, rng, toyCross } from "./model";
import { Tape, TokChip, KIND_COL } from "./tokens";

// transcribe.py on a made-up 66 s recording: three 30 s windows, a seek that jumps
// to the last complete timestamp, a temperature fallback, a skipped silent window,
// and word timestamps from cross-attention + DTW. Compression ratios are real (zlib).

const CODE = {
  mel: ["mel = log_mel_spectrogram(audio, n_mels, padding=N_SAMPLES)   # + 30 s of zeros", "content_frames = mel.shape[-1] - N_FRAMES   # real frames only"],
  lang: ["mel_segment = pad_or_trim(mel, N_FRAMES)   # first 30 s", "_, probs = model.detect_language(mel_segment)"],
  win: ["segment_size = min(N_FRAMES, content_frames - seek)", "mel_segment = pad_or_trim(mel[:, seek : seek + segment_size], N_FRAMES)", "time_offset = seek * HOP_LENGTH / SAMPLE_RATE"],
  prompt: ["decode_options['prompt'] = all_tokens[prompt_reset_since:]", "# DecodingTask keeps the last 223: [<|startofprev|>] + prompt[-223:] + sot_sequence", "if result.temperature > 0.5: prompt_reset_since = len(all_tokens)"],
  fallback: ["for t in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0):", "    result = model.decode(segment, DecodingOptions(temperature=t, ...))", "    if result.compression_ratio > 2.4: retry   # too repetitive", "    elif result.avg_logprob < -1.0: retry   # too unsure", "    else: break"],
  nospeech: ["should_skip = result.no_speech_prob > 0.6", "if result.avg_logprob > -1.0: should_skip = False", "if should_skip: seek += segment_size; continue"],
  seek: ["consecutive = where(is_ts[:-1] & is_ts[1:]) + 1   # segment boundaries", "if single_timestamp_ending: seek += segment_size", "else: seek += last_timestamp_pos * input_stride   # 2 mel frames per token"],
  words: ["w = stack([cross_qk[layer][head] for layer, head in alignment_heads])", "w = median_filter(softmax(w * qk_scale), 7); matrix = w.mean(0)", "text_idx, time_idx = dtw(-matrix)   # monotonic best path", "word start/end = time_idx where text_idx jumps × 0.02 s"],
  out: ["return dict(text=..., segments=all_segments, language=language)"],
};

const FLOW = [
  { id: "mel", label: "log-mel, whole file", sub: "+ 30 s padding", code: CODE.mel },
  { id: "lang", label: "detect language", sub: "first 30 s only", code: CODE.lang },
  {
    group: "while seek < content_frames",
    loop: "30 s windows",
    items: [
      { id: "win", label: "cut a window", sub: "mel[:, seek : seek + 3000]", code: CODE.win },
      { id: "prompt", label: "previous text → prompt", sub: "<|startofprev|> …", code: CODE.prompt },
      { id: "fallback", label: "decode_with_fallback", sub: "T = 0 → 0.2 → … 1.0", code: CODE.fallback },
      { id: "nospeech", label: "silence check", sub: "no_speech_prob > 0.6", code: CODE.nospeech },
      { id: "seek", label: "advance seek", sub: "to the last full pair", code: CODE.seek },
    ],
  },
  { id: "words", label: "word timestamps", sub: "cross-attn + DTW · optional", code: CODE.words },
  { id: "out", label: "segments + text", sub: "start · end · text", code: CODE.out },
];

const DUR = 66, PAD = 30, TOTAL = DUR + PAD;
const SPEECH = [[0.3, 2.6], [3.3, 7.68], [8.2, 10.7], [12.2, 19.4], [19.6, 27.2], [27.5, 31.4], [33, 41.8], [41.9, 46.2]];
const SEGS = [
  { a: 0, b: 7.68, w: 1, text: "And so my fellow Americans, ask not what your country can do for you," },
  { a: 7.68, b: 10.88, w: 1, text: "ask what you can do for your country." },
  { a: 12.2, b: 19.4, w: 1, text: "We observe today not a victory of party, but a celebration of freedom," },
  { a: 19.4, b: 27.36, w: 1, text: "symbolizing an end, as well as a beginning," },
  { a: 27.36, b: 31.46, w: 2, text: "signifying renewal, as well as change." },
  { a: 33.0, b: 41.8, w: 2, text: "We choose to go to the moon in this decade and do the other things," },
  { a: 41.8, b: 46.2, w: 2, text: "not because they are easy, but because they are hard." },
];
const WIN = [{ a: 0, b: 30 }, { a: 27.36, b: 57.36 }, { a: 57.36, b: 66 }];

// the recording's loudness envelope (toy)
function envelope(n) {
  const r = rng(11), out = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const t = (i / n) * TOTAL;
    let a = 0.01;
    if (t < DUR) {
      if (SPEECH.some(([x, y]) => t >= x && t <= y)) a = 0.35 + 0.55 * Math.abs(Math.sin(t * 9.1)) * r();
      else if (t > 50) a = 0.06 + 0.05 * r(); // quiet room noise
      else a = 0.02 * r();
    } else a = 0;
    out[i] = a;
  }
  return out;
}

function Timeline({ win, seek, segs = [], skip, upto = TOTAL, nextSeek, cut }) {
  const W = 760, H = 64, X = (t) => (t / TOTAL) * W;
  const env = useMemo(() => envelope(W), []);
  return (
    <div style={{ display: "grid", gap: 4 }}>
      <div style={{ position: "relative", width: "100%", maxWidth: W }}>
        <Canvas w={W} h={H} deps={[upto]} draw={(x) => {
          x.fillStyle = `repeating-linear-gradient(0)`;
          for (let i = 0; i < W; i++) {
            const t = (i / W) * TOTAL, a = env[i];
            x.fillStyle = t >= DUR ? T.rule : T.accent;
            x.globalAlpha = t >= DUR ? 0.5 : 0.85;
            x.fillRect(i, H / 2 - a * (H / 2 - 2), 1, Math.max(1, 2 * a * (H / 2 - 2)));
          }
          x.globalAlpha = 1;
          x.strokeStyle = T.rule; x.setLineDash([4, 3]);
          x.beginPath(); x.moveTo(X(DUR), 0); x.lineTo(X(DUR), H); x.stroke(); x.setLineDash([]);
        }} />
        {win && (
          <div style={{ position: "absolute", top: -4, height: H + 8, left: `${(win.a / TOTAL) * 100}%`, width: `${((win.b - win.a) / TOTAL) * 100}%`, border: `2px solid ${skip ? T.bad : T.wire}`, borderRadius: 5, background: `${skip ? T.bad : T.wire}14`, transition: "all .5s" }}>
            <span style={{ position: "absolute", top: -21, left: 0, fontSize: 13, color: skip ? T.bad : T.wire, whiteSpace: "nowrap" }}>{win.label}</span>
          </div>
        )}
        {win?.padTo && (
          <div style={{ position: "absolute", top: -4, height: H + 8, left: `${(win.b / TOTAL) * 100}%`, width: `${((win.padTo - win.b) / TOTAL) * 100}%`, border: `2px dashed ${T.dim}`, borderRadius: 5, transition: "all .5s" }} />
        )}
        {seek != null && <div style={{ position: "absolute", top: -8, height: H + 16, left: `${(seek / TOTAL) * 100}%`, width: 3, background: T.accent2, transition: "left .5s" }} />}
        {nextSeek != null && <div style={{ position: "absolute", top: -8, height: H + 16, left: `${(nextSeek / TOTAL) * 100}%`, width: 3, background: T.accent2, opacity: 0.5, borderLeft: `1px dashed ${T.accent2}` }} />}
      </div>
      <div style={{ position: "relative", height: 26, width: "100%", maxWidth: W }}>
        {segs.map((s, i) => (
          <div key={i} title={s.text} style={{ position: "absolute", top: 2, height: 20, left: `${(s.a / TOTAL) * 100}%`, width: `${((s.b - s.a) / TOTAL) * 100}%`, background: s.cut ? `${T.bad}33` : `${KIND_COL.ts}55`, border: `1px solid ${s.cut ? T.bad : KIND_COL.ts}`, borderRadius: 3, transition: "all .4s" }} />
        ))}
        {cut && <div style={{ position: "absolute", top: 2, height: 20, left: `${(cut.a / TOTAL) * 100}%`, width: `${((cut.b - cut.a) / TOTAL) * 100}%`, background: `repeating-linear-gradient(135deg, ${T.bg} 0 4px, ${T.bad}66 4px 6px)`, borderRadius: 3 }} />}
      </div>
      <div style={{ position: "relative", height: 16, width: "100%", maxWidth: W, fontSize: 12, color: T.dim }}>
        {[0, 10, 20, 30, 40, 50, 60, 66, 96].map((t) => (
          <span key={t} style={{ position: "absolute", left: `${(t / TOTAL) * 100}%`, transform: t ? (t === 96 ? "translateX(-100%)" : "translateX(-50%)") : "none", color: t === 66 ? T.soft : T.dim }}>{t === 96 ? "+pad" : `${t}s`}</span>
        ))}
      </div>
    </div>
  );
}

const SegList = ({ segs, hi }) => (
  <div style={{ display: "grid", gap: 5 }}>
    {segs.map((s, i) => (
      <div key={i} style={{ display: "flex", gap: 12, alignItems: "baseline", fontSize: 15, opacity: hi && !hi(s) ? 0.45 : 1 }}>
        <span style={{ color: s.cut ? T.bad : KIND_COL.ts, minWidth: 150, whiteSpace: "nowrap" }}>[{s.a.toFixed(2)} → {s.cut ? "…" : s.b.toFixed(2)}]</span>
        <span style={{ color: s.cut ? T.bad : T.ink, textDecoration: s.cut ? "line-through" : "none" }}>{s.text}</span>
      </div>
    ))}
  </div>
);

const Verdict = ({ T: t, ratio, lp, ok, text }) => (
  <div style={{ display: "grid", gap: 8, padding: "12px 14px", borderRadius: 10, border: `1.5px solid ${ok ? T.accent : T.bad}`, background: `${ok ? T.accent : T.bad}0e` }}>
    <Row gap={20}>
      <Stat k="temperature" v={t.toFixed(1)} tone={T.wire} />
      <Stat k="compression_ratio" v={`${ratio.toFixed(2)} ${ratio > 2.4 ? "> 2.4 ✗" : "≤ 2.4 ✓"}`} tone={ratio > 2.4 ? T.bad : T.accent} />
      <Stat k="avg_logprob" v={`${lp.toFixed(2)} ${lp < -1 ? "< −1 ✗" : "≥ −1 ✓"}`} tone={lp < -1 ? T.bad : T.accent} />
      <span style={{ fontSize: 18, fontWeight: 700, color: ok ? T.accent : T.bad }}>{ok ? "keep" : "retry"}</span>
    </Row>
    <div style={{ fontSize: 15, color: ok ? T.ink : T.bad, lineHeight: 1.5 }}>“{text}”</div>
  </div>
);

// dynamic time warping on −A: the monotonic path through tokens × frames
function dtw(A, n, m) {
  const C = new Float32Array((n + 1) * (m + 1)).fill(Infinity), B = new Int8Array((n + 1) * (m + 1));
  C[0] = 0;
  for (let i = 1; i <= n; i++)
    for (let j = 1; j <= m; j++) {
      const c = -A[(i - 1) * m + (j - 1)];
      const a = C[(i - 1) * (m + 1) + j - 1], b = C[(i - 1) * (m + 1) + j], d = C[i * (m + 1) + j - 1];
      let best = a, k = 0;
      if (b < best) { best = b; k = 1; }
      if (d < best) { best = d; k = 2; }
      C[i * (m + 1) + j] = c + best; B[i * (m + 1) + j] = k;
    }
  const path = [];
  let i = n, j = m;
  while (i > 0 && j > 0) {
    path.push([i - 1, j - 1]);
    const k = B[i * (m + 1) + j];
    if (k === 0) { i--; j--; } else if (k === 1) i--; else j--;
  }
  return path.reverse();
}

function WordAlign() {
  const ws = WORDS.slice(0, 16), n = ws.length, m = 400; // first 8 s
  const A = useMemo(() => toyCross(ws, m, 5), [ws]);
  const path = useMemo(() => dtw(A, n, m), [A, n]);
  const starts = ws.map((_, i) => path.find(([a]) => a === i)?.[1] ?? 0);
  const ends = ws.map((_, i) => { const p = path.filter(([a]) => a === i); return p.length ? p[p.length - 1][1] + 1 : 0; });
  const Wd = 620, rh = 18;
  return (
    <div style={{ display: "grid", gap: 10 }}>
      <div style={{ display: "grid", gridTemplateColumns: "100px 1fr", gap: 6 }}>
        <div style={{ display: "grid", gridAutoRows: `${rh}px`, fontSize: 12, alignItems: "center", justifyItems: "end" }}>
          {ws.map((w, i) => <span key={i} style={{ color: KIND_COL.text, whiteSpace: "pre" }}>{w.s.replace(/^ /, "·")}</span>)}
        </div>
        <div>
          <Canvas w={Wd} h={rh * n} deps={[A, path]} draw={(x) => {
            for (let i = 0; i < n; i++) {
              let mx = 0;
              for (let j = 0; j < m; j++) mx = Math.max(mx, A[i * m + j]);
              for (let j = 0; j < m; j++) { const [r, g, b] = heat(Math.sqrt(A[i * m + j] / mx) * 0.85); x.fillStyle = `rgb(${r},${g},${b})`; x.fillRect((j / m) * Wd, i * rh, Wd / m + 0.6, rh); }
            }
            x.strokeStyle = T.accent; x.lineWidth = 2.5; x.beginPath();
            path.forEach(([i, j], k) => { const px = ((j + 0.5) / m) * Wd, py = (i + 0.5) * rh; if (k) x.lineTo(px, py); else x.moveTo(px, py); });
            x.stroke();
          }} />
          <div style={{ display: "flex", justifyContent: "space-between", fontSize: 12, color: T.dim }}>{[0, 2, 4, 6, 8].map((s) => <span key={s}>{s}s</span>)}</div>
        </div>
      </div>
      <div style={{ display: "flex", gap: 6, flexWrap: "wrap" }}>
        {ws.slice(0, 8).map((w, i) => (
          <span key={i} style={{ fontSize: 14, padding: "3px 8px", borderRadius: 5, background: T.well, border: `1px solid ${T.rule}` }}>
            <span style={{ color: KIND_COL.text }}>{w.s.trim()}</span> <span style={{ color: T.wire }}>{(starts[i] * 0.02).toFixed(2)}–{(ends[i] * 0.02).toFixed(2)}</span>
          </span>
        ))}
      </div>
    </div>
  );
}

export default function TranscribeView({ A }) {
  const nMels = A.m.mels;
  const segsW1 = SEGS.filter((s) => s.w === 1);
  const cutW1 = { a: 27.36, b: 30, text: "signifying renewal, as well as", cut: true };
  const rep = "signifying renewal, as well as change. ask what you can do for your country. ask what you can do for your country. ask what you can do for your country. … (×6)";

  const steps = [
    { id: "mel", block: "mel", caption: "one log-mel for the whole file", vars: () => [["audio", "66 s"], ["mel", `[${nMels}, 9600]`], ["content_frames", 6600]] },
    { id: "lang", block: "lang", caption: "language from the first 30 s", vars: () => [["language", "en"]] },
    { id: "w1", block: "win", caption: "window 1 · seek = 0", vars: () => [["seek", 0], ["segment_size", 3000], ["time_offset", "0.00 s"]] },
    { id: "d1", block: "fallback", caption: "T = 0 works · 4 segments + a cut-off tail", vars: () => [["T", "0.0"], ["ratio", "1.58"], ["avg_logprob", "−0.18"]] },
    { id: "s1", block: "seek", caption: "ends mid-sentence → seek to the last pair", vars: () => [["last ts", "<|27.36|>"], ["seek", "1368 × 2 = 2736"]] },
    { id: "p2", block: "prompt", caption: "window 2 starts with the text so far", vars: () => [["prompt", "≤ 223 tokens"]] },
    { id: "w2", block: "win", caption: "window 2 · seek = 2736", vars: () => [["seek", 2736], ["time_offset", "27.36 s"]] },
    { id: "d2a", block: "fallback", caption: "T = 0 loops on itself", vars: () => [["ratio", "3.55 > 2.4"]] },
    { id: "d2b", block: "fallback", caption: "retry at T = 0.2", vars: () => [["ratio", "1.44"], ["T ≤ 0.5", "prompt kept"]] },
    { id: "s2", block: "seek", caption: "ends on one timestamp → jump 30 s", vars: () => [["seek", "2736 + 3000 = 5736"]] },
    { id: "w3", block: "win", caption: "window 3 · only 8.64 s left", vars: () => [["segment_size", "6600 − 5736 = 864"]] },
    { id: "ns", block: "nospeech", caption: "silence → skip the window", vars: () => [["no_speech_prob", "0.83 > 0.6"], ["avg_logprob", "−1.31"]] },
    { id: "words", block: "words", caption: "which audio token does each word read?", vars: () => [["heads", "alignment_heads"]] },
    { id: "out", block: "out", caption: "seven segments with times", vars: () => [["segments", 7]] },
  ];

  const tags = (s) => ({
    mel: [{ name: "audio", dims: "1056000", io: "in" }, { name: "mel", dims: "M 9600", io: "out" }],
    lang: [{ name: "mel[:, :3000]", dims: "M 3000", io: "in" }, { name: "lang", dims: [], io: "out" }],
    win: [{ name: "mel", dims: "M 9600", io: "in" }, { name: "mel_segment", dims: "M 3000", io: "out" }],
    prompt: [{ name: "all_tokens", dims: "≤223", io: "in" }, { name: "initial_tokens", dims: "1 n", io: "out" }],
    fallback: [{ name: "mel_segment", dims: "M 3000", io: "in" }, { name: "DecodingResult", dims: [], io: "out" }],
    nospeech: [{ name: "DecodingResult", dims: [], io: "in" }],
    seek: [{ name: "tokens", dims: "n", io: "in" }, { name: "seek", dims: [], io: "out" }],
    words: [{ name: "cross qk", dims: "heads n A", io: "in" }, { name: "word times", dims: "words 2", io: "out" }],
    out: [{ name: "segments", dims: "7", io: "out" }],
  }[s.block]);

  const scene = (s) => {
    switch (s.id) {
      case "mel":
        return (
          <div style={{ display: "grid", gap: 14 }}>
            <Timeline />
            <Row gap={24}>
              <Stat k="66 s × 100" v="6600 frames" tone={T.accent} />
              <Stat k="+ padding" v="3000 frames" tone={T.dim} />
              <Stat k="max − 8 floor" v="over the whole file" tone={T.wire} />
            </Row>
          </div>
        );
      case "lang":
        return (
          <div style={{ display: "grid", gap: 14 }}>
            <Timeline win={{ a: 0, b: 30, label: "pad_or_trim(mel, 3000)" }} />
            <Row gap={10}><TokChip id={TOK.sot} big /><Op>→ softmax over languages →</Op><TokChip id={TOK.en} big sub="0.962" /></Row>
          </div>
        );
      case "w1":
      case "w2":
      case "w3": {
        const w = WIN[{ w1: 0, w2: 1, w3: 2 }[s.id]];
        const done = s.id === "w1" ? [] : s.id === "w2" ? segsW1 : SEGS;
        return (
          <div style={{ display: "grid", gap: 14 }}>
            <Timeline win={{ ...w, label: `mel[:, ${Math.round(w.a * 100)} : ${Math.round(w.b * 100)}]`, padTo: s.id === "w3" ? w.a + 30 : undefined }} seek={w.a} segs={done} />
            <Row gap={10}>
              <Mat name="mel_segment" shape={`M ${Math.round((w.b - w.a) * 100)}`} rows={nMels} cols={Math.round((w.b - w.a) * 100)} />
              <Op>pad_or_trim →</Op>
              <Mat name="mel_segment" shape="M 3000" rows={nMels} cols={3000} glow />
              <Op>→ encoder → decode</Op>
            </Row>
            {s.id === "w3" && <Stat k="zeros added" v="3000 − 864 = 2136 frames" tone={T.dim} />}
          </div>
        );
      }
      case "d1":
        return (
          <div style={{ display: "grid", gap: 14 }}>
            <Timeline win={{ ...WIN[0], label: "window 1" }} seek={0} segs={segsW1} cut={cutW1} />
            <Verdict T={0} ratio={1.58} lp={-0.18} ok text="And so my fellow Americans, … as well as a beginning, signifying renewal, as well as" />
            <SegList segs={[...segsW1, cutW1]} />
          </div>
        );
      case "s1": {
        return (
          <div style={{ display: "grid", gap: 14 }}>
            <Timeline win={{ ...WIN[0], label: "window 1" }} seek={0} nextSeek={27.36} segs={segsW1} cut={cutW1} />
            <Tape ids={[2863, 11, TOK.ts0 + 1368, TOK.ts0 + 1368, 35516, 11, 382, 731, 382]} label="last tokens · a pair, then text with no closing time" subs={{ 2: "end", 3: "start", 8: "window ends" }} />
            <Row gap={24}>
              <Stat k="is_ts[-2:]" v="[False, False]" tone={T.bad} />
              <Stat k="last pair" v="<|27.36|> = pos 1368" tone={KIND_COL.ts} />
              <Stat k="seek +=" v="1368 × 2 = 2736 frames" tone={T.accent2} />
            </Row>
            <span style={{ fontSize: 15, color: T.dim }}>the cut-off text is thrown away and decoded again in window 2</span>
          </div>
        );
      }
      case "p2":
        return (
          <div style={{ display: "grid", gap: 14 }}>
            <Row gap={5}>
              <TokChip id={TOK.sot_prev} big />
              <span style={{ fontSize: 15, color: KIND_COL.text, padding: "4px 8px", border: `1px dashed ${KIND_COL.text}`, borderRadius: 6 }}>… a celebration of freedom, symbolizing an end, as well as a beginning,</span>
              <TokChip id={TOK.sot} big /><TokChip id={TOK.en} big /><TokChip id={TOK.transcribe} big />
            </Row>
            <Row gap={24}>
              <Stat k="prompt" v="all_tokens[prompt_reset_since:][-223:]" tone={KIND_COL.text} />
              <Stat k="why" v="keeps names and style consistent" tone={T.accent} />
              <Stat k="risk" v="repeats can carry over" tone={T.bad} />
            </Row>
          </div>
        );
      case "d2a":
      case "d2b":
        return (
          <div style={{ display: "grid", gap: 14 }}>
            <Timeline win={{ ...WIN[1], label: "window 2" }} seek={27.36} segs={s.id === "d2b" ? SEGS.filter((x) => x.w <= 2) : segsW1} />
            <Verdict T={0} ratio={3.55} lp={-0.41} ok={false} text={rep} />
            {s.id === "d2b" && <Verdict T={0.2} ratio={1.44} lp={-0.33} ok text="signifying renewal, as well as change. We choose to go to the moon … but because they are hard." />}
          </div>
        );
      case "s2":
        return (
          <div style={{ display: "grid", gap: 14 }}>
            <Timeline win={{ ...WIN[1], label: "window 2" }} seek={27.36} nextSeek={57.36} segs={SEGS} />
            <Tape ids={[1152, 13, TOK.ts0 + 942, TOK.eot]} label="last tokens · text, one timestamp, end" subs={{ 2: "18.84 s" }} />
            <Row gap={24}>
              <Stat k="is_ts[-2:]" v="[False, True]" tone={T.accent} />
              <Stat k="single_timestamp_ending" v="True" tone={T.accent} />
              <Stat k="seek +=" v="segment_size = 3000" tone={T.accent2} />
            </Row>
          </div>
        );
      case "ns":
        return (
          <div style={{ display: "grid", gap: 14 }}>
            <Timeline win={{ ...WIN[2], label: "window 3", padTo: WIN[2].a + 30 }} seek={57.36} nextSeek={66} segs={SEGS} skip />
            <Row gap={24}>
              <Stat k="no_speech_prob" v="0.83 > 0.6" tone={T.bad} />
              <Stat k="avg_logprob" v="−1.31 < −1.0" tone={T.bad} />
              <Stat k="→" v="skip · seek += 864" tone={T.accent2} />
            </Row>
            <Row gap={10}><span style={{ fontSize: 15, color: T.dim }}>p(</span><TokChip id={TOK.nospeech} /><span style={{ fontSize: 15, color: T.dim }}>) read at the {"<|sot|>"} position on call 0</span></Row>
          </div>
        );
      case "words":
        return <WordAlign />;
      case "out":
        return (
          <div style={{ display: "grid", gap: 14 }}>
            <Timeline segs={SEGS} seek={66} />
            <SegList segs={SEGS} />
          </div>
        );
      default:
        return null;
    }
  };

  return (
    <DimCtx.Provider value={{ M: ["mel bins", nMels], A: ["audio tokens", 1500] }}>
      <StepShell title="transcribe.py · long audio" subtitle="example · 66 s recording · 3 windows" blocks={FLOW} steps={steps} scene={scene} tags={tags} />
    </DimCtx.Provider>
  );
}
