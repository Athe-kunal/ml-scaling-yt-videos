import { useMemo, useState } from "react";
import { T, mono } from "./theme";
import StepShell, { Stat } from "./StepShell";
import { DimCtx, Row, Op, Mat, MM, MMNote, MatMulCard, Label, Canvas, heat } from "./draw";
import { Heat } from "./plots";
import { logMel, SAMPLE_RATE, HOP_LENGTH } from "./dsp";
import { toyStem, DT } from "./model";

// Qwen2-Audio (transformers: Qwen2AudioForConditionalGeneration), one job per step.
// The audio side is real: the shared clip → 128-mel log-mel → real frame count →
// real number of <|AUDIO|> tokens. Pictures of features / attention use toy weights.

const E = 4096, D = 1280, I = 11008, V = 156032, L_ENC = 32, L_LM = 32, H_LM = 32;
const ID = { im_start: 151644, im_end: 151645, endoftext: 151643, AUDIO: 151646, audio_bos: 151647, audio_eos: 151648 };

const CODE = {
  chat: ["messages = [{'role': 'user', 'content': [{'type': 'audio', 'audio_url': ...}, {'type': 'text', 'text': \"What's that sound?\"}]}]", "text = processor.apply_chat_template(messages, add_generation_prompt=True)", "# audio part → 'Audio 1: <|audio_bos|><|AUDIO|><|audio_eos|>\\n'"],
  feat: ["audio_inputs = feature_extractor(audio, padding='max_length', return_attention_mask=True)", "# WhisperFeatureExtractor: 16 kHz, n_fft 400, hop 160, 128 mels, 30 s → [128, 3000]", "feature_attention_mask = audio_inputs.pop('attention_mask')   # 1 on real frames"],
  count: ["L = feature_attention_mask.sum(-1)   # real mel frames", "after_conv = (L - 1) // 2 + 1        # conv2, stride 2", "num_audio_tokens = (after_conv - 2) // 2 + 1   # avg_pooler, stride 2"],
  expand: ["text = text.replace('<|AUDIO|>', '<|AUDIO|>' * num_audio_tokens)", "input_ids = tokenizer(text)   # <|AUDIO|> = 151646"],
  stem: ["x = gelu(conv1(input_features)); x = gelu(conv2(x))   # [1, 128, 3000] → [1, 1280, 1500]", "x = x.permute(0, 2, 1) + embed_positions.weight   # [1500, 1280], frozen", "# = Whisper-large-v3 encoder, initialised from it"],
  mask: ["padding_mask = arange(1500) >= audio_feat_lengths   # after conv2", "audio_attention_mask = create_bidirectional_mask(..., attention_mask=~padding_mask)", "# no causal mask; padded keys are never attended"],
  layers: ["for layer in self.layers:   # 32 × pre-LN block, d_model 1280, 20 heads, ffn 5120", "    x = layer(x, audio_attention_mask)"],
  pool: ["x = x.permute(0, 2, 1)", "x = self.avg_pooler(x)   # AvgPool1d(2, stride=2): 1500 → 750", "x = self.layer_norm(x.permute(0, 2, 1))   # [1, 750, 1280]"],
  proj: ["audio_features = self.multi_modal_projector(x)   # one Linear(1280, 4096), bias"],
  trim: ["keep = arange(750) < audio_output_lengths[:, None]", "audio_features = audio_features[keep]   # [N, 4096]; the padded tail is dropped"],
  scatter: ["inputs_embeds = embed_tokens(input_ids)   # [1, S, 4096]", "mask = (input_ids == audio_token_id).unsqueeze(-1)", "inputs_embeds = inputs_embeds.masked_scatter(mask, audio_features)   # in order"],
  llm: ["outputs = self.language_model(inputs_embeds=inputs_embeds, attention_mask=...)", "# Qwen2 decoder × 32: RMSNorm → causal self-attn (RoPE, positions 0 … S−1) → RMSNorm → SwiGLU", "# audio slots are just positions in the sequence; no cross-attention"],
  head: ["logits = self.lm_head(hidden_states[:, -1])   # [1, 156032]", "next_token = logits.argmax(-1)"],
  gen: ["# prefill: the whole prompt + audio once → KV cache", "if input_features is not None and input_ids.shape[1] != 1:   # audio merged only at prefill", "    ... merge audio ...", "# decode steps: input_ids [1, 1], reuse the cache"],
  vs: ["# Whisper:      decoder layer: self-attn(text) → cross-attn(q=text, k,v=audio[1500])", "# Qwen2-Audio:  audio[N] written into the token sequence → plain causal self-attn"],
};

const FLOW = [
  { id: "chat", label: "chat template", sub: "<|audio_bos|><|AUDIO|><|audio_eos|>", code: CODE.chat },
  {
    group: "processor",
    items: [
      { id: "feat", label: "log-mel · 128 bins", sub: "WhisperFeatureExtractor", code: CODE.feat },
      { id: "count", label: "count audio tokens", sub: "frames ÷ 2 ÷ 2", code: CODE.count },
      { id: "expand", label: "<|AUDIO|> × N", sub: "placeholders in input_ids", code: CODE.expand },
    ],
  },
  {
    group: "audio_tower · Whisper-large-v3 encoder",
    items: [
      { id: "stem", label: "convs + positions", sub: "[1500, 1280]", code: CODE.stem },
      { id: "mask", label: "mask the padding", sub: "keys past the audio", code: CODE.mask },
      { id: "layers", label: "32 encoder layers", sub: "bidirectional", code: CODE.layers },
      { id: "pool", label: "avg-pool × 2 · LN", sub: "1500 → 750 · 40 ms", code: CODE.pool },
    ],
  },
  { id: "proj", label: "projector", sub: "Linear 1280 → 4096", code: CODE.proj },
  { id: "trim", label: "keep N of 750", sub: "drop the padded tail", code: CODE.trim },
  { id: "scatter", label: "masked_scatter", sub: "audio rows → <|AUDIO|> slots", code: CODE.scatter },
  {
    group: "Qwen2 language model",
    loop: "× 32",
    items: [{ id: "llm", label: "RMSNorm · attn · SwiGLU", sub: "causal over text + audio", code: CODE.llm }],
  },
  { id: "head", label: "lm_head → next token", sub: "[1, 156032]", code: CODE.head },
  { id: "gen", label: "generate", sub: "prefill once · KV cache", code: CODE.gen },
  { id: "vs", label: "vs Whisper decoder", sub: "prefix vs cross-attn", code: CODE.vs },
];

// ---------------------------------------------------------------- the prompt
// pieces of the chat template (text splits are shown as words, not exact BPE ids)
const P = (t, k = "text", id) => ({ t, k, id });
const BEFORE = [P("<|im_start|>", "sp", ID.im_start), P("system"), P("\\n"), P("You are a helpful assistant."), P("<|im_end|>", "sp", ID.im_end), P("\\n"),
  P("<|im_start|>", "sp", ID.im_start), P("user"), P("\\n"), P("Audio 1: "), P("<|audio_bos|>", "sp", ID.audio_bos)];
const AFTER = [P("<|audio_eos|>", "sp", ID.audio_eos), P("\\n"), P("What's that sound?"), P("<|im_end|>", "sp", ID.im_end), P("\\n"), P("<|im_start|>", "sp", ID.im_start), P("assistant"), P("\\n")];
const ANSWER = { melody: ["It", "'s", " a", " piano", " melody", "."], voice: ["A", " person", " is", " singing", " vowels", "."], chirp: ["A", " rising", " electronic", " tone", "."], mic: ["Someone", " is", " speaking", "."] };
const KC = { text: "#8DB4FF", sp: "#FF8AC6", audio: "#6EF3DD", ans: "#FFC94D", ph: "#5A6B80" };
// approximate token counts for the text pieces (for the sequence length)
const NTOK = (p) => (p.k === "sp" ? 1 : Math.max(1, p.t.split(/(?=[ '?:.])/).length));

function Chip({ p, big, sub }) {
  const c = KC[p.k];
  return (
    <div style={{ display: "grid", justifyItems: "center", gap: 2 }}>
      <span style={{ fontFamily: mono, fontSize: big ? 16 : 14, padding: "3px 8px", borderRadius: 6, border: `1.5px solid ${c}`, background: `${c}1c`, color: c, whiteSpace: "pre" }}>{p.t}</span>
      {(p.id || sub) && <span style={{ fontSize: 12, color: T.dim }}>{sub ?? p.id}</span>}
    </div>
  );
}

function AudioRun({ n, label }) {
  return (
    <div style={{ display: "grid", justifyItems: "center", gap: 2 }}>
      <div style={{ display: "flex", gap: 1, padding: "3px 6px", borderRadius: 6, border: `1.5px solid ${KC.audio}`, background: `${KC.audio}1c` }}>
        {Array.from({ length: Math.min(n, 40) }, (_, i) => <span key={i} style={{ width: 4, height: 18, borderRadius: 1, background: KC.audio, opacity: 0.4 + 0.6 * ((i * 37) % 10) / 10 }} />)}
        {n > 40 && <span style={{ color: KC.audio, fontSize: 13, paddingLeft: 4 }}>…</span>}
      </div>
      <span style={{ fontSize: 12, color: KC.audio }}>{label ?? `<|AUDIO|> × ${n}`}</span>
    </div>
  );
}

// the merged sequence as a strip of cells: kind per position
function SeqStrip({ kinds, energy, w = 740, h = 26, hi }) {
  return (
    <Canvas w={w} h={h} deps={[kinds.join(""), energy, hi]} draw={(x) => {
      const u = w / kinds.length;
      kinds.forEach((k, i) => {
        const c = KC[k];
        x.fillStyle = c;
        x.globalAlpha = k === "audio" && energy ? 0.25 + 0.75 * energy[i] : 0.85;
        x.fillRect(i * u, 2, Math.max(1, u - (u > 3 ? 1 : 0)), h - 4);
      });
      x.globalAlpha = 1;
      if (hi != null) { x.strokeStyle = T.ink; x.lineWidth = 2; x.strokeRect(hi * u - 1, 0, u + 2, h); }
    }} />
  );
}

export default function QwenAudioView({ A }) {
  const lm = useMemo(() => logMel(A.y30, 128), [A.y30]);
  const stem = useMemo(() => toyStem(lm.scaled, 128), [lm]);
  const n = A.y.length;
  // WhisperFeatureExtractor: attention mask is the sample mask taken every 160 samples
  const Lf = Math.min(3000, Math.ceil(n / HOP_LENGTH));
  const afterConv = Math.floor((Lf - 1) / 2) + 1;
  const N = Math.floor((afterConv - 2) / 2) + 1;
  const ans = ANSWER[A.name] ?? ANSWER.melody;
  const nBefore = BEFORE.reduce((a, p) => a + NTOK(p), 0), nAfter = AFTER.reduce((a, p) => a + NTOK(p), 0);
  const S = nBefore + N + nAfter;
  const kinds = useMemo(() => [...Array(nBefore).fill("text").map((k, i) => (i === 0 || i === nBefore - 1 ? "sp" : k)), ...Array(N).fill("audio"), ...Array(nAfter).fill("text").map((k, i) => (i === 0 ? "sp" : k))], [nBefore, N, nAfter]);
  // pooled toy features: c2 [DT, 1500] → average pairs → 750
  const pooled = useMemo(() => { const o = new Float32Array(DT * 750); for (let c = 0; c < DT; c++) for (let t = 0; t < 750; t++) o[c * 750 + t] = (stem.c2[c * 1500 + 2 * t] + stem.c2[c * 1500 + 2 * t + 1]) / 2; return o; }, [stem]);
  const rng2 = (arr) => { let a = 1e9, b = -1e9; for (const v of arr) { a = Math.min(a, v); b = Math.max(b, v); } return [a, b]; };
  const c2r = useMemo(() => rng2(stem.c2), [stem]), pr = useMemo(() => rng2(pooled), [pooled]);
  // audio energy per kept token (from the mel), for colouring the audio slots
  const energy = useMemo(() => {
    const e = new Array(S).fill(0); let mx = 1e-9;
    for (let j = 0; j < N; j++) { let s = 0; for (let m = 0; m < 128; m++) for (let f = 4 * j; f < 4 * j + 4 && f < 3000; f++) s += lm.scaled[m * 3000 + f] + 1; e[nBefore + j] = s; mx = Math.max(mx, s); }
    return e.map((v) => v / mx);
  }, [lm, N, S, nBefore]);
  const [qSel, setQ] = useState(null);
  const sMin = useMemo(() => lm.scaled.reduce((a, v) => Math.min(a, v), 9), [lm]), sMax = (lm.max + 4) / 4;

  const steps = [
    { id: "chat", block: "chat", caption: "one placeholder marks the audio", vars: () => [["<|AUDIO|>", ID.AUDIO], ["<|audio_bos|>", ID.audio_bos]] },
    { id: "feat", block: "feat", caption: "same log-mel as Whisper large-v3", vars: () => [["mels", 128], ["real frames", Lf], ["padded", 3000]] },
    { id: "count", block: "count", caption: `${Lf} frames → ${N} audio tokens`, vars: () => [["L", Lf], ["conv2", afterConv], ["pool", N]] },
    { id: "expand", block: "expand", caption: "one slot per audio token", vars: () => [["S", S], ["audio slots", N]] },
    { id: "stem", block: "stem", caption: "the Whisper encoder, unchanged input", vars: () => [["out", "[1500, 1280]"]] },
    { id: "mask", block: "mask", caption: "padding is masked as keys", vars: () => [["valid", `${afterConv} / 1500`]] },
    { id: "layers", block: "layers", caption: "32 layers · every token sees all real ones", vars: () => [["heads", 20], ["ffn", 5120]] },
    { id: "pool", block: "pool", caption: "average neighbours: 20 ms → 40 ms", vars: () => [["out", "[750, 1280]"]] },
    { id: "proj", block: "proj", caption: "one Linear into the LLM's width", vars: () => [["W", "[1280, 4096]"]] },
    { id: "trim", block: "trim", caption: `keep ${N}, drop ${750 - N}`, vars: () => [["N", N], ["check", "N == count(<|AUDIO|>)"]] },
    { id: "scatter", block: "scatter", caption: "audio vectors take the placeholder rows", vars: () => [["inputs_embeds", `[1, ${S}, 4096]`]] },
    { id: "llm", block: "llm", caption: "text reads audio by plain self-attention", vars: () => [["positions", `0 … ${S - 1}`], ["mask", "causal"]] },
    { id: "head", block: "head", caption: "score 156 032 tokens", vars: () => [["next", ans[0]]] },
    { id: "gen", block: "gen", caption: "audio is processed once, in prefill", vars: () => [["prefill", `${S} tokens`], ["then", "1 token / step"]] },
    { id: "vs", block: "vs", caption: "two ways to give a text model ears", vars: () => [] },
  ];

  const tags = (s) => ({
    chat: [{ name: "text", dims: [], io: "out" }],
    feat: [{ name: "audio", dims: String(n), io: "in" }, { name: "input_features", dims: "1 128 3000", io: "out" }, { name: "feature_attention_mask", dims: "1 3000", io: "out" }],
    count: [{ name: "feature_attention_mask", dims: "1 3000", io: "in" }, { name: "num_audio_tokens", dims: [], io: "out" }],
    expand: [{ name: "text", dims: [], io: "in" }, { name: "input_ids", dims: "1 Q", io: "out" }],
    stem: [{ name: "input_features", dims: "1 128 3000", io: "in" }, { name: "x", dims: "1 1500 D", io: "out" }],
    mask: [{ name: "audio_attention_mask", dims: "1 1 1500 1500", io: "out" }],
    layers: [{ name: "x", dims: "1 1500 D", io: "in" }, { name: "x", dims: "1 1500 D", io: "out" }],
    pool: [{ name: "x", dims: "1 1500 D", io: "in" }, { name: "x", dims: "1 P D", io: "out" }],
    proj: [{ name: "x", dims: "1 P D", io: "in" }, { name: "audio_features", dims: "1 P E", io: "out" }],
    trim: [{ name: "audio_features", dims: "1 P E", io: "in" }, { name: "audio_features", dims: "N E", io: "out" }],
    scatter: [{ name: "embed_tokens(input_ids)", dims: "1 Q E", io: "in" }, { name: "audio_features", dims: "N E", io: "in" }, { name: "inputs_embeds", dims: "1 Q E", io: "out" }],
    llm: [{ name: "inputs_embeds", dims: "1 Q E", io: "in" }, { name: "hidden", dims: "1 Q E", io: "out" }],
    head: [{ name: "hidden[:, -1]", dims: "1 E", io: "in" }, { name: "logits", dims: "1 V", io: "out" }],
    gen: [{ name: "input_ids", dims: "1 1", io: "in" }, { name: "logits", dims: "1 V", io: "out" }],
    vs: [],
  }[s.id]);

  const matmuls = (s) => {
    if (s.id === "proj" || s.id === "trim") return <MatMulCard><MM a={["pooled", [1, 750, D]]} b={["projector.weight", [D, E]]} c={["audio_features", [1, 750, E]]} note="+ bias · then keep the first N rows" /></MatMulCard>;
    if (s.id === "pool") return <MatMulCard><MMNote>avg_pooler: x[:, 2t] / 2 + x[:, 2t+1] / 2 → [1, 750, {D}] (no weights)</MMNote></MatMulCard>;
    if (s.id === "llm") return (
      <MatMulCard footer={`prefill over S = ${S} positions · × ${L_LM} = all layers`}>
        <MM a={["RMSNorm(x)", [1, S, E]]} b={["W_q, W_k, W_v", [E, 3 * E]]} c={["q, k, v", [1, S, 3 * E]]} times={L_LM} note={`+ bias · RoPE on q, k · ${H_LM} heads of 128`} />
        <MM a={["q", [1, H_LM, S, 128]]} b={["kᵀ", [1, H_LM, 128, S]]} c={["scores", [1, H_LM, S, S]]} times={L_LM} note="÷ √128, causal −∞ above diagonal, softmax" />
        <MM a={["A", [1, H_LM, S, S]]} b={["v", [1, H_LM, S, 128]]} c={["o", [1, H_LM, S, 128]]} times={L_LM} />
        <MM a={["o", [1, S, E]]} b={["W_o", [E, E]]} c={["attn", [1, S, E]]} times={L_LM} />
        <MM a={["RMSNorm(x)", [1, S, E]]} b={["W_gate, W_up", [E, 2 * I]]} c={["g, u", [1, S, 2 * I]]} times={L_LM} note="SiLU(g) · u" />
        <MM a={["h", [1, S, I]]} b={["W_down", [I, E]]} c={["mlp", [1, S, E]]} times={L_LM} />
      </MatMulCard>
    );
    if (s.id === "head") return <MatMulCard><MM a={["norm(hidden[:, -1])", [1, E]]} b={["lm_head.weight", [E, V]]} c={["logits", [1, V]]} /></MatMulCard>;
    return null;
  };

  const promptRow = (audio) => (
    <div style={{ display: "flex", gap: 5, flexWrap: "wrap", alignItems: "flex-start" }}>
      {BEFORE.map((p, i) => <Chip key={`b${i}`} p={p} />)}
      {audio}
      {AFTER.map((p, i) => <Chip key={`a${i}`} p={p} />)}
    </div>
  );

  const scene = (s) => {
    switch (s.id) {
      case "chat":
        return (
          <div style={{ display: "grid", gap: 16 }}>
            {promptRow(<Chip p={P("<|AUDIO|>", "audio", ID.AUDIO)} big />)}
            <Row gap={24}>
              <Stat k="audio" v="1 placeholder for now" tone={KC.audio} />
              <Stat k="markers" v="<|audio_bos|> … <|audio_eos|>" tone={KC.sp} />
              <Stat k="chat format" v="<|im_start|> role … <|im_end|>" tone={KC.sp} />
            </Row>
          </div>
        );
      case "feat":
        return (
          <div style={{ display: "grid", gap: 10 }}>
            <Label>input_features [1, 128, 3000] · your clip</Label>
            <Heat data={lm.scaled} rows={128} cols={3000} h={170} lo={sMin} hi={sMax} />
            <Label color={KC.audio}>feature_attention_mask · 1 = real frame</Label>
            <div style={{ position: "relative", height: 16, borderRadius: 3, background: T.well, border: `1px solid ${T.rule}`, maxWidth: 760 }}>
              <div style={{ position: "absolute", inset: 0, width: `${(Lf / 3000) * 100}%`, background: KC.audio, borderRadius: 3 }} />
            </div>
            <Row gap={24}>
              <Stat k="real" v={`${Lf} frames = ${(n / SAMPLE_RATE).toFixed(2)} s`} tone={KC.audio} />
              <Stat k="padded" v="3000 frames = 30 s" tone={T.dim} />
              <Stat k="same as" v="Whisper large-v3 audio.py" tone={T.wire} />
            </Row>
          </div>
        );
      case "count":
        return (
          <div style={{ display: "grid", gap: 18 }}>
            {[[`${Lf}`, "mel frames (10 ms)", "#8DB4FF"], [`(${Lf} − 1) // 2 + 1 = ${afterConv}`, "after conv2, stride 2 (20 ms)", T.accent], [`(${afterConv} − 2) // 2 + 1 = ${N}`, "after avg-pool, stride 2 (40 ms)", KC.audio]].map(([v, l, c], i) => (
              <div key={i} style={{ display: "grid", gap: 4 }}>
                <span style={{ fontSize: 22, color: c, fontWeight: 700 }}>{v}</span>
                <span style={{ fontSize: 14, color: T.dim }}>{l}</span>
              </div>
            ))}
            <Row gap={24}><Stat k="1 audio token" v="40 ms" tone={KC.audio} /><Stat k="30 s" v="750 tokens max" tone={T.dim} /><Stat k="your clip" v={`${N} tokens`} tone={KC.audio} /></Row>
          </div>
        );
      case "expand":
        return (
          <div style={{ display: "grid", gap: 16 }}>
            {promptRow(<AudioRun n={N} />)}
            <Label>input_ids · {S} positions</Label>
            <SeqStrip kinds={kinds} />
            <Row gap={24}><Stat k="text + specials" v={`${S - N}`} tone={KC.text} /><Stat k="<|AUDIO|>" v={`${N}`} tone={KC.audio} /><Stat k="S" v={S} tone={T.wire} /></Row>
          </div>
        );
      case "stem":
        return (
          <div style={{ display: "grid", gap: 14 }}>
            <Row gap={10}>
              <Mat name="input_features" shape="128 3000" rows={128} cols={3000} color={T.soft} />
              <Op>conv1 · conv2 (stride 2) →</Op>
              <Mat name="x" shape="1500 D" rows={1500} cols={D} />
              <Op>+ embed_positions</Op>
              <Mat name="x" shape="1500 D" rows={1500} cols={D} glow />
            </Row>
            <Label>after conv2 · toy channels</Label>
            <Heat data={stem.c2} rows={DT} cols={1500} h={90} lo={c2r[0]} hi={c2r[1]} />
            <Row gap={24}><Stat k="architecture" v="= Whisper encoder (tab 2)" tone={T.wire} /><Stat k="positions" v="Embedding, frozen" tone={T.dim} /></Row>
          </div>
        );
      case "mask": {
        const f = afterConv / 1500;
        return (
          <div style={{ display: "grid", gap: 14 }}>
            <Row gap={20} style={{ alignItems: "flex-start" }}>
              <div style={{ display: "grid", gap: 4 }}>
                <Label>queries ↓ · keys →</Label>
                <div style={{ position: "relative", width: 240, height: 240, border: `1px solid ${T.rule}`, borderRadius: 4, background: `repeating-linear-gradient(135deg, ${T.bg} 0 5px, ${T.bad}55 5px 7px)` }}>
                  <div style={{ position: "absolute", left: 0, top: 0, bottom: 0, width: `${f * 100}%`, background: `${T.accent}aa` }} />
                </div>
              </div>
              <div style={{ display: "grid", gap: 8, alignContent: "start" }}>
                <Stat k="valid keys" v={`0 … ${afterConv - 1}`} tone={T.accent} />
                <Stat k="masked keys" v={`${afterConv} … 1499`} tone={T.bad} />
                <Stat k="Whisper" v="attends to the padding too" tone={T.dim} />
              </div>
            </Row>
          </div>
        );
      }
      case "layers":
        return (
          <div style={{ display: "grid", gap: 14 }}>
            <Row gap={10}>
              <Mat name="x" shape="1500 D" rows={1500} cols={D} />
              <Op>32 × (LN → self-attn → LN → MLP 5120)</Op>
              <Mat name="x" shape="1500 D" rows={1500} cols={D} glow />
            </Row>
            <Row gap={24}><Stat k="mask" v="padding keys only" tone={T.accent} /><Stat k="params" v="≈ 640 M" tone={T.dim} /></Row>
          </div>
        );
      case "pool":
        return (
          <div style={{ display: "grid", gap: 12 }}>
            <Label>before · 1500 tokens · 20 ms each</Label>
            <Heat data={stem.c2} rows={DT} cols={1500} c1={Math.min(1500, 2 * N + 40)} h={80} lo={c2r[0]} hi={c2r[1]} />
            <div style={{ fontSize: 17, color: T.accent2 }}>↓ AvgPool1d(2, 2) · mean of each pair · LayerNorm</div>
            <Label color={KC.audio}>after · 750 tokens · 40 ms each</Label>
            <Heat data={pooled} rows={DT} cols={750} c1={Math.min(750, N + 20)} h={80} lo={pr[0]} hi={pr[1]} vline={[[N, KC.audio]]} />
            <Row gap={24}><Stat k="shown" v={`first ${Math.min(750, N + 20)} of 750`} tone={T.dim} /><Stat k="real audio ends at" v={`token ${N}`} tone={KC.audio} /></Row>
          </div>
        );
      case "proj":
        return (
          <div style={{ display: "grid", gap: 14 }}>
            <Row gap={10}>
              <Mat name="x" shape="P D" rows={750} cols={D} />
              <Op>@</Op>
              <Mat name="projector" shape="D E" rows={D} cols={E} color={T.accent2} />
              <Op>+ b =</Op>
              <Mat name="audio_features" shape="P E" rows={750} cols={E} glow />
            </Row>
            <Row gap={24}><Stat k="params" v="1280·4096 + 4096 ≈ 5.2 M" tone={T.accent2} /><Stat k="no MLP, no Q-Former" v="one Linear" tone={T.wire} /></Row>
          </div>
        );
      case "trim":
        return (
          <div style={{ display: "grid", gap: 12 }}>
            <Label>750 rows · keep the first {N}</Label>
            <div style={{ display: "flex", height: 30, maxWidth: 760, borderRadius: 4, overflow: "hidden", border: `1px solid ${T.rule}` }}>
              <div style={{ width: `${(N / 750) * 100}%`, background: KC.audio }} />
              <div style={{ flex: 1, background: `repeating-linear-gradient(135deg, ${T.bg} 0 5px, ${T.bad}55 5px 7px)` }} />
            </div>
            <Row gap={24}><Stat k="kept" v={`${N} × 4096`} tone={KC.audio} /><Stat k="dropped" v={`${750 - N} (silence padding)`} tone={T.bad} /><Stat k="must equal" v={`${N} <|AUDIO|> slots`} tone={T.wire} /></Row>
          </div>
        );
      case "scatter":
        return (
          <div style={{ display: "grid", gap: 12 }}>
            <Label>{"embed_tokens(input_ids) · <|AUDIO|> rows hold the placeholder embedding"}</Label>
            <SeqStrip kinds={kinds.map((k) => (k === "audio" ? "ph" : k))} h={40} />
            <div style={{ fontSize: 17, color: T.accent2 }}>{"↓ masked_scatter: audio row j → j-th <|AUDIO|> slot"}</div>
            <Label color={KC.audio}>inputs_embeds · [1, {S}, 4096]</Label>
            <SeqStrip kinds={kinds} energy={energy} h={40} />
            <Row gap={24}><Stat k="grey" v="same placeholder row ×N" tone={KC.ph} /><Stat k="text rows" v="unchanged" tone={KC.text} /><Stat k="audio rows" v="projected encoder outputs" tone={KC.audio} /><Stat k="order" v="time order kept" tone={T.wire} /></Row>
          </div>
        );
      case "llm":
        return <LlmAttn kinds={kinds} energy={energy} S={S} nBefore={nBefore} N={N} q={qSel} setQ={setQ} />;
      case "head":
        return (
          <div style={{ display: "grid", gap: 14 }}>
            <SeqStrip kinds={kinds} energy={energy} hi={S - 1} />
            <span style={{ fontSize: 15, color: T.dim }}>last position: "assistant\n" → next token</span>
            <div style={{ display: "grid", gap: 6, maxWidth: 520 }}>
              {[[ans[0], 0.71], ["The", 0.12], ["This", 0.08], ["A", 0.04], ["There", 0.02]].filter((v, i, a) => a.findIndex((w) => w[0] === v[0]) === i).map(([t, p], i) => (
                <div key={t} style={{ display: "grid", gridTemplateColumns: "100px 1fr 50px", gap: 10, alignItems: "center", fontSize: 15 }}>
                  <span style={{ textAlign: "right", color: KC.ans }}>{t}</span>
                  <div style={{ height: 14, width: `${p * 100}%`, background: i === 0 ? T.accent : `${T.accent}66`, borderRadius: 3 }} />
                  <span>{p.toFixed(2)}</span>
                </div>
              ))}
            </div>
            <span style={{ fontSize: 13, color: T.dim }}>toy probabilities</span>
          </div>
        );
      case "gen":
        return (
          <div style={{ display: "grid", gap: 14 }}>
            <Label>prefill · one forward over all {S} positions</Label>
            <SeqStrip kinds={kinds} energy={energy} />
            <Label color={KC.ans}>decode · one token per step, KV cache holds the {S} above</Label>
            <Row gap={5}>
              {ans.map((t, i) => <Chip key={i} p={P(t, "ans")} sub={`step ${i + 1}`} />)}
              <Chip p={P("<|im_end|>", "sp", ID.im_end)} sub="stop" />
            </Row>
            <Row gap={24}><Stat k="audio encoder runs" v="once" tone={KC.audio} /><Stat k="input_ids.shape[1] == 1" v="skip the merge" tone={T.wire} /></Row>
          </div>
        );
      case "vs":
        return <VsWhisper N={N} S={S} />;
      default:
        return null;
    }
  };

  return (
    <DimCtx.Provider value={{ D: ["encoder width", D], E: ["LLM width", E], N: ["audio tokens kept", N], P: ["pooled audio tokens", 750], Q: ["sequence length", S], V: ["vocabulary", V], I: ["SwiGLU width", I] }}>
      <StepShell title="Qwen2-Audio · audio → LLM" subtitle={`Qwen2-Audio-7B · your clip ${(n / SAMPLE_RATE).toFixed(2)} s → ${N} audio tokens`} blocks={FLOW} steps={steps} scene={scene} tags={tags} matmuls={matmuls} />
    </DimCtx.Provider>
  );
}

// causal attention over the merged sequence (toy weights): each query spreads over
// earlier positions; text queries after the audio give the loud audio slots more weight
function LlmAttn({ kinds, energy, S, nBefore, N, q, setQ }) {
  const qt = Math.min(q ?? S - 1, S - 1);
  const A = useMemo(() => {
    const M = new Float32Array(S * S);
    for (let i = 0; i < S; i++) {
      let z = 0;
      for (let j = 0; j <= i; j++) {
        let v = 0.15 + Math.exp(-(i - j) * 0.35);
        if (kinds[j] === "sp") v += 0.4;
        if (kinds[j] === "audio") v += kinds[i] === "audio" ? 0.3 * energy[j] + 0.4 * Math.exp(-Math.abs(i - j) * 0.2) : 1.4 * energy[j];
        M[i * S + j] = v; z += v;
      }
      for (let j = 0; j <= i; j++) M[i * S + j] /= z;
    }
    return M;
  }, [kinds, energy, S]);
  const row = Array.from({ length: S }, (_, j) => A[qt * S + j]);
  const audioMass = row.slice(nBefore, nBefore + N).reduce((a, b) => a + b, 0);
  const size = 260;
  return (
    <div style={{ display: "grid", gap: 12 }}>
      <Row gap={16} wrap={false} style={{ alignItems: "flex-start" }}>
        <div style={{ display: "grid", gap: 4 }}>
          <Label>weights · {S} × {S} · causal</Label>
          <Canvas w={size} h={size} deps={[A, S, qt]} draw={(x) => {
            const u = size / S;
            for (let i = 0; i < S; i++) {
              let m = 0;
              for (let j = 0; j <= i; j++) m = Math.max(m, A[i * S + j]);
              for (let j = 0; j < S; j++) {
                if (j > i) { x.fillStyle = T.well; } else { const [r, g, b] = heat(Math.sqrt(A[i * S + j] / m)); x.fillStyle = `rgb(${r},${g},${b})`; }
                x.fillRect(j * u, i * u, u + 0.5, u + 0.5);
              }
            }
            x.strokeStyle = KC.audio; x.lineWidth = 1.5;
            x.strokeRect(nBefore * u, nBefore * u, N * u, size - nBefore * u);
            x.strokeStyle = T.ink; x.strokeRect(0, qt * u, size, Math.max(u, 2));
          }} />
          <span style={{ fontSize: 12, color: KC.audio }}>teal box: audio keys</span>
        </div>
        <div style={{ display: "grid", gap: 8, flex: 1, minWidth: 0 }}>
          <Label>sequence</Label>
          <SeqStrip kinds={kinds} energy={energy} w={460} hi={qt} />
          <Label color={T.accent}>position {qt} ({kinds[qt]}) attends to ↓</Label>
          <Canvas w={460} h={100} deps={[row.join(","), qt]} draw={(x) => {
            const u = 460 / S, m = Math.max(...row);
            row.forEach((v, j) => { x.fillStyle = KC[kinds[j]]; x.globalAlpha = j > qt ? 0.15 : 0.9; const h = (v / m) * 96; x.fillRect(j * u, 100 - h, Math.max(1, u - 0.5), h); });
            x.globalAlpha = 1;
          }} />
          <Row gap={20}><Stat k="mass on audio" v={audioMass.toFixed(2)} tone={KC.audio} /><Stat k="picture" v="toy weights" tone={T.dim} /></Row>
        </div>
      </Row>
      <div style={{ display: "flex", gap: 10, alignItems: "center", fontSize: 14, color: T.dim }}>
        query position <input type="range" min={0} max={S - 1} value={qt} onChange={(e) => setQ(+e.target.value)} style={{ width: 260 }} /><span style={{ color: T.wire }}>{qt}</span>
      </div>
      <Row gap={24}><Stat k="RoPE positions" v="0 … S−1, audio included" tone={T.wire} /><Stat k="audio ↔ audio" v="causal here too" tone={KC.audio} /><Stat k="heads" v="32 × 128" tone={T.accent2} /></Row>
    </div>
  );
}

function VsWhisper({ N, S }) {
  const row = (k, a, b) => (
    <div key={k} style={{ display: "grid", gridTemplateColumns: "170px 1fr 1fr", gap: 14, fontSize: 15, padding: "6px 0", borderTop: `1px solid ${T.rule}` }}>
      <span style={{ color: T.dim }}>{k}</span><span style={{ color: T.soft }}>{a}</span><span style={{ color: T.ink }}>{b}</span>
    </div>
  );
  return (
    <div style={{ display: "grid", gap: 14 }}>
      <Row gap={30} style={{ alignItems: "flex-start" }}>
        <div style={{ display: "grid", gap: 6 }}>
          <Label>Whisper decoder layer</Label>
          <div style={{ display: "grid", gap: 4, fontSize: 14 }}>
            <span style={{ padding: "4px 10px", border: `1px solid ${KC.text}`, borderRadius: 6, color: KC.text }}>text self-attn (causal)</span>
            <span style={{ padding: "4px 10px", border: `1px solid ${KC.audio}`, borderRadius: 6, color: KC.audio }}>cross-attn → audio [1500]</span>
            <span style={{ padding: "4px 10px", border: `1px solid ${T.rule}`, borderRadius: 6, color: T.soft }}>MLP</span>
          </div>
        </div>
        <div style={{ display: "grid", gap: 6 }}>
          <Label>Qwen2 layer</Label>
          <div style={{ display: "grid", gap: 4, fontSize: 14 }}>
            <span style={{ padding: "4px 10px", border: `1px solid ${T.wire}`, borderRadius: 6, color: T.wire }}>self-attn over [text ‖ audio {N} ‖ text] (causal)</span>
            <span style={{ padding: "4px 10px", border: `1px solid ${T.rule}`, borderRadius: 6, color: T.soft }}>SwiGLU MLP</span>
          </div>
        </div>
      </Row>
      <div>
        <div style={{ display: "grid", gridTemplateColumns: "170px 1fr 1fr", gap: 14, fontSize: 13, color: T.dim }}><span /><span>Whisper</span><span>Qwen2-Audio</span></div>
        {row("audio enters", "cross-attn keys/values, every layer", "rows in the token sequence")}
        {row("audio tokens", "1500 · 20 ms · padding kept", `${N} · 40 ms · padding dropped`)}
        {row("connector", "none (same model)", "avg-pool + Linear 1280 → 4096")}
        {row("text model", "4–32 layers, 384–1280 wide", "Qwen-7B, 32 layers, 4096 wide")}
        {row("output", "transcript + timestamps", "any text answer")}
        {row("sequence length", "text only", `text + audio = ${S}`)}
      </div>
    </div>
  );
}
