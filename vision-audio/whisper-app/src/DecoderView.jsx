import { useMemo } from "react";
import { T, mono } from "./theme";
import StepShell, { Stat } from "./StepShell";
import { DimCtx, Row, Op, Mat, MM, MMNote, MatMulCard, BlockMMs, Label, Canvas, heatCss, heat } from "./draw";
import { Plot } from "./plots";
import { PREFIX, GEN, TOK, WORDS, tokText, toyCross, vocabOf, N_TEXT_CTX, tsId } from "./model";
import { Tape, TokChip, VocabStrip, KIND_COL } from "./tokens";

// model.py · TextDecoder.forward on the example clip (jfk.flac). 12 tokens are shown;
// the shapes on labels are the real ones for the chosen size.

const CODE = {
  tokens: ["tokens = [<|startoftranscript|>, <|en|>, <|transcribe|>] + text tokens", "# timestamps on: text sits between <|t₀|> … <|t₁|> pairs"],
  vocab: ["n_vocab = 51865   # 50257 BPE + 107 specials + 1501 timestamps", "specials = ['<|endoftext|>', '<|startoftranscript|>', *languages(99),", "            '<|translate|>', '<|transcribe|>', '<|startoflm|>', '<|startofprev|>',", "            '<|nospeech|>', '<|notimestamps|>', *[f'<|{i * 0.02:.2f}|>' for i in range(1501)]]"],
  embed: ["self.token_embedding = nn.Embedding(n_vocab, n_state)", "self.positional_embedding = nn.Parameter(torch.empty(448, n_state))   # learned", "x = self.token_embedding(x) + self.positional_embedding[offset : offset + n]"],
  self: ["mask = torch.empty(448, 448).fill_(-np.inf).triu_(1)", "x = x + self.attn(self.attn_ln(x), mask=self.mask)[0]", "qk = qk + mask[:n, :n]   # −∞ above the diagonal"],
  cross: ["x = x + self.cross_attn(self.cross_attn_ln(x), xa)[0]", "q = self.query(x)        # from text   [n, D]", "k = self.key(xa); v = self.value(xa)   # from audio [1500, D]"],
  mlp: ["x = x + self.mlp(self.mlp_ln(x))   # Linear(D, 4D) → GELU → Linear(4D, D)"],
  logits: ["x = self.ln(x)", "logits = x @ self.token_embedding.weight.T   # tied: same matrix as the input table"],
  train: ["def forward(self, mel, tokens):", "    return self.decoder(tokens, self.encoder(mel))   # logits for every position", "# position i predicts token i + 1 (all at once, causal mask)"],
};

const FLOW = [
  { id: "tokens", label: "token ids", sub: "SOT · lang · task · text", code: CODE.tokens },
  { id: "vocab", label: "vocabulary", sub: "51865 ids, 3 regions", code: CODE.vocab },
  { id: "embed", label: "token + position", sub: "learned [448, D]", code: CODE.embed },
  {
    group: "ResidualAttentionBlock",
    loop: "× Ld", // replaced with the real count below
    items: [
      { id: "self", label: "masked self-attn", sub: "see only the past", code: CODE.self },
      { id: "cross", label: "cross-attn → audio", sub: "q text · k, v audio", code: CODE.cross },
      { id: "mlp", label: "LN → MLP 4D", sub: "GELU", code: CODE.mlp },
    ],
  },
  { id: "logits", label: "ln → x @ Embᵀ", sub: "[n, 51865] · tied", code: CODE.logits },
  { id: "train", label: "teacher forcing", sub: "position i → token i+1", code: CODE.train },
];

const SEQ = [...PREFIX, ...GEN.slice(0, 9)]; // 12 tokens
const n = SEQ.length;
// time spans for the toy cross-attention: text words from WORDS, timestamps at their time
const SPAN = SEQ.map((id) => {
  if (id >= TOK.ts0) { const s = (id - TOK.ts0) * 0.02; return { a: s, b: s + 0.12 }; }
  const w = WORDS.find((x) => x.id === id);
  return w ? { a: w.a, b: w.b } : {};
});

const lab = (id) => tokText(id).replace(/^ /, "·").replace("<|startoftranscript|>", "<|sot|>");

// causal self-attention: n × n grid, −∞ above the diagonal
function CausalGrid({ size = 380 }) {
  const u = size / n;
  return (
    <svg width={size + 110} height={size + 100} viewBox={`0 0 ${size + 110} ${size + 100}`} style={{ maxWidth: "100%", fontFamily: mono }}>
      {SEQ.map((id, i) => (
        <g key={i}>
          <text x={104} y={20 + i * u + u / 2 + 4} fill={KIND_COL[id >= TOK.ts0 ? "ts" : id >= TOK.eot ? "special" : "text"]} fontSize={12} textAnchor="end">{lab(id)}</text>
          <text transform={`translate(${110 + i * u + u / 2 + 4}, ${size + 26}) rotate(-50)`} fill={T.dim} fontSize={11} textAnchor="end">{lab(id)}</text>
          {SEQ.map((_, j) => {
            const ok = j <= i, w = ok ? 0.35 + 0.65 * Math.exp(-(i - j) * 0.6) : 0;
            return (
              <g key={j}>
                <rect x={110 + j * u + 1} y={20 + i * u + 1} width={u - 2} height={u - 2} rx={3} fill={ok ? heatCss(w) : T.well} stroke={ok ? "none" : `${T.bad}55`} />
                {!ok && <text x={110 + j * u + u / 2} y={20 + i * u + u / 2 + 4} fill={`${T.bad}aa`} fontSize={11} textAnchor="middle">−∞</text>}
              </g>
            );
          })}
        </g>
      ))}
      <text x={110} y={12} fill={T.dim} fontSize={12}>keys →</text>
    </svg>
  );
}

function CrossMap({ A, nA = 600, w = 640 }) {
  const rh = 22;
  return (
    <div style={{ display: "grid", gridTemplateColumns: "110px 1fr", gap: 6 }}>
      <div style={{ display: "grid", gridAutoRows: `${rh}px`, fontFamily: mono, fontSize: 12, alignItems: "center", justifyItems: "end" }}>
        {SEQ.map((id, i) => <span key={i} style={{ color: KIND_COL[id >= TOK.ts0 ? "ts" : id >= TOK.eot ? "special" : "text"] }}>{lab(id)}</span>)}
      </div>
      <div>
        <Canvas w={w} h={rh * n} deps={[A, nA, w]} draw={(x) => {
          for (let i = 0; i < n; i++) {
            let m = 0;
            for (let j = 0; j < nA; j++) m = Math.max(m, A[i * nA + j]);
            for (let j = 0; j < nA; j++) {
              const [r, g, b] = heat(Math.sqrt(A[i * nA + j] / m));
              x.fillStyle = `rgb(${r},${g},${b})`;
              x.fillRect((j / nA) * w, i * rh + 1, w / nA + 0.6, rh - 2);
            }
          }
        }} />
        <div style={{ display: "flex", justifyContent: "space-between", fontFamily: mono, fontSize: 12, color: T.dim }}>
          {[0, 2, 4, 6, 8, 10, 12].map((s) => <span key={s}>{s === 0 ? "audio token 0" : `${s * 50} · ${s}s`}</span>)}
        </div>
      </div>
    </div>
  );
}

export default function DecoderView({ A }) {
  const { m, size } = A;
  const D = m.D, h = m.h, d = D / h, V = vocabOf(size);
  const cross = useMemo(() => toyCross(SPAN), []);

  const steps = [
    { id: "tokens", block: "tokens", caption: "the prompt tells the task", vars: () => [["n", n], ["sot", TOK.sot], ["<|en|>", TOK.en], ["<|transcribe|>", TOK.transcribe]] },
    { id: "vocab", block: "vocab", caption: "words, controls, and time", vars: () => [["n_vocab", V], ["<|0.00|>", TOK.ts0], ["1 step", "20 ms"]] },
    { id: "embed", block: "embed", caption: "look up, add a learned position", vars: () => [["table", `[${V}, ${D}]`], ["pos", `[448, ${D}]`]] },
    { id: "self", block: "self", caption: "token i sees tokens 0 … i", vars: () => [["mask", "triu(−∞, 1)"]] },
    { id: "cross", block: "cross", caption: "each token reads the audio", vars: () => [["q", `[${n}, ${D}]`], ["k, v", `[1500, ${D}]`]] },
    { id: "mlp", block: "mlp", caption: "per token: D → 4D → D", vars: () => [["hidden", 4 * D]] },
    { id: "logits", block: "logits", caption: "score every id in the vocabulary", vars: () => [["logits", `[${n}, ${V}]`], ["tied", "yes"]] },
    { id: "train", block: "train", caption: "training: predict the next token", vars: () => [["loss", "cross-entropy"]] },
  ];

  const tags = (s) => ({
    tokens: [{ name: "tokens", dims: "1 n", io: "out" }],
    vocab: [{ name: "ids", dims: "V", io: "out" }],
    embed: [{ name: "tokens", dims: "1 n", io: "in" }, { name: "token_embedding", dims: "V D" }, { name: "positional_embedding", dims: "448 D" }, { name: "x", dims: "1 n D", io: "out" }],
    self: [{ name: "x", dims: "1 n D", io: "in" }, { name: "weights", dims: "1 h n n" }, { name: "x", dims: "1 n D", io: "out" }],
    cross: [{ name: "x", dims: "1 n D", io: "in" }, { name: "xa", dims: "1 A D", io: "in" }, { name: "weights", dims: "1 h n A" }, { name: "x", dims: "1 n D", io: "out" }],
    mlp: [{ name: "x", dims: "1 n D", io: "in" }, { name: "x", dims: "1 n D", io: "out" }],
    logits: [{ name: "x", dims: "1 n D", io: "in" }, { name: "Embᵀ", dims: "D V" }, { name: "logits", dims: "1 n V", io: "out" }],
    train: [{ name: "mel", dims: "1 M 3000", io: "in" }, { name: "tokens", dims: "1 n", io: "in" }, { name: "logits", dims: "1 n V", io: "out" }],
  }[s.id]);

  const matmuls = (s) => {
    if (["self", "cross", "mlp"].includes(s.id))
      return (
        <MatMulCard footer={`one decoder pass, ${m.Ld} layers · cross-attn shown separately`}>
          <MM a={["q", [1, h, n, d]]} b={["kᵀ (audio)", [1, h, d, 1500]]} c={["S", [1, h, n, 1500]]} times={m.Ld} note="cross-attn · no mask · softmax over 1500 audio tokens" />
          <MM a={["A", [1, h, n, 1500]]} b={["v (audio)", [1, h, 1500, d]]} c={["o", [1, h, n, d]]} times={m.Ld} />
          <BlockMMs B={1} T={n} D={D} h={h} d={d} M={4 * D} L={m.Ld} causal />
        </MatMulCard>
      );
    if (s.id === "logits") return <MatMulCard><MM a={["ln(x)", [1, n, D]]} b={["token_embedding.weightᵀ", [D, V]]} c={["logits", [1, n, V]]} /></MatMulCard>;
    if (s.id === "embed") return <MatMulCard><MMNote>token_embedding[ids] → [1, {n}, {D}] (a row lookup, no multiply)</MMNote></MatMulCard>;
    return null;
  };

  const top = [[" what", 437, 0.88], [" for", 337, 0.03], [" you", 291, 0.02], [",", 11, 0.02], ["<|4.40|>", tsId(4.4), 0.01]];

  const scene = (s) => {
    switch (s.id) {
      case "tokens":
        return (
          <div style={{ display: "grid", gap: 16 }}>
            <Tape ids={SEQ} label="example · jfk.flac" />
            <Row gap={18}>
              <TokChip id={TOK.sot} big sub="start" />
              <TokChip id={TOK.en} big sub="language" />
              <TokChip id={TOK.transcribe} big sub="task" />
              <TokChip id={TOK.ts0} big sub="time 0.00 s" />
              <TokChip id={400} big sub="text" />
            </Row>
            <Row gap={18}>
              <span style={{ fontSize: 15, color: T.dim }}>other prompts:</span>
              <TokChip id={TOK.translate} sub="→ English" />
              <TokChip id={TOK.notimestamps} sub="text only" />
              <TokChip id={TOK.sot_prev} sub="previous text first" />
            </Row>
          </div>
        );
      case "vocab":
        return (
          <div style={{ display: "grid", gap: 16 }}>
            <VocabStrip marks={[{ id: TOK.eot, color: KIND_COL.special }, { id: TOK.ts0, color: KIND_COL.ts }]} />
            <Row gap={24}>
              <Stat k="BPE text" v="0 … 50256" tone={KIND_COL.text} />
              <Stat k="specials" v="50257 … 50363 (107)" tone={KIND_COL.special} />
              <Stat k="timestamps" v="50364 … 51864 (1501)" tone={KIND_COL.ts} />
            </Row>
            <Row gap={10}>
              {[0, 0.02, 7.68, 30].map((t) => <TokChip key={t} id={tsId(t)} big />)}
            </Row>
            <Row gap={24}>
              <Stat k="1 timestamp step" v="0.02 s = 1 audio token" tone={KIND_COL.ts} />
              <Stat k="1501 × 0.02" v="0.00 … 30.00 s" tone={KIND_COL.ts} />
            </Row>
          </div>
        );
      case "embed":
        return (
          <div style={{ display: "grid", gap: 16 }}>
            <Tape ids={SEQ.slice(0, 8)} />
            <Row gap={10}>
              <Mat name="token_embedding" shape="V D" rows={V} cols={D} color={T.accent2} note="row = id" />
              <Op>[ids] +</Op>
              <Mat name="positional_embedding" shape="448 D" rows={448} cols={D} color={T.wire} note="learned" />
              <Op>[0 : n] =</Op>
              <Mat name="x" shape="1 n D" rows={n} cols={D} glow />
            </Row>
            <Row gap={24}>
              <Stat k="encoder positions" v="sinusoids, fixed" tone={T.dim} />
              <Stat k="decoder positions" v="learned, max 448" tone={T.wire} />
              <Stat k="with KV cache" v="pos[offset : offset + 1]" tone={T.accent} />
            </Row>
          </div>
        );
      case "self":
        return (
          <div style={{ display: "grid", gap: 10 }}>
            <CausalGrid />
            <Row gap={24}><Stat k="above diagonal" v="−∞ → weight 0" tone={T.bad} /><Stat k="real" v={`${h} heads × [n, n]`} tone={T.accent2} /></Row>
          </div>
        );
      case "cross":
        return (
          <div style={{ display: "grid", gap: 12 }}>
            <Row gap={10}>
              <Mat name="q (text)" shape="n D" rows={n} cols={D} />
              <Op>@</Op>
              <Mat name="k (audio)ᵀ" shape="D A" rows={D} cols={1500} color={T.accent} />
              <Op>→ softmax →</Op>
              <Mat name="weights" shape="n A" rows={n} cols={1500} color={T.accent2} glow />
            </Row>
            <Label>weights over the first 600 audio tokens (12 s) · toy picture</Label>
            <CrossMap A={cross} />
            <Row gap={24}><Stat k="mask" v="none" tone={T.accent} /><Stat k="words" v="look at their own time" tone={T.wire} /><Stat k="used later" v="word timestamps" tone={T.accent2} /></Row>
          </div>
        );
      case "mlp":
        return (
          <Row gap={10}>
            <Mat name="LN(x)" shape="n D" rows={n} cols={D} />
            <Op>@</Op><Mat name="W1" shape="D 4D" rows={D} cols={4 * D} color={T.accent2} />
            <Op>GELU @</Op><Mat name="W2" shape="4D D" rows={4 * D} cols={D} color={T.accent2} />
            <Op>+ x =</Op><Mat name="x" shape="n D" rows={n} cols={D} glow />
          </Row>
        );
      case "logits":
        return (
          <div style={{ display: "grid", gap: 14 }}>
            <Row gap={10}>
              <Mat name="ln(x)" shape="n D" rows={n} cols={D} />
              <Op>@</Op>
              <Mat name="token_embedding.weightᵀ" shape="D V" rows={D} cols={V} color={T.accent2} note="same table as the input" />
              <Op>=</Op>
              <Mat name="logits" shape="n V" rows={n} cols={V} glow />
            </Row>
            <Label>last row · after "{lab(SEQ[n - 1])}" · softmax</Label>
            <div style={{ display: "grid", gap: 6, maxWidth: 560 }}>
              {top.map(([t, id, p], i) => (
                <div key={id} style={{ display: "grid", gridTemplateColumns: "130px 1fr 60px", gap: 10, alignItems: "center", fontSize: 15 }}>
                  <span style={{ color: KIND_COL[id >= TOK.ts0 ? "ts" : "text"], textAlign: "right", whiteSpace: "pre" }}>{t.replace(/^ /, "·")}</span>
                  <div style={{ height: 16, width: `${Math.max(1, p * 100)}%`, background: i === 0 ? T.accent : `${T.accent}66`, borderRadius: 3 }} />
                  <span style={{ color: T.soft }}>{p.toFixed(2)}</span>
                </div>
              ))}
            </div>
          </div>
        );
      case "train": {
        const ids = SEQ;
        return (
          <div style={{ display: "grid", gap: 10 }}>
            <Label>input · tokens[:-1]</Label>
            <Tape ids={ids.slice(0, -1)} showId={false} />
            <div style={{ fontSize: 17, color: T.accent2, paddingLeft: 8 }}>↓ decoder(tokens, encoder(mel)) · one pass, causal mask</div>
            <Label color={T.accent}>target · tokens[1:]</Label>
            <Tape ids={ids.slice(1)} showId={false} />
            <Row gap={24}>
              <Stat k="loss" v="cross-entropy at every position" tone={T.accent} />
              <Stat k="inference" v="one token at a time → decoding.py" tone={T.wire} />
            </Row>
          </div>
        );
      }
      default:
        return null;
    }
  };

  return (
    <DimCtx.Provider value={{ n: ["text tokens", n], V: ["vocabulary", V], D: ["width", D], h: ["heads", h], d: ["head dim", d], A: ["audio tokens", 1500], M: ["mel bins", m.mels], "4D": ["MLP width", 4 * D] }}>
      <StepShell title="model.py · TextDecoder" subtitle={`${size} · D ${D} · ${h} heads · ${m.Ld} layers · ctx ${N_TEXT_CTX}`} blocks={FLOW.map((b) => (b.loop ? { ...b, loop: `× ${m.Ld}` } : b))} steps={steps} scene={scene} tags={tags} matmuls={matmuls} />
    </DimCtx.Provider>
  );
}

export { Plot };
