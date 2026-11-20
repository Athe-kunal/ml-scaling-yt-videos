import { useMemo, useState } from "react";
import { T, mono } from "./theme";
import StepShell, { Stat } from "./StepShell";
import { DimCtx, Row, Op, Mat, MM, MMNote, MatMulCard, BlockMMs, Label, Heatmap } from "./draw";
import { Heat, Plot, W0 } from "./plots";
import { DT, sinusoid, toyAttn } from "./model";

// model.py · AudioEncoder.forward, one job per step. Shapes are the real ones for
// the chosen size; the feature pictures use a toy 32-channel stem with random weights.

const CODE = {
  mel: ["mel = log_mel_spectrogram(audio)   # [M, 3000]", "audio_features = model.encoder(mel[None])   # batch of 1"],
  conv1: ["self.conv1 = Conv1d(n_mels, n_state, kernel_size=3, padding=1)", "x = F.gelu(self.conv1(x))   # [1, M, 3000] → [1, D, 3000]"],
  conv2: ["self.conv2 = Conv1d(n_state, n_state, kernel_size=3, stride=2, padding=1)", "x = F.gelu(self.conv2(x))   # [1, D, 3000] → [1, D, 1500]"],
  pos: ["x = x.permute(0, 2, 1)   # [1, 1500, D]", "self.register_buffer('positional_embedding', sinusoids(1500, n_state))   # fixed, not learned", "assert x.shape[1:] == self.positional_embedding.shape, 'incorrect audio shape'", "x = x + self.positional_embedding"],
  attn: ["# ResidualAttentionBlock, no mask", "x = x + self.attn(self.attn_ln(x))[0]", "q = self.query(x); k = self.key(x); v = self.value(x)   # key: Linear(bias=False)", "scale = (n_state // n_head) ** -0.25", "qk = (q * scale) @ (k * scale).transpose(-1, -2)   # = q·kᵀ / √d", "w = softmax(qk, dim=-1); out = w @ v"],
  mlp: ["x = x + self.mlp(self.mlp_ln(x))", "self.mlp = Sequential(Linear(D, 4D), GELU(), Linear(4D, D))"],
  post: ["x = self.ln_post(x)   # [1, 1500, D]", "return x   # audio_features: computed once per 30 s window"],
};

const FLOW = [
  { id: "mel", label: "mel in", sub: "[M, 3000]", code: CODE.mel },
  {
    group: "conv stem",
    items: [
      { id: "conv1", label: "conv1 · k3 · GELU", sub: "M → D channels", code: CODE.conv1 },
      { id: "conv2", label: "conv2 · k3 · stride 2", sub: "3000 → 1500 tokens", code: CODE.conv2 },
    ],
  },
  { id: "pos", label: "permute + sinusoids", sub: "[1500, D] · fixed", code: CODE.pos },
  {
    group: "ResidualAttentionBlock",
    loop: "× Le", // replaced with the real count below
    items: [
      { id: "attn", label: "LN → self-attention", sub: "no mask · all 1500", code: CODE.attn },
      { id: "mlp", label: "LN → MLP 4D", sub: "GELU", code: CODE.mlp },
    ],
  },
  { id: "post", label: "ln_post", sub: "audio_features [1500, D]", code: CODE.post },
];

function MelBox({ data, rows, cols, c0 = 0, c1, h = 150, lo, hi, children, label, sub }) {
  return (
    <div style={{ display: "grid", gap: 4 }}>
      <div style={{ display: "flex", gap: 10, alignItems: "baseline" }}><Label>{label}</Label>{sub && <span style={{ fontSize: 13, color: T.dim }}>{sub}</span>}</div>
      <div style={{ position: "relative" }}>
        <Heat data={data} rows={rows} cols={cols} c0={c0} c1={c1} h={h} lo={lo} hi={hi} />
        {children}
      </div>
    </div>
  );
}

// boxes drawn over a heatmap, in column units
const Box = ({ a, b, n, color, label }) => (
  <div style={{ position: "absolute", top: -3, bottom: -3, left: `${(a / n) * 100}%`, width: `${((b - a) / n) * 100}%`, minWidth: 3, border: `2px solid ${color}`, borderRadius: 3, transition: "all .3s" }}>
    {label && <span style={{ position: "absolute", top: -20, left: 0, whiteSpace: "nowrap", fontSize: 13, color }}>{label}</span>}
  </div>
);

// stride picture: input frames → output positions
function StrideDiagram({ stride, nIn = 12, color }) {
  const nOut = Math.ceil(nIn / stride), u = 52, W = nIn * u;
  return (
    <svg width={W + 20} height={110} viewBox={`0 0 ${W + 20} 110`} style={{ maxWidth: "100%", fontFamily: mono }}>
      {Array.from({ length: nIn }, (_, i) => (
        <g key={i}>
          <rect x={10 + i * u + 4} y={6} width={u - 8} height={22} rx={4} fill={T.well} stroke={T.rule} />
          <text x={10 + i * u + u / 2} y={22} fill={T.soft} fontSize={12} textAnchor="middle">{i * 10}ms</text>
        </g>
      ))}
      {Array.from({ length: nOut }, (_, j) => {
        const c = j * stride, cx = 10 + c * u + u / 2;
        return (
          <g key={j}>
            {[-1, 0, 1].map((k) => c + k >= 0 && c + k < nIn && <line key={k} x1={10 + (c + k) * u + u / 2} y1={30} x2={stride === 1 ? cx : 10 + c * u + u / 2} y2={74} stroke={color} strokeOpacity={j === 2 ? 1 : 0.35} strokeWidth={j === 2 ? 2 : 1} />)}
            <rect x={cx - (u * stride) / 2 + 4 + (stride === 2 ? 0 : 0)} y={76} width={u * stride - 8} height={24} rx={4} fill={`${color}${j === 2 ? "44" : "18"}`} stroke={color} />
            <text x={cx} y={93} fill={T.ink} fontSize={12} textAnchor="middle">{stride === 2 ? `tok ${j}` : `${j * 10}ms`}</text>
          </g>
        );
      })}
    </svg>
  );
}

export default function EncoderView({ A }) {
  const { lm, stem, nMels, m } = A;
  const D = m.D, h = m.h, d = D / h;
  const nTok = Math.min(1500, Math.ceil((lm.lastFrame + 1) / 2) + 40);
  const N = Math.min(nTok, 260);
  const C1 = Math.min(3000, N * 2);
  const [qSel, setQ] = useState(null);
  const att = useMemo(() => toyAttn(stem.c2, stem.n2, N), [stem, N]);
  // default query: the loudest token
  const loudTok = useMemo(() => {
    let b = 0, bv = -1e9;
    for (let t = 0; t < N; t++) { let s = 0; for (let k = 0; k < nMels; k++) s += lm.scaled[k * 3000 + 2 * t]; if (s > bv) { bv = s; b = t; } }
    return b;
  }, [lm, N, nMels]);
  const qt = Math.min(qSel ?? loudTok, N - 1);
  const ft = 2 * qt; // the mel frame under that token
  const c1lo = useMemo(() => { let a = 1e9, b = -1e9; for (const v of stem.c1) { a = Math.min(a, v); b = Math.max(b, v); } return [a, b]; }, [stem]);
  const c2lo = useMemo(() => { let a = 1e9, b = -1e9; for (const v of stem.c2) { a = Math.min(a, v); b = Math.max(b, v); } return [a, b]; }, [stem]);
  const sMin = useMemo(() => lm.scaled.reduce((a, v) => Math.min(a, v), 9), [lm]);
  const sMax = (lm.max + 4) / 4;
  const sinD = Math.min(D, 384);
  const pos = useMemo(() => { const P = new Float32Array(sinD * 1500); for (let c = 0; c < sinD; c++) for (let t = 0; t < 1500; t++) P[c * 1500 + t] = sinusoid(t, c, D); return P; }, [D, sinD]);

  const steps = [
    { id: "mel", block: "mel", caption: "80 or 128 rows × 3000 columns", vars: () => [["M", nMels], ["frames", 3000], ["1 column", "10 ms"]] },
    { id: "conv1", block: "conv1", caption: "look at 3 frames, mix all mels", vars: () => [["kernel", `[${D}, ${nMels}, 3]`], ["frame", ft]] },
    { id: "conv2", block: "conv2", caption: "stride 2 halves the time axis", vars: () => [["kernel", `[${D}, ${D}, 3]`], ["token", `${qt} = ${(qt * 0.02).toFixed(2)} s`]] },
    { id: "pos", block: "pos", caption: "fixed sin / cos per position", vars: () => [["buffer", `[1500, ${D}]`], ["learned", "no"]] },
    { id: "attn", block: "attn", caption: "every token sees all 1500", vars: () => [["heads", h], ["d", d], ["mask", "none"]] },
    { id: "attn_scale", block: "attn", caption: "scale q and k by d^−¼ each", vars: () => [["d^−¼", (d ** -0.25).toFixed(4)], ["product", `1/√${d}`]] },
    { id: "mlp", block: "mlp", caption: "per token: D → 4D → D", vars: () => [["hidden", 4 * D]] },
    { id: "post", block: "post", caption: "1500 audio vectors, ready", vars: () => [["out", `[1500, ${D}]`], ["layers", m.Le]] },
  ];

  const tags = (s) => ({
    mel: [{ name: "mel", dims: "M T", io: "out" }],
    conv1: [{ name: "mel", dims: "1 M T", io: "in" }, { name: "x", dims: "1 D T", io: "out" }],
    conv2: [{ name: "x", dims: "1 D T", io: "in" }, { name: "x", dims: "1 D A", io: "out" }],
    pos: [{ name: "x", dims: "1 D A", io: "in" }, { name: "pos", dims: "A D" }, { name: "x", dims: "1 A D", io: "out" }],
    attn: [{ name: "x", dims: "1 A D", io: "in" }, { name: "weights", dims: "1 h A A" }, { name: "x", dims: "1 A D", io: "out" }],
    attn_scale: [{ name: "q", dims: "1 h A d", io: "in" }, { name: "k", dims: "1 h A d", io: "in" }, { name: "qk", dims: "1 h A A", io: "out" }],
    mlp: [{ name: "x", dims: "1 A D", io: "in" }, { name: "hidden", dims: "1 A 4D" }, { name: "x", dims: "1 A D", io: "out" }],
    post: [{ name: "x", dims: "1 A D", io: "in" }, { name: "audio_features", dims: "1 A D", io: "out" }],
  }[s.id]);

  const matmuls = (s) => {
    if (s.id === "conv1") return <MatMulCard><MM a={["mel windows", [3000, 3 * nMels]]} b={["conv1.weight", [3 * nMels, D]]} c={["x", [3000, D]]} note="each output frame = 3 neighbouring columns, flattened" /></MatMulCard>;
    if (s.id === "conv2") return <MatMulCard><MM a={["x windows", [1500, 3 * D]]} b={["conv2.weight", [3 * D, D]]} c={["x", [1500, D]]} note="windows start at frames 0, 2, 4, … (stride 2)" /></MatMulCard>;
    if (["attn", "attn_scale", "mlp"].includes(s.id)) return <MatMulCard footer={`one encoder pass, all ${m.Le} layers`}><BlockMMs B={1} T={1500} D={D} h={h} d={d} M={4 * D} L={m.Le} /></MatMulCard>;
    return null;
  };

  const scene = (s) => {
    switch (s.id) {
      case "mel":
        return (
          <div style={{ display: "grid", gap: 12 }}>
            <MelBox data={lm.scaled} rows={nMels} cols={3000} c1={3000} lo={sMin} hi={sMax} h={180} label={`mel [${nMels}, 3000] · 30 s`} sub="most of it is padding" />
            <Row gap={24}><Stat k="1 column" v="10 ms" tone="#8DB4FF" /><Stat k="3000 columns" v="30 s" tone="#8DB4FF" /><Stat k="real audio" v={`${lm.lastFrame} columns`} tone={T.accent} /></Row>
          </div>
        );
      case "conv1":
        return (
          <div style={{ display: "grid", gap: 14 }}>
            <MelBox data={lm.scaled} rows={nMels} cols={3000} c1={C1} lo={sMin} hi={sMax} h={140} label={`mel · frames 0 … ${C1}`}>
              <Box a={ft - 1} b={ft + 2} n={C1} color={T.wire} label={`frames ${ft - 1} · ${ft} · ${ft + 1}`} />
            </MelBox>
            <div style={{ fontSize: 17, color: T.accent2 }}>↓ {nMels} × 3 numbers → 1 column of D = {D} · GELU</div>
            <MelBox data={stem.c1} rows={DT} cols={3000} c1={C1} lo={c1lo[0]} hi={c1lo[1]} h={110} label={`x · D = ${D} rows`} sub={`picture: ${DT} toy channels`}>
              <Box a={ft} b={ft + 1} n={C1} color={T.wire} />
            </MelBox>
            <Row gap={24}><Stat k="time" v="3000 → 3000" tone="#8DB4FF" /><Stat k="channels" v={`${nMels} → ${D}`} tone="#C79BFF" /></Row>
          </div>
        );
      case "conv2":
        return (
          <div style={{ display: "grid", gap: 14 }}>
            <StrideDiagram stride={2} color={T.accent} />
            <MelBox data={stem.c1} rows={DT} cols={3000} c1={C1} lo={c1lo[0]} hi={c1lo[1]} h={90} label={`x after conv1 · ${C1} frames`}>
              <Box a={ft - 1} b={ft + 2} n={C1} color={T.wire} />
            </MelBox>
            <div style={{ fontSize: 17, color: T.accent2 }}>↓ stride 2 · GELU</div>
            <MelBox data={stem.c2} rows={DT} cols={1500} c1={N} lo={c2lo[0]} hi={c2lo[1]} h={90} label={`x after conv2 · ${N} tokens`}>
              <Box a={qt} b={qt + 1} n={N} color={T.wire} label={`token ${qt}`} />
            </MelBox>
            <Row gap={24}><Stat k="1 token" v="20 ms" tone={T.accent} /><Stat k="30 s" v="1500 tokens" tone={T.accent} /><Stat k="TOKENS_PER_SECOND" v="50" tone={T.wire} /></Row>
          </div>
        );
      case "pos":
        return (
          <div style={{ display: "grid", gap: 12 }}>
            <MelBox data={pos} rows={sinD} cols={1500} c1={1500} lo={-1} hi={1} h={200} label={`sinusoids(1500, ${D})`} sub={sinD < D ? `first ${sinD} of ${D} channels` : "channels ↑ · positions →"} />
            <Plot xs={Array.from({ length: 300 }, (_, i) => i)} h={110} xticks={[[0, "pos 0"], [100, "100"], [200, "200"], [299, "300"]]}
              series={[0, 8, 40, D / 2].map((c, k) => ({ y: Array.from({ length: 300 }, (_, t) => sinusoid(t, c, D)), color: [T.accent, T.wire, T.accent2, "#8DB4FF"][k], lw: 1.6 }))} ymin={-1.1} ymax={1.1} />
            <Row gap={18}>
              <span style={{ fontSize: 14, color: T.accent }}>ch 0 fast</span><span style={{ fontSize: 14, color: T.wire }}>ch 8</span><span style={{ fontSize: 14, color: T.accent2 }}>ch 40 slow</span><span style={{ fontSize: 14, color: "#8DB4FF" }}>ch {D / 2} = cos</span>
            </Row>
            <div style={{ padding: "10px 14px", borderRadius: 8, border: `1px solid ${T.bad}`, background: `${T.bad}10`, fontSize: 16 }}>
              <span style={{ color: T.bad }}>assert</span> x.shape[1:] == (1500, {D}) <span style={{ color: T.dim }}>→ input must be exactly 3000 frames → pad every clip to 30 s</span>
            </div>
          </div>
        );
      case "attn":
      case "attn_scale": {
        const row = Array.from({ length: N }, (_, j) => att[qt * N + j]);
        if (s.id === "attn_scale") {
          const sc = d ** -0.25;
          return (
            <div style={{ display: "grid", gap: 14 }}>
              <Row gap={10}>
                <Mat name="q" shape="h A d" rows={1500} cols={d} />
                <Op>× {sc.toFixed(3)}</Op>
                <Op color={T.dim}>@</Op>
                <Mat name="k" shape="h A d" rows={1500} cols={d} color={T.wire} />
                <Op>× {sc.toFixed(3)} ᵀ</Op>
                <Op>=</Op>
                <Mat name="qk" shape="h A A" rows={1500} cols={1500} color={T.accent2} glow />
              </Row>
              <Row gap={24}>
                <Stat k="d" v={d} />
                <Stat k="d^−¼ · d^−¼" v={`= 1/√${d} = ${(1 / Math.sqrt(d)).toFixed(4)}`} tone={T.wire} />
                <Stat k="why split it" v="smaller values in fp16" tone={T.accent2} />
                <Stat k="key bias" v="none" tone={T.bad} />
              </Row>
            </div>
          );
        }
        return (
          <div style={{ display: "grid", gap: 10 }}>
            <Row gap={16} wrap={false} style={{ alignItems: "flex-start" }}>
              <div style={{ display: "grid", gap: 4 }}>
                <Label>weights · first {N} tokens</Label>
                <Heatmap A={att} n={N} size={250} mark={qt} />
              </div>
              <div style={{ display: "grid", gap: 6, flex: 1, minWidth: 0 }}>
                <Label>mel under the tokens</Label>
                <Heat data={lm.scaled} rows={nMels} cols={3000} c1={C1} lo={sMin} hi={sMax} h={80} w={480} col={ft} />
                <Label color={T.accent}>token {qt} attends to →</Label>
                <Plot xs={Array.from({ length: N }, (_, j) => j)} w={480} h={110} bars={{ y: row, color: (j) => (j === qt ? T.wire : T.accent) }} ymin={0} xticks={[[0, "0"], [N - 1, `${N - 1}`]]} pad={{ l: 8, r: 8, t: 6, b: 24 }} />
              </div>
            </Row>
            <Row gap={24}><Stat k="mask" v="none" tone={T.accent} /><Stat k="real size" v={`${h} × [1500, 1500]`} tone={T.accent2} /><Stat k="picture" v="toy weights" tone={T.dim} /></Row>
            <div style={{ display: "flex", gap: 10, alignItems: "center", fontSize: 14, color: T.dim }}>
              query token <input type="range" min={0} max={N - 1} value={qt} onChange={(e) => setQ(+e.target.value)} style={{ width: 240 }} /><span style={{ color: T.wire }}>{qt} · {(qt * 0.02).toFixed(2)} s</span>
            </div>
          </div>
        );
      }
      case "mlp":
        return (
          <div style={{ display: "grid", gap: 16 }}>
            <Row gap={10}>
              <Mat name="LN(x)" shape="A D" rows={1500} cols={D} />
              <Op>@</Op><Mat name="W1" shape="D 4D" rows={D} cols={4 * D} color={T.accent2} />
              <Op>GELU</Op><Mat name="hidden" shape="A 4D" rows={1500} cols={4 * D} color={T.wire} />
              <Op>@</Op><Mat name="W2" shape="4D D" rows={4 * D} cols={D} color={T.accent2} />
              <Op>+ x =</Op><Mat name="x" shape="A D" rows={1500} cols={D} glow />
            </Row>
            <Row gap={24}><Stat k="each token" v="on its own" tone={T.accent} /><Stat k="pre-LN" v="LN before attn and MLP" tone={T.wire} /></Row>
          </div>
        );
      case "post":
        return (
          <div style={{ display: "grid", gap: 16 }}>
            <Row gap={10}>
              <Mat name="mel" shape="M T" rows={nMels} cols={3000} color={T.soft} />
              <Op>→ convs →</Op>
              <Op>{m.Le} blocks</Op>
              <Op>→ ln_post →</Op>
              <Mat name="audio_features" shape="A D" rows={1500} cols={D} glow />
            </Row>
            <MelBox data={stem.c2} rows={DT} cols={1500} c1={1500} lo={c2lo[0]} hi={c2lo[1]} h={90} label="1500 vectors · 30 s" sub="toy picture" />
            <Row gap={24}><Stat k="run" v="once per 30 s window" tone={T.accent} /><Stat k="decoder reads it" v="every token, via cross-attn" tone={T.accent2} /></Row>
          </div>
        );
      default:
        return null;
    }
  };

  return (
    <DimCtx.Provider value={{ M: ["mel bins", nMels], T: ["mel frames", 3000], A: ["audio tokens", 1500], D: ["width", D], h: ["heads", h], d: ["head dim", d], "4D": ["MLP width", 4 * D] }}>
      <StepShell title="model.py · AudioEncoder" subtitle={`${A.size} · D ${D} · ${h} heads · ${m.Le} layers`} blocks={FLOW.map((b) => (b.loop ? { ...b, loop: `× ${m.Le}` } : b))} steps={steps} scene={scene} tags={tags} matmuls={matmuls} />
    </DimCtx.Provider>
  );
}

export { W0 };
