import { useMemo } from "react";
import { T, mono } from "./theme";
import { useProgress, ease, clamp01, lerp } from "./anim";
import StepShell from "./StepShell";
import { MM, MMNote, MatMulCard, BlockMMs } from "./draw";
import { SUBJECTS, renderSubject } from "./model";
import { PatchScene } from "./ModernVlmView";

// PaliGemma-3B-224, following hkproj/pytorch-paligemma:
// processor → SigLIP So400m/14 (27 × 1152, no CLS) → one Linear 1152→2048 →
// masked_scatter into the <image> slots → ÷√d / ×√d → Gemma-2B (18 × 2048, 8 Q / 1 KV)
// → greedy decoding with a KV cache. One dog picture + "caption en" throughout.

const PG = { res: 224, P: 14, g: 16, nImg: 256, vD: 1152, vL: 27, vH: 16, vM: 4304, D: 2048, L: 18, H: 8, KV: 1, hd: 256, M: 16384, V: 257216 };
const S = 260; // 256 <image> + <bos> + "caption" + "▁en" + "\n"
const SQ = Math.sqrt(PG.D); // 45.25
const C_IMG = "#8DB4FF", C_BOS = "#FF8AC6", C_TXT = "#FFC94D", C_ANS = "#9BE564";
const PROMPT = [["<bos>", C_BOS, 257], ["caption", C_TXT, 258], ["▁en", C_TXT, 259], ["\\n", C_TXT, 260]];
const ANSWER = ["▁a", "▁dog", "▁on", "▁grass", "<eos>"];
// toy next-token distributions (greedy picks the first)
const CANDS = [
  [["▁a", 0.58], ["▁the", 0.14], ["▁dog", 0.09], ["▁two", 0.05]],
  [["▁dog", 0.81], ["▁puppy", 0.07], ["▁cat", 0.03], ["▁small", 0.02]],
  [["▁on", 0.64], ["▁sitting", 0.12], ["▁in", 0.08], ["▁with", 0.05]],
  [["▁grass", 0.71], ["▁the", 0.1], ["▁a", 0.06], ["▁green", 0.04]],
  [["<eos>", 0.88], [".", 0.05], ["▁and", 0.03], [",", 0.02]],
];

const seg = (p, a, b) => ease(clamp01((p - a) / (b - a)));
const Txt = ({ x, y, s, size = 14, c = T.dim, a = "middle", w = 400, op = 1 }) => (
  <text x={x} y={y} textAnchor={a} fontFamily={mono} fontSize={size} fontWeight={w} fill={c} opacity={op}>{s}</text>
);
const Chip = ({ x, y, w, s, c, op = 1, h = 30, size = 14 }) => (
  <g opacity={op}>
    <rect x={x} y={y} width={w} height={h} rx={6} fill={`${c}22`} stroke={c} strokeWidth={1.3} />
    <Txt x={x + w / 2} y={y + h / 2 + 5} s={s} size={size} w={700} c={T.ink} />
  </g>
);
const Arrow = ({ x1, y1, x2, y2, op = 1, c = T.dim }) => {
  const a = Math.atan2(y2 - y1, x2 - x1), hx = x2 - 7 * Math.cos(a), hy = y2 - 7 * Math.sin(a);
  return (
    <g opacity={op}>
      <line x1={x1} y1={y1} x2={hx} y2={hy} stroke={c} strokeWidth={1.4} />
      <path d={`M${x2},${y2} L${hx - 4 * Math.sin(a)},${hy + 4 * Math.cos(a)} L${hx + 4 * Math.sin(a)},${hy - 4 * Math.cos(a)} z`} fill={c} />
    </g>
  );
};
const Svg = ({ h, children }) => <svg viewBox={`0 0 700 ${h}`} width="100%" style={{ maxWidth: 780, display: "block" }}>{children}</svg>;

// dog picture: url + 16×16 patch colours (one per SigLIP token)
function usePic() {
  return useMemo(() => {
    const img = renderSubject(SUBJECTS[0], 224), d = img.data, out = [];
    for (let r = 0; r < 16; r++) for (let c = 0; c < 16; c++) {
      const a = [0, 0, 0];
      for (let y = 0; y < 14; y += 2) for (let x = 0; x < 14; x += 2) { const o = ((r * 14 + y) * 224 + c * 14 + x) * 4; a[0] += d[o]; a[1] += d[o + 1]; a[2] += d[o + 2]; }
      out.push(a.map((v) => Math.round(v / 49)));
    }
    return { url: img.canvas.toDataURL(), cols: out };
  }, []);
}
const rgb = (c) => `rgb(${c.join(",")})`;
const mix = (c, k) => c.map((v) => Math.round(v * (1 - k) + 150 * k));

// the 260-token sequence drawn as 8 image chips (standing for 256) + 4 prompt chips
function SeqChips({ x, y, p = 1, cw = 34, img = true, ans = 0, h = 30 }) {
  const items = [
    ...Array.from({ length: 8 }, (_, i) => ["img", C_IMG, i]),
    ...PROMPT.map(([s, c]) => [s, c]),
    ...ANSWER.slice(0, ans).map((s) => [s, C_ANS]),
  ];
  let cx = x;
  return (
    <g>
      {items.map(([s, c], i) => {
        const w = s === "img" ? 22 : Math.max(cw, s.length * 9 + 14), el = (
          <g key={i} opacity={clamp01(p * items.length - i + 1)}>
            {s === "img"
              ? <rect x={cx} y={y} width={w - 3} height={h} rx={4} fill={`${C_IMG}33`} stroke={C_IMG} strokeWidth={1.1} />
              : <Chip x={cx} y={y} w={w - 4} h={h} s={s} c={c} size={13} />}
          </g>
        );
        cx += w;
        return el;
      })}
      {img && <Txt x={x + 86} y={y + h + 18} s="<image> × 256" size={13} c={C_IMG} />}
    </g>
  );
}

// ------------------------------------------------------------ 1 · prompt
function PromptScene({ pic }) {
  const p = useProgress("prompt", 3200), a = seg(p, 0, 0.35), b = seg(p, 0.35, 0.75), c = seg(p, 0.75, 1);
  return (
    <Svg h={330}>
      <image href={pic.url} x={10} y={20} width={120} height={120} />
      <Txt x={70} y={160} s="224 × 224 × 3" size={14} c={T.soft} />
      <Arrow x1={140} y1={80} x2={196} y2={80} op={a} />
      <g opacity={a}>
        <Txt x={206} y={58} a="start" s="÷ 255" size={15} c={T.ink} w={700} />
        <Txt x={206} y={82} a="start" s="(x − 0.5) / 0.5" size={15} c={T.ink} w={700} />
        <Txt x={206} y={106} a="start" s="→ [−1, 1]" size={14} c={T.soft} />
        <Txt x={420} y={82} a="start" s="pixel_values" size={15} w={700} c={C_IMG} />
        <Txt x={420} y={104} a="start" s="[1, 3, 224, 224]" size={14} c={T.soft} />
      </g>
      <g opacity={b}>
        <rect x={10} y={190} width={196} height={30} rx={6} fill={T.well} stroke={T.rule} />
        <Txt x={20} y={210} a="start" s={'"caption en"'} size={15} c={C_TXT} w={700} />
        <Arrow x1={108} y1={222} x2={108} y2={244} />
      </g>
      <SeqChips x={10} y={250} p={b} />
      <g opacity={c}>
        <Txt x={10} y={322} a="start" s="ids [1, 260] · image first, then <bos>" size={14} c={T.soft} />
        <Txt x={520} y={270} a="start" s={`S = 256 + 1 + 2 + 1`} size={14} c={T.soft} />
        <Txt x={520} y={292} a="start" s={`= ${S}`} size={18} w={700} c={T.ink} />
      </g>
    </Svg>
  );
}

// ------------------------------------------------------------ 3 · SigLIP blocks
function SiglipScene({ pic }) {
  const p = useProgress("siglip", 3600), a = seg(p, 0, 0.2), b = clamp01((p - 0.2) / 0.55), c = seg(p, 0.75, 1);
  const cs = 9, G = 16 * cs;
  const grid = (x, y, k) => pic.cols.map((col, i) => <rect key={i} x={x + (i % 16) * cs} y={y + Math.floor(i / 16) * cs} width={cs - 1} height={cs - 1} fill={rgb(mix(col, k))} />);
  return (
    <Svg h={330}>
      <Txt x={10} y={22} a="start" s="256 patch vectors" size={15} w={700} c={T.ink} />
      {grid(10, 34, 0)}
      <Txt x={10 + G / 2} y={34 + G + 22} s="[256, 1152]" size={14} c={C_IMG} />
      <g opacity={a}>
        <rect x={10} y={232} width={86} height={30} rx={6} fill="none" stroke={T.bad} strokeDasharray="4 3" />
        <Txt x={53} y={252} s="CLS" size={14} c={T.bad} />
        <line x1={14} x2={92} y1={258} y2={236} stroke={T.bad} strokeWidth={1.6} />
        <Txt x={106} y={252} a="start" s="no CLS · keep all 256" size={14} c={T.soft} />
      </g>
      <Arrow x1={166} y1={106} x2={214} y2={106} op={a} />
      <rect x={220} y={30} width={240} height={190} rx={10} fill={`${C_IMG}10`} stroke={C_IMG} strokeWidth={1.3} />
      {Array.from({ length: 27 }, (_, i) => (
        <rect key={i} x={234} y={40 + i * 6.3} width={212} height={4} rx={2} fill={b * 27 > i ? C_IMG : T.rule} opacity={b * 27 > i ? 0.85 : 1} />
      ))}
      <Txt x={340} y={244} s="27 blocks · 1152 wide" size={14} c={T.soft} />
      <Txt x={340} y={264} s="LN → attn 16 heads → LN → MLP" size={13} c={T.dim} />
      <Txt x={340} y={284} s="every patch sees every patch" size={13} c={C_IMG} op={b} />
      <Arrow x1={466} y1={106} x2={514} y2={106} op={c} />
      <g opacity={c}>
        <Txt x={520} y={22} a="start" s="post_layernorm" size={15} w={700} c={T.ink} />
        {grid(520, 34, 0.35)}
        <Txt x={520 + G / 2} y={34 + G + 22} s="[256, 1152]" size={14} c={C_IMG} />
        <Txt x={520 + G / 2} y={34 + G + 42} s="now with context" size={13} c={T.soft} />
      </g>
    </Svg>
  );
}

// ------------------------------------------------------------ 4 · one Linear
function LinearScene({ pic }) {
  const p = useProgress("project", 3200), a = seg(p, 0, 0.25), row = Math.min(255, Math.floor(seg(p, 0.25, 0.85) * 256)), c = seg(p, 0.85, 1);
  const sc = 0.1, h = 128, X = 10, Y = 50, xw = PG.vD * sc, ww = PG.D * sc;
  const ry = Y + (row / 256) * h;
  return (
    <Svg h={290}>
      <Txt x={X} y={30} a="start" s="image features" size={15} w={700} c={T.ink} />
      <rect x={X} y={Y} width={xw} height={h} fill={`${C_IMG}22`} stroke={C_IMG} />
      {pic.cols.filter((_, i) => i % 16 === 0).map((col, i) => <rect key={i} x={X} y={Y + i * 8} width={6} height={7} fill={rgb(col)} />)}
      <rect x={X} y={ry} width={xw} height={2.5} fill={T.ink} opacity={a} />
      <Txt x={X + xw / 2} y={Y + h + 22} s="[256, 1152]" size={14} c={C_IMG} />
      <Txt x={X + xw + 22} y={Y + h / 2 + 6} s="@" size={22} w={700} c={T.accent2} />
      <g opacity={a}>
        <Txt x={160} y={30} a="start" s="W  (learned)" size={15} w={700} c={T.ink} />
        <rect x={160} y={Y - 10} width={ww} height={xw} fill={`${T.accent2}14`} stroke={T.accent2} />
        <Txt x={160 + ww / 2} y={Y + xw + 12} s="[1152, 2048] + bias" size={14} c={T.accent2} />
      </g>
      <Txt x={160 + ww + 20} y={Y + h / 2 + 6} s="=" size={22} w={700} c={T.dim} op={a} />
      <g opacity={a}>
        <Txt x={400} y={30} a="start" s="Gemma-width rows" size={15} w={700} c={T.ink} />
        <rect x={400} y={Y} width={ww} height={h} fill={`${T.accent}12`} stroke={T.accent} />
        <rect x={400} y={Y} width={ww} height={ry - Y + 2.5} fill={`${T.accent}40`} />
        <rect x={400} y={ry} width={ww} height={2.5} fill={T.accent} />
        <Txt x={400 + ww / 2} y={Y + h + 22} s="[256, 2048]" size={14} c={T.accent} />
      </g>
      <g opacity={c}>
        <Txt x={X} y={262} a="start" s="one Linear · no MLP · no pooling" size={16} w={700} c={T.ink} />
        <Txt x={X} y={284} a="start" s="256 image tokens = 256 LLM tokens" size={14} c={T.soft} />
      </g>
    </Svg>
  );
}

// ------------------------------------------------------------ 5 · fill the <image> slots
function MergeScene({ pic }) {
  const p = useProgress("merge", 3600), a = seg(p, 0, 0.3), b = seg(p, 0.3, 0.7), c = seg(p, 0.7, 1);
  const X0 = 40, colW = 22, items = [...Array.from({ length: 8 }, () => "img"), ...PROMPT.map((q) => q[0])];
  let cx = X0;
  const xs = items.map((s) => { const x = cx; cx += s === "img" ? colW : Math.max(34, s.length * 9 + 14); return x; });
  const vec = (x, y, col, k, op = 1) => <g opacity={op}>{Array.from({ length: 9 }, (_, j) => <rect key={j} x={x} y={y + j * 11} width={16} height={9} rx={2} fill={col} opacity={0.35 + 0.6 * Math.abs(Math.sin(k * 3.1 + j * 1.7))} />)}</g>;
  return (
    <Svg h={330}>
      <Txt x={10} y={22} a="start" s="ids" size={14} c={T.dim} />
      <SeqChips x={X0} y={6} img={false} h={26} />
      <Txt x={10} y={62} a="start" s="embed_tokens [257216, 2048] → one row per id" size={14} c={T.soft} op={a} />
      {items.map((s, i) => {
        const isImg = s === "img", x = xs[i] + (isImg ? 0 : (Math.max(34, s.length * 9 + 14) - 20) / 2);
        return (
          <g key={i}>
            {vec(x, 80, isImg ? T.dim : (i === 8 ? C_BOS : C_TXT), i, a * (isImg ? 1 - b : 1))}
            {isImg && vec(x, lerp(-120, 80, b), C_IMG, i + 0.4, b)}
          </g>
        );
      })}
      <g opacity={b}>
        <Txt x={X0 + 86} y={202} s="image rows drop in" size={14} c={C_IMG} />
        <Txt x={X0 + 86} y={222} s="ids == <image>" size={13} c={T.soft} />
      </g>
      <g opacity={c}>
        <rect x={X0 - 4} y={74} width={cx - X0 + 4} height={108} rx={6} fill="none" stroke={T.accent} strokeWidth={1.4} />
        <Txt x={cx + 10} y={124} a="start" s="x" size={18} w={700} c={T.accent} />
        <Txt x={cx + 10} y={146} a="start" s={`[1, ${S}, 2048]`} size={15} w={700} c={T.ink} />
        <Txt x={10} y={270} a="start" s="masked_scatter: 256 rows replaced" size={16} w={700} c={T.ink} />
        <Txt x={10} y={292} a="start" s="text rows: from the embedding table" size={14} c={T.soft} />
      </g>
    </Svg>
  );
}

// ------------------------------------------------------------ 6 · ÷√d then ×√d
function ScaleScene() {
  const p = useProgress("scale", 3600), a = seg(p, 0.1, 0.45), b = seg(p, 0.55, 0.9);
  const img = [40, 40 / SQ, 40], txt = [0.9, 0.9, 0.9 * SQ];
  const W = 160, sx = (v) => (v / 45) * W;
  const cols = [[10, "after Linear / embed", 1], [240, `image ÷ √2048`, a], [470, `all × √2048`, b]];
  const val = (arr, k, t) => (k === 0 ? arr[0] : k === 1 ? lerp(arr[0], arr[1], t) : lerp(arr[1], arr[2], t));
  return (
    <Svg h={300}>
      <Txt x={10} y={22} a="start" s={`√2048 = ${SQ.toFixed(2)}  ·  ‖row‖ (toy)`} size={15} w={700} c={T.ink} />
      {cols.map(([x, lab, t], k) => (
        <g key={k} opacity={k === 0 ? 1 : 0.25 + 0.75 * t}>
          <Txt x={x} y={62} a="start" s={lab} size={14} w={700} c={k === 0 ? T.soft : T.ink} />
          {[[img, C_IMG, "image"], [txt, C_TXT, "text"]].map(([arr, col, nm], r) => {
            const v = val(arr, k, t);
            return (
              <g key={nm}>
                <Txt x={x} y={98 + r * 70} a="start" s={nm} size={13} c={col} />
                <rect x={x} y={106 + r * 70} width={W} height={20} rx={4} fill={T.well} />
                <rect x={x} y={106 + r * 70} width={Math.max(2, sx(v))} height={20} rx={4} fill={col} />
                <Txt x={x + W + 6} y={121 + r * 70} a="start" s={v.toFixed(v < 10 ? 2 : 1)} size={14} w={700} c={T.ink} />
              </g>
            );
          })}
        </g>
      ))}
      <g opacity={b}>
        <Txt x={10} y={258} a="start" s="image rows: net × 1 (unchanged)" size={15} w={700} c={C_IMG} />
        <Txt x={10} y={282} a="start" s="text rows: × 45.25 → same scale" size={15} w={700} c={C_TXT} />
      </g>
    </Svg>
  );
}

// ------------------------------------------------------------ 7 · prefix mask + MQA
function Mask({ x, y, n, kinds, rule, title, sub, p }) {
  const cs = 13;
  const col = (k) => (k === "i" ? C_IMG : k === "b" ? C_BOS : k === "t" ? C_TXT : C_ANS);
  return (
    <g>
      <Txt x={x} y={y - 26} a="start" s={title} size={15} w={700} c={T.ink} />
      <Txt x={x} y={y - 8} a="start" s={sub} size={13} c={T.soft} />
      {kinds.map((k, i) => <rect key={`r${i}`} x={x} y={y + 6 + i * cs} width={5} height={cs - 2} fill={col(k)} />)}
      {kinds.map((k, j) => <rect key={`c${j}`} x={x + 9 + j * cs} y={y} width={cs - 2} height={4} fill={col(k)} />)}
      {kinds.map((_, i) => kinds.map((__, j) => {
        const ok = rule(i, j), shown = p * n > i;
        return <rect key={`${i}-${j}`} x={x + 9 + j * cs} y={y + 6 + i * cs} width={cs - 2} height={cs - 2} rx={2} fill={ok ? T.accent : T.well} opacity={shown ? (ok ? 0.85 : 1) : 0.15} />;
      }))}
    </g>
  );
}
function MaskScene() {
  const p = useProgress("mask", 3400), a = seg(p, 0, 0.45), b = seg(p, 0.35, 0.8), c = seg(p, 0.7, 1);
  const pre = ["i", "i", "i", "i", "b", "t", "t", "t"], full = [...pre, "a", "a", "a", "a"], np = pre.length;
  return (
    <Svg h={330}>
      <Mask x={10} y={60} n={8} kinds={pre} rule={() => true} title="inference (repo)" sub="prompt: all see all" p={a} />
      <Mask x={170} y={60} n={12} kinds={full} rule={(i, j) => (j < np ? true : i >= np && j <= i)} title="training" sub="answer: causal" p={b} />
      <g opacity={b}>
        <Txt x={170} y={248} a="start" s="prefix ↔ prefix · answer → left" size={13} c={T.soft} />
      </g>
      <g opacity={0.9}>
        {[["image", C_IMG], ["<bos>", C_BOS], ["prompt", C_TXT], ["answer", C_ANS]].map(([s, col], k) => (
          <g key={s}><rect x={10 + k * 82} y={292} width={10} height={10} fill={col} /><Txt x={24 + k * 82} y={302} a="start" s={s} size={13} c={T.soft} /></g>
        ))}
      </g>
      {/* MQA: 8 query heads share 1 K/V head */}
      <g opacity={c}>
        <Txt x={400} y={34} a="start" s="8 Q heads · 1 KV head" size={15} w={700} c={T.ink} />
        {Array.from({ length: 8 }, (_, h) => (
          <g key={h}>
            <rect x={400 + h * 34} y={56} width={28} height={28} rx={5} fill={`${T.accent}22`} stroke={T.accent} />
            <Txt x={414 + h * 34} y={75} s={`q${h}`} size={13} c={T.ink} />
            <line x1={414 + h * 34} y1={84} x2={530} y2={146} stroke={T.dim} strokeOpacity={0.6} />
          </g>
        ))}
        <rect x={496} y={146} width={34} height={30} rx={5} fill={`${T.wire}22`} stroke={T.wire} />
        <Txt x={513} y={166} s="k" size={14} w={700} c={T.ink} />
        <rect x={538} y={146} width={34} height={30} rx={5} fill={`${T.wire}22`} stroke={T.wire} />
        <Txt x={555} y={166} s="v" size={14} w={700} c={T.ink} />
        <Txt x={400} y={206} a="start" s="repeat_kv × 8" size={14} c={T.soft} />
        <Txt x={400} y={228} a="start" s="q [1, 8, 260, 256]" size={14} c={T.ink} />
        <Txt x={400} y={248} a="start" s="k, v [1, 1, 260, 256]" size={14} c={T.wire} />
        <Txt x={400} y={270} a="start" s="KV cache 8× smaller" size={14} w={700} c={T.wire} />
      </g>
    </Svg>
  );
}

// ------------------------------------------------------------ 8 · one Gemma block
function BlockScene() {
  const p = useProgress("block", 4200);
  const boxes = [
    [10, 40, 120, "RMSNorm", "x · (1 + w)", T.soft],
    [150, 40, 150, "attention", "RoPE on q, k", T.accent],
    [320, 40, 60, "+ x", "residual", T.dim],
    [10, 150, 120, "RMSNorm", "x · (1 + w)", T.soft],
    [150, 150, 230, "GeGLU", "gelu(gate) · up → down", T.wire],
    [400, 150, 60, "+ x", "residual", T.dim],
  ];
  const k = Math.min(boxes.length - 1, Math.floor(p * boxes.length * 1.05));
  const ang = seg(p, 0.15, 0.45) * 1.1;
  return (
    <Svg h={320}>
      <Txt x={10} y={22} a="start" s="x [260, 2048]  ·  × 18 blocks" size={15} w={700} c={T.ink} />
      {boxes.map(([x, y, w, t, s, col], i) => (
        <g key={i} opacity={i <= k ? 1 : 0.3}>
          <rect x={x} y={y} width={w} height={56} rx={8} fill={i === k ? `${col}26` : T.well} stroke={col} strokeWidth={i === k ? 2 : 1.2} />
          <Txt x={x + w / 2} y={y + 24} s={t} size={15} w={700} c={T.ink} />
          <Txt x={x + w / 2} y={y + 44} s={s} size={13} c={col === T.dim ? T.soft : col} />
        </g>
      ))}
      <Arrow x1={130} y1={68} x2={150} y2={68} />
      <Arrow x1={300} y1={68} x2={320} y2={68} />
      <path d="M380,68 H392 V128 H70 V150" fill="none" stroke={T.dim} strokeWidth={1.3} />
      <Arrow x1={130} y1={178} x2={150} y2={178} />
      <Arrow x1={380} y1={178} x2={400} y2={178} />
      {/* GeGLU widths */}
      <g opacity={k >= 4 ? 1 : 0.3}>
        <Txt x={150} y={232} a="start" s="2048 → 16384 (gate, up)" size={14} c={T.wire} />
        <Txt x={150} y={252} a="start" s="16384 → 2048 (down)" size={14} c={T.wire} />
      </g>
      {/* RoPE dial: a (q) pair rotated by position × θ */}
      <g opacity={k >= 1 ? 1 : 0.3}>
        <Txt x={560} y={22} s="RoPE" size={15} w={700} c={T.ink} />
        <circle cx={560} cy={100} r={60} fill="none" stroke={T.rule} />
        <line x1={560} y1={100} x2={620} y2={100} stroke={T.dim} strokeDasharray="3 3" />
        <line x1={560} y1={100} x2={560 + 60 * Math.cos(-ang)} y2={100 + 60 * Math.sin(-ang)} stroke={T.accent} strokeWidth={2.4} />
        <circle cx={560 + 60 * Math.cos(-ang)} cy={100 + 60 * Math.sin(-ang)} r={4} fill={T.accent} />
        <Txt x={560} y={186} s="each (q, k) pair" size={13} c={T.soft} />
        <Txt x={560} y={204} s="turned by pos · θ" size={13} c={T.soft} />
      </g>
      <Txt x={10} y={300} a="start" s="final RMSNorm → logits" size={14} c={T.soft} op={seg(p, 0.85, 1)} />
    </Svg>
  );
}

// ------------------------------------------------------------ 9 · greedy decoding + KV cache
function GenScene() {
  const p = useProgress("generate", 8000);
  const t = p * 5.4, k = Math.min(4, Math.floor(t)), f = clamp01(t - Math.floor(t)), ans = Math.min(5, Math.floor(t + 0.15));
  const cache = S + k;
  const cand = CANDS[k];
  return (
    <Svg h={340}>
      <Txt x={10} y={22} a="start" s={k === 0 ? "prefill: 260 tokens in" : "decode: 1 token in"} size={16} w={700} c={T.ink} />
      <SeqChips x={10} y={36} ans={ans} img={false} />
      <Txt x={96} y={88} s="<image> × 256" size={13} c={C_IMG} />
      {/* KV cache bar */}
      <Txt x={10} y={128} a="start" s="KV cache (per layer)" size={14} c={T.soft} />
      <rect x={10} y={138} width={400} height={22} rx={4} fill={T.well} />
      <rect x={10} y={138} width={(S / 266) * 400} height={22} rx={4} fill={`${T.wire}66`} />
      <rect x={10 + (S / 266) * 400} y={138} width={(k / 266) * 400 + f * (1 / 266) * 400} height={22} fill={C_ANS} />
      <Txt x={420} y={154} a="start" s={`${cache} × [1, 256]`} size={15} w={700} c={T.ink} />
      <Txt x={10} y={184} a="start" s={`new token position = ${S + k + 1}`} size={14} c={T.soft} />
      {/* greedy pick */}
      <Txt x={10} y={222} a="start" s="next-token probabilities (toy)" size={14} c={T.soft} />
      {cand.map(([s, pr], i) => (
        <g key={s + i}>
          <Txt x={10} y={250 + i * 22} a="start" s={s} size={14} w={i === 0 ? 700 : 400} c={i === 0 ? C_ANS : T.soft} />
          <rect x={110} y={238 + i * 22} width={pr * 260} height={15} rx={3} fill={i === 0 ? C_ANS : T.dim} />
          <Txt x={116 + pr * 260} y={250 + i * 22} a="start" s={pr.toFixed(2)} size={13} c={T.ink} />
        </g>
      ))}
      <Txt x={440} y={250} a="start" s="argmax →" size={15} c={T.soft} />
      <Chip x={530} y={234} w={90} s={cand[0][0]} c={C_ANS} />
      <Txt x={440} y={300} a="start" s={ans >= 5 ? '"a dog on grass"' : "stop at <eos>"} size={15} w={700} c={ans >= 5 ? C_ANS : T.dim} />
    </Svg>
  );
}

// ------------------------------------------------------------ the view
const CODE = {
  prompt: [
    'return f"{image_token * image_seq_len}{bos_token}{prefix_prompt}\\n"',
    "rescale_factor=1 / 255.0,",
    "image_mean=IMAGENET_STANDARD_MEAN,   # 0.5",
  ],
  patches: [
    "patch_embeds = self.patch_embedding(pixel_values)   # Conv2d(3, 1152, 14, stride=14)",
    "embeddings = embeddings + self.position_embedding(self.position_ids)",
  ],
  siglip: [
    "hidden_states, _ = self.self_attn(hidden_states=hidden_states)",
    "last_hidden_state = self.post_layernorm(last_hidden_state)",
  ],
  project: ["self.linear = nn.Linear(config.vision_config.hidden_size, config.vision_config.projection_dim, bias=True)"],
  merge: [
    "image_mask = input_ids == self.config.image_token_index",
    "final_embedding = final_embedding.masked_scatter(image_mask_expanded, scaled_image_features)",
  ],
  scale: [
    "scaled_image_features = image_features / (self.config.hidden_size**0.5)",
    "normalizer = torch.tensor(self.config.hidden_size**0.5, dtype=hidden_states.dtype)",
    "hidden_states = hidden_states * normalizer   # GemmaModel",
  ],
  mask: [
    "causal_mask = torch.full((batch_size, q_len, q_len), fill_value=0, dtype=dtype, device=device)",
    "key_states = repeat_kv(key_states, self.num_key_value_groups)",
    "# training (not in this repo): causal over the answer",
  ],
  block: [
    "output = output * (1.0 + self.weight.float())",
    "q_embed = (q * cos) + (rotate_half(q) * sin)",
    'return self.down_proj(nn.functional.gelu(self.gate_proj(x), approximate="tanh") * self.up_proj(x))',
  ],
  generate: [
    "self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=-2)",
    "self.lm_head.weight = self.model.embed_tokens.weight   # tied",
    "# do_sample=False → next_token = argmax(logits[:, -1])",
  ],
};

const FLOW = [
  { group: "1 · prompt", items: [{ id: "prompt", label: "image + text → ids", sub: "[1, 260]", code: CODE.prompt }] },
  { group: "2 · SigLIP", items: [
    { id: "patches", label: "patches → 256 vectors", sub: "14 px · 1152 wide", code: CODE.patches },
    { id: "siglip", label: "27 blocks · no CLS", sub: "[256, 1152]", code: CODE.siglip },
  ] },
  { group: "3 · bridge", items: [
    { id: "project", label: "one Linear → 2048", sub: "no MLP", code: CODE.project },
    { id: "merge", label: "fill the image slots", sub: "masked_scatter", code: CODE.merge },
    { id: "scale", label: "÷ √d, then × √d", sub: "45.25", code: CODE.scale },
  ] },
  { group: "4 · Gemma", items: [
    { id: "mask", label: "prefix mask · 8Q / 1KV", sub: "who sees whom", code: CODE.mask },
    { id: "block", label: "inside one block", sub: "RMSNorm · RoPE · GeGLU", code: CODE.block },
    { id: "generate", label: "greedy + KV cache", sub: "→ a dog on grass", code: CODE.generate },
  ] },
];

const STEPS = [
  { id: "prompt", block: "prompt", caption: "image + “caption en” → 260 token ids" },
  { id: "patches", block: "patches", caption: "one 14 × 14 patch → one 1152 vector" },
  { id: "siglip", block: "siglip", caption: "27 SigLIP blocks · all 256 kept" },
  { id: "project", block: "project", caption: "one Linear: 1152 → 2048" },
  { id: "merge", block: "merge", caption: "image rows fill the <image> slots" },
  { id: "scale", block: "scale", caption: "÷ √2048 for images, × √2048 for all" },
  { id: "mask", block: "mask", caption: "prefix sees all · answer is causal" },
  { id: "block", block: "block", caption: "one Gemma block, 18 times" },
  { id: "generate", block: "generate", caption: "greedy, one token at a time" },
];

const PATCH_M = { id: "pali", col: C_IMG, patch: PG.P, pp: PG.P * PG.P * 3, vitD: PG.vD, vitL: PG.vL, res: PG.res };

function Scene({ st }) {
  const pic = usePic();
  const body = {
    prompt: <PromptScene pic={pic} />,
    patches: <PatchScene m={PATCH_M} />,
    siglip: <SiglipScene pic={pic} />,
    project: <LinearScene pic={pic} />,
    merge: <MergeScene pic={pic} />,
    scale: <ScaleScene />,
    mask: <MaskScene />,
    block: <BlockScene />,
    generate: <GenScene />,
  }[st.id];
  return <div style={{ minHeight: 360 }}>{body}</div>;
}

function matmuls(st) {
  const body = {
    prompt: <MMNote>pixel_values [1, 3, 224, 224] · input_ids [1, {S}] · positions 1 … {S}</MMNote>,
    patches: <>
      <MM a={["patches", [256, 588]]} b={["W_patch", [588, 1152]]} c={["x", [256, 1152]]} note="= Conv2d(3, 1152, kernel 14, stride 14)" />
      <MMNote>x + position_embedding [256, 1152]</MMNote>
    </>,
    siglip: <BlockMMs B={1} T={256} D={1152} h={16} d={72} M={4304} L={27} act="GELU (tanh)" />,
    project: <MM a={["image_features", [256, 1152]]} b={["W", [1152, 2048]]} c={["image_rows", [256, 2048]]} note="+ bias" />,
    merge: <>
      <MM a={["one_hot(ids)", [S, PG.V]]} b={["embed_tokens", [PG.V, 2048]]} c={["x", [S, 2048]]} note="really a lookup" cost="0 (a gather)" />
      <MMNote>masked_scatter: 256 rows ← image_rows [256, 2048] → x [1, {S}, 2048]</MMNote>
    </>,
    scale: <MMNote>image_rows ÷ {SQ.toFixed(2)} · then x [1, {S}, 2048] × {SQ.toFixed(2)}</MMNote>,
    mask: <>
      <MM a={["q", [8, S, 256]]} b={["kᵀ", [8, 256, S]]} c={["scores", [8, S, S]]} times={18} note="k: 1 head repeated × 8 · + mask · softmax" />
      <MM a={["A", [8, S, S]]} b={["v", [8, S, 256]]} c={["o", [8, S, 256]]} times={18} note="concat → [260, 2048]" />
    </>,
    block: <>
      <MM a={["x", [S, 2048]]} b={["W_q", [2048, 2048]]} c={["q", [S, 2048]]} times={18} note="8 heads × 256" />
      <MM a={["x", [S, 2048]]} b={["W_k, W_v", [2048, 256]]} c={["k, v", [S, 256]]} times={36} note="1 KV head" />
      <MM a={["x", [S, 2048]]} b={["W_gate, W_up", [2048, 16384]]} c={["g, u", [S, 16384]]} times={36} />
      <MM a={["gelu(g)·u", [S, 16384]]} b={["W_down", [16384, 2048]]} c={["y", [S, 2048]]} times={18} />
    </>,
    generate: <>
      <MM a={["h[last]", [1, 2048]]} b={["embed_tokensᵀ", [2048, PG.V]]} c={["logits", [1, PG.V]]} note="tied weights · argmax" />
      <MM a={["q (decode)", [8, 1, 256]]} b={["kᵀ cache", [8, 256, S + 1]]} c={["scores", [8, 1, S + 1]]} times={18} note="1 new query vs the whole cache" />
    </>,
  };
  return <MatMulCard>{body[st.id]}</MatMulCard>;
}

export default function PaliGemmaView() {
  return (
    <StepShell
      title="PaliGemma"
      subtitle="PaliGemma-3B-224 · one dog picture + “caption en” · toy values, real shapes"
      blocks={FLOW}
      steps={STEPS}
      scene={(st) => <Scene key={st.id} st={st} />}
      matmuls={matmuls}
      controls={null}
    />
  );
}
