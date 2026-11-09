import { T, mono } from "./theme";
import { EXPERT_COLORS as EC } from "./moeModel";
import { useProgress, ease, lerp, clamp01 } from "./anim";

// The primary picture for the "All-to-all" tab: the whole MoE block, bottom
// (previous layer) to top (next layer) — the same shape as the paper figure
// this was modelled on. A yellow highlighter-style glow marks whichever
// stage the current step belongs to, and a travelling dot shows a token's
// vector actually moving along that stage's path: Router → All-to-All
// dispatch → an expert → × (gate weight) → All-to-All combine → +. Hatched
// expert boxes are the ones living on the device you're following.

const W = 640, H = 520, CX = W / 2;
const ADD2_Y = 58, COMB_Y = 86, COMB_H = 44, MUL_Y = 158, EXP_Y = 182, EXP_H = 56;
const DISP_Y = 266, DISP_H = 44, ROUTER_Y = 334, ROUTER_H = 44, ADD1_Y = 402, ATTN_Y = 436, ATTN_H = 44;
const BOX_W = 290, BOX_X = CX - BOX_W / 2;

function Glow({ x, y, w, h, on }) {
  if (!on) return null;
  return <rect x={x - 9} y={y - 9} width={w + 18} height={h + 18} rx={13} fill={T.wire} fillOpacity={0.3} style={{ transition: "opacity .25s" }} />;
}

function Box({ x, y, w, h, label, sub, on }) {
  return (
    <g>
      <Glow x={x} y={y} w={w} h={h} on={on} />
      <rect x={x} y={y} width={w} height={h} rx={9} fill={T.panel} stroke={on ? T.ink : T.rule} strokeWidth={on ? 2.2 : 1.4} style={{ transition: "all .25s" }} />
      <text x={x + w / 2} y={y + h / 2 + (sub ? -4 : 5)} textAnchor="middle" fontFamily={mono} fontSize="14" fontWeight="700" fill={T.ink}>{label}</text>
      {sub && <text x={x + w / 2} y={y + h / 2 + 14} textAnchor="middle" fontFamily={mono} fontSize="10.5" fill={T.dim}>{sub}</text>}
    </g>
  );
}

function Arrow({ x1, y1, x2, y2, dashed, on }) {
  return <path d={`M${x1},${y1} L${x2},${y2}`} fill="none" stroke={on ? T.wire : T.rule} strokeWidth={on ? 2.4 : 1.4} strokeDasharray={dashed ? "4 4" : "none"} markerEnd={`url(#mbm-arrow${on ? "-on" : ""})`} opacity={on ? 1 : 0.85} />;
}

// stage ∈ "router" | "dispatch" | "expert" | "combine"
function stageOf(step) {
  if (step.sid === "combine" || step.back != null) return "combine";
  if (step.ffn || step.sid === "moe") return "expert";
  if (step.sid === "start" || step.sid === "addr") return "router";
  return "dispatch"; // keep, the D-1 send/recv rounds, done
}

export default function MoEBlockMap({ step, Ed, E, sel }) {
  const stage = stageOf(step);
  const p = ease(useProgress(`${step.id}:${stage}`));
  const ew = Math.max(30, Math.min(84, 540 / E)), gap = 6;
  const totalW = E * ew + (E - 1) * gap, ex0 = CX - totalW / 2;
  const expX = (i) => ex0 + i * (ew + gap);
  const litExperts = Array.from({ length: E }, (_, i) => Math.floor(i / Ed) === sel);
  const anyLit = litExperts.some(Boolean);
  const firstLit = litExperts.indexOf(true);
  const firstLitCx = firstLit >= 0 ? expX(firstLit) + ew / 2 : CX;

  // the travelling dot: a 2-leg path per stage, p in [0,1] crossing both legs
  let dot = null;
  if (stage === "router") dot = [CX, lerp(ADD1_Y - 13, ROUTER_Y + ROUTER_H, p)];
  else if (stage === "dispatch") {
    dot = p < 0.45
      ? [CX, lerp(ROUTER_Y, DISP_Y + DISP_H, p / 0.45)]
      : [lerp(CX, firstLitCx, (p - 0.45) / 0.55), lerp(DISP_Y, EXP_Y + EXP_H, (p - 0.45) / 0.55)];
  } else if (stage === "expert") dot = [firstLitCx, lerp(EXP_Y, MUL_Y + 11, p)];
  else if (stage === "combine") {
    dot = p < 0.4
      ? [lerp(firstLitCx, CX, p / 0.4), lerp(MUL_Y - 11, COMB_Y + COMB_H, p / 0.4)]
      : p < 0.75
        ? [CX, lerp(COMB_Y, ADD2_Y + 13, (p - 0.4) / 0.35)]
        : [CX, lerp(ADD2_Y - 13, 20, (p - 0.75) / 0.25)];
  }

  return (
    <div style={{ background: T.panel, border: `1px solid ${T.rule}`, borderRadius: 12, padding: "14px 18px" }}>
      <div style={{ fontFamily: mono, fontSize: 11, letterSpacing: "0.12em", color: T.dim, marginBottom: 8 }}>HOW ALL-TO-ALL FITS IN THE MoE BLOCK</div>
      <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", maxWidth: 440, display: "block", margin: "0 auto" }}>
        <defs>
          <marker id="mbm-arrow" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto">
            <path d="M0,0 L8,4 L0,8 z" fill={T.rule} />
          </marker>
          <marker id="mbm-arrow-on" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto">
            <path d="M0,0 L8,4 L0,8 z" fill={T.wire} />
          </marker>
        </defs>

        {/* residual skip: branches off add1, bypasses router/dispatch/experts/combine, re-enters at add2 */}
        <path d={`M${BOX_X - 1},${ADD1_Y} L40,${ADD1_Y} L40,${ADD2_Y} L${BOX_X - 1},${ADD2_Y}`} fill="none" stroke={T.rule} strokeWidth={1.3} markerEnd="url(#mbm-arrow)" />

        {/* from previous layer → self-attention → add1 */}
        <text x={CX} y={H - 8} textAnchor="middle" fontFamily={mono} fontSize="11" fill={T.dim}>from previous layer</text>
        <Arrow x1={CX} y1={H - 18} x2={CX} y2={ATTN_Y + ATTN_H} />
        <Box x={CX - 115} y={ATTN_Y} w={230} h={ATTN_H} label="Self-Attention" />
        <Arrow x1={CX} y1={ATTN_Y} x2={CX} y2={ADD1_Y + 13} />
        <circle cx={CX} cy={ADD1_Y} r={13} fill={T.panel} stroke={T.rule} strokeWidth={1.4} />
        <text x={CX} y={ADD1_Y + 5} textAnchor="middle" fontFamily={mono} fontSize="14" fill={T.ink}>+</text>

        {/* add1 → router */}
        <Arrow x1={CX} y1={ADD1_Y - 13} x2={CX} y2={ROUTER_Y + ROUTER_H} on={stage === "router"} />
        <Box x={CX - 95} y={ROUTER_Y} w={190} h={ROUTER_H} label="Router" sub="softmax(w_g·x)" on={stage === "router"} />

        {/* router → all-to-all dispatch */}
        <Arrow x1={CX} y1={ROUTER_Y} x2={CX} y2={DISP_Y + DISP_H} on={stage === "dispatch"} />
        <Box x={BOX_X} y={DISP_Y} w={BOX_W} h={DISP_H} label="All-to-All dispatch" on={stage === "dispatch"} />

        {/* dispatch → every expert (fan), hatched = lives on the followed device */}
        {Array.from({ length: E }, (_, i) => {
          const cx = expX(i) + ew / 2, dev = Math.floor(i / Ed), lit = litExperts[i];
          return (
            <g key={i}>
              <Arrow x1={CX} y1={DISP_Y} x2={cx} y2={EXP_Y + EXP_H} on={stage === "dispatch" && lit} dashed={!lit} />
              <Glow x={expX(i)} y={EXP_Y} w={ew} h={EXP_H} on={stage === "expert" && lit} />
              <rect x={expX(i)} y={EXP_Y} width={ew} height={EXP_H} rx={6} fill={lit ? `${EC[dev]}33` : T.panel} stroke={stage === "expert" && lit ? T.ink : EC[dev]} strokeWidth={stage === "expert" && lit ? 2.2 : 1.4} />
              {lit && Array.from({ length: Math.ceil(EXP_H / 9) }, (_, k) => (
                <line key={k} x1={expX(i)} y1={EXP_Y + k * 9} x2={expX(i) + Math.min(ew, k * 9)} y2={EXP_Y + Math.max(0, k * 9 - ew)} stroke={EC[dev]} strokeOpacity={0.4} strokeWidth={1.2} />
              ))}
              {ew >= 34 && <text x={cx} y={EXP_Y + EXP_H / 2 + 4} textAnchor="middle" fontFamily={mono} fontSize="11.5" fontWeight="700" fill={T.ink}>E{i}</text>}
              {lit && (
                <>
                  <Arrow x1={cx} y1={EXP_Y} x2={cx} y2={MUL_Y + 12} on={stage === "expert"} />
                  <circle cx={cx} cy={MUL_Y} r={12} fill={T.panel} stroke={stage === "combine" ? T.ink : T.rule} strokeWidth={stage === "combine" ? 2.2 : 1.3} />
                  <text x={cx} y={MUL_Y + 4.5} textAnchor="middle" fontFamily={mono} fontSize="13" fill={T.ink}>×</text>
                  <Arrow x1={cx} y1={MUL_Y - 12} x2={CX} y2={COMB_Y + COMB_H} on={stage === "combine"} dashed={cx !== CX} />
                </>
              )}
            </g>
          );
        })}

        {/* all-to-all combine (the return collective) → add2 → next layer */}
        <Box x={BOX_X} y={COMB_Y} w={BOX_W} h={COMB_H} label="All-to-All combine" on={stage === "combine"} />
        <Arrow x1={CX} y1={COMB_Y} x2={CX} y2={ADD2_Y + 13} on={stage === "combine"} />
        <circle cx={CX} cy={ADD2_Y} r={13} fill={T.panel} stroke={stage === "combine" ? T.ink : T.rule} strokeWidth={stage === "combine" ? 2.2 : 1.4} />
        <text x={CX} y={ADD2_Y + 5} textAnchor="middle" fontFamily={mono} fontSize="14" fill={T.ink}>+</text>
        <Arrow x1={CX} y1={ADD2_Y - 13} x2={CX} y2={24} on={stage === "combine"} />
        <text x={CX} y={14} textAnchor="middle" fontFamily={mono} fontSize="11" fill={T.dim}>to next layer</text>

        {/* the travelling token: one vector's actual path through this stage */}
        {dot && <circle cx={dot[0]} cy={dot[1]} r={6} fill={T.accent2} stroke={T.bg} strokeWidth={1.5} />}
      </svg>
      <div style={{ fontSize: 11, color: T.dim, marginTop: 6, textAlign: "center" }}>
        hatched · {anyLit ? `the ${Ed > 1 ? "experts" : "expert"} living on device ${sel}` : "no local expert"} · <span style={{ color: T.wire }}>yellow</span> = this step · <span style={{ color: T.accent2 }}>dot</span> = a token's vector in flight
      </div>
    </div>
  );
}
