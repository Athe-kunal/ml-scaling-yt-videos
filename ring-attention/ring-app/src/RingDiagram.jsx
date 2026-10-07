import { T, mono } from "./theme";
import { DEV_COLORS, PHASE, mod, sub, deviceStep, heldKV, inFlight, ownChunks } from "./ringModel";

// N GPUs on a circle, clockwise arcs r -> r+1 (send_rank). Each box shows
// the stationary Q shard and the KV block currently in `k, v`; packets ride
// the arcs while isend/irecv are in flight and land on the wait phase.

const W = 620;
const BW = 132;
const BH = 92;

function Chip({ x, y, w, label, color, on }) {
  return (
    <g opacity={on ? 1 : 0.3} style={{ transition: "opacity .3s" }}>
      <rect x={x} y={y} width={w} height={17} rx={4} fill={`${color}26`} stroke={color} strokeWidth={on ? 1.4 : 1} />
      <text x={x + w / 2} y={y + 12.5} textAnchor="middle" fontFamily={mono} fontSize="11" fontWeight="600" fill={color}>{label}</text>
    </g>
  );
}

function statusText(variant, N, r, phase) {
  const { type, s } = phase;
  if (type === "init") return { t: "Q, K, V local", c: T.dim };
  if (type === "final") return { t: "out ready", c: T.soft };
  const d = deviceStep(variant, N, r, s);
  if (type === "send") return { t: `isend KV${sub(d.j)} → ${mod(r + 1, N)}`, c: T.wire };
  if (type === "wait") return { t: "k, v = next_k, next_v", c: T.wire };
  if (!d.active) return { t: "idle · masked", c: T.bad };
  if (type === "merge") return { t: d.branch === "hi" ? "merge out[L/2:]" : s === 0 ? "out = block_out" : "merge out, lse", c: T.accent2 };
  const lbl = variant === "ring"
    ? d.branch === "0" ? "causal ◣ diag" : `full Q${sub(r)}·K${sub(d.j)}`
    : d.branch === "0" ? "causal ◣ local" : d.branch === "lo" ? "q · k0  (K 1st half)" : "q1 · k  (Q 2nd half)";
  return { t: lbl, c: T.accent };
}

export default function RingDiagram({ variant, N, phase, sel, onSel }) {
  const R = N <= 2 ? 150 : N <= 4 ? 170 : N <= 6 ? 215 : 238;
  const H = 2 * R + BH + 70;
  const cx = W / 2, cy = H / 2 + 6;
  const span = (2 * Math.PI) / N;
  const ang = (r) => -Math.PI / 2 + span * r;
  const pos = (r) => [cx + R * Math.cos(ang(r)), cy + R * Math.sin(ang(r))];
  const delta = Math.min(span * 0.4, 88 / R);
  const arcPt = (r, t) => {
    const th = ang(r) + delta + t * (span - 2 * delta);
    return [cx + R * Math.cos(th), cy + R * Math.sin(th)];
  };
  const flying = inFlight(N, phase);
  const landed = phase.type === "wait";
  const tProg = { send: 0.18, compute: 0.5, merge: 0.8 }[phase.type] ?? 1;
  const prev = mod(sel - 1, N), next = mod(sel + 1, N);
  const ph = PHASE[phase.type];

  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", display: "block" }}>
      <defs>
        {[["a-dim", T.rule], ["a-wire", T.wire]].map(([id, c]) => (
          <marker key={id} id={`${variant}-${id}`} markerUnits="userSpaceOnUse" markerWidth="10" markerHeight="10" refX="8" refY="5" orient="auto">
            <path d="M0,0 L10,5 L0,10 z" fill={c} />
          </marker>
        ))}
      </defs>

      {/* ring arcs r -> r+1 */}
      {Array.from({ length: N }, (_, r) => {
        const [x1, y1] = arcPt(r, 0);
        const [x2, y2] = arcPt(r, 1);
        const live = flying || landed;
        const selEdge = r === sel || r === prev;
        return (
          <path
            key={r}
            d={`M${x1},${y1} A${R},${R} 0 0 1 ${x2},${y2}`}
            fill="none" stroke={live ? T.wire : T.rule}
            strokeWidth={live && selEdge ? 2.6 : live ? 1.6 : 1.2}
            strokeDasharray={flying ? "6 5" : "none"}
            opacity={live ? (selEdge ? 1 : 0.55) : 0.8}
            markerEnd={`url(#${variant}-${live ? "a-wire" : "a-dim"})`}
          />
        );
      })}

      {/* center readout */}
      <text x={cx} y={cy - 22} textAnchor="middle" fontFamily={mono} fontSize="12" fill={T.dim} letterSpacing="0.12em">
        {phase.type === "init" ? "BEFORE LOOP" : phase.type === "final" ? "AFTER LOOP" : `STEP ${phase.s} / ${N - 1}`}
      </text>
      <text x={cx} y={cy + 2} textAnchor="middle" fontFamily={mono} fontSize="17" fontWeight="700" fill={ph.color}>{ph.label}</text>
      <text x={cx} y={cy + 24} textAnchor="middle" fontFamily={mono} fontSize="11" fill={T.dim}>
        {flying ? "KV in flight" : landed ? "KV landed" : phase.type === "init" || phase.type === "final" ? "" : "last step: no send"}
      </text>

      {/* devices */}
      {Array.from({ length: N }, (_, r) => {
        const [x, y] = pos(r);
        const bx = x - BW / 2, by = y - BH / 2;
        const col = DEV_COLORS[r];
        const held = heldKV(N, r, phase);
        const d = phase.s >= 0 ? deviceStep(variant, N, r, phase.s) : null;
        const working = d && (phase.type === "compute" || phase.type === "merge");
        const qOn = (i) => !working || d.useQ.includes(i);
        const kOn = (i) => !working || d.useKV.includes(i);
        const qc = ownChunks(variant, N, r);
        const kc = ownChunks(variant, N, held);
        const st = statusText(variant, N, r, phase);
        const isSel = r === sel;
        const tag = r === prev && r === next ? "prev = next" : r === prev ? "prev · recv_rank" : r === next ? "next · send_rank" : null;
        const zig = variant === "zigzag";
        const chipW = zig ? 40 : 84;
        return (
          <g key={r} onClick={() => onSel(r)} style={{ cursor: "pointer" }}>
            <rect
              x={bx} y={by} width={BW} height={BH} rx={9}
              fill={isSel ? `${col}1c` : T.well} stroke={isSel ? col : working && d.active ? T.accent : T.rule}
              strokeWidth={isSel ? 2.4 : 1.3} opacity={working && !d.active ? 0.55 : 1}
              style={{ transition: "all .3s" }}
            />
            <text x={bx + 9} y={by + 16} fontFamily={mono} fontSize="11.5" fontWeight="700" fill={col}>GPU {r}</text>
            <text x={bx + BW - 9} y={by + 16} textAnchor="end" fontFamily={mono} fontSize="9.5" fill={T.dim}>rank {r}</text>

            <text x={bx + 9} y={by + 37} fontFamily={mono} fontSize="9.5" fill={T.dim}>q</text>
            {qc.map((c, i) => (
              <Chip key={i} x={bx + 30 + i * (chipW + 6)} y={by + 25} w={chipW} label={zig ? `Q${sub(c)}` : `Q${sub(r)}`} color={col} on={qOn(i)} />
            )).slice(0, zig ? 2 : 1)}

            <text x={bx + 9} y={by + 59} fontFamily={mono} fontSize="9.5" fill={T.dim}>kv</text>
            {kc.map((c, i) => (
              <Chip key={`${held}-${i}`} x={bx + 30 + i * (chipW + 6)} y={by + 47} w={chipW} label={zig ? `KV${sub(c)}` : `KV${sub(held)}`} color={DEV_COLORS[held]} on={kOn(i)} />
            )).slice(0, zig ? 2 : 1)}

            <text x={x} y={by + 82} textAnchor="middle" fontFamily={mono} fontSize="9.5" fill={st.c}>{st.t}</text>

            {tag && (
              <g>
                <rect x={x - 54} y={by - 10} width={108} height={16} rx={8} fill={T.bg} stroke={T.wire} strokeWidth={1} />
                <text x={x} y={by + 1.5} textAnchor="middle" fontFamily={mono} fontSize="9" fill={T.wire}>{tag}</text>
              </g>
            )}
          </g>
        );
      })}

      {/* packets in flight: GPU r sends the KV it holds to r+1 */}
      {(flying || landed) && Array.from({ length: N }, (_, r) => {
        const j = mod(r - phase.s, N);
        const [px, py] = arcPt(r, landed ? 1 : tProg);
        const lbl = variant === "zigzag" ? `KV${sub(j)},${sub(2 * N - 1 - j)}` : `KV${sub(j)}`;
        const w = variant === "zigzag" ? 64 : 42;
        const hot = r === sel || r === prev;
        return (
          <g
            key={`${phase.s}-${r}`}
            style={{ transform: `translate(${px}px, ${py}px)`, transition: "transform .6s cubic-bezier(.3,.7,.3,1), opacity .4s" }}
            opacity={landed ? 0 : 1}
          >
            <rect x={-w / 2} y={-10} width={w} height={20} rx={10} fill={T.bg} stroke={DEV_COLORS[j]} strokeWidth={hot ? 2 : 1.2} />
            <text x={0} y={4} textAnchor="middle" fontFamily={mono} fontSize="10.5" fontWeight="700" fill={DEV_COLORS[j]}>{lbl}</text>
          </g>
        );
      })}
    </svg>
  );
}
