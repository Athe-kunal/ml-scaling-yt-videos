import { T, mono } from "./theme";
import { DEV_COLORS, PHASE, mod, deviceStep, heldKV, inFlight, mergedRows, shardLabel, tokensOf } from "./ringModel";

// N GPUs on a circle, clockwise arcs r -> r+1 (send_rank). Each box shows
// the stationary Q shard, the KV block currently in `k, v`, and the out / lse
// rows of its own queries (o,ℓ): those never move, they are merged in place,
// one square per source GPU filling as its KV block is merged. Packets ride
// the arcs while isend/irecv are in flight and land on the wait phase.

const W = 620;
const BW = 132;
const BH = 116;

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
  if (type === "init") return { t: "Q, K, V local · out = None", c: T.dim };
  if (type === "final") return { t: "return out, lse (stay here)", c: T.soft };
  const d = deviceStep(variant, N, r, s);
  if (type === "send") return { t: `isend k, v → GPU ${mod(r + 1, N)}`, c: T.wire };
  if (type === "wait") return { t: "k, v = next_k, next_v", c: T.wire };
  if (!d.active) return { t: "idle · masked", c: T.bad };
  if (type === "merge") return { t: d.branch === "hi" ? (variant === "stripe" ? "merge out[1:]" : "merge out[L/2:]") : s === 0 ? "out = block_out" : "merge out, lse", c: T.accent2 };
  if (variant === "stripe") return { t: d.branch === "lo" ? "causal  p' ≤ p" : "shifted q[1:]·k[:-1]", c: T.accent };
  const lbl = variant === "ring"
    ? d.branch === "0" ? "causal ◣ diag" : "full: 4 chunk²"
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
        // token ids per chip: zig-zag keeps its two halves apart (k0/q1 dim one)
        const split = (ts) => (variant === "zigzag" ? [ts.slice(0, 2), ts.slice(2)] : [ts]);
        const qc = split(tokensOf(variant, N, r)).map((ts) => ts.join(","));
        const kc = split(tokensOf(variant, N, held)).map((ts) => ts.join(","));
        const st = statusText(variant, N, r, phase);
        const merged = mergedRows(variant, N, r, phase);
        const isSel = r === sel;
        const tag = r === prev && r === next ? "prev = next" : r === prev ? "prev · recv_rank" : r === next ? "next · send_rank" : null;
        const chipW = variant === "zigzag" ? 42 : 88;
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
            {qc.map((label, i) => (
              <Chip key={i} x={bx + 30 + i * (chipW + 6)} y={by + 25} w={chipW} label={label} color={col} on={qOn(i)} />
            ))}

            <text x={bx + 9} y={by + 59} fontFamily={mono} fontSize="9.5" fill={T.dim}>kv</text>
            {kc.map((label, i) => (
              <Chip key={`${held}-${i}`} x={bx + 30 + i * (chipW + 6)} y={by + 47} w={chipW} label={label} color={DEV_COLORS[held]} on={kOn(i)} />
            ))}

            {/* out / lse rows of this GPU's queries: merged in place, never sent */}
            <text x={bx + 9} y={by + 81} fontFamily={mono} fontSize="9.5" fill={T.dim}>o,ℓ</text>
            {qc.map((label, i) => {
              const cx0 = bx + 30 + i * (chipW + 6);
              const got = merged.rows[i];
              const fresh = merged.fresh != null && got.includes(merged.fresh) && phase.type === "merge";
              const pw = Math.min(8, (chipW - (N - 1) * 1.5) / N);
              return (
                <g key={`o${i}`}>
                  <rect x={cx0} y={by + 69} width={chipW} height={17} rx={4} fill={got.length ? `${col}14` : "transparent"}
                    stroke={fresh ? T.accent2 : col} strokeWidth={fresh ? 2 : 1} strokeDasharray={got.length ? "none" : "3 3"} style={{ transition: "all .3s" }} />
                  <text x={cx0 + chipW / 2} y={by + 81.5} textAnchor="middle" fontFamily={mono} fontSize="11" fontWeight="600" fill={col} opacity={got.length ? 1 : 0.45}>{label}</text>
                  {Array.from({ length: N }, (_, j) => {
                    const on = got.includes(j);
                    return <rect key={j} x={cx0 + j * (pw + 1.5)} y={by + 89} width={pw} height={5} rx={1}
                      fill={on ? DEV_COLORS[j] : "transparent"} stroke={j === merged.fresh && fresh ? T.ink : DEV_COLORS[j]} strokeWidth={0.8} opacity={on ? 1 : 0.35}>
                      <title>{on ? `KV from GPU ${j} merged` : `KV from GPU ${j} not merged`}</title>
                    </rect>;
                  })}
                </g>
              );
            })}

            <text x={x} y={by + 108} textAnchor="middle" fontFamily={mono} fontSize="9.5" fill={st.c}>{st.t}</text>

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
        const lbl = shardLabel(variant, N, j, "KV");
        const w = Math.max(64, lbl.length * 6.6 + 14);
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

// One line under the ring: what the o,ℓ row of the followed GPU holds right now.
export function OutLegend({ variant, N, phase, sel }) {
  const { rows, fresh } = mergedRows(variant, N, sel, phase);
  const toks = tokensOf(variant, N, sel);
  const halves = variant === "zigzag" ? [toks.slice(0, 2), toks.slice(2)] : [toks];
  const list = (js) => (js.length ? `KV of GPU${js.length > 1 ? "s" : ""} ${js.join(", ")}` : "nothing yet");
  const parts = rows.map((js, i) => (variant === "zigzag" ? `rows t[${halves[i].join(",")}]: ${list(js)}` : list(js)));
  const all = rows.every((js) => js.length === rows[0].length);
  return (
    <div style={{ fontFamily: mono, fontSize: 11.5, color: T.dim, lineHeight: 1.6, marginTop: 4 }}>
      <span style={{ color: DEV_COLORS[sel], fontWeight: 700 }}>o,ℓ</span> = <span style={{ color: T.soft }}>out</span> and <span style={{ color: T.soft }}>lse</span> of GPU {sel}'s own query rows t[{toks.join(",")}]. They never travel: each step merges the new block's (block_out, block_lse) into them in place; a square fills when that GPU's KV has been merged.
      <div style={{ color: T.soft }}>
        GPU {sel} now: {parts.join(" · ")}
        {fresh != null && <span style={{ color: T.accent2 }}> · just merged GPU {fresh}</span>}
        {phase.type === "final" && <span style={{ color: T.accent }}> · returned as out [{toks.length}, H, D], lse [H, {toks.length}]{variant === "zigzag" && !all ? " (early rows saw fewer blocks: causal)" : ""}</span>}
      </div>
    </div>
  );
}
