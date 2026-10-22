import { T, mono } from "./theme";
import { DEV_COLORS } from "./ringModel";
import { TREE_PHASE, PER, log2, fmt } from "./treeModel";

// Binomial reduction tree. Leaves = GPUs (bottom), node (lvl, k) = result
// over GPUs [k 2^lvl, (k+1) 2^lvl), held by GPU k 2^lvl. Its left child has
// the same owner (no message, dashed); the right child's owner sends.

const W = 700;
const LVL_H = 74;
const LEAF_H = 92;
const NODE_W = 92;
const NODE_H = 30;

export default function TreeDiagram({ N, phase, state, data, sel, onSel }) {
  const L = log2(N);
  const top = 26;
  const leafY = top + L * LVL_H + 18;
  const H = leafY + LEAF_H + 14;
  const colW = W / N;
  const leafW = Math.min(132, colW - 8);
  const cx = (a, b) => ((a + b) / 2) * colW; // center of GPU range [a, b)
  const nodeY = (lvl) => top + (L - lvl) * LVL_H;
  const isComm = phase.type === "up" || phase.type === "down";
  const op = isComm ? phase.op : phase.type === "rescale" || phase.type === "final" ? "sum" : "max";
  const showNodes = phase.type !== "init" && phase.type !== "local";
  const pc = TREE_PHASE[phase.type].color;

  // top/bottom anchor of node (lvl, k); lvl 0 is a leaf box
  const anchor = (lvl, k, end) => {
    const x = cx(k * 2 ** lvl, (k + 1) * 2 ** lvl);
    if (lvl === 0) return [x, leafY];
    return [x, end === "top" ? nodeY(lvl) : nodeY(lvl) + NODE_H];
  };

  const edges = [];
  for (let lvl = 1; lvl <= L; lvl++) {
    for (let k = 0; k < N / 2 ** lvl; k++) {
      for (const side of [0, 1]) {
        const ck = 2 * k + side;
        const [x1, y1] = anchor(lvl - 1, ck, "top");
        const [x2, y2] = anchor(lvl, k, "bottom");
        const live = isComm && phase.lvl === lvl;
        const msg = live && side === 1;
        edges.push({ key: `${lvl}-${k}-${side}`, x1, y1, x2, y2, live, msg, side, from: ck * 2 ** (lvl - 1), to: k * 2 ** lvl });
      }
    }
  }

  const nodeVal = (lvl, k) => state.nodes[`${op}:${lvl}:${k}`];

  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", display: "block" }}>
      <defs>
        {[["up", T.wire], ["down", "#5FD3F3"]].map(([id, c]) => (
          <marker key={id} id={`tree-${id}`} markerUnits="userSpaceOnUse" markerWidth="10" markerHeight="10" refX="8" refY="5" orient="auto">
            <path d="M0,0 L10,5 L0,10 z" fill={c} />
          </marker>
        ))}
      </defs>

      {/* level labels */}
      {Array.from({ length: L }, (_, i) => {
        const lvl = i + 1;
        const on = isComm && phase.lvl === lvl;
        return (
          <text key={lvl} x={4} y={nodeY(lvl) - 6} fontFamily={mono} fontSize="9.5" fill={on ? pc : T.dim}>
            {`level ${lvl} · step ${2 ** (lvl - 1)}`}
          </text>
        );
      })}

      {/* direction of the current collective */}
      {isComm && (
        <text x={W - 4} y={14} textAnchor="end" fontFamily={mono} fontSize="10.5" fontWeight="700" fill={pc}>
          {phase.type === "up" ? "reduce: data flows ↑ to GPU 0" : "broadcast: result flows ↓ to everyone"}
        </text>
      )}

      {/* edges */}
      {edges.map((e) => {
        const up = phase.type === "up";
        const color = e.msg ? pc : T.dim;
        const [sx, sy, tx, ty] = up || !e.msg ? [e.x1, e.y1, e.x2, e.y2] : [e.x2, e.y2, e.x1, e.y1];
        return (
          <line
            key={e.key} x1={sx} y1={sy} x2={tx} y2={ty}
            stroke={color} strokeWidth={e.msg ? 2.4 : 1.2}
            strokeDasharray={e.side === 0 ? "4 4" : "none"}
            opacity={e.live ? 1 : 0.45}
            markerEnd={e.msg ? `url(#tree-${phase.type})` : undefined}
            style={{ transition: "all .3s" }}
          />
        );
      })}

      {/* message pills on the active level */}
      {isComm && state.msgs.map((m) => {
        const e = edges.find((x) => x.msg && (phase.type === "up" ? x.from === m.from && x.to === m.to : x.from === m.to && x.to === m.from));
        if (!e) return null;
        const px = (e.x1 + e.x2) / 2, py = (e.y1 + e.y2) / 2;
        const hot = m.from === sel || m.to === sel;
        const label = `${m.from}→${m.to}  ${op === "max" ? "m" : "d"}=${fmt(m.val)}`;
        const w = label.length * 6.4 + 12;
        return (
          <g key={`${m.from}-${m.to}`}>
            <rect x={px - w / 2} y={py - 10} width={w} height={20} rx={10} fill={T.bg} stroke={pc} strokeWidth={hot ? 2 : 1} />
            <text x={px} y={py + 4} textAnchor="middle" fontFamily={mono} fontSize="10.5" fontWeight="700" fill={pc}>{label}</text>
          </g>
        );
      })}

      {/* internal nodes */}
      {Array.from({ length: L }, (_, i) => i + 1).map((lvl) =>
        Array.from({ length: N / 2 ** lvl }, (_, k) => {
          const v = nodeVal(lvl, k);
          const owner = k * 2 ** lvl;
          const [x] = anchor(lvl, k, "top");
          const y = nodeY(lvl);
          const done = showNodes && v != null;
          const active = isComm && phase.type === "up" && phase.lvl === lvl;
          const root = lvl === L;
          const sym = op === "max" ? "max" : "Σd";
          return (
            <g key={`${lvl}-${k}`}>
              <rect
                x={x - NODE_W / 2} y={y} width={NODE_W} height={NODE_H} rx={7}
                fill={active ? `${pc}22` : T.well} stroke={active ? pc : done ? (root ? T.accent : T.soft) : T.rule}
                strokeWidth={active || root ? 1.8 : 1} strokeDasharray={done ? "none" : "3 4"} opacity={done ? 1 : 0.5}
                style={{ transition: "all .3s" }}
              />
              <text x={x} y={y + 12} textAnchor="middle" fontFamily={mono} fontSize="9" fill={T.dim}>
                {`GPU ${owner} · [${owner}..${owner + 2 ** lvl - 1}]`}
              </text>
              <text x={x} y={y + 25} textAnchor="middle" fontFamily={mono} fontSize="11" fontWeight="700" fill={done ? (root ? T.accent : T.ink) : T.rule}>
                {done ? `${sym} ${fmt(v)}` : "·"}
              </text>
            </g>
          );
        })
      )}

      {/* leaves = GPUs: tokens, the allreduce buffer, and whose data it already includes */}
      {Array.from({ length: N }, (_, i) => {
        const x = cx(i, i + 1) - leafW / 2;
        const g = data.gpus[i], s = state.gpus[i];
        const col = DEV_COLORS[i];
        const isSel = i === sel;
        const involved = isComm && state.msgs.some((m) => m.from === i || m.to === i);
        const small = N > 4;
        const fs = small ? 9 : 10.5;
        const toks = Array.from({ length: PER }, (_, t) => PER * i + t);
        const inSum = phase.type === "rescale" || phase.op === "sum" || phase.type === "final";
        const bufName = inSum ? "den" : "lse_max";
        const bufVal = inSum ? s.den : s.mx;
        const all = s.cov === (1 << N) - 1;
        const rows = [];
        rows.push([small ? `t${toks[0]}–${toks[PER - 1]}` : `tokens ${toks.join(",")}`, T.dim]);
        if (s.local) rows.push([`lse ${fmt(g.lse)}`, T.accent]);
        if (phase.type === "final") rows.push(["out = n/d ✓", T.accent]);
        else if (s.local) rows.push([s.changed ? `${fmt(s.prevVal)}→${fmt(bufVal)}` : `${small ? (inSum ? "d" : "m") : bufName} ${fmt(bufVal)}`, s.changed ? pc : inSum ? T.accent2 : T.wire]);
        const sq = Math.min(9, (leafW - 16) / N - 1);
        return (
          <g key={i} onClick={() => onSel(i)} style={{ cursor: "pointer" }}>
            <rect
              x={x} y={leafY} width={leafW} height={LEAF_H} rx={8}
              fill={isSel ? `${col}1c` : T.well} stroke={isSel ? col : involved ? pc : T.rule} strokeWidth={isSel ? 2.4 : involved ? 1.6 : 1.2}
              style={{ transition: "all .3s" }}
            />
            <text x={x + 8} y={leafY + 16} fontFamily={mono} fontSize="11" fontWeight="700" fill={col}>GPU {i}</text>
            {all && s.local && phase.type !== "local" && <text x={x + leafW - 7} y={leafY + 16} textAnchor="end" fontFamily={mono} fontSize="11" fontWeight="700" fill={T.accent}>✓</text>}
            {rows.map(([t, c], r) => (
              <text key={r} x={x + 8} y={leafY + 33 + r * 15} fontFamily={mono} fontSize={fs} fill={c}>{t}</text>
            ))}
            {s.local && phase.type !== "final" && (
              <g>
                {Array.from({ length: N }, (_, k) => (
                  <rect key={k} x={x + 8 + k * (sq + 1)} y={leafY + LEAF_H - 15} width={sq} height={8} rx={1.5}
                    fill={s.cov & (1 << k) ? DEV_COLORS[k] : "transparent"} stroke={s.cov & (1 << k) ? "none" : T.rule} strokeWidth={0.8} />
                ))}
              </g>
            )}
          </g>
        );
      })}
    </svg>
  );
}
