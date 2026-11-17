import { T, mono } from "./theme";
import { useProgress, ease } from "./anim";
import { Label, Row } from "./draw";
import { SUBJECTS } from "./model";
import { Thumb } from "./ClipView";

// Chunked SigLIP loss, the way the ring-attention app passes KV around:
// D devices on a circle. Device r keeps its b images (and their rows of the
// loss matrix) forever; the text chunks travel r → r+1. At step s device r
// holds texts of device (r − s) mod D and fills block (r, (r − s) mod D).
// Step 0 = the diagonal blocks (positives + local negatives); every later
// step is negatives only. No device ever holds the whole n × n matrix.

export const DEVS = 4;
export const DEV_C = ["#6EF3DD", "#FF8AC6", "#FFC94D", "#8DB4FF"];
const mod = (a, m) => ((a % m) + m) % m;
export const heldBy = (r, s) => mod(r - s, DEVS); // text chunk device r holds at step s

function Ring({ n, s, final }) {
  const b = n / DEVS, W = 400, H = 380, cx = W / 2, cy = H / 2, R = 128, BW = 128, BH = 74;
  const sending = !final && s < DEVS - 1;
  const p = useProgress(`ring${s}${final}`, 1600), t = sending ? ease(Math.max(0, (p - 0.45) / 0.55)) : 0;
  const ang = (r) => -Math.PI / 2 + (2 * Math.PI * r) / DEVS;
  const pos = (r) => [cx + R * Math.cos(ang(r)), cy + R * Math.sin(ang(r))];
  const dl = 0.42;
  const arc = (r, u) => { const th = ang(r) + dl + u * (Math.PI / 2 - 2 * dl); return [cx + R * Math.cos(th), cy + R * Math.sin(th)]; };
  const emo = (dev) => SUBJECTS.slice(dev * b, dev * b + b).map((x) => x.emoji).join("");
  return (
    <svg width={W} height={H} style={{ flexShrink: 0 }}>
      <defs><marker id="sr-a" markerUnits="userSpaceOnUse" markerWidth="10" markerHeight="10" refX="8" refY="5" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill={sending ? T.wire : T.rule} /></marker></defs>
      {Array.from({ length: DEVS }, (_, r) => {
        const [x1, y1] = arc(r, 0), [x2, y2] = arc(r, 1);
        return <path key={r} d={`M${x1},${y1} A${R},${R} 0 0 1 ${x2},${y2}`} fill="none" stroke={sending ? T.wire : T.rule} strokeWidth={sending ? 2 : 1.2} strokeDasharray={sending ? "6 5" : "none"} markerEnd="url(#sr-a)" />;
      })}
      <text x={cx} y={cy - 6} textAnchor="middle" fontFamily={mono} fontSize={14} fill={T.dim}>{final ? "ALL-REDUCE" : `STEP ${s} / ${DEVS - 1}`}</text>
      <text x={cx} y={cy + 16} textAnchor="middle" fontFamily={mono} fontSize={15} fontWeight={700} fill={final ? T.accent : sending && p > 0.45 ? T.wire : T.accent}>
        {final ? "sum the losses" : sending && p > 0.45 ? "send texts →" : "compute block"}
      </text>
      {Array.from({ length: DEVS }, (_, r) => {
        const [x, y] = pos(r), c = DEV_C[r], h = heldBy(r, s);
        return (
          <g key={r}>
            <rect x={x - BW / 2} y={y - BH / 2} width={BW} height={BH} rx={9} fill={`${c}14`} stroke={c} strokeWidth={1.8} />
            <text x={x - BW / 2 + 9} y={y - BH / 2 + 18} fontFamily={mono} fontSize={14} fontWeight={700} fill={c}>GPU {r}</text>
            <text x={x - BW / 2 + 9} y={y - BH / 2 + 41} fontFamily={mono} fontSize={14} fill={T.soft}>img {emo(r)}</text>
            {!final && <>
              <rect x={x - BW / 2 + 8} y={y - BH / 2 + 50} width={BW - 16} height={19} rx={5} fill={`${DEV_C[h]}26`} stroke={DEV_C[h]} opacity={sending && p > 0.45 ? 0.3 : 1} />
              <text x={x - BW / 2 + 14} y={y - BH / 2 + 64} fontFamily={mono} fontSize={14} fill={DEV_C[h]} opacity={sending && p > 0.45 ? 0.3 : 1}>txt {emo(h)}</text>
            </>}
          </g>
        );
      })}
      {sending && p > 0.45 && Array.from({ length: DEVS }, (_, r) => {
        const [x, y] = arc(r, t), h = heldBy(r, s);
        return (
          <g key={`k${r}`} transform={`translate(${x},${y})`}>
            <rect x={-32} y={-12} width={64} height={24} rx={12} fill={T.bg} stroke={DEV_C[h]} strokeWidth={1.8} />
            <text x={0} y={5} textAnchor="middle" fontFamily={mono} fontSize={14} fill={DEV_C[h]}>{emo(h)}</text>
          </g>
        );
      })}
    </svg>
  );
}

// the n × n cell-loss matrix: block (r, c) is computed by device r at the step
// when it holds text chunk c. Done blocks show ℓ, the current ones are outlined.
function BlockMatrix({ n, toy, s, final }) {
  const b = n / DEVS, cell = n > 4 ? 44 : 60;
  const stepOf = (i, j) => mod(Math.floor(i / b) - Math.floor(j / b), DEVS); // step at which cell (i,j) is computed
  const idx = Array.from({ length: n }, (_, k) => k);
  const done = (i, j) => final || stepOf(i, j) <= s;
  const rowSoFar = idx.map((i) => idx.reduce((a, j) => a + (done(i, j) ? toy.ell[i][j] : 0), 0));
  return (
    <div style={{ display: "grid", gridTemplateColumns: `40px repeat(${n}, ${cell}px) 62px`, gap: 2, alignItems: "center" }}>
      <span />
      {idx.map((j) => <span key={j} style={{ textAlign: "center", fontSize: 17 }}>{SUBJECTS[j].emoji}</span>)}
      <span style={{ fontFamily: mono, fontSize: 13, color: T.wire, textAlign: "center" }}>Σ row</span>
      {idx.map((i) => [
        <div key={`t${i}`} style={{ justifySelf: "center" }}><Thumb i={i} size={28} ring={false} /></div>,
        ...idx.map((j) => {
          const st = stepOf(i, j), now = !final && st === s, d = done(i, j), dev = Math.floor(i / b);
          return (
            <div key={j} style={{
              height: cell - 12, borderRadius: 3, display: "grid", placeItems: "center", fontFamily: mono, fontSize: 13, fontWeight: i === j ? 700 : 400,
              background: d ? `${DEV_C[dev]}${now ? "55" : "22"}` : "transparent", color: d ? T.ink : T.rule,
              border: `1.5px ${d ? "solid" : "dashed"} ${now ? DEV_C[dev] : d ? `${DEV_C[dev]}55` : T.rule}`, transition: "all .4s",
            }}>{d ? toy.ell[i][j].toFixed(2) : ""}</div>
          );
        }),
        <span key={`s${i}`} style={{ fontFamily: mono, fontSize: 14, color: DEV_C[Math.floor(i / b)], textAlign: "right" }}>{rowSoFar[i].toFixed(2)}</span>,
      ])}
    </div>
  );
}

export function RingScene({ n, toy, s, final }) {
  const b = n / DEVS;
  const devSum = Array.from({ length: DEVS }, (_, r) => toy.rowSum.slice(r * b, r * b + b).reduce((a, c) => a + c, 0));
  return (
    <div style={{ display: "grid", gap: 12, padding: 6 }}>
      <Row gap={22} wrap={false} style={{ alignItems: "flex-start", overflowX: "auto" }}>
        <Ring n={n} s={s} final={final} />
        <div style={{ display: "grid", gap: 10 }}>
          <Label>ℓ matrix · row = image (stays) · column = caption (travels)</Label>
          <BlockMatrix n={n} toy={toy} s={s} final={final} />
        </div>
      </Row>
      {final ? (
        <div style={{ fontFamily: mono, fontSize: 17, color: T.ink, lineHeight: 1.7 }}>
          {devSum.map((v, r) => <span key={r}><span style={{ color: DEV_C[r] }}>{v.toFixed(2)}</span>{r < DEVS - 1 ? " + " : ""}</span>)} = {toy.total.toFixed(2)}
          <span style={{ color: T.dim }}> ÷ {n} = </span><b style={{ color: T.accent }}>{toy.loss.toFixed(3)}</b>
          <span style={{ color: T.soft, fontSize: 15 }}>  same loss as before</span>
        </div>
      ) : (
        <div style={{ fontFamily: mono, fontSize: 16, color: T.soft, lineHeight: 1.6 }}>
          {s === 0 ? <>own texts first: diagonal blocks hold the <b style={{ color: T.accent }}>positives</b></> : <>texts from {s} hop{s > 1 ? "s" : ""} away: <b style={{ color: T.bad }}>negatives only</b></>}
          <span style={{ color: T.dim }}> · each GPU holds one [{b}, {b}] block, never [{n}, {n}]</span>
        </div>
      )}
    </div>
  );
}
