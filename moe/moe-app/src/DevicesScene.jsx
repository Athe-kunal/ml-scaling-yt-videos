import { T, mono } from "./theme";
import { CFG, EXPERT_COLORS as EC, MODEL } from "./moeModel";

// The MoE layer across D = G = E devices. Device d holds group d's tokens
// (top) and expert d's input buffer [G, C, M] (bottom: one row per source
// group). Tickets ride the all-to-all: group -> expert (dispatch), expert
// computes, then back (combine). Empty buffer slots are wasted compute.
//
// stage: 0 split · 1 dispatch · 2 expert FFN · 3 combine · 4 output

const { G, S, E, C } = CFG;
const W = 860, H = 430;
const DW = 190, DX = (d) => 20 + d * 210, BY = 16, DH = 376;
const chipXY = (g, s) => [DX(g) + 17 + s * 31, BY + 64];
const rowCY = (g) => BY + 160 + g * 52;
const slotXY = (e, g, k) => [DX(e) + 40 + k * 46 + 20, rowCY(g)];

export default function DevicesScene({ stage, sel }) {
  const tickets = [];
  MODEL.groups.forEach((gm, g) =>
    gm.toks.forEach((k) => k.final.forEach((f) => tickets.push({ g, s: k.s, e: f.e, slot: f.slot, second: f.second }))),
  );
  const outStage = stage >= 4;

  // all-to-all lanes for the followed group
  const lanes = Array.from({ length: E }, (_, e) => tickets.filter((t) => t.g === sel && t.e === e).length);

  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", display: "block" }}>
      <defs>
        {MODEL.groups.flatMap((gm, g) =>
          gm.toks.filter((k) => k.final.length).map((k) => {
            const [a, b] = k.final;
            const w = a.w;
            return (
              <linearGradient key={`${g}-${k.s}`} id={`out-${g}-${k.s}`} x1="0" x2="1" y1="0" y2="0">
                <stop offset="0" stopColor={EC[a.e]} />
                <stop offset={b ? w : 1} stopColor={EC[a.e]} />
                <stop offset={b ? w : 1} stopColor={EC[(b || a).e]} />
                <stop offset="1" stopColor={EC[(b || a).e]} />
              </linearGradient>
            );
          }),
        )}
      </defs>

      {/* devices */}
      {Array.from({ length: G }, (_, d) => {
        const busy = tickets.filter((t) => t.e === d).length;
        const live = stage === 1 || stage === 2;
        return (
          <g key={d}>
            <rect
              x={DX(d)} y={BY} width={DW} height={DH} rx={10} fill={T.panel}
              stroke={d === sel ? T.soft : T.rule} strokeWidth={d === sel ? 1.8 : 1.2}
            />
            <text x={DX(d) + 12} y={BY + 22} fontFamily={mono} fontSize="12.5" fontWeight="700" fill={T.ink}>Device {d}</text>
            <text x={DX(d) + DW - 12} y={BY + 22} textAnchor="end" fontFamily={mono} fontSize="10" fill={d === sel ? T.soft : T.dim}>group {d}</text>
            <text x={DX(d) + 12} y={BY + 40} fontFamily={mono} fontSize="9.5" fill={T.dim}>x [S, M]</text>

            <line x1={DX(d) + 10} x2={DX(d) + DW - 10} y1={BY + 98} y2={BY + 98} stroke={T.rule} />
            <text x={DX(d) + 12} y={BY + 116} fontFamily={mono} fontSize="11" fontWeight="700" fill={EC[d]}>expert E{d}</text>
            <text x={DX(d) + DW - 12} y={BY + 116} textAnchor="end" fontFamily={mono} fontSize="9.5" fill={stage === 2 ? EC[d] : T.dim}>
              {stage === 2 ? "relu(x·wi)·wo" : "buffer [G, C, M]"}
            </text>

            {/* buffer rows, one per source group */}
            {Array.from({ length: G }, (_, g) => (
              <g key={g} opacity={g === sel ? 1 : 0.55}>
                <text x={DX(d) + 12} y={rowCY(g) + 4} fontFamily={mono} fontSize="10" fill={T.dim}>g{g}</text>
                {Array.from({ length: C }, (_, k) => {
                  const held = tickets.some((t) => t.g === g && t.e === d && t.slot === k);
                  const fill = live && held;
                  return (
                    <rect
                      key={k} x={DX(d) + 40 + k * 46} y={rowCY(g) - 15} width={40} height={30} rx={5}
                      fill={fill ? `${EC[d]}55` : "transparent"} stroke={EC[d]} strokeOpacity={fill ? 1 : 0.3}
                      strokeDasharray={fill ? "none" : "3 3"} style={{ transition: "all .4s" }}
                    />
                  );
                })}
              </g>
            ))}

            {(stage === 1 || stage === 2) && (
              <text x={DX(d) + DW / 2} y={BY + DH - 8} textAnchor="middle" fontFamily={mono} fontSize="10.5" fontWeight="600" fill={busy >= G * C ? T.bad : T.soft}>
                {busy} / {G * C} slots busy
              </text>
            )}
          </g>
        );
      })}

      {/* all-to-all lanes for the followed group */}
      {(stage === 1 || stage === 3) &&
        lanes.map((n, e) => {
          if (!n) return null;
          const [x0, y0] = [DX(sel) + DW / 2, BY + 80];
          const x1 = DX(e) + DW / 2, y1 = rowCY(sel) - 18;
          const my = e === sel ? y0 + 30 : (y0 + y1) / 2 - 36;
          return (
            <path
              key={e} d={`M${x0},${y0} Q${(x0 + x1) / 2},${my} ${x1},${y1}`} fill="none" stroke={T.wire}
              strokeWidth={1 + n * 0.9} opacity={0.55} className={stage === 3 ? "lane back" : "lane"}
            />
          );
        })}

      {/* token chips */}
      {MODEL.groups.map((gm, g) =>
        gm.toks.map((k) => {
          const [x, y] = chipXY(g, k.s);
          const gone = k.final.length === 0;
          return (
            <g key={`${g}-${k.s}`} opacity={g === sel ? 1 : 0.7}>
              <circle
                cx={x} cy={y} r={11}
                fill={outStage ? (gone ? T.well : `url(#out-${g}-${k.s})`) : T.well}
                stroke={outStage && gone ? T.bad : g === sel ? T.soft : T.rule}
                strokeWidth={1.6} strokeDasharray={outStage && gone ? "3 2.5" : "none"} style={{ transition: "fill .4s" }}
              />
              <text x={x} y={y + 3.5} textAnchor="middle" fontFamily={mono} fontSize="9.5" fontWeight="700" fill={outStage && !gone ? "#0A0E14" : T.soft}>{k.s}</text>
            </g>
          );
        }),
      )}
      {outStage && (
        <g fontFamily={mono} fontSize="9.5" fill={T.dim}>
          {Array.from({ length: G }, (_, d) => (
            <text key={d} x={DX(d) + 12} y={BY + 84}>y [S, M]</text>
          ))}
        </g>
      )}

      {/* tickets in flight / parked */}
      {tickets.map((t, i) => {
        const [sx, sy] = chipXY(t.g, t.s);
        const [dx, dy] = slotXY(t.e, t.g, t.slot);
        const atDest = stage === 1 || stage === 2;
        const [x, y] = atDest ? [dx, dy] : [sx, sy];
        const visible = stage >= 1 && stage <= 3;
        const focus = t.g === sel;
        return (
          <g
            key={i}
            style={{
              transform: `translate(${x}px, ${y}px)`,
              opacity: visible ? (focus ? 1 : 0.3) : 0,
              transition: `transform .85s cubic-bezier(.3,.7,.3,1) ${(t.s * 40 + t.g * 25) % 300}ms, opacity .35s`,
            }}
          >
            <circle r={focus ? 8.5 : 6.5} fill={t.second ? T.bg : EC[t.e]} stroke={EC[t.e]} strokeWidth={t.second ? 2 : 0} />
            {focus && <text textAnchor="middle" y={3.5} fontFamily={mono} fontSize="9.5" fontWeight="700" fill={t.second ? EC[t.e] : "#0A0E14"}>{t.s}</text>}
          </g>
        );
      })}
    </svg>
  );
}
