import { T, mono, sans } from "./theme";
import { AXIS_COLOR, codeHtml, parseDims } from "./notation";

// Draws each tensor as a matrix over its LAST TWO dims (rows, cols). Any
// leading dims become a stack of offset cards plus "extra dim" chips. A dim
// sharded over Y or Z splits that side of the matrix into 2 bands (4 for YZ):
// the first band is this chip's shard (solid), the rest live on other chips
// (hatched), and the dividers are tinted with the shard axis colour.

const BOX_W = 118;
const BOX_H = 80;
const LABEL_H = 22;

const splits = (shard) => (shard === "YZ" ? 4 : shard ? 2 : 1);

function DimLabel({ d, style }) {
  return (
    <span style={{ fontFamily: mono, fontSize: 13, color: d.shard ? AXIS_COLOR[d.shard] : T.soft, fontWeight: d.shard ? 700 : 400, ...style }}>
      {d.name}
      {d.shard && <sub style={{ fontSize: 9.5 }}>{d.shard}</sub>}
    </span>
  );
}

function Matrix({ t }) {
  const { dims, unreduced } = parseDims(t.d);
  const [rowD, colD] = dims.slice(-2);
  const extra = dims.slice(0, -2);
  const nr = splits(rowD.shard);
  const nc = splits(colD.shard);
  const hatch = `repeating-linear-gradient(45deg, ${T.rule}55 0 1px, transparent 1px 6px)`;

  const ghost = (dx, alpha, dashed) => (
    <div style={{ position: "absolute", left: dx, top: -dx, width: BOX_W, height: BOX_H, borderRadius: 8, border: `1.2px ${dashed ? "dashed" : "solid"} ${T.dim}`, background: T.well, opacity: alpha }} />
  );

  return (
    <div style={{ display: "flex", flexDirection: "column", alignItems: "center", paddingTop: extra.length ? 12 : 0, paddingRight: extra.length || unreduced ? 12 : 0 }}>
      <div style={{ height: LABEL_H, display: "flex", alignItems: "center" }}><DimLabel d={colD} /></div>
      <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
        <div style={{ width: 34, textAlign: "right" }}><DimLabel d={rowD} /></div>
        <div style={{ position: "relative", width: BOX_W, height: BOX_H }}>
          {extra.length > 0 && (<>{ghost(12, 0.45)}{ghost(6, 0.75)}</>)}
          {unreduced && ghost(8, 0.6, true)}
          <div
            style={{
              position: "absolute", inset: 0, borderRadius: 8, overflow: "hidden", background: T.panel,
              border: `1.5px solid ${T.soft}`,
              display: "grid", gridTemplateColumns: `repeat(${nc}, 1fr)`, gridTemplateRows: `repeat(${nr}, 1fr)`,
            }}
          >
            {Array.from({ length: nr * nc }, (_, i) => {
              const r = Math.floor(i / nc);
              const c = i % nc;
              const mine = r === 0 && c === 0;
              return (
                <div
                  key={i}
                  style={{
                    background: mine ? "rgba(245,248,251,.2)" : hatch,
                    borderRight: c < nc - 1 ? `2px solid ${AXIS_COLOR[colD.shard] || T.rule}` : "none",
                    borderBottom: r < nr - 1 ? `2px solid ${AXIS_COLOR[rowD.shard] || T.rule}` : "none",
                  }}
                />
              );
            })}
          </div>
          {unreduced && (
            <div style={{ position: "absolute", right: 6, top: 6, width: 20, height: 20, borderRadius: 10, background: T.well, border: `1.3px solid ${T.bad}`, color: T.bad, fontFamily: mono, fontSize: 12, fontWeight: 700, display: "grid", placeItems: "center" }}>Σ</div>
          )}
        </div>
      </div>
      <div style={{ display: "flex", gap: 6, alignItems: "center", marginTop: 8, minHeight: 20 }}>
        {extra.length > 0 && <span style={{ fontFamily: mono, fontSize: 10, color: T.dim }}>+</span>}
        {extra.map((d, i) => (
          <span key={i} style={{ padding: "1px 7px", borderRadius: 5, border: `1px solid ${d.shard ? AXIS_COLOR[d.shard] : T.rule}`, background: T.well }}>
            <DimLabel d={d} style={{ fontSize: 12 }} />
          </span>
        ))}
      </div>
      <div style={{ fontFamily: mono, fontSize: 12.5, color: T.ink, fontWeight: 600, marginTop: 2 }} dangerouslySetInnerHTML={{ __html: codeHtml(t.t) }} />
    </div>
  );
}

export default function MatrixShapes({ row, accent }) {
  return (
    <div>
      <div style={{ display: "flex", alignItems: "flex-start", gap: 14, flexWrap: "wrap" }}>
        {row.map((r, i) =>
          r.op ? (
            <span key={i} style={{ fontFamily: sans, fontSize: 24, fontWeight: 600, color: accent, paddingTop: LABEL_H + BOX_H / 2 - 4 }}>{r.op}</span>
          ) : (
            <Matrix key={i} t={r} />
          )
        )}
      </div>
      <div style={{ fontFamily: sans, fontSize: 11, color: T.dim, marginTop: 12, lineHeight: 1.6 }}>
        Matrix = last two dims. Extra leading dims show as stacked cards plus chips. Solid band = this chip's shard, hatched = other chips'.
        Divider colour = shard axis (<span style={{ color: T.accent }}>Y</span> · <span style={{ color: T.accent2 }}>Z</span> · <span style={{ color: T.wire }}>YZ</span>).
        Dashed card with <span style={{ fontFamily: mono, color: T.bad }}>Σ</span> = unreduced partial sum.
      </div>
    </div>
  );
}
