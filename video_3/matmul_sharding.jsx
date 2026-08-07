import React, { useState, useEffect, useCallback, useMemo } from "react";

/* ------------------------------------------------------------------ tokens */
const T = {
  bg: "#080B10",
  panel: "#10161F",
  well: "#0C1119",
  ink: "#E6EDF3",
  soft: "#7D8B9A",
  dim: "#4A5766",
  rule: "#1E2732",
  axX: "#5EEAD4",
  axY: "#F472B6",
  wire: "#FBBF24",
  bad: "#F87171",
};

const mono = "'IBM Plex Mono', ui-monospace, SFMono-Regular, monospace";
const sans = "'IBM Plex Sans', ui-sans-serif, system-ui, sans-serif";
const cond = "'IBM Plex Sans Condensed', 'IBM Plex Sans', sans-serif";

/* shard identity: same index -> same hue everywhere, so you can track a chunk */
const shardHue = (k, n) => (((k / Math.max(n, 1)) * 0.82 + 0.03) % 1) * 360;
const shardColor = (k, n) => `hsl(${shardHue(k, n)} 58% 60%)`;

/* ------------------------------------------------------------------ blocks */
function Block({ uid, x, y, w, h, spec, label }) {
  const s = spec.state;
  const parts = [];

  if (s === "empty") {
    parts.push(
      <rect key="e" x={x} y={y} width={w} height={h} fill="none" stroke={T.rule}
        strokeWidth="1.2" strokeDasharray="4 4" rx="2" />
    );
  }

  if (s === "replicated") {
    parts.push(
      <rect key="r" x={x} y={y} width={w} height={h} fill="#2A3542" opacity="0.7"
        stroke={T.rule} strokeWidth="1" rx="2" />,
      <text key="rt" x={x + w / 2} y={y + h / 2} textAnchor="middle" fontFamily={mono}
        fontSize="8.5" fill={T.soft}>
        <tspan x={x + w / 2} dy="-4">{spec.shape}</tspan>
        <tspan x={x + w / 2} dy="12">replicated</tspan>
      </text>
    );
  }

  if (s === "reduced") {
    parts.push(
      <rect key="f" x={x} y={y} width={w} height={h} fill={T.axX} opacity="0.9" rx="2" />,
      <text key="ft" x={x + w / 2} y={y + h / 2 + 3} textAnchor="middle" fontFamily={mono}
        fontSize="9.5" fontWeight="600" fill={T.bg}>{spec.shape}</text>
    );
  }

  if (s === "partial") {
    parts.push(
      <rect key="p" x={x} y={y} width={w} height={h} fill={`url(#hatch-${uid})`}
        stroke={T.wire} strokeWidth="1.6" rx="2" />,
      <text key="pt" x={x + w / 2} y={y + h / 2} textAnchor="middle" fontFamily={mono}
        fontSize="8" fontWeight="600" fill={T.wire}>
        <tspan x={x + w / 2} dy="-4">{spec.shape}</tspan>
        <tspan x={x + w / 2} dy="12">unreduced</tspan>
      </text>
    );
  }

  if (s === "mesh2d") {
    const { nRow, nCol, rowIdx, colIdx } = spec;
    const cw = w / nCol, ch = h / nRow;
    for (let r = 0; r < nRow; r++) {
      for (let c = 0; c < nCol; c++) {
        const cx = x + c * cw, cy = y + r * ch;
        const own = r === rowIdx && c === colIdx;
        parts.push(
          <rect key={`c${r}-${c}`} x={cx} y={cy} width={cw} height={ch}
            fill={own ? T.axX : T.rule} opacity={own ? 0.9 : 0.16}
            stroke={own ? "none" : T.rule} strokeWidth="0.5" strokeDasharray="3 3" />
        );
        if (own) {
          parts.push(
            <line key={`hx${r}`} x1={cx} y1={cy} x2={cx + cw} y2={cy} stroke={T.axX} strokeWidth="2.4" />,
            <line key={`hx2${r}`} x1={cx} y1={cy + ch} x2={cx + cw} y2={cy + ch} stroke={T.axX} strokeWidth="2.4" />,
            <line key={`vy${c}`} x1={cx} y1={cy} x2={cx} y2={cy + ch} stroke={T.axY} strokeWidth="2.4" />,
            <line key={`vy2${c}`} x1={cx + cw} y1={cy} x2={cx + cw} y2={cy + ch} stroke={T.axY} strokeWidth="2.4" />
          );
        }
      }
    }
  }

  if (s === "held" || s === "gathered" || s === "shard") {
    const { axis, n, idx, edge } = spec;
    for (let k = 0; k < n; k++) {
      const cw = axis === "col" ? w / n : w;
      const ch = axis === "col" ? h : h / n;
      const cx = axis === "col" ? x + k * cw : x;
      const cy = axis === "col" ? y : y + k * ch;
      const own = k === idx;
      const fill = shardColor(k, n);

      if (s === "held") {
        parts.push(
          <rect key={`k${k}`} x={cx} y={cy} width={cw} height={ch}
            fill={fill} opacity={own ? 1 : 0.1}
            stroke={own ? edge : T.rule} strokeWidth={own ? 2.2 : 0.7}
            strokeDasharray={own ? "none" : "3 3"} />
        );
      } else if (s === "gathered") {
        parts.push(
          <rect key={`k${k}`} x={cx} y={cy} width={cw} height={ch}
            fill={fill} opacity="0.92"
            stroke={own ? edge : T.bg} strokeWidth={own ? 2.4 : 0.8} />
        );
      } else if (s === "shard" && own) {
        parts.push(
          <rect key={`k${k}`} x={cx} y={cy} width={cw} height={ch}
            fill={fill} stroke={edge} strokeWidth="2.2" />
        );
      }
    }
  }

  parts.push(
    <rect key="o" x={x} y={y} width={w} height={h} fill="none" stroke={T.rule} strokeWidth="1" rx="2" />
  );
  parts.push(
    <text key="l" x={x + w / 2} y={y - 7} textAnchor="middle" fontFamily={mono}
      fontSize="9" fontWeight="600" fill={T.soft} letterSpacing="0.08em">{label}</text>
  );

  return <g className="blk">{parts}</g>;
}

/* ------------------------------------------------------------ device panel */
function DevicePanel({ uid, specs, label }) {
  const BS = 96, GAP = 32;
  const xA = 0, xB = BS + GAP, xC = 2 * (BS + GAP);
  const W = 3 * BS + 2 * GAP;

  return (
    <div style={{
      background: T.well, border: `1px solid ${T.rule}`, borderRadius: 8,
      padding: "16px 12px 10px",
    }}>
      <svg viewBox={`-3 -22 ${W + 6} ${BS + 30}`} style={{ width: "100%", display: "block" }}>
        <defs>
          <pattern id={`hatch-${uid}`} width="7" height="7" patternUnits="userSpaceOnUse"
            patternTransform="rotate(45)">
            <rect width="7" height="7" fill={T.wire} opacity="0.14" />
            <line x1="0" y1="0" x2="0" y2="7" stroke={T.wire} strokeWidth="1.6" opacity="0.55" />
          </pattern>
        </defs>
        <Block uid={uid} x={xA} y={0} w={BS} h={BS} spec={specs.A} label="A" />
        <text x={xA + BS + GAP / 2} y={BS / 2 + 5} textAnchor="middle" fontSize="15" fill={T.dim}>×</text>
        <Block uid={uid} x={xB} y={0} w={BS} h={BS} spec={specs.B} label="B" />
        <text x={xB + BS + GAP / 2} y={BS / 2 + 5} textAnchor="middle" fontSize="15" fill={T.dim}>=</text>
        <Block uid={uid} x={xC} y={0} w={BS} h={BS} spec={specs.C} label="C" />
      </svg>
      <div style={{
        fontFamily: mono, fontSize: 10, color: T.dim, textAlign: "center",
        marginTop: 6, letterSpacing: "0.06em",
      }}>{label}</div>
    </div>
  );
}

/* --------------------------------------------------- the ring (signature) */
function LinkRing({ n, kind }) {
  const R = 46, cx = 62, cy = 62;
  const pts = Array.from({ length: n }, (_, k) => {
    const a = (k / n) * 2 * Math.PI - Math.PI / 2;
    return { x: cx + R * Math.cos(a), y: cy + R * Math.sin(a), k };
  });
  const inbound = kind === "ReduceScatter" || kind === "AllReduce";

  return (
    <svg viewBox="0 0 124 124" style={{ width: 124, height: 124, flexShrink: 0 }}>
      <circle cx={cx} cy={cy} r={R} fill="none" stroke={T.rule} strokeWidth="1" />
      <circle cx={cx} cy={cy} r={R} fill="none" stroke={T.wire} strokeWidth="2"
        strokeDasharray="5 9" strokeLinecap="round" className="ring-cw" opacity="0.85" />
      <circle cx={cx} cy={cy} r={R - 7} fill="none" stroke={T.wire} strokeWidth="1.4"
        strokeDasharray="3 11" strokeLinecap="round" className="ring-ccw" opacity="0.45" />
      {pts.map((p) => (
        <g key={p.k}>
          <circle cx={p.x} cy={p.y} r="6.5" fill={T.bg} stroke={shardColor(p.k, n)} strokeWidth="2" />
          <circle cx={p.x} cy={p.y} r="2.5" fill={shardColor(p.k, n)} />
        </g>
      ))}
      <text x={cx} y={cy - 3} textAnchor="middle" fontFamily={mono} fontSize="8.5"
        fontWeight="600" fill={T.wire} letterSpacing="0.04em">
        {inbound ? "sum" : "copy"}
      </text>
      <text x={cx} y={cy + 8} textAnchor="middle" fontFamily={mono} fontSize="8" fill={T.dim}>
        {n} hops
      </text>
    </svg>
  );
}

/* ---------------------------------------------------------- cost read-out */
function CostTape({ rows }) {
  const max = Math.max(...rows.map((r) => r.bytes), 1);
  return (
    <div style={{ display: "grid", gap: 8 }}>
      {rows.map((r) => (
        <div key={r.label} style={{ display: "grid", gridTemplateColumns: "minmax(0,1fr) auto", gap: 12, alignItems: "center" }}>
          <div>
            <div style={{
              fontFamily: mono, fontSize: 10.5, color: r.win ? T.axX : T.soft,
              letterSpacing: "0.04em", marginBottom: 4,
            }}>{r.label}</div>
            <div style={{ height: 6, background: T.rule, borderRadius: 3, overflow: "hidden" }}>
              <div style={{
                height: "100%", width: `${(r.bytes / max) * 100}%`,
                background: r.win ? T.axX : T.dim,
                borderRadius: 3, transition: "width .35s cubic-bezier(.2,.7,.3,1)",
              }} />
            </div>
          </div>
          <div style={{ fontFamily: mono, fontSize: 10.5, color: T.ink, whiteSpace: "nowrap" }}>
            {r.expr}
          </div>
        </div>
      ))}
    </div>
  );
}

/* ------------------------------------------------------------------ chrome */
function Slider({ label, value, min, max, onChange }) {
  return (
    <label style={{ display: "block" }}>
      <div style={{
        display: "flex", justifyContent: "space-between", fontFamily: mono,
        fontSize: 10.5, color: T.soft, letterSpacing: "0.06em", marginBottom: 6,
      }}>
        <span>{label}</span><span style={{ color: T.axX }}>{value}</span>
      </div>
      <input type="range" min={min} max={max} value={value}
        onChange={(e) => onChange(Number(e.target.value))}
        style={{ width: "100%", accentColor: T.axX }} />
    </label>
  );
}

function Choice({ label, options, value, onChange }) {
  return (
    <div>
      <div style={{
        fontFamily: mono, fontSize: 10.5, color: T.soft,
        letterSpacing: "0.06em", marginBottom: 6,
      }}>{label}</div>
      <div style={{ display: "flex", gap: 6, flexWrap: "wrap" }}>
        {options.map((o) => (
          <button key={o.value} onClick={() => onChange(o.value)}
            style={{
              fontFamily: mono, fontSize: 11, padding: "6px 11px", borderRadius: 6,
              cursor: "pointer", transition: "all .16s",
              border: `1px solid ${value === o.value ? T.axX : T.rule}`,
              background: value === o.value ? "rgba(94,234,212,0.1)" : "transparent",
              color: value === o.value ? T.axX : T.soft,
            }}>{o.label}</button>
        ))}
      </div>
    </div>
  );
}

function StepBar({ step, total, onStep, title, body, wire }) {
  const tagStyle = {
    None: { c: T.axX, t: "no communication" },
    INVALID: { c: T.bad, t: "invalid sharding" },
  };
  const tag = wire === null ? tagStyle.None : wire === "INVALID" ? tagStyle.INVALID : { c: T.wire, t: wire };

  return (
    <div style={{
      background: T.panel, border: `1px solid ${T.rule}`, borderRadius: 10, padding: 16,
    }}>
      <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 12 }}>
        <button onClick={() => onStep(step - 1)} disabled={step === 0} aria-label="Previous step"
          style={navBtn(step === 0)}>←</button>
        <div style={{ display: "flex", gap: 5, flex: 1, justifyContent: "center" }}>
          {Array.from({ length: total }, (_, i) => (
            <button key={i} onClick={() => onStep(i)} aria-label={`Step ${i + 1}`}
              style={{
                width: i === step ? 22 : 7, height: 7, borderRadius: 4, border: "none",
                cursor: "pointer", padding: 0,
                background: i === step ? T.axX : i < step ? T.dim : T.rule,
                transition: "all .25s cubic-bezier(.2,.7,.3,1)",
              }} />
          ))}
        </div>
        <button onClick={() => onStep(step + 1)} disabled={step === total - 1} aria-label="Next step"
          style={navBtn(step === total - 1)}>→</button>
      </div>

      <div style={{
        display: "inline-block", fontFamily: mono, fontSize: 9.5, letterSpacing: "0.12em",
        textTransform: "uppercase", padding: "3px 9px", borderRadius: 5, marginBottom: 9,
        color: tag.c, border: `1px solid ${tag.c}55`, background: `${tag.c}14`,
      }}>{tag.t}</div>

      <div style={{
        fontFamily: cond, fontSize: 17, fontWeight: 700, color: T.ink,
        letterSpacing: "0.01em", marginBottom: 6,
      }}>{title}</div>
      <div style={{ fontFamily: sans, fontSize: 14, lineHeight: 1.55, color: "#B6C2CF" }}
        dangerouslySetInnerHTML={{ __html: body }} />
    </div>
  );
}

const navBtn = (disabled) => ({
  fontFamily: mono, fontSize: 15, width: 34, height: 30, borderRadius: 6,
  border: `1px solid ${T.rule}`, background: disabled ? "transparent" : T.well,
  color: disabled ? T.rule : T.ink, cursor: disabled ? "default" : "pointer",
  transition: "all .16s",
});

function Grid({ children, cols }) {
  return (
    <div style={{
      display: "grid", gap: 12, marginTop: 14,
      gridTemplateColumns: `repeat(auto-fit, minmax(${cols > 2 ? 260 : 300}px, 1fr))`,
    }}>{children}</div>
  );
}

function Formula({ children }) {
  return (
    <div style={{
      fontFamily: mono, fontSize: 13, color: T.ink, background: T.well,
      border: `1px solid ${T.rule}`, borderRadius: 8, padding: "11px 14px",
      letterSpacing: "0.01em", overflowX: "auto", whiteSpace: "nowrap",
    }}>{children}</div>
  );
}

function Controls({ children }) {
  return (
    <div style={{
      display: "grid", gap: 18, marginTop: 14, marginBottom: 14,
      gridTemplateColumns: "repeat(auto-fit, minmax(180px, 1fr))",
      background: T.panel, border: `1px solid ${T.rule}`, borderRadius: 10, padding: 16,
    }}>{children}</div>
  );
}

/* =========================================================== CASE 1 ====== */
function Case1() {
  const [x, setX] = useState(2), [y, setY] = useState(2), [step, setStep] = useState(0);
  const steps = [
    {
      title: "Where the data starts",
      wire: null,
      body: "Each device holds one row-chunk of <b>A</b> (cut by mesh axis x) and one column-chunk of <b>B</b> (cut by mesh axis y). The contracting dimension J is whole on both operands, on every device.",
    },
    {
      title: "Multiply in place",
      wire: null,
      body: "Nothing is missing, so every device just multiplies what it has. The output block it produces is exactly the block the target sharding asked for. Zero bytes cross a link — this is the case you want your matmuls to land in.",
    },
  ];
  useKeys(setStep, steps.length);
  const s = steps[Math.min(step, steps.length - 1)];

  const panels = [];
  for (let dx = 0; dx < x; dx++) {
    for (let dy = 0; dy < y; dy++) {
      panels.push(
        <DevicePanel key={`${dx}-${dy}`} uid={`c1-${dx}-${dy}`} label={`device  x=${dx}  y=${dy}`}
          specs={{
            A: { state: "held", axis: "row", n: x, idx: dx, edge: T.axX },
            B: { state: "held", axis: "col", n: y, idx: dy, edge: T.axY },
            C: step === 0 ? { state: "empty" }
              : { state: "mesh2d", nRow: x, nCol: y, rowIdx: dx, colIdx: dy },
          }} />
      );
    }
  }

  return (
    <>
      <Formula>A[I<sub>x</sub>, J] · B[J, K<sub>y</sub>] → C[I<sub>x</sub>, K<sub>y</sub>]</Formula>
      <Controls>
        <Slider label="MESH X — CUTS I" value={x} min={1} max={4} onChange={setX} />
        <Slider label="MESH Y — CUTS K" value={y} min={1} max={4} onChange={setY} />
      </Controls>
      <StepBar step={step} total={steps.length} onStep={setStep} {...s} />
      <Grid cols={y}>{panels}</Grid>
    </>
  );
}

/* =========================================================== CASE 2 ====== */
function Case2() {
  const [n, setN] = useState(4), [strat, setStrat] = useState("gather"), [step, setStep] = useState(0);

  const gatherSteps = [
    {
      title: "Where the data starts",
      wire: null,
      body: "<b>A</b> is cut along J, the contracting dimension — each device owns one slice of the sum. <b>B</b> is replicated, so every device already holds all of it.",
    },
    {
      title: "AllGather A along x",
      wire: "AllGather",
      body: "Each device passes its J-slice around the ring until everyone holds the whole of <b>A</b>. Watch the shard colors fill in: after this, the contraction is no longer split.",
    },
    {
      title: "Multiply in place",
      wire: null,
      body: "Every device now holds full <b>A</b> and full <b>B</b>, so each computes the same complete result. Note what that costs: the FLOPs are <i>replicated</i>, not shared. Every device does the entire matmul.",
    },
  ];

  const reduceSteps = [
    {
      title: "Where the data starts",
      wire: null,
      body: "Same setup: <b>A</b> cut along J, <b>B</b> replicated. But replication hides an option — every device already holds the rows of <b>B</b> that match its own J-slice.",
    },
    {
      title: "Slice B locally — free",
      wire: null,
      body: "Reinterpret B[J, K] as B[J<sub>x</sub>, K]. Each device just drops the rows it won't touch. Nothing is sent; this is a pointer change. You are now in Case 3.",
    },
    {
      title: "Multiply the local slices",
      wire: null,
      body: "Each device contracts over its own slice of J only, producing a full-shape but incomplete <b>C</b>. The FLOPs are genuinely split x ways — this is the payoff.",
    },
    {
      title: "AllReduce C along x",
      wire: "AllReduce",
      body: "Sum the partial results across the ring so every device ends with the finished <b>C</b>. AllReduce is a ReduceScatter followed by an AllGather, so it moves twice the bytes of a plain gather.",
    },
  ];

  const steps = strat === "gather" ? gatherSteps : reduceSteps;
  const st = Math.min(step, steps.length - 1);
  useKeys(setStep, steps.length);
  useEffect(() => { setStep(0); }, [strat]);

  const panels = Array.from({ length: n }, (_, k) => {
    let A, B, C;
    if (strat === "gather") {
      A = { state: st === 0 ? "held" : "gathered", axis: "col", n, idx: k, edge: T.axX };
      B = { state: "replicated", shape: "J × K" };
      C = st < 2 ? { state: "empty" } : { state: "reduced", shape: "I × K" };
    } else {
      A = { state: "held", axis: "col", n, idx: k, edge: T.axX };
      B = st === 0 ? { state: "replicated", shape: "J × K" }
        : { state: "held", axis: "row", n, idx: k, edge: T.axX };
      C = st < 2 ? { state: "empty" }
        : st === 2 ? { state: "partial", shape: "I × K" }
          : { state: "reduced", shape: "I × K" };
    }
    return <DevicePanel key={k} uid={`c2-${k}-${strat}`} label={`device ${k}`} specs={{ A, B, C }} />;
  });

  return (
    <>
      <Formula>A[I, J<sub>x</sub>] · B[J, K] → C[I, K]</Formula>
      <Controls>
        <Slider label="MESH X — CUTS J OF A" value={n} min={2} max={8} onChange={setN} />
        <Choice label="STRATEGY" value={strat} onChange={setStrat}
          options={[
            { value: "gather", label: "AllGather A first" },
            { value: "reduce", label: "Local matmul + AllReduce" },
          ]} />
      </Controls>
      <StepBar step={st} total={steps.length} onStep={setStep} {...steps[st]} />

      <div style={{ display: "flex", gap: 16, alignItems: "center", marginTop: 14, flexWrap: "wrap" }}>
        {steps[st].wire && steps[st].wire !== "INVALID" && <LinkRing n={n} kind={steps[st].wire} />}
        <div style={{ flex: 1, minWidth: 260 }}>
          <div style={{
            fontFamily: mono, fontSize: 9.5, letterSpacing: "0.12em", color: T.dim,
            textTransform: "uppercase", marginBottom: 10,
          }}>Bytes on the wire, bf16</div>
          <CostTape rows={[
            { label: "AllGather A", bytes: 2, expr: "2·I·J / W", win: strat === "gather" },
            { label: "AllReduce C", bytes: 4, expr: "4·I·K / W", win: strat === "reduce" },
          ]} />
          <div style={{
            fontFamily: sans, fontSize: 12.5, color: T.soft, marginTop: 12, lineHeight: 1.5,
          }}>
            The reduce path wins when <span style={{ fontFamily: mono, color: T.axX }}>J &gt; 2K</span> —
            and it also cuts each device's FLOPs by {n}×. If you can leave the output sharded, swap
            AllReduce for ReduceScatter and the comparison becomes J vs K flat.
          </div>
        </div>
      </div>

      <Grid cols={2}>{panels}</Grid>
    </>
  );
}

/* =========================================================== CASE 3 ====== */
function OuterProduct() {
  return (
    <svg viewBox="0 0 560 130" style={{ width: "100%", maxWidth: 620, display: "block", margin: "14px 0" }}>
      <defs>
        <pattern id="hatch-op" width="7" height="7" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
          <rect width="7" height="7" fill={T.wire} opacity="0.14" />
          <line x1="0" y1="0" x2="0" y2="7" stroke={T.wire} strokeWidth="1.6" opacity="0.55" />
        </pattern>
      </defs>
      <rect x="8" y="20" width="22" height="90" fill={shardColor(0, 3)} rx="1" />
      <text x="19" y="124" textAnchor="middle" fontFamily={mono} fontSize="8.5" fill={T.dim}>col of A</text>
      <text x="52" y="70" textAnchor="middle" fontSize="16" fill={T.dim}>⊗</text>
      <rect x="74" y="54" width="90" height="22" fill={shardColor(1, 3)} rx="1" />
      <text x="119" y="94" textAnchor="middle" fontFamily={mono} fontSize="8.5" fill={T.dim}>row of B</text>
      <text x="186" y="70" textAnchor="middle" fontSize="16" fill={T.dim}>=</text>
      <rect x="208" y="20" width="90" height="90" fill="url(#hatch-op)" stroke={T.wire} strokeWidth="1.5" rx="1" />
      <text x="253" y="68" textAnchor="middle" fontFamily={mono} fontSize="8.5" fontWeight="600" fill={T.wire}>
        <tspan x="253" dy="0">dense I × K</tspan>
        <tspan x="253" dy="11">every entry hit</tspan>
      </text>
      <line x1="316" y1="65" x2="392" y2="65" stroke={T.dim} strokeWidth="1.2" markerEnd="url(#ar)" />
      <marker id="ar" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto">
        <path d="M0,0 L7,3.5 L0,7 z" fill={T.dim} />
      </marker>
      <text x="354" y="56" textAnchor="middle" fontFamily={mono} fontSize="8.5" fill={T.dim}>
        <tspan x="354" dy="0">sum over the j's</tspan>
        <tspan x="354" dy="10.5">this device owns</tspan>
      </text>
      <rect x="404" y="20" width="90" height="90" fill="url(#hatch-op)" stroke={T.wire} strokeWidth="1.5" rx="1" />
      <text x="449" y="68" textAnchor="middle" fontFamily={mono} fontSize="8.5" fontWeight="600" fill={T.wire}>
        <tspan x="449" dy="0">one partial</tspan>
        <tspan x="449" dy="11">C[I, K]</tspan>
      </text>
    </svg>
  );
}

function Case3() {
  const [n, setN] = useState(4), [op, setOp] = useState("allreduce"), [step, setStep] = useState(0);
  const rs = op === "reducescatter";

  const steps = [
    {
      title: "Where the data starts",
      wire: null,
      body: "Both operands are cut along J on the <i>same</i> mesh axis, so device k holds J-slice k of <b>A</b> and J-slice k of <b>B</b>. The slices line up — the local multiply is well-defined.",
    },
    {
      title: "Multiply the local slices",
      wire: null,
      body: "Each device sums over its own piece of J. The result has the full I × K shape but is only a term of the sum. Every device holds a different, incomplete <b>C</b> — written C[I,K]{U<sub>x</sub>}, \"unreduced along x\".",
    },
    rs ? {
      title: "ReduceScatter along x",
      wire: "ReduceScatter",
      body: "Sum the partials and hand each device only its own slice of the answer. Same bytes as an AllGather — half the cost of an AllReduce — and usually all you need, since the next layer wants a sharded input anyway.",
    } : {
      title: "AllReduce along x",
      wire: "AllReduce",
      body: "Sum the partials so every device lands on the identical, finished <b>C</b>. Under the hood this is a ReduceScatter then an AllGather, which is why it costs twice a plain gather.",
    },
  ];
  useKeys(setStep, steps.length);
  const st = Math.min(step, steps.length - 1);

  const panels = Array.from({ length: n }, (_, k) => (
    <DevicePanel key={k} uid={`c3-${k}`} label={`device ${k}`} specs={{
      A: { state: "held", axis: "col", n, idx: k, edge: T.axX },
      B: { state: "held", axis: "row", n, idx: k, edge: T.axX },
      C: st === 0 ? { state: "empty" }
        : st === 1 ? { state: "partial", shape: "I × K" }
          : rs ? { state: "shard", axis: "col", n, idx: k, edge: T.axX }
            : { state: "reduced", shape: "I × K" },
    }} />
  ));

  return (
    <>
      <Formula>A[I, J<sub>x</sub>] ·<sub>LOCAL</sub> B[J<sub>x</sub>, K] → C[I, K] {"{U"}<sub>x</sub>{"}"}</Formula>
      <Controls>
        <Slider label="MESH X — CUTS J OF BOTH" value={n} min={2} max={8} onChange={setN} />
        <Choice label="HOW TO RESOLVE THE PARTIALS" value={op} onChange={setOp}
          options={[
            { value: "allreduce", label: "AllReduce → replicated" },
            { value: "reducescatter", label: "ReduceScatter → sharded" },
          ]} />
      </Controls>
      <StepBar step={st} total={steps.length} onStep={setStep} {...steps[st]} />

      {st === 1 && (
        <div style={{
          background: T.panel, border: `1px solid ${T.rule}`, borderRadius: 10,
          padding: "12px 16px", marginTop: 14,
        }}>
          <div style={{
            fontFamily: mono, fontSize: 9.5, letterSpacing: "0.12em", color: T.dim,
            textTransform: "uppercase",
          }}>Why a partial result is dense, not partial in shape</div>
          <OuterProduct />
        </div>
      )}

      {st === 2 && (
        <div style={{ display: "flex", gap: 16, alignItems: "center", marginTop: 14, flexWrap: "wrap" }}>
          <LinkRing n={n} kind={steps[st].wire} />
          <div style={{ flex: 1, minWidth: 260 }}>
            <div style={{
              fontFamily: mono, fontSize: 9.5, letterSpacing: "0.12em", color: T.dim,
              textTransform: "uppercase", marginBottom: 10,
            }}>Bytes on the wire, bf16</div>
            <CostTape rows={[
              { label: "ReduceScatter → C[I, K_x]", bytes: 2, expr: "2·I·K / W", win: rs },
              { label: "AllReduce → C[I, K]", bytes: 4, expr: "4·I·K / W", win: !rs },
            ]} />
          </div>
        </div>
      )}

      <Grid cols={2}>{panels}</Grid>
    </>
  );
}

/* =========================================================== CASE 4 ====== */
function Case4() {
  const [n, setN] = useState(4), [pick, setPick] = useState("A"), [step, setStep] = useState(0);

  const steps = [
    {
      title: "Both operands claim mesh axis x",
      wire: "INVALID",
      body: "<b>A</b> cuts I on x and <b>B</b> cuts K on x. Device k can only produce block (k, k) of the output — the diagonal. The off-diagonal blocks exist on no device at all, so the sharding is rejected before it runs.",
    },
    pick === "A" ? {
      title: "AllGather A along x",
      wire: "AllGather",
      body: "Gather the row-chunks of <b>A</b> so every device holds it whole. Axis x is now spent only on <b>B</b>, and the conflict is gone.",
    } : {
      title: "AllGather B along x",
      wire: "AllGather",
      body: "Gather the column-chunks of <b>B</b> so every device holds it whole. Axis x is now spent only on <b>A</b>, and the conflict is gone.",
    },
    pick === "A" ? {
      title: "Multiply in place → C[I, K_x]",
      wire: null,
      body: "The output inherits <b>B</b>'s sharding: cut along K. Which operand you gathered decides how the result comes out, so pick based on what the next layer wants — and on which matrix is smaller.",
    } : {
      title: "Multiply in place → C[I_x, K]",
      wire: null,
      body: "The output inherits <b>A</b>'s sharding: cut along I. Which operand you gathered decides how the result comes out, so pick based on what the next layer wants — and on which matrix is smaller.",
    },
  ];
  useKeys(setStep, steps.length);
  const st = Math.min(step, steps.length - 1);

  const panels = Array.from({ length: n }, (_, k) => {
    const gA = st >= 1 && pick === "A";
    const gB = st >= 1 && pick === "B";
    return (
      <DevicePanel key={k} uid={`c4-${k}-${pick}`} label={`device ${k}`} specs={{
        A: { state: gA ? "gathered" : "held", axis: "row", n, idx: k, edge: T.axX },
        B: { state: gB ? "gathered" : "held", axis: "col", n, idx: k, edge: T.axX },
        C: st < 2 ? { state: "empty" }
          : { state: "shard", axis: pick === "A" ? "col" : "row", n, idx: k, edge: T.axX },
      }} />
    );
  });

  return (
    <>
      <Formula>A[I<sub>x</sub>, J] · B[J, K<sub>x</sub>] → C[I<sub>x</sub>, K<sub>x</sub>]  ✗</Formula>
      <Controls>
        <Slider label="MESH X — CUTS I AND K" value={n} min={2} max={8} onChange={setN} />
        <Choice label="WHICH OPERAND TO GATHER" value={pick} onChange={setPick}
          options={[{ value: "A", label: "Gather A" }, { value: "B", label: "Gather B" }]} />
      </Controls>
      <StepBar step={st} total={steps.length} onStep={setStep} {...steps[st]} />

      {st === 1 && (
        <div style={{ display: "flex", gap: 16, alignItems: "center", marginTop: 14, flexWrap: "wrap" }}>
          <LinkRing n={n} kind="AllGather" />
          <div style={{ flex: 1, minWidth: 260 }}>
            <div style={{
              fontFamily: mono, fontSize: 9.5, letterSpacing: "0.12em", color: T.dim,
              textTransform: "uppercase", marginBottom: 10,
            }}>Bytes on the wire, bf16</div>
            <CostTape rows={[
              { label: "Gather A", bytes: 2, expr: "2·I·J / W", win: pick === "A" },
              { label: "Gather B", bytes: 2, expr: "2·J·K / W", win: pick === "B" },
            ]} />
            <div style={{ fontFamily: sans, fontSize: 12.5, color: T.soft, marginTop: 12, lineHeight: 1.5 }}>
              Neither is cheaper by rule — compare I·J against J·K and gather the smaller matrix,
              unless the output sharding you need forces the choice.
            </div>
          </div>
        </div>
      )}

      <Grid cols={2}>{panels}</Grid>
    </>
  );
}

/* ------------------------------------------------------------------ keys */
function useKeys(setStep, total) {
  const onKey = useCallback((e) => {
    if (e.target.tagName === "INPUT") return;
    if (e.key === "ArrowRight") setStep((s) => Math.min(s + 1, total - 1));
    if (e.key === "ArrowLeft") setStep((s) => Math.max(s - 1, 0));
  }, [setStep, total]);
  useEffect(() => {
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onKey]);
}

/* ------------------------------------------------------------------- app */
const TABS = [
  { id: 0, num: "01", name: "Contraction intact", sub: "no comms", C: Case1 },
  { id: 1, num: "02", name: "One side cut on J", sub: "gather first", C: Case2 },
  { id: 2, num: "03", name: "Both sides cut on J", sub: "reduce after", C: Case3 },
  { id: 3, num: "04", name: "Axis reused", sub: "invalid", C: Case4 },
];

export default function ShardedMatmul() {
  const [tab, setTab] = useState(0);
  const Active = TABS[tab].C;

  return (
    <div style={{
      background: `radial-gradient(1100px 500px at 12% -12%, rgba(94,234,212,0.055), transparent), ${T.bg}`,
      minHeight: "100%", color: T.ink, padding: "26px 20px 48px",
    }}>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans+Condensed:wght@600;700&family=IBM+Plex+Sans:wght@400;500&display=swap');
        .ring-cw { animation: dashmove 1.5s linear infinite; }
        .ring-ccw { animation: dashmove 2.1s linear infinite reverse; }
        @keyframes dashmove { to { stroke-dashoffset: -28; } }
        .blk rect, .blk line { transition: opacity .3s ease, fill .3s ease, stroke .3s ease; }
        button:focus-visible, input:focus-visible { outline: 2px solid ${T.axX}; outline-offset: 2px; }
        input[type=range] { height: 4px; background: ${T.rule}; border-radius: 2px; }
        @media (prefers-reduced-motion: reduce) {
          .ring-cw, .ring-ccw { animation: none; }
          .blk rect, .blk line { transition: none; }
        }
      `}</style>

      <div style={{ maxWidth: 1180, margin: "0 auto" }}>
        <header style={{ marginBottom: 24 }}>
          <div style={{
            fontFamily: mono, fontSize: 10, letterSpacing: "0.22em", color: T.dim,
            textTransform: "uppercase", marginBottom: 8,
          }}>A[I, J] · B[J, K] → C[I, K] &nbsp;/&nbsp; four shardings</div>
          <h1 style={{
            fontFamily: cond, fontWeight: 700, fontSize: "clamp(27px, 4.4vw, 44px)",
            letterSpacing: "0.005em", margin: 0, lineHeight: 1.05,
          }}>
            What each device holds,<br />
            <span style={{ color: T.axX }}>and what has to move</span>
          </h1>
          <p style={{
            fontFamily: sans, fontSize: 14.5, color: T.soft, maxWidth: 620,
            marginTop: 12, lineHeight: 1.55,
          }}>
            Set the mesh, then step through. Solid blocks are data the device physically has;
            faded dashed blocks are data it doesn't. A shard keeps its color wherever it travels,
            so you can follow one chunk across a collective. Arrow keys work.
          </p>
        </header>

        <nav style={{
          display: "grid", gap: 8, marginBottom: 20,
          gridTemplateColumns: "repeat(auto-fit, minmax(150px, 1fr))",
        }}>
          {TABS.map((t) => {
            const on = t.id === tab;
            return (
              <button key={t.id} onClick={() => setTab(t.id)}
                style={{
                  textAlign: "left", padding: "11px 13px", borderRadius: 9, cursor: "pointer",
                  border: `1px solid ${on ? T.axX : T.rule}`,
                  background: on ? "rgba(94,234,212,0.08)" : T.panel,
                  transition: "all .18s cubic-bezier(.2,.7,.3,1)",
                }}>
                <div style={{
                  fontFamily: mono, fontSize: 10, letterSpacing: "0.16em",
                  color: on ? T.axX : T.dim, marginBottom: 5,
                }}>{t.num} · {t.sub}</div>
                <div style={{
                  fontFamily: cond, fontWeight: 600, fontSize: 15,
                  color: on ? T.ink : T.soft, lineHeight: 1.2,
                }}>{t.name}</div>
              </button>
            );
          })}
        </nav>

        <Active key={tab} />

        <footer style={{
          fontFamily: sans, fontSize: 12.5, color: T.dim, marginTop: 30,
          paddingTop: 16, borderTop: `1px solid ${T.rule}`, lineHeight: 1.6,
        }}>
          Border color marks which mesh axis made the cut —{" "}
          <span style={{ color: T.axX, fontFamily: mono }}>teal for x</span>,{" "}
          <span style={{ color: T.axY, fontFamily: mono }}>pink for y</span>.{" "}
          Hatched amber is an unreduced partial sum. Byte counts assume bf16 and a full
          bidirectional ring, where a gather or scatter costs V/W and an AllReduce costs 2V/W.
        </footer>
      </div>
    </div>
  );
}