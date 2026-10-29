import { useEffect, useRef } from "react";
import { T, mono } from "../theme";
import { FNS, calleeOf } from "../flow/dsl";
import { MODS } from "../flow/world";
import { pyHtml } from "../pyhl";
import { tokText } from "./Panels";

// The whole program as one nested flowchart, starting at each thread's root
// function. The node a frame is on, if it calls something, opens into the
// callee's flowchart (a subgraph) right there, so the call stack reads top to
// bottom. The running frame is drawn large with its variables; callers shrink.
//
// levels: "focus" = the running frame, "near" = its caller, "far" = older
// callers (trimmed to the path unless showAll), "peek" = a callee opened by click.

const ICON = { CALL: "▸", IF: "◆", WHILE: "⟳", FOR: "⟳", RET: "↩", BRK: "⤓", CONT: "↺", AWAIT: "⏸" };

// ---------------------------------------------------------------- values

function pyVal(v, inTokens) {
  if (v === null || v === undefined) return "None";
  if (v === true) return "True";
  if (v === false) return "False";
  if (typeof v === "number") return inTokens && v === -1 ? "?" : String(v);
  if (typeof v === "string") {
    if (/^[0-9a-f]{8}$/.test(v)) return `#${v.slice(0, 4)}`;
    if (v === "<e>") return "⏎";
    return `'${v}'`;
  }
  if (v.$r) return v.$r;
  if (v.$sr) return `SR(${v.$sr.join(", ")})`;
  if (v.$t) return `Tensor[${v.$t.join(", ")}]`;
  if (v.$x) return v.$x;
  if (Array.isArray(v)) {
    const toks = v.some((x) => typeof x === "string");
    const xs = v.slice(0, 12).map((x) => (toks && typeof x === "string" && x.length <= 3 ? tokText(x) : pyVal(x, toks)));
    return `[${xs.join(", ")}${v.length > 12 ? ", …" : ""}]`;
  }
  return `{${Object.entries(v).map(([k, x]) => `${k}: ${pyVal(x)}`).join(", ")}}`;
}

function Vars({ frame, prev, big, color }) {
  const entries = Object.entries(frame.vars);
  if (!entries.length) return null;
  return (
    <div style={{ display: "flex", flexWrap: "wrap", gap: big ? 5 : 3, margin: big ? "6px 0 8px" : "3px 0 5px" }}>
      {entries.map(([k, v]) => {
        const changed = prev && JSON.stringify(prev.vars[k]) !== JSON.stringify(v);
        return (
          <span key={k} className={changed && big ? "var-flash" : ""} style={{
            fontFamily: mono, fontSize: big ? 11.5 : 9.5, padding: big ? "2px 7px" : "0 5px", borderRadius: 4, "--c": color,
            border: `1px solid ${changed && big ? color : T.rule}`, background: T.bg, color: T.soft, maxWidth: big ? 520 : 260,
            overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap",
          }}>
            <span style={{ color: T.dim }}>{k}</span> = <span style={{ color: v && v.$t ? "#8AB4FF" : T.ink }}>{pyVal(v)}</span>
          </span>
        );
      })}
    </div>
  );
}

// ---------------------------------------------------------------- one function

export function Frame({ fn, ctx, d, level }) {
  const spec = FNS[fn];
  const mod = MODS[spec.mod];
  const frame = ctx.frames ? ctx.frames[d] : null;
  const prev = frame && ctx.prevFrames && ctx.prevFrames[d] && ctx.prevFrames[d].fn === fn ? ctx.prevFrames[d] : null;
  const big = level === "focus";
  const sub = { ...ctx, d, level, frame, mod, trim: level === "far" && !ctx.showAll };
  return (
    <div className={`fr fr-${level}`} style={{
      "--c": mod.color, borderColor: big ? mod.color : level === "peek" ? `${mod.color}66` : `${mod.color}55`,
      borderStyle: level === "peek" ? "dashed" : "solid",
      background: big ? `${mod.color}0d` : T.well, padding: big ? "8px 12px 12px" : "4px 8px 6px",
    }}>
      <div style={{ display: "flex", alignItems: "center", gap: 8, flexWrap: "wrap" }}>
        <span style={{ fontFamily: mono, fontSize: big ? 10 : 9, fontWeight: 700, color: T.bg, background: mod.color, borderRadius: 4, padding: "0 5px" }}>{mod.label}</span>
        <code style={{ fontFamily: mono, fontSize: big ? 13 : 10.5, color: T.ink, whiteSpace: "pre" }} dangerouslySetInnerHTML={{ __html: spec.sig.split("\n").map(pyHtml).join("\n") }} />
        {frame && frame.label && <span style={{ fontFamily: mono, fontSize: 9.5, color: mod.color }}>({frame.label})</span>}
        {big && <span style={{ fontFamily: mono, fontSize: 9.5, color: mod.color, marginLeft: "auto" }}>● running</span>}
      </div>
      {frame && <Vars frame={frame} prev={prev} big={big} color={mod.color} />}
      <div style={{ marginTop: frame ? 0 : 6 }}>
        <Nodes nodes={spec.body} ctx={sub} />
      </div>
    </div>
  );
}

// ---------------------------------------------------------------- statements

function onPath(n, frame) {
  return !!frame && (frame.node === n.id || n.desc.includes(frame.node));
}

function Nodes({ nodes, ctx }) {
  const out = [];
  let gap = 0;
  const flush = (k) => {
    if (gap) out.push(<Gap key={`g${k}`} n={gap} />);
    gap = 0;
  };
  nodes.forEach((n, k) => {
    if (ctx.trim && !onPath(n, ctx.frame)) {
      gap++;
      return;
    }
    flush(k);
    if (out.length) out.push(<Conn key={`c${k}`} />);
    out.push(<Node key={n.id} n={n} ctx={ctx} />);
  });
  flush("end");
  return <div style={{ display: "grid", justifyItems: "start" }}>{out}</div>;
}

const Gap = ({ n }) => <span style={{ fontFamily: mono, fontSize: 9, color: T.dim, padding: "0 6px", margin: "1px 0", border: `1px dashed ${T.rule}`, borderRadius: 8 }}>⋯ {n}</span>;
const Conn = () => <span className="conn" />;

function Node({ n, ctx }) {
  const { frame } = ctx;
  const cur = !!frame && frame.node === n.id;
  const callee = calleeOf(n);
  const childLive = cur && callee && ctx.frames && ctx.frames[ctx.d + 1] && ctx.frames[ctx.d + 1].fn === callee;
  const peekKey = `${ctx.thread}|${ctx.d}|${n.id}`;
  const peekOpen = !childLive && callee && ctx.peek.has(peekKey);

  const child = childLive ? (
    <Sub color={MODS[FNS[callee].mod].color}>
      <Frame fn={callee} ctx={ctx} d={ctx.d + 1} level={levelOf(ctx, ctx.d + 1)} />
    </Sub>
  ) : peekOpen ? (
    <Sub color={MODS[FNS[callee].mod].color} dashed>
      <Frame fn={callee} ctx={{ ...ctx, frames: null }} d={0} level="peek" />
    </Sub>
  ) : null;

  const box = <Box n={n} ctx={ctx} cur={cur} onPeek={callee && !childLive ? () => ctx.togglePeek(peekKey) : null} peekOpen={peekOpen} />;

  if (n.t === "IF") {
    const dec = frame ? frame.dec[n.id] : undefined;
    return (
      <div style={{ display: "grid", justifyItems: "start" }}>
        {box}
        {child}
        <Branch label="yes" nodes={n.then} ctx={ctx} dim={dec === false} />
        {n.els && <Branch label="else" nodes={n.els} ctx={ctx} dim={dec === true} />}
      </div>
    );
  }
  if (n.t === "WHILE" || n.t === "FOR") {
    const inside = onPath(n, frame);
    const it = frame && frame.iters[n.id];
    const tot = frame && frame.totals[n.id];
    const c = ctx.mod.color;
    return (
      <div className="loop" style={{ borderColor: inside ? `${c}aa` : T.rule, "--c": inside ? c : T.rule, background: inside ? `${c}08` : "transparent" }}>
        <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
          {box}
          {it !== undefined && it > 0 && (
            <span style={{ fontFamily: mono, fontSize: ctx.level === "focus" ? 10.5 : 9, fontWeight: 700, color: T.bg, background: c, borderRadius: 999, padding: "0 7px", whiteSpace: "nowrap" }}>
              iter {it}{tot ? `/${tot}` : ""}
            </span>
          )}
        </div>
        {child}
        {n.body.length > 0 && <><Conn /><Nodes nodes={n.body} ctx={ctx} /></>}
        <span className="backedge" />
      </div>
    );
  }
  return (
    <div style={{ display: "grid", justifyItems: "start" }}>
      {box}
      {child}
    </div>
  );
}

function Branch({ label, nodes, ctx, dim }) {
  if (ctx.trim && !nodes.some((n) => onPath(n, ctx.frame))) return null;
  return (
    <div className="branch" style={{ opacity: dim ? 0.38 : 1 }}>
      <span className="branch-label">{label}</span>
      <Nodes nodes={nodes} ctx={ctx} />
    </div>
  );
}

const Sub = ({ color, dashed, children }) => (
  <div className="sub" style={{ borderLeft: `2px ${dashed ? "dashed" : "solid"} ${color}`, "--c": color }}>{children}</div>
);

function levelOf(ctx, d) {
  if (!ctx.live) return d === ctx.frames.length - 1 ? "near" : "far";
  const last = ctx.frames.length - 1;
  return d === last ? "focus" : d === last - 1 ? "near" : "far";
}

function Box({ n, ctx, cur, onPeek, peekOpen }) {
  const { frame, level } = ctx;
  const note = frame ? frame.notes[n.id] : undefined;
  const big = level === "focus";
  const c = ctx.mod.color;
  const callee = calleeOf(n);
  const calleeColor = callee ? MODS[FNS[callee].mod].color : null;
  const lines = n.text.split("\n");
  const shape = n.t === "IF" ? 16 : n.t === "RET" ? 12 : 6;
  return (
    <div style={{ display: "grid", justifyItems: "start", maxWidth: "100%" }}>
      <div
        data-cur={cur ? (big ? "focus" : "1") : undefined}
        className={`nd ${cur ? "nd-cur" : ""}`}
        onClick={onPeek || undefined}
        title={onPeek ? (peekOpen ? "close" : `peek inside ${callee}`) : undefined}
        style={{
          "--c": c, fontSize: big ? 12.5 : level === "near" ? 10.5 : 9.5, padding: big ? "4px 10px" : "1px 7px", borderRadius: shape,
          border: `1px solid ${cur ? c : n.t === "IF" ? "#FFC94D44" : n.t === "AWAIT" ? "#5FD3F355" : T.rule}`,
          borderLeft: callee ? `3px solid ${calleeColor}` : undefined,
          background: cur ? `${c}24` : note !== undefined ? T.panel : level === "peek" ? T.well : T.panel,
          boxShadow: cur ? `0 0 0 1px ${c}, 0 0 16px ${c}55` : "none",
          opacity: frame && note === undefined && !cur ? 0.62 : 1,
          cursor: onPeek ? "pointer" : "default",
        }}
      >
        <span style={{ color: n.t === "IF" ? "#FFC94D" : callee ? calleeColor : T.dim, width: big ? 12 : 9, flexShrink: 0, textAlign: "center" }}>{ICON[n.t] || "·"}</span>
        <code style={{ whiteSpace: "pre", fontFamily: mono, color: T.ink }} dangerouslySetInnerHTML={{ __html: lines.map(pyHtml).join("\n") }} />
        {onPeek && <span style={{ color: calleeColor, fontSize: 9, alignSelf: "center", marginLeft: 4 }}>{peekOpen ? "▾" : "▸"}</span>}
      </div>
      {note !== undefined && note !== "" && (
        <span className={cur && big ? "title-in" : ""} style={{
          fontFamily: mono, fontSize: big ? 11.5 : 9, color: cur ? "#9FE8D9" : "#7FBFB2", fontStyle: "italic",
          margin: big ? "2px 0 0 24px" : "0 0 0 18px", whiteSpace: big ? "pre-wrap" : "nowrap", maxWidth: big ? 640 : 360,
          overflow: "hidden", textOverflow: "ellipsis",
        }}># {note}</span>
      )}
    </div>
  );
}

// ---------------------------------------------------------------- whole canvas

export function FlowCanvas({ ev, prev, peek, togglePeek, showAll }) {
  const box = useRef(null);
  useEffect(() => {
    const el = box.current && (box.current.querySelector("[data-cur='focus']") || box.current.querySelector("[data-cur='1']"));
    if (el) el.scrollIntoView({ block: "center", inline: "nearest", behavior: "smooth" });
  }, [ev]);

  return (
    <div ref={box} className="flow-canvas">
      {ev.threads.map((t) => {
        const live = t.name === ev.thread;
        const pt = prev && prev.threads.find((x) => x.name === t.name);
        const color = { main: MODS.main.color, engine: MODS.engine.color, client: MODS.async.color }[t.kind];
        return (
          <div key={t.name} className="thread" style={{ opacity: live ? 1 : 0.8 }}>
            <div style={{ display: "flex", alignItems: "center", gap: 8, margin: "0 0 4px 2px" }}>
              <span style={{ width: 8, height: 8, borderRadius: 4, background: live ? color : "transparent", border: `1.5px solid ${color}`, boxShadow: live ? `0 0 8px ${color}` : "none" }} />
              <span style={{ fontFamily: mono, fontSize: 11, fontWeight: 700, color }}>{t.name}</span>
              <span style={{ fontFamily: mono, fontSize: 10, color: T.dim }}>{live ? "running" : t.state === "done" ? "finished" : t.state === "blocked" ? "blocked" : "paused"}</span>
            </div>
            {t.frames.length ? (
              <Frame fn={t.frames[0].fn} d={0} level={levelOf({ live, frames: t.frames }, 0)}
                ctx={{ frames: t.frames, prevFrames: pt && pt.frames, live, thread: t.name, peek, togglePeek, showAll }} />
            ) : (
              <span style={{ fontFamily: mono, fontSize: 10, color: T.dim, marginLeft: 16 }}>returned</span>
            )}
          </div>
        );
      })}
    </div>
  );
}
