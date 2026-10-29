import { useMemo, useState } from "react";
import { T, mono, sans } from "./theme";
import { navBtn } from "./styles";
import { runTrace } from "./flow/run";
import { DEFAULTS, MODS } from "./flow/world";
import { FNS } from "./flow/dsl";
import { FlowCanvas } from "./components/Flow";
import { DebugBar, NowCard, useDebugger } from "./components/Controls";
import {
  AsyncView, BudgetBar, FreeList, HashTable, ModelView, Outputs, Packed, PagePool, Pipeline, QueueBoard, RequestRows, SampleView, Section, TokenLegend,
} from "./components/Panels";

export default function App() {
  const [cfg, setCfg] = useState(DEFAULTS);
  const trace = useMemo(() => runTrace(cfg), [cfg]);
  return <Run key={JSON.stringify(cfg)} cfg={cfg} setCfg={setCfg} trace={trace} />;
}

function Choice({ label, value, options, onChange }) {
  return (
    <span style={{ display: "inline-flex", alignItems: "center", gap: 3 }}>
      <span style={{ fontFamily: mono, fontSize: 10.5, color: T.dim, marginRight: 3 }}>{label}</span>
      {options.map((o) => (
        <button key={o} onClick={() => onChange(o)} style={{ ...navBtn(false), height: 24, padding: "0 8px", fontSize: 11,
          background: o === value ? T.accent : T.well, color: o === value ? T.bg : T.soft, borderColor: o === value ? T.accent : T.rule }}>{o}</button>
      ))}
    </span>
  );
}

function Num({ label, value, min, max, onChange }) {
  return (
    <span style={{ display: "inline-flex", alignItems: "center", gap: 3 }}>
      <span style={{ fontFamily: mono, fontSize: 10.5, color: T.dim, marginRight: 3 }}>{label}</span>
      <button onClick={() => onChange(Math.max(min, value - 1))} disabled={value <= min} style={{ ...navBtn(value <= min), height: 24, padding: "0 7px" }}>−</button>
      <span style={{ fontFamily: mono, fontSize: 12, color: T.ink, minWidth: 18, textAlign: "center" }}>{value}</span>
      <button onClick={() => onChange(Math.min(max, value + 1))} disabled={value >= max} style={{ ...navBtn(value >= max), height: 24, padding: "0 7px" }}>+</button>
    </span>
  );
}

function Run({ cfg, setCfg, trace }) {
  const { events, hashInfo, halted } = trace;
  const dbg = useDebugger(events);
  const ev = dbg.ev;
  const prev = events[dbg.i - 1];
  const [peek, setPeek] = useState(() => new Set());
  const [showAll, setShowAll] = useState(false);
  const togglePeek = (k) => setPeek((s) => { const n = new Set(s); n.has(k) ? n.delete(k) : n.add(k); return n; });
  const set = (k) => (v) => setCfg({ ...cfg, [k]: v });

  const w = ev.world;
  const hl = ev.hl;
  const top = ev.threads.find((t) => t.name === ev.thread);
  const frames = top.frames;
  const mod = FNS[ev.fn].mod;
  const color = MODS[mod].color;
  const sched = frames.find((f) => f.fn === "Scheduler.schedule");
  const plan = sched && sched.vars.plan ? sched.vars.plan : Object.fromEntries((w.next_step ? w.next_step.batch : []).map((s) => [s.rid, s.num_new]));
  const startup = ev.group === 0 && mod !== "engine";

  const context = {
    runner: <Section title="Packed batch · ModelRunner" color={color}><Packed w={w} frames={frames} /></Section>,
    model: <Section title="Model forward · attention" color={color}><ModelView w={w} layers={cfg.layers} /></Section>,
    sample: <Section title="Sampling" color={color}><SampleView w={w} frames={frames} /></Section>,
    async: <Section title="HTTP side · inbox and streams" color={color}><AsyncView w={w} /></Section>,
    detok: <Section title="Outputs · detokenized" color={color}><Outputs w={w} /></Section>,
  }[mod];

  return (
    <div style={{ height: "100vh", display: "flex", flexDirection: "column", background: T.bg, color: T.ink, fontFamily: sans }}>
      <header style={{ padding: "10px clamp(12px, 2vw, 24px)", borderBottom: `1px solid ${T.rule}`, display: "grid", gap: 8 }}>
        <div style={{ display: "flex", alignItems: "center", gap: 16, flexWrap: "wrap" }}>
          <div>
            <div style={{ fontFamily: mono, fontSize: 11, color: T.accent, letterSpacing: 1.4 }}>TINYSERVE · END TO END</div>
            <div style={{ fontSize: 17, fontWeight: 600 }}>main() → Engine → schedule → pages → forward → sample</div>
          </div>
          <div style={{ display: "flex", gap: 12, flexWrap: "wrap", alignItems: "center", marginLeft: "auto" }}>
            <Choice label="command" value={cfg.mode} options={["generate", "serve"]} onChange={set("mode")} />
            <Choice label="block_size" value={cfg.bs} options={[2, 4, 8]} onChange={set("bs")} />
            <Num label="num_blocks" value={cfg.numBlocks} min={4} max={16} onChange={set("numBlocks")} />
            <Num label="batched_tokens" value={cfg.budget} min={2} max={16} onChange={set("budget")} />
            <Num label="max_seqs" value={cfg.maxSeqs} min={1} max={4} onChange={set("maxSeqs")} />
            <Num label="layers" value={cfg.layers} min={1} max={3} onChange={set("layers")} />
            {JSON.stringify(cfg) !== JSON.stringify(DEFAULTS) && <button onClick={() => setCfg(DEFAULTS)} style={{ ...navBtn(false), height: 24, fontSize: 11 }}>defaults</button>}
          </div>
        </div>
        <DebugBar dbg={dbg} events={events} />
        {halted && <span style={{ fontFamily: mono, fontSize: 11, color: "#FF8A8A" }}>trace {halted} (the cache may be too small for this config)</span>}
      </header>

      <div className="split">
        <div className="pane-left">
          <div style={{ display: "flex", gap: 10, alignItems: "center", marginBottom: 8, flexWrap: "wrap" }}>
            {Object.entries(MODS).map(([k, m]) => (
              <span key={k} style={{ display: "inline-flex", alignItems: "center", gap: 4, fontFamily: mono, fontSize: 9.5, color: k === mod ? m.color : T.dim }}>
                <span style={{ width: 8, height: 8, borderRadius: 2, background: m.color, opacity: k === mod ? 1 : 0.5 }} />{m.label}
              </span>
            ))}
            <button onClick={() => setShowAll(!showAll)} style={{ ...navBtn(false), height: 22, fontSize: 10, marginLeft: "auto" }}>
              callers: {showAll ? "full" : "trimmed"}
            </button>
            {peek.size > 0 && <button onClick={() => setPeek(new Set())} style={{ ...navBtn(false), height: 22, fontSize: 10 }}>close peeks</button>}
          </div>
          <FlowCanvas ev={ev} prev={prev} peek={peek} togglePeek={togglePeek} showAll={showAll} />
        </div>

        <div className="pane-right">
          <NowCard ev={ev} />
          {context}
          {startup && (
            <Section title="Toy config" color={color}>
              <div style={{ fontFamily: mono, fontSize: 11, color: T.soft, lineHeight: 1.7 }}>
                {cfg.layers} layers · H={4} Hkv={2} D={8} · hidden 32<br />
                {cfg.numBlocks} pages × {cfg.bs} slots (page 0 = null) · max_num_batched_tokens {cfg.budget} · max_num_seqs {cfg.maxSeqs}<br />
                {w.runner ? `CUDA graphs: ${w.runner.graphs.join(", ") || "none yet"}` : "ModelRunner not built yet"}
              </div>
            </Section>
          )}
          <Section title="Engine.step pipeline · CPU one step ahead of GPU" color={mod === "engine" ? color : undefined}><Pipeline w={w} /></Section>
          <Section title="Scheduler · queues and token budget" color={mod === "sched" ? color : undefined}>
            <QueueBoard w={w} hl={hl} color={color} />
            <div style={{ height: 8 }} />
            <BudgetBar w={w} plan={plan} budget={cfg.budget} />
          </Section>
          <Section title="Requests · logical blocks → pages" right={<TokenLegend />}><RequestRows w={w} hl={hl} color={color} bs={cfg.bs} /></Section>
          <Section title="KV pages · free list (LRU)" color={mod === "kv" ? color : undefined}>
            <PagePool w={w} hl={hl} color={color} />
            <FreeList w={w} hl={hl} color={color} />
          </Section>
          <Section title="hash_to_block · prefix cache"><HashTable w={w} hl={hl} hashInfo={hashInfo} /></Section>
          {mod !== "detok" && mod !== "async" && <Section title="Outputs"><Outputs w={w} /></Section>}
        </div>
      </div>
    </div>
  );
}
