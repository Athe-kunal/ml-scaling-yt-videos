import { useMemo, useState } from "react";
import { T, mono, sans } from "../theme";
import { panel, h2, navBtn } from "../styles";
import { DEFAULT_SIM, runSim } from "../sim/scheduler";
import { KV_SRC, SCHED_SRC } from "../codeSources";
import { CodeBlock } from "../CodePanel";
import { BudgetBar, Frames, FreeList, HashTable, PagePool, QueueBoard, RequestRows } from "../components/Viz";
import { Choice, NumCtl, StepperControls, useStepper } from "../components/Stepper";

export default function SchedView() {
  const [cfg, setCfg] = useState(DEFAULT_SIM);
  const sim = useMemo(() => runSim(cfg), [cfg]);
  return <SchedRun key={JSON.stringify(cfg)} cfg={cfg} setCfg={setCfg} sim={sim} />;
}

function SchedRun({ cfg, setCfg, sim }) {
  const { events, hashInfo } = sim;
  const st = useStepper(events);
  const ev = events[st.i];
  const prev = events[st.i - 1];
  const set = (k) => (v) => setCfg({ ...cfg, [k]: v });
  const isDefault = JSON.stringify(cfg) === JSON.stringify(DEFAULT_SIM);

  return (
    <div style={{ maxWidth: 1560, margin: "0 auto", padding: "22px clamp(16px, 3vw, 36px) 60px", color: T.ink, fontFamily: sans }}>
      <header style={{ marginBottom: 14 }}>
        <div style={{ fontFamily: mono, fontSize: 12, color: T.accent, letterSpacing: 1.4 }}>TINYSERVE · SCHEDULER</div>
        <h1 style={{ margin: "6px 0 0", fontSize: 26, fontWeight: 600 }}>Continuous batching over a paged KV cache</h1>
      </header>

      <div style={{ ...panel, display: "grid", gap: 12, marginBottom: 18, padding: "12px 16px" }}>
        <div style={{ display: "flex", flexWrap: "wrap", gap: 18, alignItems: "center" }}>
          <Choice label="block_size" value={cfg.bs} options={[2, 4, 8]} onChange={set("bs")} />
          <NumCtl label="num_blocks" value={cfg.numBlocks} min={3} max={16} onChange={set("numBlocks")} />
          <NumCtl label="max_num_batched_tokens" value={cfg.budget} min={1} max={16} onChange={set("budget")} />
          <NumCtl label="max_num_seqs" value={cfg.maxSeqs} min={1} max={4} onChange={set("maxSeqs")} />
          {!isDefault && <button onClick={() => setCfg(DEFAULT_SIM)} style={{ ...navBtn(false), height: 26 }}>defaults</button>}
        </div>
        <StepperControls events={events} st={st} groupLabel="step" />
      </div>

      <div className="ra-grid">
        <main style={{ display: "grid", gap: 18, minWidth: 0 }}>
          <div className="ra-two">
            <section style={panel}>
              <div style={h2}>Queues · engine step {ev.snap.step < 0 ? "–" : ev.snap.step}</div>
              <QueueBoard ev={ev} />
              <div style={{ height: 12 }} />
              <BudgetBar ev={ev} budget={cfg.budget} />
            </section>
            <section style={panel}>
              <div style={h2}>Variables · call stack</div>
              <Frames ev={ev} prev={prev} />
            </section>
          </div>

          <section style={panel}>
            <div style={{ ...h2, display: "flex", justifyContent: "space-between", flexWrap: "wrap", gap: 8 }}>
              <span>Requests · logical blocks → pages</span>
              <Legend />
            </div>
            <RequestRows ev={ev} bs={cfg.bs} visible={(r) => r.arrival <= Math.max(ev.snap.step, -1) || ev.snap.waiting.includes(r.id)} />
          </section>

          <section style={panel}>
            <div style={h2}>Physical KV pages</div>
            <PagePool ev={ev} />
            <div style={{ height: 12 }} />
            <FreeList ev={ev} />
          </section>

          <section style={panel}>
            <div style={h2}>hash_to_block · prefix cache</div>
            <HashTable ev={ev} hashInfo={hashInfo} />
          </section>
        </main>

        <aside className="ra-side">
          <CodeBlock title="Scheduler · Engine.step" lines={SCHED_SRC} focus={ev.focus.sched || {}} notes={ev.notes.sched}
            maxHeight="50vh" />
          <CodeBlock title="KVCacheManager" lines={KV_SRC} focus={ev.focus.kv || {}} notes={ev.notes.kv}
            maxHeight="40vh" />
        </aside>
      </div>
    </div>
  );
}

export function Legend() {
  const item = (label, style) => (
    <span style={{ display: "inline-flex", alignItems: "center", gap: 5, textTransform: "none", letterSpacing: 0, fontWeight: 400 }}>
      <span style={{ width: 12, height: 12, borderRadius: 3, ...style }} />
      <span style={{ fontFamily: mono, fontSize: 10, color: T.dim }}>{label}</span>
    </span>
  );
  return (
    <span style={{ display: "inline-flex", gap: 12, flexWrap: "wrap" }}>
      {item("K/V written", { background: "#5FD3F338", border: "1px solid #5FD3F3" })}
      {item("from cache", { background: "#A5D6A730", border: "1px solid #A5D6A7" })}
      {item("this step", { border: "1.5px dashed #5FD3F3" })}
      {item("no K/V", { border: `1px solid ${T.rule}` })}
      {item("num_computed", { width: 3, background: T.accent })}
    </span>
  );
}
