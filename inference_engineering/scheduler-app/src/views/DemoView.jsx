import { useMemo, useState } from "react";
import { T, mono, sans } from "../theme";
import { panel, h2 } from "../styles";
import { runDemo, TOKEN_LEGEND } from "../sim/demo";
import { KV_SRC, demoSrc } from "../codeSources";
import { CodeBlock } from "../CodePanel";
import { Frames, FreeList, HashTable, PagePool, RequestRows } from "../components/Viz";
import { Choice, StepperControls, useStepper } from "../components/Stepper";
import { Legend } from "./SchedView";

const SECTIONS = ["setup", "1 chunked prefill", "2 decode", "3 shared block", "4 num_tokens - 1", "5 free A", "6 eviction", "7 no pages"];

export default function DemoView() {
  const [bs, setBs] = useState(4);
  const sim = useMemo(() => runDemo(bs), [bs]);
  return <DemoRun key={bs} bs={bs} setBs={setBs} sim={sim} />;
}

function DemoRun({ bs, setBs, sim }) {
  const events = useMemo(() => sim.events.map((e) => ({ ...e, group: SECTIONS[e.group] })), [sim]);
  const st = useStepper(events);
  const ev = events[st.i];
  const prev = events[st.i - 1];
  const src = useMemo(() => demoSrc(bs), [bs]);

  return (
    <div style={{ maxWidth: 1560, margin: "0 auto", padding: "22px clamp(16px, 3vw, 36px) 60px", color: T.ink, fontFamily: sans }}>
      <header style={{ marginBottom: 14 }}>
        <div style={{ fontFamily: mono, fontSize: 12, color: T.accent, letterSpacing: 1.4 }}>TINYSERVE · kv_cache_demo()</div>
        <h1 style={{ margin: "6px 0 0", fontSize: 26, fontWeight: 600 }}>KVCacheManager: pages, ref counts, hashes, LRU</h1>
      </header>

      <div style={{ ...panel, display: "grid", gap: 12, marginBottom: 18, padding: "12px 16px" }}>
        <div style={{ display: "flex", flexWrap: "wrap", gap: 18, alignItems: "center" }}>
          <Choice label="B" value={bs} options={[2, 4, 8]} onChange={setBs} />
          <span style={{ display: "inline-flex", gap: 10, flexWrap: "wrap" }}>
            {TOKEN_LEGEND.map(([ids, l]) => (
              <span key={ids} style={{ fontFamily: mono, fontSize: 11, color: T.dim }}>{ids} <span style={{ color: T.soft }}>→ {l}</span></span>
            ))}
          </span>
        </div>
        <StepperControls events={events} st={st} groupLabel="section" />
      </div>

      <div className="ra-grid">
        <main style={{ display: "grid", gap: 18, minWidth: 0 }}>
          <section style={panel}>
            <div style={{ ...h2, display: "flex", justifyContent: "space-between", flexWrap: "wrap", gap: 8 }}>
              <span>Requests · logical blocks → pages</span>
              <Legend />
            </div>
            <RequestRows ev={ev} bs={bs} />
          </section>

          <div className="ra-two">
            <section style={panel}>
              <div style={h2}>Physical KV pages</div>
              <PagePool ev={ev} />
              <div style={{ height: 12 }} />
              <FreeList ev={ev} />
            </section>
            <section style={panel}>
              <div style={h2}>Variables · call stack</div>
              <Frames ev={ev} prev={prev} />
            </section>
          </div>

          <section style={panel}>
            <div style={h2}>hash_to_block · prefix cache</div>
            <HashTable ev={ev} hashInfo={sim.hashInfo} />
          </section>
        </main>

        <aside className="ra-side">
          <CodeBlock title="kv_cache_demo()" lines={src} focus={ev.focus.demo || {}} notes={ev.notes.demo}
            maxHeight="42vh" />
          <CodeBlock title="KVCacheManager" lines={KV_SRC} focus={ev.focus.kv || {}} notes={ev.notes.kv}
            maxHeight="48vh" />
        </aside>
      </div>
    </div>
  );
}
