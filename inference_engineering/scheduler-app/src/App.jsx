import { useState } from "react";
import { T, mono } from "./theme";
import SchedView from "./views/SchedView";
import DemoView from "./views/DemoView";

const VIEWS = [
  { id: "demo", label: "kv_cache_demo()", sub: "KVCacheManager · 7 sections" },
  { id: "sched", label: "Scheduler", sub: "4 requests · chunking · prefix hits · preemption" },
];

export default function App() {
  const [view, setView] = useState("demo");
  return (
    <div style={{ minHeight: "100vh", background: T.bg }}>
      <nav style={{ display: "flex", gap: 6, flexWrap: "wrap", padding: "16px clamp(16px, 3vw, 36px) 0", maxWidth: 1560, margin: "0 auto" }}>
        {VIEWS.map((v) => {
          const active = v.id === view;
          return (
            <button
              key={v.id} onClick={() => setView(v.id)}
              style={{ fontFamily: mono, fontSize: 12, padding: "8px 14px", borderRadius: 8, textAlign: "left", cursor: "pointer",
                border: `1px solid ${active ? T.accent : T.rule}`, background: active ? `${T.accent}18` : T.panel, color: active ? T.accent : T.soft }}
            >
              <div style={{ fontWeight: 700 }}>{v.label}</div>
              <div style={{ fontSize: 9.5, opacity: 0.75, marginTop: 2, color: active ? T.accent : T.dim }}>{v.sub}</div>
            </button>
          );
        })}
      </nav>
      {view === "demo" ? <DemoView /> : <SchedView />}
    </div>
  );
}
