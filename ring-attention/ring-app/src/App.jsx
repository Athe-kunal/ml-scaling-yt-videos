import { useState } from "react";
import { T, mono } from "./theme";
import RingView from "./RingView";

const VIEWS = [
  { id: "ring", label: "Ring attention", sub: "contiguous blocks · causal imbalance" },
  { id: "zigzag", label: "Zig-zag ring attention", sub: "chunks r and 2N-1-r · balanced" },
];

export default function App() {
  const [view, setView] = useState("ring");
  return (
    <div style={{ minHeight: "100vh", background: T.bg }}>
      <nav style={{ display: "flex", gap: 6, flexWrap: "wrap", padding: "16px clamp(16px, 3vw, 36px) 0", maxWidth: 1480, margin: "0 auto" }}>
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
      <RingView key={view} variant={view} />
    </div>
  );
}
