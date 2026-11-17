import { useState } from "react";
import { T, mono } from "./theme";
import ViTView from "./ViTView";
import ClipView from "./ClipView";

const VIEWS = [
  { id: "vit", label: "Vision Transformer", sub: "patches · CLS · encoder blocks · head", C: ViTView },
  { id: "clip", label: "CLIP", sub: "two encoders · contrastive loss · zero-shot", C: ClipView },
];

export default function App() {
  const [view, setView] = useState(() => (location.hash === "#clip" ? "clip" : "vit"));
  const { C } = VIEWS.find((v) => v.id === view);
  const pick = (id) => { setView(id); history.replaceState(null, "", `#${id}`); };
  return (
    <div style={{ minHeight: "100vh", background: T.bg, color: T.ink, fontFamily: mono }}>
      <nav style={{ display: "flex", gap: 6, flexWrap: "wrap", padding: "16px clamp(16px, 3vw, 36px) 0", maxWidth: 1480, margin: "0 auto" }}>
        {VIEWS.map((v) => {
          const active = v.id === view;
          return (
            <button key={v.id} onClick={() => pick(v.id)}
              style={{ fontFamily: mono, fontSize: 12, padding: "8px 14px", borderRadius: 8, textAlign: "left", cursor: "pointer",
                border: `1px solid ${active ? T.accent : T.rule}`, background: active ? `${T.accent}18` : T.panel, color: active ? T.accent : T.soft }}>
              <div style={{ fontWeight: 700 }}>{v.label}</div>
              <div style={{ fontSize: 9.5, marginTop: 2, color: active ? T.accent : T.dim, opacity: 0.85 }}>{v.sub}</div>
            </button>
          );
        })}
      </nav>
      <C key={view} />
    </div>
  );
}
