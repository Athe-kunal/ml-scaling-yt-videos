import { useState } from "react";
import { T, mono } from "./theme";
import ViTView from "./ViTView";
import ClipView from "./ClipView";
import SizesView from "./SizesView";
import SiglipView from "./SiglipView";
import VlmView from "./VlmView";

const VIEWS = [
  { id: "vit", label: "Vision Transformer", sub: "patches · CLS · encoder blocks · head", C: ViTView },
  { id: "names", label: "Model names", sub: "what B/32, B/16, L/14 mean", C: SizesView },
  { id: "clip", label: "CLIP", sub: "two encoders · contrastive loss · zero-shot", C: ClipView },
  { id: "siglip", label: "SigLIP", sub: "sigmoid loss · ring · SigLIP 2", C: SiglipView },
  { id: "vlm", label: "Vision → LLM", sub: "LLaVA · Flamingo · Q-Former · DeepStack", C: VlmView },
];

export default function App() {
  const [view, setView] = useState(() => (["#clip", "#siglip", "#names", "#vlm"].includes(location.hash) ? location.hash.slice(1) : "vit"));
  const { C } = VIEWS.find((v) => v.id === view);
  const pick = (id) => { setView(id); history.replaceState(null, "", `#${id}`); };
  return (
    <div style={{ minHeight: "100vh", background: T.bg, color: T.ink, fontFamily: mono }}>
      <nav style={{ display: "flex", gap: 6, flexWrap: "wrap", padding: "16px clamp(16px, 3vw, 36px) 0", maxWidth: 1480, margin: "0 auto" }}>
        {VIEWS.map((v) => {
          const active = v.id === view;
          return (
            <button key={v.id} onClick={() => pick(v.id)}
              style={{ fontFamily: mono, fontSize: 15, padding: "8px 14px", borderRadius: 8, textAlign: "left", cursor: "pointer",
                border: `1px solid ${active ? T.accent : T.rule}`, background: active ? `${T.accent}18` : T.panel, color: active ? T.accent : T.soft }}>
              <div style={{ fontWeight: 700 }}>{v.label}</div>
              <div style={{ fontSize: 12, marginTop: 2, color: active ? T.accent : T.dim, opacity: 0.85 }}>{v.sub}</div>
            </button>
          );
        })}
      </nav>
      <C key={view} />
    </div>
  );
}
