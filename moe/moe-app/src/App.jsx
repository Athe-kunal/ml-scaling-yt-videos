import { useState } from "react";
import { T, mono } from "./theme";
import DenseToMoEView from "./DenseToMoEView";
import GatingView from "./GatingView";
import CapacityView from "./CapacityView";
import A2AView from "./A2AView";
import EplbView from "./EplbView";
import DeepSeekView from "./DeepSeekView";
import KimiView from "./KimiView";
import CompareView from "./CompareView";
import RouterZLossView from "./RouterZLossView";
import GatingFunctionsView from "./GatingFunctionsView";

const VIEWS = [
  { id: "dense", label: "Dense → MoE", sub: "set sparsity, watch the FFN split into experts", C: DenseToMoEView },
  { id: "gating", label: "MoE layer", sub: "top-2 gating · balance · sharding", C: GatingView },
  { id: "capacity", label: "Capacity & ℓ_aux", sub: "factor · overflow · CV · the aux-loss sum", C: CapacityView },
  { id: "a2a", label: "All-to-all", sub: "the collective behind dispatch / combine", C: A2AView },
  { id: "gatefn", label: "Gating functions", sub: "softmax squeezes · sigmoid stays independent", C: GatingFunctionsView },
  { id: "eplb", label: "EPLB placement", sub: "where experts live · hot ones get copies", C: EplbView },
  { id: "ds", label: "DeepSeek-V3 balancing", sub: "aux loss → loss-free bias", C: DeepSeekView },
  { id: "kimi", label: "Kimi K3 balancing", sub: "quantile-based bias", C: KimiView },
  { id: "compare", label: "DeepSeek vs Kimi", sub: "the two balancing methods side by side", C: CompareView },
  { id: "zloss", label: "Router z-loss", sub: "keep router logits small and bf16-safe", C: RouterZLossView },
];

export default function App() {
  const [view, setView] = useState("dense");
  const { C } = VIEWS.find((v) => v.id === view);
  return (
    <div style={{ minHeight: "100vh", background: T.bg, color: T.ink, fontFamily: mono }}>
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
              <div style={{ fontSize: 9.5, marginTop: 2, color: active ? T.accent : T.dim, opacity: 0.85 }}>{v.sub}</div>
            </button>
          );
        })}
      </nav>
      <C key={view} />
    </div>
  );
}
