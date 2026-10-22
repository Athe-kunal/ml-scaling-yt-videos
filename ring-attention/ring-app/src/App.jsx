import { useState } from "react";
import { T, mono } from "./theme";
import RingView from "./RingView";
import TreeView from "./TreeView";
import TreeFirstPrinciples from "./TreeFirstPrinciples";
import CompareView from "./CompareView";
import ShardingView from "./ShardingView";
import DistributeView from "./DistributeView";

const VIEWS = [
  { id: "dist", label: "Q, K, V → GPUs", sub: "animated layouts · 3 causal matrices" },
  { id: "shard", label: "Sharding", sub: "who gets which tokens · extract_local" },
  { id: "ring", label: "Ring attention", sub: "contiguous blocks · causal imbalance" },
  { id: "zigzag", label: "Zig-zag ring attention", sub: "chunks r and 2N-1-r · balanced" },
  { id: "stripe", label: "Stripe attention", sub: "token t on GPU t mod N · shifted mask" },
  { id: "compare", label: "Zig-zag vs Stripe", sub: "one stepper · both side by side" },
  { id: "treefp", label: "Tree attention · first principles", sub: "7 steps from the paper" },
  { id: "tree", label: "Tree attention", sub: "decoding · allreduce in log N" },
];

export default function App() {
  const [view, setView] = useState("dist");
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
      {view === "treefp" ? <TreeFirstPrinciples /> : view === "dist" ? <DistributeView /> : view === "shard" ? <ShardingView /> : view === "tree" ? <TreeView /> : view === "compare" ? <CompareView /> : <RingView key={view} variant={view} />}
    </div>
  );
}
