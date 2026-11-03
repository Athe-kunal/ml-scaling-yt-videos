import { useMemo, useState } from "react";
import { T, btn } from "./theme";
import { CFG, EXPERT_COLORS as EC, MODEL } from "./moeModel";
import { FLOW } from "./steps";
import { buildFlat } from "./gshardFlat";
import Stepper from "./Stepper";
import GroupScene from "./GroupScene";
import DevicesScene from "./DevicesScene";
import TensorView from "./TensorView";
import { GLOSSARY, shapesOf } from "./gshardShapes";

export default function GatingView() {
  const [view, setView] = useState("token");
  const [g, setG] = useState(0);
  const gm = MODEL.groups[g];
  const flat = useMemo(() => buildFlat(gm), [gm]);

  return (
    <Stepper
      title="GShard MoE · top-2 gating"
      subtitle={`G=${CFG.G} groups · S=${CFG.S} tokens · E=${CFG.E} experts · C=${CFG.C}`}
      blocks={FLOW}
      flat={flat}
      header={() => (
        <>
          <div style={{ display: "flex", gap: 4, alignItems: "center" }}>
            <span style={{ fontSize: 10.5, color: T.dim, marginRight: 4 }}>follow group</span>
            {Array.from({ length: CFG.G }, (_, k) => (
              <button key={k} onClick={() => setG(k)} style={{ ...btn(k === g, false), height: 28, padding: "0 10px" }}>{k}</button>
            ))}
          </div>
          <div style={{ display: "flex", gap: 4 }}>
            <button onClick={() => setView("token")} style={btn(view === "token", false)}>tokens</button>
            <button onClick={() => setView("tensor")} style={btn(view === "tensor", false)}>tensors</button>
          </div>
        </>
      )}
      scene={(item) =>
        view === "tensor" ? <TensorView step={item} gm={gm} />
          : item.scene === "group" ? <GroupScene gm={gm} stage={item.stage} loop={item.loop} />
          : <DevicesScene stage={item.stage} sel={g} />
      }
      glossary={GLOSSARY}
      shapes={shapesOf}
      legend={(item) =>
        item.scene === "devices" && view === "token" ? (
          <>
            <span><b style={{ color: T.soft }}>●</b> 1st choice ticket</span>
            <span><b style={{ color: T.soft }}>○</b> 2nd choice ticket</span>
            {EC.map((c, e) => <span key={e} style={{ color: c }}>■ E{e}</span>)}
            <span style={{ color: T.wire }}>━ lanes of followed group {g}</span>
          </>
        ) : null
      }
    />
  );
}
