import { useCallback, useEffect, useMemo, useState } from "react";
import { T, mono, sans, cond } from "./lib/theme";
import { TOPICS, DEFAULT_TOPIC_ID, resolveTopicId } from "./lib/topics";
import DeviceDiagram from "./components/DeviceDiagram";
import FSDPDiagram from "./components/FSDPDiagram";
import MatrixShapes from "./components/MatrixShapes";
import StepPanel from "./components/StepPanel";

const ENV_TOPIC = (typeof import.meta !== "undefined" && import.meta.env && import.meta.env.VITE_TOPIC) || DEFAULT_TOPIC_ID;

function useArrowKeys(setStep, total) {
  const onKey = useCallback(
    (e) => {
      if (e.target.tagName === "INPUT") return;
      if (e.key === "ArrowRight") setStep((s) => Math.min(s + 1, total - 1));
      if (e.key === "ArrowLeft") setStep((s) => Math.max(s - 1, 0));
    },
    [setStep, total]
  );
  useEffect(() => {
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onKey]);
}

function DiagramFor({ diagram }) {
  if (diagram.kind === "fsdp" || diagram.kind === "tp" || diagram.kind === "fsdp_tp") return <FSDPDiagram diagram={diagram} />;
  return <DeviceDiagram diagram={diagram} />;
}

export default function App() {
  const [topicId, setTopicId] = useState(resolveTopicId(ENV_TOPIC));
  const [step, setStep] = useState(0);

  const topic = useMemo(() => TOPICS.find((t) => t.id === topicId) || TOPICS[0], [topicId]);
  const clampedStep = Math.min(step, topic.steps.length - 1);
  const currentStep = topic.steps[clampedStep];
  useArrowKeys(setStep, topic.steps.length);

  function selectTopic(id) {
    setTopicId(id);
    setStep(0);
  }

  function jumpToSection(sectionId) {
    const idx = topic.steps.findIndex((s) => s.section === sectionId);
    if (idx >= 0) setStep(idx);
  }

  return (
    <div style={{ minHeight: "100vh", background: T.bg, color: T.ink, fontFamily: sans }}>
      <div style={{ maxWidth: 1000, margin: "0 auto", padding: "28px 20px 60px" }}>
        <header style={{ marginBottom: 22 }}>
          <div style={{ fontFamily: mono, fontSize: 10.5, letterSpacing: "0.16em", color: T.dim, textTransform: "uppercase", marginBottom: 6 }}>
            Scaling ML Models
          </div>
          <h1 style={{ fontFamily: cond, fontSize: 30, fontWeight: 700, margin: 0, color: T.ink }}>
            Parallelism Visualizer
          </h1>
        </header>

        <nav style={{ display: "flex", gap: 6, flexWrap: "wrap", marginBottom: 18 }}>
          {TOPICS.map((t) => {
            const active = t.id === topic.id;
            return (
              <button
                key={t.id}
                onClick={() => selectTopic(t.id)}
                style={{
                  fontFamily: mono,
                  fontSize: 12,
                  padding: "8px 14px",
                  borderRadius: 8,
                  border: `1px solid ${active ? T.accent : T.rule}`,
                  background: active ? `${T.accent}18` : T.panel,
                  color: active ? T.accent : T.soft,
                  cursor: "pointer",
                  transition: "all .16s",
                }}
              >
                <div style={{ fontWeight: 700 }}>{t.label}</div>
                <div style={{ fontSize: 9.5, opacity: 0.75, marginTop: 2, color: active ? T.accent : T.dim }}>{t.sub}</div>
              </button>
            );
          })}
        </nav>

        {topic.sections && (
          <div style={{ display: "flex", gap: 5, flexWrap: "wrap", marginBottom: 18 }}>
            {topic.sections.map((sec) => {
              const active = currentStep.section === sec.id;
              return (
                <button
                  key={sec.id}
                  onClick={() => jumpToSection(sec.id)}
                  style={{
                    fontFamily: mono,
                    fontSize: 10.5,
                    padding: "5px 10px",
                    borderRadius: 6,
                    border: `1px solid ${active ? T.wire : T.rule}`,
                    background: active ? `${T.wire}18` : "transparent",
                    color: active ? T.wire : T.dim,
                    cursor: "pointer",
                    transition: "all .16s",
                  }}
                >
                  {sec.label}
                </button>
              );
            })}
          </div>
        )}

        <div style={{ display: "grid", gridTemplateColumns: "1fr", gap: 18 }}>
          <DiagramFor diagram={currentStep.diagram} />
          {currentStep.matrices && <MatrixShapes items={currentStep.matrices} />}
          <StepPanel steps={topic.steps} step={clampedStep} onStep={setStep} />
        </div>
      </div>
    </div>
  );
}
