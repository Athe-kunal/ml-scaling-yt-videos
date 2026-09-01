import { T, mono, cond } from "../lib/theme";
import { TOPICS } from "../lib/topics";

export default function TopicSwitcher({ topicId, onSelect }) {
  return (
    <nav
      style={{
        display: "grid",
        gap: 8,
        marginBottom: 16,
        gridTemplateColumns: "repeat(auto-fit, minmax(180px, 1fr))",
      }}
    >
      {TOPICS.map((t) => {
        const on = t.id === topicId;
        return (
          <button
            key={t.id}
            onClick={() => onSelect(t.id)}
            style={{
              textAlign: "left",
              padding: "11px 13px",
              borderRadius: 9,
              cursor: "pointer",
              border: `1px solid ${on ? T.accent : T.rule}`,
              background: on ? "rgba(94,234,212,0.08)" : T.panel,
              transition: "all .18s cubic-bezier(.2,.7,.3,1)",
            }}
          >
            <div
              style={{
                fontFamily: mono,
                fontSize: 10,
                letterSpacing: "0.14em",
                color: on ? T.accent : T.dim,
                marginBottom: 4,
                textTransform: "uppercase",
              }}
            >
              {t.sub}
            </div>
            <div style={{ fontFamily: cond, fontWeight: 600, fontSize: 15, color: on ? T.ink : T.soft }}>{t.label}</div>
          </button>
        );
      })}
    </nav>
  );
}
