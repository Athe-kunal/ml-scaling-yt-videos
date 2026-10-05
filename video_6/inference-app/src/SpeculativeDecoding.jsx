import { useState } from "react";
import { T, mono, sans } from "./theme";
import { withInlineMath, katexHtml } from "./latex";

// Interactive version of speculative decoding's validation step (Leviathan
// et al., "Fast Inference from Transformers via Speculative Decoding",
// arXiv:2302.01318): a cheap draft model proposes K tokens autoregressively,
// then ONE parallel forward pass of the target model scores all K+1
// positions at once. Tokens are accepted/rejected left to right by actual
// random sampling — the thing this page makes concrete — using the paper's
// rule: accept x_i iff u_i <= min(1, p_i(x_i)/q_i(x_i)); the first
// rejection stops the chain and resamples from the corrected residual
// distribution norm(max(0, p-q)); if every draft token survives, a bonus
// token is sampled straight from p for free.

const Math_ = ({ tex, display }) => <span dangerouslySetInnerHTML={{ __html: katexHtml(tex, !!display) }} />;
const Prose = ({ html, style }) => <div style={style} dangerouslySetInnerHTML={{ __html: withInlineMath(html) }} />;

const panel = { background: T.panel, border: `1px solid ${T.rule}`, borderRadius: 10, padding: 18 };
const h2 = { margin: "0 0 10px", fontFamily: sans, fontSize: 13, letterSpacing: 1.2, textTransform: "uppercase", color: T.dim, fontWeight: 600 };

function Slider({ label, value, onChange, min, max, step = 1, display, color = T.accent, small }) {
  return (
    <label style={{ display: "block", marginBottom: small ? 4 : 12 }}>
      <div style={{ display: "flex", justifyContent: "space-between", fontFamily: mono, fontSize: small ? 10 : 12, color: T.soft, marginBottom: 2 }}>
        <span>{label}</span>
        <span style={{ color: T.ink }}>{display ?? value}</span>
      </div>
      <input
        type="range" min={min} max={max} step={step} value={value}
        onChange={(e) => onChange(Number(e.target.value))}
        style={{ width: "100%", accentColor: color, height: small ? 14 : undefined }}
      />
    </label>
  );
}

function randIn(a, b) {
  return a + Math.random() * (b - a);
}
function clamp(x, a, b) {
  return Math.max(a, Math.min(b, x));
}

// Two illustrative presets for the (q, p) pair at each draft position: how
// likely the draft model was to propose this token (q) vs. how likely the
// target model thinks it actually is (p). Good overlap -> high acceptance.
function presetSlot(mode) {
  if (mode === "confident") {
    const q = randIn(0.55, 0.9);
    return { q, p: clamp(q + randIn(-0.1, 0.25), 0.15, 0.97) };
  }
  const q = randIn(0.45, 0.9); // "shaky": draft is sure of itself, target often disagrees
  return { q, p: clamp(q * randIn(0.2, 0.65), 0.05, 0.9) };
}
function makeSlots(k, mode) {
  return Array.from({ length: k }, () => presetSlot(mode));
}

function acceptRatio(s) {
  return Math.min(1, s.p / s.q);
}

function runTrial(slots) {
  const us = slots.map(() => Math.random());
  let firstReject = -1;
  for (let i = 0; i < slots.length; i++) {
    if (us[i] > acceptRatio(slots[i])) {
      firstReject = i;
      break;
    }
  }
  const allAccepted = firstReject === -1;
  const bonus = allAccepted ? { u: Math.random(), p: clamp(randIn(0.3, 0.9), 0.05, 0.97) } : null;
  return { us, firstReject, allAccepted, bonus };
}

function Bar({ frac, color, label }) {
  return (
    <div style={{ marginBottom: 3 }}>
      <div style={{ display: "flex", justifyContent: "space-between", fontFamily: mono, fontSize: 9.5, color: T.dim, marginBottom: 1 }}>
        <span>{label}</span>
        <span>{frac.toFixed(2)}</span>
      </div>
      <div style={{ height: 7, borderRadius: 4, background: T.well, border: `1px solid ${T.rule}`, overflow: "hidden" }}>
        <div style={{ width: `${frac * 100}%`, height: "100%", background: color, transition: "width .15s" }} />
      </div>
    </div>
  );
}

function slotState(i, trial) {
  if (!trial) return "pending";
  const { firstReject, allAccepted } = trial;
  if (allAccepted) return "accepted";
  if (i < firstReject) return "accepted";
  if (i === firstReject) return "rejected";
  return "skipped";
}

const STATE_STYLE = {
  pending: { border: T.rule, bg: T.well, label: T.dim },
  accepted: { border: T.accent, bg: `${T.accent}1a`, label: T.accent },
  rejected: { border: T.bad, bg: `${T.bad}1a`, label: T.bad },
  skipped: { border: T.rule, bg: "transparent", label: T.rule },
};

// Color the min(1, p/q) readout by how favorable it is: a low score means
// most random draws will land above it (likely rejected).
function scoreColor(ratio) {
  if (ratio >= 0.66) return T.accent;
  if (ratio >= 0.33) return T.wire;
  return T.bad;
}

function Slot({ i, slot, trial, onChange }) {
  const st = slotState(i, trial);
  const style = STATE_STYLE[st];
  const u = trial && st !== "skipped" ? trial.us[i] : null;
  const ratio = acceptRatio(slot);
  return (
    <div style={{ width: 140, flexShrink: 0, border: `1.5px solid ${style.border}`, background: style.bg, borderRadius: 9, padding: "9px 10px", opacity: st === "skipped" ? 0.4 : 1, transition: "all .2s" }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 6 }}>
        <span style={{ fontFamily: mono, fontSize: 11, color: T.soft }}>draft {i + 1}</span>
        {st !== "pending" && (
          <span style={{ fontFamily: mono, fontSize: 9.5, fontWeight: 700, color: style.label, textTransform: "uppercase" }}>
            {st === "skipped" ? "—" : st === "rejected" ? "✗ reject" : "✓ accept"}
          </span>
        )}
      </div>
      <Bar frac={slot.q} color={T.wire} label="q (draft)" />
      <Bar frac={slot.p} color={T.accent2} label="p (target)" />
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline", marginTop: 6, padding: "4px 7px", borderRadius: 5, background: T.well, border: `1px solid ${T.rule}` }}>
        <span style={{ fontFamily: mono, fontSize: 9, color: T.dim, textTransform: "uppercase", letterSpacing: "0.04em" }}>score min(1,p/q)</span>
        <span style={{ fontFamily: mono, fontSize: 13, fontWeight: 700, color: scoreColor(ratio) }}>{ratio.toFixed(2)}</span>
      </div>
      <div style={{ fontFamily: mono, fontSize: 10, color: T.dim, marginTop: 4 }}>
        accept if u ≤ {ratio.toFixed(2)}
        {u !== null && (
          <span style={{ color: style.label }}> · u={u.toFixed(2)}</span>
        )}
      </div>
      <div style={{ marginTop: 6 }}>
        <Slider label="q" value={Number(slot.q.toFixed(2))} min={0.05} max={0.97} step={0.01} onChange={(v) => onChange({ ...slot, q: v })} color={T.wire} small />
        <Slider label="p" value={Number(slot.p.toFixed(2))} min={0.05} max={0.97} step={0.01} onChange={(v) => onChange({ ...slot, p: v })} color={T.accent2} small />
      </div>
    </div>
  );
}

function BonusSlot({ trial }) {
  if (!trial || !trial.allAccepted) {
    return (
      <div style={{ width: 140, flexShrink: 0, border: `1.5px dashed ${T.rule}`, borderRadius: 9, padding: "9px 10px", display: "flex", alignItems: "center", justifyContent: "center", textAlign: "center", color: T.rule, fontFamily: mono, fontSize: 10 }}>
        bonus token (only if all K accepted)
      </div>
    );
  }
  return (
    <div style={{ width: 140, flexShrink: 0, border: `1.5px solid ${T.accent2}`, background: `${T.accent2}1a`, borderRadius: 9, padding: "9px 10px" }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 6 }}>
        <span style={{ fontFamily: mono, fontSize: 11, color: T.soft }}>bonus</span>
        <span style={{ fontFamily: mono, fontSize: 9.5, fontWeight: 700, color: T.accent2, textTransform: "uppercase" }}>free!</span>
      </div>
      <Bar frac={trial.bonus.p} color={T.accent2} label="p (target)" />
      <div style={{ fontFamily: mono, fontSize: 10, color: T.dim, marginTop: 4 }}>
        sampled directly from <Math_ tex="p_{K+1}" />
      </div>
    </div>
  );
}

export default function SpeculativeDecoding() {
  const [K, setK] = useState(5);
  const [slots, setSlots] = useState(() => makeSlots(5, "confident"));
  const [trial, setTrial] = useState(null);

  const setK_ = (k) => {
    setK(k);
    setSlots((prev) => {
      if (k <= prev.length) return prev.slice(0, k);
      return [...prev, ...makeSlots(k - prev.length, "confident")];
    });
    setTrial(null);
  };
  const regen = (mode) => {
    setSlots(makeSlots(K, mode));
    setTrial(null);
  };
  const updateSlot = (i, next) => {
    setSlots((prev) => prev.map((s, j) => (j === i ? next : s)));
    setTrial(null);
  };

  const alphaBar = slots.reduce((acc, s) => acc + acceptRatio(s), 0) / slots.length;
  const expectedTokens = (alpha, k) => (alpha >= 0.999 ? k + 1 : (1 - Math.pow(alpha, k + 1)) / (1 - alpha));
  const ET = expectedTokens(alphaBar, K);

  const acceptedCount = !trial ? null : trial.allAccepted ? K + 1 : trial.firstReject + 1; // +1: the resampled replacement for the rejected slot

  return (
    <div style={{ minHeight: "100vh", background: T.bg, color: T.ink, fontFamily: sans, padding: "28px clamp(16px, 3vw, 40px) 60px" }}>
      <header style={{ maxWidth: 1180, margin: "0 auto 22px" }}>
        <div style={{ fontFamily: mono, fontSize: 12, color: T.accent, letterSpacing: 1.4 }}>LEVIATHAN ET AL. 2023 · SPECULATIVE DECODING</div>
        <h1 style={{ margin: "6px 0 8px", fontSize: 28, fontWeight: 600 }}>Draft, score, and validate — one random trial at a time</h1>
        <Prose
          style={{ color: T.soft, maxWidth: 900, lineHeight: 1.55, fontSize: 15 }}
          html="A cheap draft model $q$ proposes $K$ tokens autoregressively; one parallel forward pass of the target model $p$ scores all $K+1$ positions at once. Walking left to right, draft token $i$ is accepted iff a fresh $u_i\sim\text{Uniform}(0,1)$ satisfies $u_i \le \min(1, p_i/q_i)$. The first rejection stops the chain — that token is replaced with a sample from the corrected residual $\\operatorname{norm}(\\max(0, p-q))$ — and everything downstream of it never happened. If every draft token survives, a bonus token is sampled straight from $p_{K+1}$, for free."
        />
      </header>

      <div style={{ maxWidth: 1180, margin: "0 auto", display: "grid", gap: 20 }}>
        <section style={panel}>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-end", flexWrap: "wrap", gap: 14, marginBottom: 16 }}>
            <div style={{ flex: "1 1 220px", maxWidth: 320 }}>
              <Slider label="K (draft tokens)" value={K} min={2} max={8} step={1} onChange={setK_} />
            </div>
            <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
              <button onClick={() => regen("confident")} style={btnStyle(T.accent)}>confident draft</button>
              <button onClick={() => regen("shaky")} style={btnStyle(T.wire)}>shaky draft</button>
              <button onClick={() => setTrial(runTrial(slots))} style={{ ...btnStyle(T.accent2), fontWeight: 700 }}>draw a trial →</button>
            </div>
          </div>

          <div style={{ fontFamily: mono, fontSize: 11, letterSpacing: 1.2, color: T.dim, textTransform: "uppercase", marginBottom: 10 }}>
            Draft tokens, scored in one target-model pass
          </div>
          <div style={{ display: "flex", gap: 10, overflowX: "auto", paddingBottom: 6 }}>
            {slots.map((s, i) => (
              <Slot key={i} i={i} slot={s} trial={trial} onChange={(next) => updateSlot(i, next)} />
            ))}
            <BonusSlot trial={trial} />
          </div>

          <div style={{ marginTop: 16, borderTop: `1px solid ${T.rule}`, paddingTop: 14 }}>
            {!trial ? (
              <div style={{ color: T.dim, fontSize: 13.5 }}>Draw a trial to sample $u_i$ for each slot and see how far the draft survives.</div>
            ) : trial.allAccepted ? (
              <Prose
                style={{ color: T.soft, fontSize: 13.5, lineHeight: 1.6 }}
                html={`All $K=${K}$ draft tokens were accepted — a lucky trial. Plus the free bonus token, this step emitted <b style="color:${T.accent}">${K + 1} tokens</b> from a single target forward pass.`}
              />
            ) : (
              <Prose
                style={{ color: T.soft, fontSize: 13.5, lineHeight: 1.6 }}
                html={`Draft token ${trial.firstReject + 1} was rejected ($u=${trial.us[trial.firstReject].toFixed(2)} > ${acceptRatio(slots[trial.firstReject]).toFixed(2)} = \\min(1,p/q)$). Everything after it is discarded, and it's replaced with one sample from $\\operatorname{norm}(\\max(0,p-q))$ instead. This step emitted <b style="color:${T.accent}">${acceptedCount} token${acceptedCount === 1 ? "" : "s"}</b> from a single target forward pass.`}
              />
            )}
          </div>
        </section>

        <section style={panel}>
          <div style={h2}>Expected tokens per iteration</div>
          <Prose
            style={{ color: T.soft, fontSize: 14, lineHeight: 1.6, marginBottom: 10 }}
            html="Averaging the current slots' accept probabilities gives a rough per-token acceptance rate $\bar\alpha$. Treating every slot as if it had that same rate, the paper's closed form for expected tokens emitted per target forward pass is:"
          />
          <Math_ display tex={"E[\\text{tokens}] = \\frac{1-\\bar\\alpha^{K+1}}{1-\\bar\\alpha}"} />
          <div style={{ display: "flex", gap: 24, flexWrap: "wrap", marginTop: 14 }}>
            <Stat label="ᾱ (mean accept rate)" value={alphaBar.toFixed(3)} color={T.wire} />
            <Stat label="E[tokens] per step" value={ET.toFixed(2)} color={T.accent} />
            <Stat label="vs. 1 token/step (no spec.)" value={`${ET.toFixed(2)}×`} color={T.accent2} />
          </div>
          <Prose
            style={{ color: T.dim, fontSize: 12.5, lineHeight: 1.6, marginTop: 12 }}
            html="This is the speedup ceiling <i>before</i> accounting for the draft model's own cost — the paper's actual speedup factor divides this by $(1 + Kc)$, where $c$ is the draft step's cost relative to one target step. A cheap-but-accurate draft model is the whole game: push $\bar\alpha$ up with the 'confident draft' preset above and watch $E[\\text{tokens}]$ climb toward $K+1$."
          />
        </section>
      </div>
    </div>
  );
}

function btnStyle(color) {
  return {
    fontFamily: mono, fontSize: 12, padding: "8px 14px", borderRadius: 7, cursor: "pointer",
    background: `${color}14`, color, border: `1px solid ${color}66`,
  };
}

function Stat({ label, value, color }) {
  return (
    <div>
      <div style={{ fontSize: 10.5, color: T.dim, fontFamily: mono, marginBottom: 4, textTransform: "uppercase" }}>{label}</div>
      <div style={{ fontSize: 24, color, fontFamily: mono, lineHeight: 1.1 }}>{value}</div>
    </div>
  );
}
