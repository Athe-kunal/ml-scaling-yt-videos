import { useMemo, useState } from "react";
import { T, mono, btn } from "./theme";
import { SIGNALS, toInt16, fromInt16, padOrTrim, logMel, recordMic, play } from "./dsp";
import { SIZES, toyStem } from "./model";
import AudioView from "./AudioView";
import EncoderView from "./EncoderView";
import DecoderView from "./DecoderView";
import DecodingView from "./DecodingView";
import TranscribeView from "./TranscribeView";
import QwenAudioView from "./QwenAudioView";

const VIEWS = [
  { id: "audio", label: "audio.py", sub: "waveform → log-mel [M, 3000]", C: AudioView },
  { id: "encoder", label: "AudioEncoder", sub: "2 convs · sinusoids · blocks → [1500, D]", C: EncoderView },
  { id: "decoder", label: "TextDecoder", sub: "tokens · causal · cross-attn · logits", C: DecoderView },
  { id: "decoding", label: "decoding.py", sub: "language · KV cache · timestamp rules", C: DecodingView },
  { id: "transcribe", label: "transcribe.py", sub: "30 s windows · seek · fallback", C: TranscribeView },
  { id: "qwen", label: "Qwen2-Audio", sub: "encoder → pool → Linear → LLM", C: QwenAudioView },
];

// the shared audio: one signal, one model size, computed once for every tab
function useAudio() {
  const [name, setName] = useState("melody");
  const [mic, setMic] = useState(null);
  const [size, setSize] = useState("tiny");
  const [recording, setRecording] = useState(false);
  const nMels = SIZES[size].mels;
  const raw = useMemo(() => (name === "mic" && mic ? mic : SIGNALS[name === "mic" ? "melody" : name]()), [name, mic]);
  const q = useMemo(() => toInt16(raw), [raw]);
  const y = useMemo(() => fromInt16(q), [q]);
  const y30 = useMemo(() => padOrTrim(y), [y]);
  const lm = useMemo(() => logMel(y30, nMels), [y30, nMels]);
  const stem = useMemo(() => toyStem(lm.scaled, nMels), [lm, nMels]);
  const rec = async () => {
    setRecording(true);
    try { setMic(await recordMic(3)); setName("mic"); } catch (e) { console.warn(e); }
    setRecording(false);
  };
  return { name, setName, size, setSize, nMels, q, y, y30, lm, stem, rec, recording, hasMic: !!mic, m: SIZES[size] };
}

function Controls({ A }) {
  return (
    <div style={{ display: "flex", gap: 14, flexWrap: "wrap", alignItems: "flex-end", padding: "10px clamp(16px, 3vw, 36px) 0", maxWidth: 1480, margin: "0 auto" }}>
      <div style={{ display: "grid", gap: 4 }}>
        <span style={{ fontSize: 13, color: T.dim, letterSpacing: "0.08em" }}>AUDIO</span>
        <div style={{ display: "flex", gap: 4 }}>
          {["melody", "voice", "chirp"].map((s) => <button key={s} onClick={() => A.setName(s)} style={{ ...btn(A.name === s), height: 30, fontSize: 15 }}>{s}</button>)}
          <button onClick={A.rec} disabled={A.recording} style={{ ...btn(A.name === "mic"), height: 30, fontSize: 15, color: A.recording ? T.bad : undefined }}>
            {A.recording ? "● recording 3 s" : A.hasMic ? "mic ↻" : "● mic"}
          </button>
          <button onClick={() => play(A.y)} style={{ ...btn(false), height: 30, fontSize: 15 }}>▶ hear</button>
        </div>
      </div>
      <div style={{ display: "grid", gap: 4 }}>
        <span style={{ fontSize: 13, color: T.dim, letterSpacing: "0.08em" }}>MODEL SIZE</span>
        <div style={{ display: "flex", gap: 4, flexWrap: "wrap" }}>
          {Object.keys(SIZES).map((s) => <button key={s} onClick={() => A.setSize(s)} style={{ ...btn(A.size === s), height: 30, fontSize: 15 }}>{s}</button>)}
        </div>
      </div>
      <span style={{ fontSize: 14, color: T.dim, paddingBottom: 6 }}>
        {A.m.mels} mels · D {A.m.D} · {A.m.h} heads · enc {A.m.Le} / dec {A.m.Ld} layers · {A.m.params}
      </span>
    </div>
  );
}

export default function App() {
  const ids = VIEWS.map((v) => `#${v.id}`);
  const [view, setView] = useState(() => (ids.includes(location.hash) ? location.hash.slice(1) : "audio"));
  const { C } = VIEWS.find((v) => v.id === view);
  const pick = (id) => { setView(id); history.replaceState(null, "", `#${id}`); };
  const A = useAudio();
  return (
    <div style={{ minHeight: "100vh", background: T.bg, color: T.ink, fontFamily: mono }}>
      <nav style={{ display: "flex", gap: 6, flexWrap: "wrap", padding: "16px clamp(16px, 3vw, 36px) 0", maxWidth: 1480, margin: "0 auto" }}>
        {VIEWS.map((v, i) => {
          const active = v.id === view;
          return (
            <button key={v.id} onClick={() => pick(v.id)}
              style={{ fontFamily: mono, fontSize: 15, padding: "8px 14px", borderRadius: 8, textAlign: "left", cursor: "pointer",
                border: `1px solid ${active ? T.accent : T.rule}`, background: active ? `${T.accent}18` : T.panel, color: active ? T.accent : T.soft }}>
              <div style={{ fontWeight: 700 }}><span style={{ color: T.dim, marginRight: 6 }}>{i + 1}</span>{v.label}</div>
              <div style={{ fontSize: 12, marginTop: 2, color: active ? T.accent : T.dim, opacity: 0.85 }}>{v.sub}</div>
            </button>
          );
        })}
      </nav>
      <Controls A={A} />
      <C key={view} A={A} />
    </div>
  );
}
