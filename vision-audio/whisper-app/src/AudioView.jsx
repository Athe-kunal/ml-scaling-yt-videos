import { useMemo, useState } from "react";
import { T, mono } from "./theme";
import StepShell, { Stat } from "./StepShell";
import { DimCtx, Row, Op, Mat, MM, MMNote, MatMulCard, Label } from "./draw";
import { Wave, Heat, Plot, Hist, W0 } from "./plots";
import { HANN, N_FFT, HOP_LENGTH, N_FREQ, N_SAMPLES, SAMPLE_RATE, PAD, frame, windowed, dft, reflectAt, binHz } from "./dsp";

// audio.py, one job per step: int16 samples → 30 s → frames → |DFT|² → mel → log → scale.
// Every picture is computed from the selected signal with the real constants.

const CODE = {
  load: ["cmd = ['ffmpeg', '-i', file, '-f', 's16le', '-ac', '1', '-ar', '16000', '-']", "out = run(cmd, capture_output=True).stdout", "np.frombuffer(out, np.int16).astype(np.float32) / 32768.0"],
  pad: ["N_SAMPLES = 30 * 16000   # 480000", "pad_or_trim(audio, N_SAMPLES)   # zeros on the right, or cut"],
  hann: ["window = torch.hann_window(400)   # periodic: 0.5 − 0.5·cos(2πn/400)"],
  frames: ["torch.stft(audio, 400, 160, window=window, return_complex=True)", "# center=True: 200 reflected samples on each side", "# frames = 1 + 480000 // 160 = 3001"],
  dft: ["# per frame: X[k] = Σ_n w[n]·x[n]·e^(−2πi·k·n/400),  k = 0 … 200", "magnitudes = stft.abs() ** 2"],
  power: ["magnitudes = stft[..., :-1].abs() ** 2   # [201, 3001] → [201, 3000]"],
  filters: ["filters = mel_filters(device, n_mels)   # librosa.filters.mel(sr=16000, n_fft=400, n_mels)", "# saved in assets/mel_filters.npz: mel_80 [80, 201], mel_128 [128, 201]"],
  melmm: ["mel_spec = filters @ magnitudes   # [M, 201] @ [201, 3000]"],
  log: ["log_spec = torch.clamp(mel_spec, min=1e-10).log10()"],
  floor: ["log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)"],
  scale: ["log_spec = (log_spec + 4.0) / 4.0   # → model.encoder(mel)"],
};

const FLOW = [
  { id: "load", label: "load_audio", sub: "ffmpeg → 16 kHz mono int16", code: CODE.load },
  { id: "pad", label: "pad_or_trim", sub: "→ 480 000 samples = 30 s", code: CODE.pad },
  {
    group: "torch.stft",
    items: [
      { id: "hann", label: "Hann window", sub: "400 samples = 25 ms", code: CODE.hann },
      { id: "frames", label: "slide by 160", sub: "hop 10 ms → 3001 frames", code: CODE.frames },
      { id: "dft", label: "DFT · |X|²", sub: "400 samples → 201 freqs", code: CODE.dft },
      { id: "power", label: "spectrogram", sub: "[201, 3000] · drop last", code: CODE.power },
    ],
  },
  {
    group: "mel",
    items: [
      { id: "filters", label: "mel filterbank", sub: "[M, 201] triangles", code: CODE.filters },
      { id: "melmm", label: "filters @ power", sub: "[M, 3000]", code: CODE.melmm },
    ],
  },
  {
    group: "log + scale",
    items: [
      { id: "log", label: "log10 · clamp 1e-10", sub: "silence → −10", code: CODE.log },
      { id: "floor", label: "max − 8", sub: "keep 8 decades = 80 dB", code: CODE.floor },
      { id: "scale", label: "(x + 4) / 4", sub: "≈ [−1, 1]", code: CODE.scale },
    ],
  },
];

const f2 = (v, d = 2) => (Math.abs(v) >= 1e4 || (Math.abs(v) < 1e-3 && v !== 0) ? v.toExponential(1) : v.toFixed(d));
const secTicks = (n, every = 1) => Array.from({ length: Math.floor(n / SAMPLE_RATE / every) + 1 }, (_, i) => [i * every * SAMPLE_RATE, `${i * every} s`]);
const dB = (v) => 10 * Math.log10(v + 1e-10);

// a heatmap with frequency labels on the left and time ticks below
function SpecBox({ title, rows, top, bottom, children, c0, c1, cols = 3000, every }) {
  const n = c1 - c0, step = every ?? (n > 1500 ? 500 : n > 400 ? 100 : 50);
  const ticks = [];
  for (let c = Math.ceil(c0 / step) * step; c <= c1; c += step) ticks.push(c);
  return (
    <div style={{ display: "grid", gap: 4 }}>
      {title && <Label>{title}</Label>}
      <div style={{ display: "grid", gridTemplateColumns: "56px 1fr", gap: 6 }}>
        <div style={{ display: "flex", flexDirection: "column", justifyContent: "space-between", alignItems: "flex-end", fontSize: 12, color: T.dim }}>
          <span>{top}</span><span style={{ writingMode: "vertical-rl", transform: "rotate(180deg)", color: T.soft }}>{rows}</span><span>{bottom}</span>
        </div>
        <div>{children}</div>
      </div>
      <div style={{ marginLeft: 62, position: "relative", height: 16, fontSize: 12, color: T.dim }}>
        {ticks.map((c) => (
          <span key={c} style={{ position: "absolute", left: `${((c - c0) / n) * 100}%`, transform: c === c0 ? "none" : c === c1 ? "translateX(-100%)" : "translateX(-50%)" }}>
            {c === 0 ? "frame 0" : `${c}`}<span style={{ color: T.rule }}>{c ? ` · ${(c * 0.01).toFixed(c % 100 ? 1 : 0)}s` : ""}</span>
          </span>
        ))}
      </div>
      {cols && null}
    </div>
  );
}

export default function AudioView({ A }) {
  const { y, q, y30, lm, nMels } = A;
  const n = y.length;
  // the frame to inspect: the loudest one; the mel bin: the brightest one in that frame
  const loud = useMemo(() => {
    let best = 0, bt = 0;
    for (let t = 0; t <= lm.lastFrame; t++) { let s = 0; for (let k = 0; k < N_FREQ; k++) s += lm.power[k * 3000 + t]; if (s > best) { best = s; bt = t; } }
    return bt;
  }, [lm]);
  const [tSel, setT] = useState(null);
  const t = Math.min(tSel ?? loud, lm.lastFrame);
  const melBright = useMemo(() => { let b = 0; for (let m = 1; m < nMels; m++) if (lm.mel[m * 3000 + t] > lm.mel[b * 3000 + t]) b = m; return b; }, [lm, t, nMels]);
  const [mSel, setM] = useState(null);
  const mb = Math.min(mSel ?? melBright, nMels - 1);
  const C1 = Math.min(3000, Math.ceil((lm.lastFrame + 30) / 10) * 10);

  const fr = useMemo(() => frame(y30, t), [y30, t]);
  const win = useMemo(() => windowed(fr), [fr]);
  const spec = useMemo(() => dft(win), [win]);
  const fb = lm.FB;
  const filt = (m) => Array.from({ length: N_FREQ }, (_, k) => fb.W[m * N_FREQ + k]);
  const col = Array.from({ length: N_FREQ }, (_, k) => lm.power[k * 3000 + t]);
  const melRange = [fb.pts[mb], fb.pts[mb + 1], fb.pts[mb + 2]];
  const sMin = useMemo(() => lm.scaled.reduce((a, v) => Math.min(a, v), 9), [lm]);
  const sMax = (lm.max + 4) / 4;
  const pMaxDb = useMemo(() => dB(lm.power.reduce((a, v) => Math.max(a, v), 0)), [lm]);
  const hzTicks = [0, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000].map((f) => [f / 40, f ? `${f / 1000}k` : "0 Hz"]);

  const steps = [
    { id: "load", block: "load", caption: "int16 samples ÷ 32768", vars: () => [["sr", "16 000 Hz"], ["samples", n.toLocaleString()], ["length", `${(n / SAMPLE_RATE).toFixed(2)} s`]] },
    { id: "pad", block: "pad", caption: "every clip becomes exactly 30 s", vars: () => [["real", n.toLocaleString()], ["zeros", (N_SAMPLES - n).toLocaleString()]] },
    { id: "hann", block: "hann", caption: "cut 25 ms, fade the edges", vars: () => [["frame", t], ["start", `${t * HOP_LENGTH - PAD}`]] },
    { id: "frames", block: "frames", caption: "a new frame every 10 ms", vars: () => [["n_fft", 400], ["hop", 160], ["frames", "3001"]] },
    { id: "dft", block: "dft", caption: "one frame → 201 frequencies", vars: () => [["bin width", "40 Hz"], ["top bin", "8000 Hz"]] },
    { id: "power", block: "power", caption: "all frames side by side", vars: () => [["shape", "[201, 3000]"]] },
    { id: "filters", block: "filters", caption: "narrow low, wide high", vars: () => [["filter", mb], ["span", `${melRange[0].toFixed(0)}–${melRange[2].toFixed(0)} Hz`]] },
    { id: "melmm", block: "melmm", caption: "201 bins → M mel bins", vars: () => [[`mel[${mb}, ${t}]`, f2(lm.mel[mb * 3000 + t], 3)]] },
    { id: "log", block: "log", caption: "loudness is logarithmic", vars: () => [["max", f2(lm.max)], ["silence", "−10"]] },
    { id: "floor", block: "floor", caption: "nothing quieter than max − 8", vars: () => [["floor", f2(lm.max - 8)]] },
    { id: "scale", block: "scale", caption: "the encoder's input", vars: () => [["min", f2(sMin)], ["max", f2(sMax)]] },
  ];

  const tags = (s) => ({
    load: [{ name: "file", dims: [], io: "in" }, { name: "int16", dims: String(n) }, { name: "audio", dims: String(n), io: "out" }],
    pad: [{ name: "audio", dims: String(n), io: "in" }, { name: "audio", dims: "S", io: "out" }],
    hann: [{ name: "frame", dims: "400", io: "in" }, { name: "window", dims: "400" }, { name: "frame·window", dims: "400", io: "out" }],
    frames: [{ name: "audio", dims: "S", io: "in" }, { name: "frames", dims: "3001 400", io: "out" }],
    dft: [{ name: "frame·window", dims: "400", io: "in" }, { name: "|X|²", dims: "F", io: "out" }],
    power: [{ name: "frames", dims: "3001 400", io: "in" }, { name: "magnitudes", dims: "F T", io: "out" }],
    filters: [{ name: "filters", dims: "M F", io: "out" }],
    melmm: [{ name: "filters", dims: "M F", io: "in" }, { name: "magnitudes", dims: "F T", io: "in" }, { name: "mel_spec", dims: "M T", io: "out" }],
    log: [{ name: "mel_spec", dims: "M T", io: "in" }, { name: "log_spec", dims: "M T", io: "out" }],
    floor: [{ name: "log_spec", dims: "M T", io: "in" }, { name: "log_spec", dims: "M T", io: "out" }],
    scale: [{ name: "log_spec", dims: "M T", io: "in" }, { name: "mel", dims: "M T", io: "out" }],
  }[s.id]);

  const matmuls = (s) => {
    if (s.id === "dft") return (
      <MatMulCard>
        <MM a={["frame·window", [1, 400]]} b={["e^(−2πi·k·n/400)", [400, 201]]} c={["X", [1, 201]]} />
        <MMNote>|X|² → power · ×3000 frames</MMNote>
      </MatMulCard>
    );
    if (s.id === "melmm") return (
      <MatMulCard>
        <MM a={["filters", [nMels, 201]]} b={["magnitudes", [201, 3000]]} c={["mel_spec", [nMels, 3000]]} />
      </MatMulCard>
    );
    return null;
  };

  const scene = (s) => {
    switch (s.id) {
      case "load": {
        const z0 = t * HOP_LENGTH, zs = Array.from({ length: 5 }, (_, i) => z0 + 8 * i);
        return (
          <div style={{ display: "grid", gap: 12 }}>
            <Wave y={y} ticks={secTicks(n, 0.5)} boxes={[{ a: z0, b: z0 + 48, color: T.wire }]} h={130} />
            <Label color={T.wire}>zoom · 48 samples = 3 ms</Label>
            <Wave y={y} from={z0} to={z0 + 48} stems={(i) => (zs.includes(i) ? T.wire : T.accent)} h={120} />
            <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
              {zs.map((i) => (
                <span key={i} style={{ fontSize: 15, padding: "4px 10px", borderRadius: 6, background: T.well, border: `1px solid ${T.rule}` }}>
                  <span style={{ color: T.dim }}>{q[i]}</span><span style={{ color: T.accent2 }}> ÷ 32768 = </span><span style={{ color: T.wire }}>{y[i].toFixed(4)}</span>
                </span>
              ))}
            </div>
          </div>
        );
      }
      case "pad":
        return (
          <div style={{ display: "grid", gap: 10 }}>
            <Wave y={y30} ticks={secTicks(N_SAMPLES, 5)} shade={{ a: n, b: N_SAMPLES }} boxes={[{ a: 0, b: n, color: T.accent, fill: "transparent" }]} h={150} />
            <Row gap={24}>
              <Stat k="audio" v={`${n.toLocaleString()} samples`} tone={T.accent} />
              <Stat k="+ zeros" v={(N_SAMPLES - n).toLocaleString()} tone={T.dim} />
              <Stat k="= 30 s × 16 000" v="480 000" tone={T.wire} />
            </Row>
          </div>
        );
      case "hann": {
        const xs = Array.from({ length: N_FFT }, (_, i) => i);
        const m = Math.max(...fr.map(Math.abs), 1e-6);
        const ticks = [[0, "0"], [200, "12.5 ms"], [399, "25 ms"]];
        return (
          <div style={{ display: "grid", gap: 6 }}>
            <Label>frame {t} · 400 samples</Label>
            <Plot xs={xs} h={120} series={[{ y: Array.from(fr), color: T.soft }]} ymin={-m} ymax={m} xticks={ticks} />
            <Label color={T.wire}>× Hann window</Label>
            <Plot xs={xs} h={90} series={[{ y: Array.from(HANN), color: T.wire, lw: 2.5, fill: T.wire, fillOpacity: 0.12 }]} ymin={0} ymax={1} xticks={ticks} />
            <Label color={T.accent}>= edges fade to 0</Label>
            <Plot xs={xs} h={120} series={[{ y: Array.from(win), color: T.accent }]} ymin={-m} ymax={m} xticks={ticks} />
          </div>
        );
      }
      case "frames": {
        // the first 1600 samples of the padded signal: 200 mirrored + the start
        const P = Float32Array.from({ length: 1600 }, (_, j) => reflectAt(y30, j));
        const cols = [T.wire, T.accent, T.accent2, "#8DB4FF", "#9BE564", "#C79BFF", T.wire];
        const boxes = Array.from({ length: 7 }, (_, k) => ({ a: k * 160, b: k * 160 + 400, color: cols[k], y0: 4 + k * 14, y1: 4 + k * 14 + 12, fill: `${cols[k]}55` }));
        const mid = t * HOP_LENGTH;
        return (
          <div style={{ display: "grid", gap: 10 }}>
            <Label>start of the clip · frames 0 … 6</Label>
            <Wave y={P} boxes={[{ a: 0, b: 200, color: T.accent2, fill: `${T.accent2}18`, y0: 104, y1: 228 }, ...boxes]} h={230}
              ticks={[[0, "−200"], [200, "sample 0"], [600, "400"], [1000, "800"], [1400, "1200"]]} color={T.soft} />
            <Row gap={24}>
              <Stat k="mirrored" v="200 samples" tone={T.accent2} />
              <Stat k="frame t starts at" v="t·160 − 200" tone={T.wire} />
              <Stat k="1 + 480000 / 160" v="3001 frames" tone={T.accent} />
            </Row>
            <Label>around frame {t}</Label>
            <Wave y={y30} from={mid - 600} to={mid + 1000} h={110} color={T.soft}
              boxes={[-2, -1, 0, 1, 2].map((d) => ({ a: (t + d) * 160 - 200, b: (t + d) * 160 + 200, color: d === 0 ? T.accent : T.dim, y0: 4 + (d + 2) * 8, y1: 106 - (2 - d) * 0, fill: d === 0 ? `${T.accent}20` : "transparent", lw: d === 0 ? 2.5 : 1 }))} />
          </div>
        );
      }
      case "dft": {
        const xs = Array.from({ length: N_FREQ }, (_, k) => k);
        const d = Array.from(spec.pw).map(dB), top = Math.max(...d);
        const peaks = xs.filter((k) => k > 0 && k < 200 && d[k] > d[k - 1] && d[k] >= d[k + 1] && d[k] > top - 25).sort((a, b) => d[b] - d[a]).slice(0, 4);
        return (
          <div style={{ display: "grid", gap: 8 }}>
            <Label>frame {t} · after the window</Label>
            <Plot xs={Array.from({ length: N_FFT }, (_, i) => i)} h={90} series={[{ y: Array.from(win), color: T.accent }]} xticks={[[0, "0"], [399, "400 samples"]]} />
            <div style={{ fontSize: 22, color: T.accent2, paddingLeft: 40 }}>↓ DFT</div>
            <Label>|X[k]|² in dB · k = 0 … 200 · 40 Hz per bin</Label>
            <Plot xs={xs} h={200} bars={{ y: d.map((v) => Math.max(v, top - 80)), color: (k) => (peaks.includes(k) ? T.wire : T.accent) }} ymin={top - 80} ymax={top + 3} xticks={hzTicks}>
              {({ X, Y }) => peaks.map((k) => <text key={k} x={X(k)} y={Y(d[k]) - 6} fill={T.wire} fontSize={13} textAnchor="middle">{binHz(k)} Hz</text>)}
            </Plot>
          </div>
        );
      }
      case "power":
        return (
          <div style={{ display: "grid", gap: 10 }}>
            <SpecBox title="power in dB · linear frequency" rows="201 bins" top="8 kHz" bottom="0" c0={0} c1={C1}>
              <Heat data={lm.power} rows={N_FREQ} cols={3000} c1={C1} h={260} lo={pMaxDb - 80} hi={pMaxDb} map={dB} col={t} />
            </SpecBox>
            <Row gap={24}>
              <Stat k="column" v={`frame ${t} = ${(t * 0.01).toFixed(2)} s`} tone={T.accent} />
              <Stat k="stft frames" v="3001 → 3000" tone={T.wire} />
              {C1 < 3000 && <Stat k="frames " v={`${C1} … 2999 = silence`} tone={T.dim} />}
            </Row>
          </div>
        );
      case "filters": {
        const xs = Array.from({ length: N_FREQ }, (_, k) => k);
        const all = Array.from({ length: nMels }, (_, m) => ({ y: filt(m), color: m === mb ? T.accent2 : m % 2 ? "#8DB4FF" : T.accent, lw: m === mb ? 3 : 1, opacity: m === mb ? 1 : 0.45, fill: m === mb ? T.accent2 : undefined }));
        return (
          <div style={{ display: "grid", gap: 10 }}>
            <Plot xs={xs} h={240} series={all} xticks={hzTicks} ymin={0} />
            <Label>zoom 0 – 1 kHz</Label>
            <Plot xs={xs.slice(0, 26)} h={130} series={all.map((s) => ({ ...s, y: s.y.slice(0, 26) }))} xticks={[[0, "0"], [5, "200"], [10, "400"], [15, "600"], [20, "800"], [25, "1k Hz"]]} ymin={0} />
            <Row gap={24}>
              <Stat k={`filter ${mb}`} v={`${melRange[0].toFixed(0)} – ${melRange[2].toFixed(0)} Hz`} tone={T.accent2} />
              <Stat k="filter 0" v={`${fb.pts[0].toFixed(0)} – ${fb.pts[2].toFixed(0)} Hz`} />
              <Stat k={`filter ${nMels - 1}`} v={`${fb.pts[nMels - 1].toFixed(0)} – ${fb.pts[nMels + 1].toFixed(0)} Hz`} />
              <Stat k="area of each" v="same (Slaney norm)" tone={T.wire} />
            </Row>
            <MelPicker n={nMels} m={mb} set={setM} />
          </div>
        );
      }
      case "melmm": {
        const xs = Array.from({ length: N_FREQ }, (_, k) => k);
        const f = filt(mb), fm = Math.max(...f);
        const prod = col.map((v, k) => v * f[k]);
        const lo = Math.max(0, Math.floor(melRange[0] / 40) - 2), hi = Math.min(200, Math.ceil(melRange[2] / 40) + 2);
        const zEnd = Math.min(200, Math.max(40, hi * 2)), cz = Math.max(...col.slice(0, zEnd + 1), 1e-12);
        return (
          <div style={{ display: "grid", gap: 10 }}>
            <Row gap={18} wrap={false} style={{ alignItems: "flex-start" }}>
              <div style={{ display: "grid", gap: 4, flex: 1, minWidth: 0 }}>
                <Label>power column {t} · filter {mb} · bins 0 – {zEnd}</Label>
                <Plot xs={xs.slice(0, zEnd + 1)} w={420} h={170} bars={{ y: col.slice(0, zEnd + 1).map((v) => v / cz), color: (k) => (f[k] > 0 ? T.accent : `${T.accent}55`) }} series={[{ y: f.slice(0, zEnd + 1).map((v) => v / fm), color: T.accent2, lw: 2.5 }]} xticks={[0, 0.25, 0.5, 0.75, 1].map((q) => { const k = Math.round(q * zEnd); return [k, `${binHz(k)}${q === 1 ? " Hz" : ""}`]; })} ymin={0} ymax={1.05} />
              </div>
              <div style={{ display: "grid", gap: 4 }}>
                <Label>bins {lo} – {hi}: power × weight</Label>
                <Plot xs={xs.slice(lo, hi + 1)} w={300} h={170} pad={{ l: 44, r: 34, t: 10, b: 30 }} bars={{ y: prod.slice(lo, hi + 1), color: () => T.wire }} ymin={0} xticks={[[lo, `${binHz(lo)}`], [hi, `${binHz(hi)} Hz`]]} />
              </div>
            </Row>
            <div style={{ fontSize: 17 }}>
              <span style={{ color: T.dim }}>Σ = </span><b style={{ color: T.wire }}>mel[{mb}, {t}] = {f2(lm.mel[mb * 3000 + t], 4)}</b>
            </div>
            <SpecBox title={`mel_spec · log display · ${nMels} rows`} rows={`${nMels} mels`} top="8 kHz" bottom="0" c0={0} c1={C1}>
              <Heat data={lm.mel} rows={nMels} cols={3000} c1={C1} h={200} lo={lm.max - 8} hi={lm.max} map={(v) => Math.log10(Math.max(v, 1e-10))} cell={[mb, t]} />
            </SpecBox>
            <MelPicker n={nMels} m={mb} set={setM} />
          </div>
        );
      }
      case "log":
      case "floor":
      case "scale": {
        const which = s.id === "log" ? lm.log : s.id === "floor" ? lm.floored : lm.scaled;
        const lo = s.id === "scale" ? -2 : -11, hi = s.id === "scale" ? 1.5 : Math.ceil(lm.max) + 1;
        const vl = s.id === "log" ? [[-10, T.bad, "clamp → −10"], [lm.max, T.wire, `max ${f2(lm.max)}`]]
          : s.id === "floor" ? [[lm.max - 8, T.accent2, `max − 8 = ${f2(lm.max - 8)}`], [lm.max, T.wire, "max"]]
          : [[sMin, T.accent2, f2(sMin)], [sMax, T.wire, f2(sMax)]];
        const tk = s.id === "scale" ? [-2, -1, 0, 1].map((v) => [v, `${v}`]) : [-10, -8, -6, -4, -2, 0].map((v) => [v, `${v}`]);
        const full = s.id === "scale";
        return (
          <div style={{ display: "grid", gap: 10 }}>
            <Label>all {nMels} × 3000 values · histogram</Label>
            <Hist data={which} lo={lo} hi={hi} vlines={vl} xticks={tk} color={s.id === "log" ? "#8DB4FF" : s.id === "floor" ? T.accent2 : T.accent} />
            <SpecBox title={full ? "the whole 30 s · what the encoder sees" : "log-mel"} rows={`${nMels} mels`} top="8 kHz" bottom="0" c0={0} c1={full ? 3000 : C1}>
              <Heat data={which} rows={nMels} cols={3000} c1={full ? 3000 : C1} h={190} lo={s.id === "log" ? -10 : s.id === "floor" ? lm.max - 8 : sMin} hi={s.id === "scale" ? sMax : lm.max} />
            </SpecBox>
            {s.id === "floor" && <Row gap={24}><Stat k="before" v="−10 … max" tone={T.dim} /><Stat k="after" v={`${f2(lm.max - 8)} … ${f2(lm.max)}`} tone={T.accent2} /><Stat k="10·8 dB" v="80 dB range" tone={T.wire} /></Row>}
            {s.id === "scale" && (
              <Row gap={10}>
                <Mat name="mel" shape="M T" rows={nMels} cols={3000} glow />
                <Op>→ AudioEncoder</Op>
                <Stat k="silence" v={`= ${f2(sMin)}`} tone={T.accent2} />
              </Row>
            )}
          </div>
        );
      }
      default:
        return null;
    }
  };

  const controls = (
    <div style={{ display: "flex", gap: 10, alignItems: "center", fontSize: 14, color: T.dim }}>
      frame
      <input type="range" min={0} max={lm.lastFrame} value={t} onChange={(e) => setT(+e.target.value)} style={{ width: 180 }} />
      <span style={{ color: T.accent, width: 90 }}>{t} · {(t * 0.01).toFixed(2)} s</span>
    </div>
  );

  return (
    <DimCtx.Provider value={{ S: ["samples", N_SAMPLES], F: ["frequency bins", 201], M: ["mel bins", nMels], T: ["frames (10 ms)", 3000] }}>
      <StepShell title="audio.py · log_mel_spectrogram" subtitle={`${A.name} · ${(n / SAMPLE_RATE).toFixed(2)} s · ${nMels} mels`} blocks={FLOW} steps={steps} scene={scene} tags={tags} matmuls={matmuls} controls={controls} />
    </DimCtx.Provider>
  );
}

function MelPicker({ n, m, set }) {
  return (
    <div style={{ display: "flex", gap: 10, alignItems: "center", fontSize: 14, color: T.dim, fontFamily: mono }}>
      mel bin
      <input type="range" min={0} max={n - 1} value={m} onChange={(e) => set(+e.target.value)} style={{ width: 220 }} />
      <span style={{ color: T.accent2 }}>{m}</span>
    </div>
  );
}

export { W0 };
