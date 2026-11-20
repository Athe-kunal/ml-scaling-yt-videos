// audio.py, in JS, with the same constants. Everything here is the real math:
// int16 quantisation, pad/trim to 30 s, periodic Hann window, reflect-padded STFT,
// |X|², Slaney mel filterbank (= librosa.filters.mel), log10 / floor / scale.

export const SAMPLE_RATE = 16000;
export const N_FFT = 400;
export const HOP_LENGTH = 160;
export const CHUNK_LENGTH = 30;
export const N_SAMPLES = CHUNK_LENGTH * SAMPLE_RATE; // 480000
export const N_FRAMES = N_SAMPLES / HOP_LENGTH; // 3000
export const N_FREQ = N_FFT / 2 + 1; // 201
export const N_MELS = 80;

// ---------------------------------------------------------------- signals
// a few seconds of sound, synthesised at 16 kHz
const TAU = Math.PI * 2;
const midi = (m) => 440 * Math.pow(2, (m - 69) / 12);

function melody() {
  // a short tune of plucked notes, each with harmonics and a decaying envelope
  const notes = [[60, 0.0], [64, 0.45], [67, 0.9], [72, 1.35], [71, 1.9], [67, 2.25], [69, 2.7], [64, 3.15]];
  const n = Math.round(4.0 * SAMPLE_RATE), y = new Float32Array(n);
  for (const [m, t0] of notes) {
    const f = midi(m), s0 = Math.round(t0 * SAMPLE_RATE), len = Math.round(0.8 * SAMPLE_RATE);
    for (let i = 0; i < len && s0 + i < n; i++) {
      const t = i / SAMPLE_RATE, env = Math.min(1, t / 0.008) * Math.exp(-t * 4.5);
      let v = 0;
      for (let h = 1; h <= 8; h++) if (f * h < 7800) v += Math.sin(TAU * f * h * t) / (h * h * 0.7 + 0.3);
      y[s0 + i] += 0.22 * env * v;
    }
  }
  return y;
}

function voice() {
  // a vowel-like voice: pitch glides, three formants move a → i → u, three syllables
  const n = Math.round(3.6 * SAMPLE_RATE), y = new Float32Array(n);
  const syl = [[0.15, 0.95], [1.25, 2.05], [2.4, 3.3]];
  const F = [[[730, 1090, 2440], [270, 2290, 3010]], [[300, 870, 2240], [530, 1840, 2480]], [[660, 1720, 2410], [730, 1090, 2440]]];
  let phase = 0;
  for (let i = 0; i < n; i++) {
    const t = i / SAMPLE_RATE;
    const k = syl.findIndex(([a, b]) => t >= a && t < b);
    if (k < 0) continue;
    const [a, b] = syl[k], u = (t - a) / (b - a);
    const f0 = 120 + 50 * Math.sin(Math.PI * u) + 15 * k;
    phase += TAU * f0 / SAMPLE_RATE;
    const env = Math.sin(Math.PI * u) ** 0.6;
    const fm = F[k][0].map((v, j) => v + (F[k][1][j] - v) * u);
    let v = 0;
    for (let h = 1; h * f0 < 7600; h++) {
      const fh = h * f0;
      let g = 0;
      fm.forEach((fc, j) => { g += (1 / (1 + ((fh - fc) / (80 + 40 * j)) ** 2)) / (j + 1); });
      v += g * Math.sin(h * phase) / Math.sqrt(h);
    }
    y[i] = 0.18 * env * v;
  }
  return y;
}

function chirp() {
  // one exponential sweep 100 Hz → 7 kHz over 2.5 s
  const n = Math.round(3.0 * SAMPLE_RATE), y = new Float32Array(n), d = 2.5, f0 = 100, f1 = 7000, k = Math.log(f1 / f0) / d;
  for (let i = 0; i < Math.round(d * SAMPLE_RATE); i++) {
    const t = i / SAMPLE_RATE, ph = TAU * f0 * (Math.exp(k * t) - 1) / k;
    const env = Math.min(1, t / 0.02, (d - t) / 0.02);
    y[i + 4000] = 0.5 * env * Math.sin(ph);
  }
  return y;
}

export const SIGNALS = { melody, voice, chirp };

// load_audio: ffmpeg -f s16le -ac 1 -ar 16000 → int16 → float32 / 32768
export function toInt16(y) {
  const q = new Int16Array(y.length);
  for (let i = 0; i < y.length; i++) q[i] = Math.max(-32768, Math.min(32767, Math.round(y[i] * 32768)));
  return q;
}
export function fromInt16(q) {
  const y = new Float32Array(q.length);
  for (let i = 0; i < q.length; i++) y[i] = q[i] / 32768;
  return y;
}

// pad_or_trim(array, 480000)
export function padOrTrim(y, length = N_SAMPLES) {
  const out = new Float32Array(length);
  out.set(y.length > length ? y.subarray(0, length) : y);
  return out;
}

// torch.hann_window(400) is periodic: 0.5 − 0.5·cos(2πn / 400)
export const HANN = Float32Array.from({ length: N_FFT }, (_, i) => 0.5 - 0.5 * Math.cos((TAU * i) / N_FFT));

// torch.stft(center=True, pad_mode="reflect"): 200 reflected samples on each side
export const PAD = N_FFT / 2;
export function reflectAt(y, j) {
  // sample j of the padded signal (j = 0 is 200 samples before y[0])
  let i = j - PAD;
  const n = y.length;
  if (i < 0) i = -i;
  if (i >= n) i = 2 * (n - 1) - i;
  return y[i];
}
export function frame(y, t) {
  // frame t: padded[t·160 : t·160 + 400], before the window
  const f = new Float32Array(N_FFT);
  for (let i = 0; i < N_FFT; i++) f[i] = reflectAt(y, t * HOP_LENGTH + i);
  return f;
}

const COS = new Float32Array(N_FREQ * N_FFT), SIN = new Float32Array(N_FREQ * N_FFT);
for (let k = 0; k < N_FREQ; k++)
  for (let i = 0; i < N_FFT; i++) {
    const a = (TAU * k * i) / N_FFT;
    COS[k * N_FFT + i] = Math.cos(a);
    SIN[k * N_FFT + i] = -Math.sin(a);
  }

// one frame: window, DFT (400 → 201 complex), |X|²
export function dft(win) {
  const re = new Float32Array(N_FREQ), im = new Float32Array(N_FREQ), pw = new Float32Array(N_FREQ);
  for (let k = 0; k < N_FREQ; k++) {
    let a = 0, b = 0;
    const o = k * N_FFT;
    for (let i = 0; i < N_FFT; i++) { a += win[i] * COS[o + i]; b += win[i] * SIN[o + i]; }
    re[k] = a; im[k] = b; pw[k] = a * a + b * b;
  }
  return { re, im, pw };
}
export const windowed = (f) => f.map((v, i) => v * HANN[i]);

// ---------------------------------------------------------------- mel filterbank
// librosa.filters.mel(sr=16000, n_fft=400, n_mels=80): Slaney mel scale + Slaney norm
const F_SP = 200 / 3, MIN_LOG_HZ = 1000, MIN_LOG_MEL = MIN_LOG_HZ / F_SP, LOGSTEP = Math.log(6.4) / 27;
export const hzToMel = (f) => (f >= MIN_LOG_HZ ? MIN_LOG_MEL + Math.log(f / MIN_LOG_HZ) / LOGSTEP : f / F_SP);
export const melToHz = (m) => (m >= MIN_LOG_MEL ? MIN_LOG_HZ * Math.exp(LOGSTEP * (m - MIN_LOG_MEL)) : F_SP * m);
export const binHz = (k) => (k * SAMPLE_RATE) / N_FFT; // 40 Hz per bin

export function melFilters(nMels = N_MELS) {
  const mMax = hzToMel(SAMPLE_RATE / 2);
  const pts = Array.from({ length: nMels + 2 }, (_, i) => melToHz((mMax * i) / (nMels + 1)));
  const W = new Float32Array(nMels * N_FREQ);
  for (let m = 0; m < nMels; m++) {
    const lo = pts[m], c = pts[m + 1], hi = pts[m + 2], enorm = 2 / (hi - lo);
    for (let k = 0; k < N_FREQ; k++) {
      const f = binHz(k);
      const w = Math.max(0, Math.min((f - lo) / (c - lo), (hi - f) / (hi - c)));
      W[m * N_FREQ + k] = w * enorm;
    }
  }
  return { W, pts };
}
export const MEL = melFilters();
const MELS = { 80: MEL };
export const melBank = (n) => (MELS[n] ??= melFilters(n));

// ---------------------------------------------------------------- log_mel_spectrogram
// returns every intermediate so the scenes can show each stage
export function logMel(y30, nMels = N_MELS) {
  // y30: 480000 samples (already pad_or_trim'd)
  const FB = melBank(nMels);
  const nFrames = N_FRAMES; // 3001 frames, the last one is dropped: stft[..., :-1]
  const power = new Float32Array(N_FREQ * nFrames); // [201, 3000]
  // frames whose 400 samples are all zero give zero power; skip the DFT for them
  let last = y30.length - 1;
  while (last > 0 && y30[last] === 0) last--;
  const lastFrame = Math.min(nFrames - 1, Math.ceil((last + PAD) / HOP_LENGTH));
  for (let t = 0; t <= lastFrame; t++) {
    const { pw } = dft(windowed(frame(y30, t)));
    for (let k = 0; k < N_FREQ; k++) power[k * nFrames + t] = pw[k];
  }
  const mel = new Float32Array(nMels * nFrames); // filters @ magnitudes
  for (let m = 0; m < nMels; m++)
    for (let k = 0; k < N_FREQ; k++) {
      const w = FB.W[m * N_FREQ + k];
      if (!w) continue;
      for (let t = 0; t <= lastFrame; t++) mel[m * nFrames + t] += w * power[k * nFrames + t];
    }
  const log = mel.map((v) => Math.log10(Math.max(v, 1e-10)));
  let mx = -Infinity;
  for (const v of log) mx = Math.max(mx, v);
  const floored = log.map((v) => Math.max(v, mx - 8));
  const scaled = floored.map((v) => (v + 4) / 4);
  return { power, mel, log, floored, scaled, max: mx, lastFrame, nFrames, nMels, FB };
}

// content length in frames / tokens
export const framesOf = (nSamples) => Math.round(nSamples / HOP_LENGTH);

// ---------------------------------------------------------------- microphone
// record a few seconds; decodeAudioData into a 16 kHz context resamples like ffmpeg -ar 16000
export async function recordMic(seconds = 3) {
  const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
  const rec = new MediaRecorder(stream), chunks = [];
  rec.ondataavailable = (e) => chunks.push(e.data);
  const done = new Promise((r) => (rec.onstop = r));
  rec.start();
  await new Promise((r) => setTimeout(r, seconds * 1000));
  rec.stop();
  await done;
  stream.getTracks().forEach((t) => t.stop());
  const buf = await new Blob(chunks).arrayBuffer();
  const ctx = new AudioContext({ sampleRate: SAMPLE_RATE });
  const audio = await ctx.decodeAudioData(buf);
  ctx.close();
  // -ac 1: average the channels
  const y = new Float32Array(audio.length);
  for (let c = 0; c < audio.numberOfChannels; c++) {
    const d = audio.getChannelData(c);
    for (let i = 0; i < y.length; i++) y[i] += d[i] / audio.numberOfChannels;
  }
  return y;
}

export function play(y) {
  const ctx = new AudioContext({ sampleRate: SAMPLE_RATE });
  const b = ctx.createBuffer(1, y.length, SAMPLE_RATE);
  b.copyToChannel(y, 0);
  const s = ctx.createBufferSource();
  s.buffer = b;
  s.connect(ctx.destination);
  s.onended = () => ctx.close();
  s.start();
}
