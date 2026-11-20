// Whisper sizes, the multilingual vocabulary layout, the example transcript, and a
// tiny toy encoder (random fixed weights) so the attention pictures come from the
// actual audio. Shapes on labels are always the real ones for the chosen size.

export const SIZES = {
  tiny: { mels: 80, D: 384, h: 6, Le: 4, Ld: 4, params: "39 M" },
  base: { mels: 80, D: 512, h: 8, Le: 6, Ld: 6, params: "74 M" },
  small: { mels: 80, D: 768, h: 12, Le: 12, Ld: 12, params: "244 M" },
  medium: { mels: 80, D: 1024, h: 16, Le: 24, Ld: 24, params: "769 M" },
  "large-v3": { mels: 128, D: 1280, h: 20, Le: 32, Ld: 32, params: "1550 M" },
  turbo: { mels: 128, D: 1280, h: 20, Le: 32, Ld: 4, params: "809 M" },
};
export const N_AUDIO_CTX = 1500, N_TEXT_CTX = 448;
export const vocabOf = (size) => (SIZES[size].mels === 128 ? 51866 : 51865);

// ---------------------------------------------------------------- vocabulary (multilingual, 99 languages)
export const TOK = {
  eot: 50257, sot: 50258, en: 50259, de: 50261, es: 50262, fr: 50265, nl: 50271,
  translate: 50358, transcribe: 50359, sot_lm: 50360, sot_prev: 50361, nospeech: 50362, notimestamps: 50363, ts0: 50364,
};
export const tsId = (sec) => TOK.ts0 + Math.round(sec / 0.02);
export const tsOf = (id) => ((id - TOK.ts0) * 0.02).toFixed(2);
export const VOCAB_REGIONS = [
  { name: "text (BPE)", a: 0, b: 50256, color: "#8DB4FF", w: 0.52 },
  { name: "<|endoftext|> <|sot|> languages · task · …", a: 50257, b: 50363, color: "#FF8AC6", w: 0.18 },
  { name: "timestamps <|0.00|> … <|30.00|>", a: 50364, b: 51864, color: "#FFC94D", w: 0.30 },
];
export const SPECIAL_NAME = {
  [TOK.eot]: "<|endoftext|>", [TOK.sot]: "<|startoftranscript|>", [TOK.en]: "<|en|>", [TOK.translate]: "<|translate|>", [TOK.transcribe]: "<|transcribe|>",
  [TOK.sot_prev]: "<|startofprev|>", [TOK.nospeech]: "<|nospeech|>", [TOK.notimestamps]: "<|notimestamps|>",
};

// the example clip: the JFK sample shipped with the whisper repo (tests/jfk.flac, 11 s)
// text token ids come from multilingual.tiktoken
const W = (s, id, a, b) => ({ s, id, a, b });
export const WORDS = [
  W(" And", 400, 0.32, 0.6), W(" so", 370, 0.6, 1.0), W(" my", 452, 1.0, 1.3), W(" fellow", 7177, 1.3, 1.72), W(" Americans", 6280, 1.72, 2.5), W(",", 11, 2.5, 2.62),
  W(" ask", 1029, 3.3, 3.8), W(" not", 406, 3.8, 4.4), W(" what", 437, 4.6, 5.0), W(" your", 428, 5.0, 5.4), W(" country", 1941, 5.4, 6.0),
  W(" can", 393, 6.1, 6.5), W(" do", 360, 6.5, 6.9), W(" for", 337, 6.9, 7.3), W(" you", 291, 7.3, 7.6), W(",", 11, 7.6, 7.68),
  W(" ask", 1029, 8.2, 8.5), W(" what", 437, 8.5, 8.8), W(" you", 291, 8.8, 9.0), W(" can", 393, 9.0, 9.3), W(" do", 360, 9.3, 9.5),
  W(" for", 337, 9.5, 9.8), W(" your", 428, 9.8, 10.1), W(" country", 1941, 10.1, 10.6), W(".", 13, 10.6, 10.7),
];
export const PREFIX = [TOK.sot, TOK.en, TOK.transcribe];
// generated tokens, with timestamps: <|0.00|> seg 1 <|7.68|><|7.68|> seg 2 <|10.88|> <|endoftext|>
export const GEN = [tsId(0), ...WORDS.slice(0, 16).map((w) => w.id), tsId(7.68), tsId(7.68), ...WORDS.slice(16).map((w) => w.id), tsId(10.88), TOK.eot];
const TEXT_OF = { ...Object.fromEntries(WORDS.map((w) => [w.id, w.s])), 2863: " beginning", 35516: " renewal", 382: " as", 731: " well", 1152: " hard" };
export const tokText = (id) => SPECIAL_NAME[id] ?? (id >= TOK.ts0 ? `<|${tsOf(id)}|>` : TEXT_OF[id] ?? `#${id}`);
export const kindOf = (id) => (id >= TOK.ts0 ? "ts" : id >= TOK.eot ? "special" : "text");

// ---------------------------------------------------------------- seeded random
export function rng(seed = 1) {
  let a = seed >>> 0;
  return () => {
    a |= 0; a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
const gauss = (r) => Math.sqrt(-2 * Math.log(r() + 1e-12)) * Math.cos(2 * Math.PI * r());
export const gelu = (x) => 0.5 * x * (1 + Math.tanh(0.7978845608 * (x + 0.044715 * x * x * x)));

// model.py sinusoids(length, channels): [sin(t·inv) | cos(t·inv)], inv = 10000^(−i/(C/2−1))
export function sinusoid(t, c, C) {
  const half = C / 2, inc = Math.log(10000) / (half - 1);
  const i = c < half ? c : c - half, v = t * Math.exp(-inc * i);
  return c < half ? Math.sin(v) : Math.cos(v);
}

// ---------------------------------------------------------------- toy encoder
// conv1 (k=3, pad 1) → GELU → conv2 (k=3, stride 2, pad 1) → GELU, Dt toy channels
export const DT = 32;
export function toyStem(scaled, nMels, nFrames = 3000) {
  const r = rng(7), w1 = Float32Array.from({ length: DT * nMels * 3 }, () => gauss(r) / Math.sqrt(nMels * 3));
  const w2 = Float32Array.from({ length: DT * DT * 3 }, () => gauss(r) / Math.sqrt(DT * 3));
  const c1 = new Float32Array(DT * nFrames);
  for (let o = 0; o < DT; o++)
    for (let t = 0; t < nFrames; t++) {
      let s = 0;
      for (let k = 0; k < 3; k++) {
        const tt = t + k - 1;
        if (tt < 0 || tt >= nFrames) continue;
        const wo = (o * nMels) * 3;
        for (let m = 0; m < nMels; m++) s += w1[wo + m * 3 + k] * scaled[m * nFrames + tt];
      }
      c1[o * nFrames + t] = gelu(s);
    }
  const n2 = nFrames / 2, c2 = new Float32Array(DT * n2);
  for (let o = 0; o < DT; o++)
    for (let t = 0; t < n2; t++) {
      let s = 0;
      for (let k = 0; k < 3; k++) {
        const tt = 2 * t + k - 1;
        if (tt < 0 || tt >= nFrames) continue;
        for (let i = 0; i < DT; i++) s += w2[(o * DT + i) * 3 + k] * c1[i * nFrames + tt];
      }
      c2[o * n2 + t] = gelu(s);
    }
  return { c1, c2, n2 };
}

// one toy self-attention map over the first N audio tokens: x = c2ᵀ + pos, softmax(x xᵀ · s)
export function toyAttn(c2, n2, N, posW = 0.6, sharp = 6) {
  const x = new Float32Array(N * DT);
  for (let t = 0; t < N; t++) {
    let nrm = 0;
    for (let c = 0; c < DT; c++) { const v = c2[c * n2 + t] + posW * sinusoid(t, c, DT); x[t * DT + c] = v; nrm += v * v; }
    nrm = Math.sqrt(nrm) || 1;
    for (let c = 0; c < DT; c++) x[t * DT + c] /= nrm;
  }
  const A = new Float32Array(N * N);
  for (let i = 0; i < N; i++) {
    let mx = -1e9;
    for (let j = 0; j < N; j++) { let s = 0; for (let c = 0; c < DT; c++) s += x[i * DT + c] * x[j * DT + c]; A[i * N + j] = s * sharp; mx = Math.max(mx, s * sharp); }
    let z = 0;
    for (let j = 0; j < N; j++) { A[i * N + j] = Math.exp(A[i * N + j] - mx); z += A[i * N + j]; }
    for (let j = 0; j < N; j++) A[i * N + j] /= z;
  }
  return A;
}

// toy cross-attention for the example transcript: each word looks at its own time span
export function toyCross(tokens, nAudio = 600, seed = 3) {
  const r = rng(seed), A = new Float32Array(tokens.length * nAudio);
  tokens.forEach((tk, i) => {
    let z = 0;
    for (let j = 0; j < nAudio; j++) {
      const t = j * 0.02;
      let v = 0.02 + 0.02 * r();
      if (tk.a != null) { const c = (tk.a + tk.b) / 2, s = Math.max(0.08, (tk.b - tk.a) / 2); v += Math.exp(-(((t - c) / s) ** 2)); }
      A[i * nAudio + j] = v; z += v;
    }
    for (let j = 0; j < nAudio; j++) A[i * nAudio + j] /= z;
  });
  return A;
}
