// SigLIP's text tokenizer: a SentencePiece *unigram* model (32,000 pieces), as
// run by Hugging Face's SiglipTokenizer. public/siglip_spm_vocab.txt is the real
// vocab: one line per id, "piece \t log-prob score \t type".
//   1. clean: lowercase, delete ASCII punctuation, collapse whitespace
//   2. "▁" marks a word start: "▁" + text with spaces → "▁"
//   3. Viterbi: of all ways to cut the string into vocab pieces, keep the one
//      with the highest total score (sum of log-probabilities)
//   4. characters not in the vocab fall back to their UTF-8 bytes <0xXX>
//   5. append </s> (id 1) and pad with </s> to 64
// Differences from the C++ library: NFKC via String.normalize (SentencePiece's
// nmt_nfkc adds a few control-character rules that don't matter for captions).

export const EOS = 1, CTX_S = 64;
const PUNCT = /[!"#$%&'()*+,\-./:;<=>?@[\\\]^_`{|}~]/g; // Python's string.punctuation
const NORMAL = 1, BYTE = 6;

export function buildSpm(text) {
  const lines = text.split("\n").filter(Boolean);
  const pieces = [], score = new Map(), id = new Map(), byteId = new Map();
  let minScore = Infinity, maxLen = 0;
  lines.forEach((l, i) => {
    const [p, s, t] = l.split("\t");
    pieces.push(p);
    id.set(p, i);
    if (+t === NORMAL) {
      score.set(p, +s);
      minScore = Math.min(minScore, +s);
      maxLen = Math.max(maxLen, [...p].length);
    }
    if (+t === BYTE) byteId.set(parseInt(p.slice(3, 5), 16), i);
  });
  const UNK = minScore - 10; // SentencePiece's unknown-piece penalty

  const clean = (t) => t.normalize("NFKC").toLowerCase().replace(PUNCT, "").replace(/\s+/g, " ").trim();
  const marked = (t) => "▁" + clean(t).replace(/ /g, "▁");

  // Viterbi over code points. best[i] = best score of any cut of chars[0..i)
  function viterbi(s) {
    const ch = [...s], n = ch.length;
    const best = new Array(n + 1).fill(-Infinity), back = new Array(n + 1);
    best[0] = 0;
    for (let i = 0; i < n; i++) {
      if (best[i] === -Infinity) continue;
      let single = false;
      let cand = "";
      for (let L = 1; L <= maxLen && i + L <= n; L++) {
        cand += ch[i + L - 1];
        const sc = score.get(cand);
        if (sc === undefined) continue;
        if (L === 1) single = true;
        if (best[i] + sc > best[i + L]) { best[i + L] = best[i] + sc; back[i + L] = { from: i, piece: cand, score: sc }; }
      }
      if (!single && best[i] + UNK > best[i + 1]) best[i + 1] = best[i] + UNK, back[i + 1] = { from: i, piece: ch[i], score: UNK, unk: true };
    }
    const path = [];
    for (let k = n; k > 0; k = back[k].from) path.unshift(back[k]);
    return { path, total: best[n] };
  }

  // pieces → ids, with byte fallback for unknown characters
  function toIds(path) {
    const out = [];
    for (const p of path) {
      if (!p.unk) { out.push({ piece: p.piece, id: id.get(p.piece), score: p.score }); continue; }
      for (const b of new TextEncoder().encode(p.piece)) out.push({ piece: pieces[byteId.get(b)], id: byteId.get(b), score: p.score, byte: true });
    }
    return out;
  }

  // every vocab piece that matches at each position of one word (for the lattice picture)
  function lattice(word) {
    const ch = [...word], edges = [];
    for (let i = 0; i < ch.length; i++) {
      let cand = "";
      for (let L = 1; L <= maxLen && i + L <= ch.length; L++) {
        cand += ch[i + L - 1];
        if (score.has(cand)) edges.push({ from: i, to: i + L, piece: cand, score: score.get(cand) });
      }
    }
    return edges;
  }

  function encode(t) {
    const s = marked(t);
    const { path, total } = viterbi(s);
    return { clean: clean(t), marked: s, toks: toIds(path), total };
  }

  // HF: input_ids = pieces + [</s>], truncated to 63 + </s>, padded with </s> to 64
  function tokenize(t, ctx = CTX_S) {
    let ids = encode(t).toks.map((x) => x.id);
    const rawLen = ids.length + 1, truncated = rawLen > ctx;
    ids = [...ids.slice(0, ctx - 1), EOS];
    return { ids: [...ids, ...Array(ctx - ids.length).fill(EOS)], len: ids.length, rawLen, truncated };
  }

  // one word (with its ▁) segmented on its own: the k best cuts, exactly
  // (k-best Viterbi: keep the top k partial cuts ending at every position)
  function wordCuts(word, k = 4) {
    const ch = [...word], n = ch.length;
    const top = Array.from({ length: n + 1 }, () => []);
    top[0] = [{ pieces: [], total: 0 }];
    for (let i = 0; i < n; i++) {
      if (!top[i].length) continue;
      let cand = "";
      for (let L = 1; L <= maxLen && i + L <= n; L++) {
        cand += ch[i + L - 1];
        const sc = score.get(cand);
        if (sc === undefined) continue;
        for (const p of top[i]) top[i + L].push({ pieces: [...p.pieces, cand], total: p.total + sc });
        top[i + L].sort((a, b) => b.total - a.total);
        top[i + L].length = Math.min(top[i + L].length, k);
      }
    }
    return top[n];
  }

  return { encode, tokenize, viterbi, lattice, wordCuts, clean, marked, vocabSize: pieces.length, pieces, score, id };
}

let promise = null;
export function loadSpm() {
  promise ??= fetch(`${import.meta.env?.BASE_URL ?? "/"}siglip_spm_vocab.txt`).then((r) => r.text()).then(buildSpm);
  return promise;
}
