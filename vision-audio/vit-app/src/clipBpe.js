// A line-for-line port of OpenAI CLIP's simple_tokenizer.py, using the real
// merge file (public/clip_bpe_merges.txt = the 48,894 merges CLIP actually uses).
// Differences: ftfy.fix_text is skipped (only matters for mojibake / ligatures like "ﬁ");
// html.unescape is done twice, as basic_clean() does, for the common entities.
// bpe() also returns a trace of every merge, for the walkthrough animation.

// bytes_to_unicode(): printable bytes map to themselves, the rest to U+0100…
function bytesToUnicode() {
  const bs = [];
  const range = (a, b) => { for (let i = a; i <= b; i++) bs.push(i); };
  range(33, 126); range(161, 172); range(174, 255);
  const cs = [...bs];
  let n = 0;
  for (let b = 0; b < 256; b++) if (!bs.includes(b)) { bs.push(b); cs.push(256 + n); n++; }
  return bs.map((b, i) => [b, String.fromCharCode(cs[i])]);
}

const PAT = /<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+/giu;
export const SOT = "<|startoftext|>", EOT = "<|endoftext|>";

export function buildTokenizer(mergesText) {
  const merges = mergesText.split("\n").filter(Boolean).map((l) => l.split(" "));
  const pairs = bytesToUnicode();
  const byteEnc = new Map(pairs);
  const base = pairs.map(([, c]) => c);
  const vocab = [...base, ...base.map((c) => c + "</w>"), ...merges.map(([a, b]) => a + b), SOT, EOT];
  const encoder = new Map(vocab.map((v, i) => [v, i]));
  const ranks = new Map(merges.map(([a, b], i) => [`${a} ${b}`, i]));

  // one pre-split word -> pieces, recording each merge as it fires
  function bpe(token) {
    let word = [...token];
    word[word.length - 1] += "</w>";
    const trace = [{ word: [...word] }];
    while (word.length > 1) {
      let best = null, bestRank = Infinity;
      for (let i = 0; i < word.length - 1; i++) {
        const r = ranks.get(`${word[i]} ${word[i + 1]}`);
        if (r !== undefined && r < bestRank) { bestRank = r; best = [word[i], word[i + 1]]; }
      }
      if (!best) break;
      const next = [];
      for (let i = 0; i < word.length; i++) {
        if (i < word.length - 1 && word[i] === best[0] && word[i + 1] === best[1]) { next.push(best[0] + best[1]); i++; }
        else next.push(word[i]);
      }
      word = next;
      trace.push({ pair: best, rank: bestRank, word: [...word] });
    }
    return { pieces: word, trace };
  }

  const ENT = { amp: "&", lt: "<", gt: ">", quot: '"', apos: "'", nbsp: "\u00a0" };
  const unescape = (t) => t.replace(/&(#x[0-9a-f]+|#\d+|[a-z]+);/gi, (m, e) =>
    e[0] === "#" ? String.fromCodePoint(e[1] === "x" || e[1] === "X" ? parseInt(e.slice(2), 16) : +e.slice(1)) : ENT[e.toLowerCase()] ?? m);
  function words(text) {
    const clean = unescape(unescape(text)).trim().replace(/\s+/g, " ").trim().toLowerCase();
    return clean.match(PAT) ?? [];
  }

  // full encode with per-word detail: [{ word, bytes, pieces, ids, trace }]
  function encodeDetailed(text) {
    return words(text).map((w) => {
      if (w === SOT || w === EOT) return { word: w, pieces: [w], ids: [encoder.get(w)], trace: [] };
      const bytes = [...new TextEncoder().encode(w)].map((b) => byteEnc.get(b)).join("");
      const { pieces, trace } = bpe(bytes);
      return { word: w, pieces, ids: pieces.map((p) => encoder.get(p)), trace };
    });
  }

  // tokenize() from clip.py: <sot> + ids + <eot>, zero-padded to ctx. Too long: clip.tokenize
  // raises by default; with truncate=True it cuts to ctx and forces <eot> into the last slot
  // (what we draw). rawLen = length before cutting.
  function tokenize(text, ctx = 77) {
    let ids = [encoder.get(SOT), ...encodeDetailed(text).flatMap((d) => d.ids), encoder.get(EOT)];
    const rawLen = ids.length, truncated = rawLen > ctx;
    if (truncated) ids = [...ids.slice(0, ctx - 1), encoder.get(EOT)];
    return { ids: [...ids, ...Array(ctx - ids.length).fill(0)], len: ids.length, rawLen, truncated };
  }

  return { encodeDetailed, tokenize, vocabSize: vocab.length, nMerges: merges.length, encoder, ranks };
}

let promise = null;
export function loadTokenizer() {
  promise ??= fetch(`${import.meta.env?.BASE_URL ?? "/"}clip_bpe_merges.txt`).then((r) => r.text()).then(buildTokenizer);
  return promise;
}
