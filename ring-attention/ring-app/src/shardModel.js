// How each method shards the sequence, as a step-by-step walk through the
// sharding code. Every step returns `blocks` of token rows to draw; a token
// item is { t, own, hl, dim, isNew } (own = GPU it ends up on).

const parse = (src) =>
  src.replace(/^\n/, "").replace(/\n\s*$/, "").split("\n").map((line) => {
    const m = line.match(/^(.*?)\s*#@(.*)$/);
    return m ? { text: m[1], tags: m[2].trim().split(/\s+/) } : { text: line, tags: [] };
  });

export const SHARD_SRC = {
  ring: parse(String.raw`
def extract_local(value, cu_seqlens, rank, world_size):  #@def
    local_values = []  #@init
    for i in range(len(cu_seqlens) - 1):  #@loop
        start, end = cu_seqlens[i], cu_seqlens[i + 1]  #@seq
        local_value = value[start:end].chunk(world_size, dim=0)[rank].detach().clone()  #@chunk pick
        local_values.append(local_value)  #@pick
    return torch.cat(local_values, dim=0).contiguous()  #@cat
`),
  zigzag: parse(String.raw`
def extract_local(value, cu_seqlens, rank, world_size):  #@def
    local_values = []  #@init
    for i in range(len(cu_seqlens) - 1):  #@loop
        start, end = cu_seqlens[i], cu_seqlens[i + 1]  #@seq
        local_value = value[start:end].chunk(2 * world_size, dim=0)  #@chunk
        local_values.extend(  #@pick
            [  #@pick
                local_value[rank].detach().clone(),  #@pick
                local_value[2 * world_size - 1 - rank].detach().clone(),  #@pick
            ]  #@pick
        )  #@pick
    return torch.cat(local_values, dim=0).contiguous()  #@cat
`),
  stripe: parse(String.raw`
def extract_local(value, rank, world_size, dim=1):  #@def
    # sequence length split on dim=1
    value = torch.stack(value.split(world_size, dim=dim), dim=dim).transpose(  #@split stack transpose
        dim, dim + 1  #@transpose
    )  #@transpose
    slicer = [rank if i == dim else slice(None) for i in range(len(value.shape))]  #@slice
    return value[slicer].contiguous()  #@slice
`),
  tree: parse(String.raw`
# decoding (illustrative): one new query token, a long KV cache
def shard_for_decode(q, k, v, rank, world_size, dim=1):  #@def
    q_local = q  #@q
    k_local = k.chunk(world_size, dim=dim)[rank]  #@kv
    v_local = v.chunk(world_size, dim=dim)[rank]  #@kv
    return q_local, k_local, v_local  #@ret
`),
};

export const PREP_SRC = parse(String.raw`
# typical way to build a varlen batch (not from the repo)
seqs = [seq0, seq1, seq2]  #@lens
seqlens = torch.tensor([s.shape[0] for s in seqs])  #@lens
value = torch.cat(seqs, dim=0)  #@pack
cu_seqlens = F.pad(seqlens.cumsum(0), (1, 0)).to(torch.int32)  #@cu
max_seqlen = seqlens.max().item()  #@cu
`);

// Sequence lengths per method / mode. Lengths are multiples of 2N so both
// ring (.chunk(N)) and zig-zag (.chunk(2N)) split them evenly.
export function seqLens(method, N, packed) {
  if (method === "tree") return [3 * N];
  if (method === "stripe" || !packed) return [4 * N];
  return [2 * N, 4 * N, 2 * N];
}

export const cumsum = (ls) => ls.reduce((a, l) => [...a, a[a.length - 1] + l], [0]);

// Which GPU token t ends up on.
export function ownerOf(method, N, lens, t) {
  const cu = cumsum(lens);
  const i = cu.findIndex((c, k) => t >= c && t < cu[k + 1]);
  const len = lens[i], p = t - cu[i];
  if (method === "stripe") return t % N;
  if (method === "zigzag") {
    const c = Math.floor(p / (len / (2 * N)));
    return c < N ? c : 2 * N - 1 - c;
  }
  return Math.floor(p / (len / N)); // ring, tree (KV)
}

// Tokens a rank holds, in local order.
export function localTokens(method, N, lens, r) {
  const cu = cumsum(lens);
  const out = [];
  lens.forEach((len, i) => {
    const toks = Array.from({ length: len }, (_, p) => cu[i] + p);
    if (method === "stripe") out.push(...toks.filter((t) => t % N === r));
    else if (method === "zigzag") {
      const c = len / (2 * N);
      out.push(...toks.slice(r * c, (r + 1) * c), ...toks.slice((2 * N - 1 - r) * c, (2 * N - r) * c));
    } else {
      const c = len / N;
      out.push(...toks.slice(r * c, (r + 1) * c));
    }
  });
  return out;
}

const item = (method, N, lens, t, extra = {}) => ({ t, own: ownerOf(method, N, lens, t), ...extra });

export function program(method, N, packed, sel) {
  const lens = seqLens(method, N, packed);
  const cu = cumsum(lens);
  const S = cu[cu.length - 1];
  const all = Array.from({ length: S }, (_, t) => t);
  const steps = [];
  // full sequence row; with `marks`, every cu_seqlens boundary gets a labelled tick
  const fullRow = (fn = () => ({}), marks = false) => {
    const items = [];
    all.forEach((t) => {
      if (cu.includes(t) && (marks || t > 0)) items.push({ gap: true, mark: marks ? t : undefined });
      items.push(item(method, N, lens, t, fn(t)));
    });
    if (marks) items.push({ gap: true, mark: S });
    return items;
  };
  const toks = (ts) => (ts.length > 6 ? `${ts[0]}…${ts[ts.length - 1]}` : ts.join(","));
  const shape = (n) => `[${n}, H, D]`;

  if (method === "ring" || method === "zigzag") {
    const k = method === "ring" ? N : 2 * N;
    const picks = (r) => (method === "ring" ? [r] : [r, 2 * N - 1 - r]);
    const locals = Array.from({ length: N }, () => []); // per rank: list of pieces
    const localBlock = (newSeq = -1) => ({
      title: "local_values (one list per rank)",
      rows: locals.map((pieces, r) => ({
        label: `rank ${r}`, rank: r,
        items: pieces.length
          ? pieces.flatMap((pc, j) => [...(j ? [{ gap: true }] : []), ...pc.toks.map((t) => item(method, N, lens, t, { isNew: pc.seq === newSeq }))])
          : [{ empty: "[]" }],
      })),
    });
    const n = (m, t, after) => ({ m, t, after });
    const locStr = () => (locals[sel].length ? `[${locals[sel].map((pc) => `t[${toks(pc.toks)}]`).join(", ")}]` : "[]");
    const defNote = n("def extract_local", `rank = ${sel}, world_size = ${N}, value = qkv: [${S}, 3, H, D] (dim 0 = tokens)`);
    const loopNote = (i) => n("for i in range", `len(cu_seqlens) − 1 = ${lens.length} → i = ${i}`);
    const seNote = (i) => n("start, end = cu_seqlens", `cu_seqlens = [${cu.join(", ")}] → start = cu_seqlens[${i}] = ${cu[i]}, end = cu_seqlens[${i + 1}] = ${cu[i + 1]}`);
    const appendNote = () => n(method === "ring" ? "local_values.append" : "local_values.extend(", `local_values = ${locStr()}`);
    const chunkList = (chs) => (chs.length <= 8 ? chs.map((ch, j) => `c${j}=t[${ch.join(",")}]`).join(" ") : `c0=t[${chs[0].join(",")}] … c${chs.length - 1}=t[${chs[chs.length - 1].join(",")}]`);
    const maxLen = Math.max(...lens);
    const prepFinal = packed
      ? [n("seqlens = torch.tensor", `tensor([${lens.join(", ")}])`), n("value = torch.cat", `[${S}, H, D]`), n("cu_seqlens = F.pad", `cumsum = [${cu.slice(1).join(", ")}] → put a 0 in front → [${cu.join(", ")}]`), n("max_seqlen", `max(${lens.join(", ")}) = ${maxLen}`)]
      : [n("value = seq0", `[${S}, H, D]`), n("cu_seqlens = torch.tensor", `S = ${S} → [0, ${S}]`), n("max_seqlen = S", `${S}`)];

    // ---- prep: where cu_seqlens comes from ----
    if (packed) {
      steps.push({
        code: "prep", tag: "lens", title: "Three sequences, three lengths",
        desc: `Padding every sequence to the longest (${maxLen}) wastes <b>${lens.length * maxLen - S}</b> of ${lens.length * maxLen} slots (hatched).`,
        blocks: [{ title: `padded batch  [${lens.length}, ${maxLen}, H, D]`, rows: lens.map((len, i) => ({ label: `seq ${i}`, items: [...Array.from({ length: len }, (_, p) => item(method, N, lens, cu[i] + p, { neutral: true })), ...Array.from({ length: maxLen - len }, () => ({ pad: true }))] })) }],
        prepNotes: [n("seqs = [", `lengths ${lens.join(", ")}`), n("seqlens = torch.tensor", `tensor([${lens.join(", ")}])`)],
      });
      steps.push({
        code: "prep", tag: "pack", title: "Pack: concatenate along the token axis",
        desc: `Varlen kernels skip the padding: concatenate all sequences into one tensor of ${S} tokens. The kernel needs to know where each one starts.`,
        blocks: [{ title: `packed value  [${S}, H, D]`, rows: [{ label: "value", items: fullRow(() => ({ neutral: true })) }] }],
        prepNotes: [n("seqlens = torch.tensor", `tensor([${lens.join(", ")}])`), n("value = torch.cat", `[${S}, H, D]: ${lens.map((l, i) => `seq ${i} at ${cu[i]}..${cu[i + 1] - 1}`).join(", ")}`)],
      });
      steps.push({
        code: "prep", tag: "cu", title: "cu_seqlens: the start of every sequence, plus the end",
        desc: `<code>cu_seqlens = [${cu.join(", ")}]</code>: the running sum of the lengths, starting at 0. Sequence $i$ is <code>value[cu_seqlens[i]:cu_seqlens[i+1]]</code>.`,
        blocks: [{ title: "value with the cu_seqlens boundaries", rows: [{ label: "value", items: fullRow(() => ({ neutral: true }), true) }] }],
        prepNotes: prepFinal,
      });
    } else {
      steps.push({
        code: "prep", tag: "cu", title: "One sequence = a batch of one",
        desc: `With one sequence of ${S} tokens, <code>cu_seqlens = [0, ${S}]</code>. The same varlen code runs, with one loop iteration.`,
        blocks: [{ title: `value  [${S}, H, D]`, rows: [{ label: "value", items: fullRow(() => ({ neutral: true }), true) }] }],
        prepNotes: prepFinal,
      });
    }

    const lcu = cu.map((x) => x / N);
    const maxL = Math.max(...lens), lmax = maxL / N, ns = lens.length;
    const mineAll = localTokens(method, N, lens, sel);
    const mineSeq = lens.map((len, i) => mineAll.filter((t) => t >= cu[i] && t < cu[i + 1]));
    const dn = {
      qkv: [n("qkv = torch.randn", `[${S}, 3, H, D]: q, k, v of all ${S} tokens, packed`), n("dist.broadcast", "identical on every rank"), n("local_cu_seqlens =", `[${cu.join(", ")}] // ${N} = [${lcu.join(", ")}]`), n("local_max_seqlen =", `${maxL} // ${N} = ${lmax}`)],
      lse: [n("for lse in extract_lse", `${lens.map((l) => `[H, ${l}]`).join(", ")}: one piece per sequence`), n(method === "ring" ? "local_lse = lse.chunk" : "torch.cat(", `rank ${sel}: ${mineSeq.map((m) => `t[${toks(m)}]`).join(" | ")} → [H, ${S / N}]`)],
    };
    const qkvDone = n("local_qkv = extract_local", `[${S / N}, 3, H, D] = t[${mineAll.join(",")}]`);
    const drv = (...ks) => ks.flatMap((k) => (k === "qkvDone" ? [qkvDone] : dn[k]));
    steps.push({
      code: "test", tag: "t_qkv", title: "The test: local_qkv = extract_local(qkv, …)",
      desc: `Every rank starts with the same packed <code>qkv</code> of ${S} tokens. Each rank calls <code>extract_local</code> to keep its share, and its <code>cu_seqlens</code> becomes <code>[${lcu.join(", ")}]</code>.`,
      blocks: [{ title: `qkv  [${S}, 3, H, D]`, rows: [{ label: "qkv", items: fullRow(() => ({}), true) }] }],
      driverNotes: drv("qkv"),
    });
    steps.push({
      tag: "def", focus: ["def", "init"], title: `extract_local(value, cu_seqlens, rank=${sel}, world_size=${N})`,
      desc: `Each rank calls this on the full tensor and gets back only its share. Colours show the GPU each token ends up on.`,
      blocks: [{ title: `value  [${S}, H, D]`, rows: [{ label: "value", items: fullRow(() => ({}), true) }] }, localBlock()],
      notes: [defNote, n("local_values = []", "starts empty")],
      driverNotes: drv("qkv"),
    });
    lens.forEach((len, i) => {
      const start = cu[i], end = cu[i + 1], c = len / k;
      steps.push({
        tag: "seq", focus: ["loop", "seq"], title: `i = ${i}: start, end = cu_seqlens[${i}], cu_seqlens[${i + 1}]`,
        desc: `<code>value[${start}:${end}]</code> is sequence ${i}. Each sequence is sharded on its own, so every rank gets a piece of <b>every</b> sequence.`,
        blocks: [{ title: `value[${start}:${end}]  ${shape(len)}`, rows: [{ label: "value", items: fullRow((t) => ({ dim: t < start || t >= end, hl: t >= start && t < end }), true) }] }, localBlock()],
        notes: [defNote, loopNote(i), seNote(i), appendNote()],
        driverNotes: drv("qkv"),
      });
      const chunks = Array.from({ length: k }, (_, j) => Array.from({ length: c }, (_, p) => start + j * c + p));
      const mine = picks(sel);
      steps.push({
        tag: "chunk", focus: ["loop", "chunk"], title: `.chunk(${method === "ring" ? "world_size" : "2 * world_size"}) → ${k} chunks of ${c}`,
        desc: method === "ring"
          ? `<code>value[${start}:${end}].chunk(${N}, dim=0)</code> cuts sequence ${i} into ${N} chunks of ${c} tokens.`
          : `<code>value[${start}:${end}].chunk(${2 * N}, dim=0)</code> cuts sequence ${i} into ${2 * N} chunks of ${c} tokens.`,
        blocks: [{
          title: `value[${start}:${end}].chunk(${k})  →  ${k} × ${shape(c)}`,
          rows: [{ label: `seq ${i}`, items: chunks.flatMap((ch, j) => [...(j ? [{ gap: true }] : []), ...ch.map((t) => item(method, N, lens, t, { hl: mine.includes(j) }))]) }],
          chunkLabels: chunks.map((ch, j) => ({ j, own: method === "ring" ? j : j < N ? j : 2 * N - 1 - j, mine: mine.includes(j) })),
        }, localBlock()],
        driverNotes: drv("qkv"),
        notes: [defNote, loopNote(i), seNote(i), n("local_value = value[start:end]", method === "ring" ? `value[${start}:${end}] (${len} tokens).chunk(${N}) → ${N} × ${c} tokens` : `value[${start}:${end}] (${len} tokens).chunk(${2 * N}) → ${2 * N} × ${c}: ${chunkList(chunks)}`), appendNote()],
      });
      for (let r = 0; r < N; r++) locals[r].push({ seq: i, toks: picks(r).flatMap((j) => chunks[j]) });
      steps.push({
        tag: "pick", focus: ["loop", "pick"], title: method === "ring" ? `[rank] → append` : `[rank], [2 * world_size − 1 − rank] → extend`,
        desc: method === "ring"
          ? `<code>[rank]</code> keeps chunk ${sel} (tokens ${chunks[sel].join(", ")}) as an independent copy, <code>.detach().clone()</code>. Every rank does the same.`
          : `Rank ${sel} keeps <code>local_value[${sel}]</code> (tokens ${chunks[mine[0]].join(", ")}) and <code>local_value[${2 * N - 1 - sel}]</code> (tokens ${chunks[mine[1]].join(", ")}).`,
        blocks: [localBlock(i)],
        driverNotes: drv("qkv"),
        notes: [defNote, loopNote(i), seNote(i), ...(method === "ring" ? [n("local_value = value[start:end]", `[rank=${sel}] = t[${chunks[sel].join(",")}]`)] : [n("local_value[rank]", `local_value[${sel}] = t[${chunks[mine[0]].join(",")}]`), n("local_value[2 * world_size", `2·${N} − 1 − ${sel} = ${mine[1]} → t[${chunks[mine[1]].join(",")}]`)]), appendNote()],
      });
    });
    steps.push({
      tag: "cat", title: "torch.cat(local_values) → each rank's local tensor",
      desc: `Each rank concatenates its pieces into ${S / N} tokens. Every length must be divisible by ${method === "ring" ? "world_size" : "2 * world_size"}.${packed ? " The varlen kernel gets the local <code>cu_seqlens</code> on every rank." : ""}`,
      blocks: [{
        title: `returned tensor per rank  ${shape(S / N)}`,
        rows: Array.from({ length: N }, (_, r) => ({ label: `rank ${r}`, rank: r, items: withLocalGaps(localTokens(method, N, lens, r).map((t) => item(method, N, lens, t, { hl: r === sel })), lcu) })),
        localIdx: true,
      }],
      notes: [defNote, n("return torch.cat", `[${S / N}, H, D] = t[${localTokens(method, N, lens, sel).join(",")}]  ·  local cu_seqlens = cu_seqlens // ${N} = [${lcu.join(", ")}]`)],
      lcu,
      driverNotes: drv("qkv", "qkvDone"),
    });

    // ---- lse: one value per query row, so it is cut exactly like q ----
    const owner = (t) => ownerOf(method, N, lens, t);
    const seqToks = lens.map((len, i) => Array.from({ length: len }, (_, p) => cu[i] + p));
    steps.push({
      code: "test", tag: "t_lse", title: "lse is sharded like q: per sequence, then the same chunks",
      desc: `lse has one value per query row, so it is cut with the <b>same rule</b> as <code>qkv</code>. Each rank ends with <code>[H, ${S / N}]</code>, and lse never moves during the ring.`,
      blocks: [
        { title: `lse  [H, ${S}] · coloured by the GPU that owns the query row`, rows: [{ label: "lse", items: [...seqToks.flatMap((ts, j) => [{ gap: true, mark: cu[j], markName: "cu_seqlens" }, ...ts.map((t) => item(method, N, lens, t, { lse: true, hl: owner(t) === sel }))]), { gap: true, mark: cu.at(-1), markName: "cu_seqlens" }] }] },
        { title: `local_lse per rank  [H, ${S / N}]`, rows: Array.from({ length: N }, (_, r) => ({ label: `rank ${r}`, rank: r, items: withLocalGaps(localTokens(method, N, lens, r).map((t) => item(method, N, lens, t, { lse: true, hl: r === sel })), lcu) })) },
      ],
      driverNotes: drv("qkv", "qkvDone", "lse"),
    });
    return { lens, cu, S, steps };
  }

  if (method === "stripe") {
    const G = S / N; // rows after stack
    const groups = Array.from({ length: G }, (_, p) => Array.from({ length: N }, (_, q) => p * N + q));
    const n = (m, t, after) => ({ m, t, after });
    const defNote = n("def extract_local", `rank = ${sel}, world_size = ${N}, value: [B, ${S}, H, D]`);
    const mine = groups.map((g) => g[sel]);
    steps.push({
      tag: "def", title: "Input: value [B, S, H, D]",
      notes: [defNote],
      desc: `One sequence of ${S} tokens on <code>dim=1</code>. Stripe has no varlen version in the repo.`,
      blocks: [{ title: `value  [B, ${S}, H, D]`, rows: [{ label: "value", items: fullRow() }] }],
    });
    steps.push({
      tag: "split", notes: [defNote, n("value = torch.stack", `value.split(${N}, dim=1) → ${G} pieces of ${N} tokens: t[${groups[0].join(",")}], t[${groups[1].join(",")}], …`)],
      title: `value.split(world_size) → ${G} pieces of ${N}`,
      desc: `<code>split(${N})</code> cuts the sequence into pieces of <b>world_size</b> consecutive tokens. Each piece holds one token per rank.`,
      blocks: [{ title: `tuple of ${G} tensors  [B, ${N}, H, D]`, rows: [{ label: "pieces", items: groups.flatMap((g, p) => [...(p ? [{ gap: true }] : []), ...g.map((t) => item(method, N, lens, t))]) }] }],
    });
    steps.push({
      tag: "stack", notes: [defNote, n("value = torch.stack", `${G} pieces of [B, ${N}] → stack(dim=1) → [B, ${G}, ${N}, H, D]: row p = t[pN … pN+${N - 1}]`)],
      title: `torch.stack(..., dim=1) → [B, ${G}, ${N}, H, D]`,
      desc: `Stacking the pieces adds a new axis: row $p$ holds tokens $pN \\dots pN+${N - 1}$. Each <b>column</b> already has a single colour.`,
      blocks: [{ title: `[B, ${G} (piece), ${N} (position in piece), H, D]`, rows: groups.map((g, p) => ({ label: `p=${p}`, items: g.map((t) => item(method, N, lens, t)) })) }],
    });
    steps.push({
      tag: "transpose", notes: [defNote, n("value = torch.stack", `[B, ${G}, ${N}, H, D] → transpose(1, 2) → [B, ${N}, ${G}, H, D]`), n("dim, dim + 1", `swap axes 1 and 2 → row r = t[r, r+${N}, r+${2 * N}, …]`)],
      title: `.transpose(1, 2) → [B, ${N}, ${G}, H, D]`,
      desc: `Swapping the two axes turns the columns into rows: row $r$ is tokens $r, r+N, r+2N, \\dots$. One row per rank.`,
      blocks: [{ title: `[B, ${N} (rank), ${G} (local position), H, D]`, rows: Array.from({ length: N }, (_, r) => ({ label: `r=${r}`, rank: r, items: groups.map((g) => item(method, N, lens, g[r], { hl: r === sel })) })) }],
    });
    steps.push({
      tag: "slice", notes: [defNote, n("value = torch.stack", `→ [B, ${N}, ${G}, H, D]`), n("slicer =", `[slice(None), ${sel}, slice(None), slice(None), slice(None)], i.e. [:, ${sel}]`), n("return value[slicer]", `[B, ${G}, H, D] = t[${mine.join(",")}] (same as value[:, ${sel}::${N}])`)],
      title: `value[:, rank] → local [B, ${G}, H, D]`,
      desc: `<code>slicer</code> picks index <code>rank</code> on <code>dim=1</code>. Rank ${sel} gets tokens ${groups.map((g) => g[sel]).join(", ")}. Local position $p$ is token $pN + r$, which the shifted causal mask relies on.`,
      blocks: [{ title: `local tensor per rank  [B, ${G}, H, D]`, rows: Array.from({ length: N }, (_, r) => ({ label: `rank ${r}`, rank: r, items: localTokens(method, N, lens, r).map((t) => item(method, N, lens, t, { hl: r === sel })) })), localIdx: true }],
    });
    return { lens, cu, S, steps };
  }

  // tree: q replicated, KV cache split contiguously
  const c = S / N;
  const n = (m, t, after) => ({ m, t, after });
  const tdef = n("def shard_for_decode", `rank = ${sel}, world_size = ${N}, q: [B, 1, H, D], k, v: [B, ${S}, H, D]`);
  const kNote = n("k_local = k.chunk", `${N} chunks of ${c} → [${sel}] = t[${localTokens("ring", N, lens, sel).join(",")}]`);
  steps.push({
    tag: "def", title: "Input: a KV cache and one new query token",
    notes: [tdef],
    desc: `The KV cache holds ${S} past tokens. The new token $q$ (position ${S}) attends to all of them.`,
    blocks: [{ title: `k, v  [B, ${S}, H, D]`, rows: [{ label: "k, v", items: fullRow() }] }, { title: "q  [B, 1, H, D]", rows: [{ label: "q", items: [{ t: S, own: -1, q: true }] }] }],
  });
  steps.push({
    tag: "q", notes: [tdef, n("q_local = q", `the same token (t${S}) on every rank`)],
    title: "q is replicated",
    desc: "Every rank gets the same query token. It is tiny, and every shard needs it.",
    blocks: [{ title: "q on every rank", rows: Array.from({ length: N }, (_, r) => ({ label: `rank ${r}`, rank: r, items: [{ t: S, own: -1, q: true, hl: r === sel }] })) }],
  });
  steps.push({
    tag: "kv", notes: [tdef, n("q_local = q", `t${S}`), kNote, n("v_local = v.chunk", "same split as k")],
    title: `k.chunk(world_size)[rank] → ${c} tokens each`,
    desc: `The cache is cut into $N=${N}$ contiguous shards. No causal balancing is needed, since the one query sees every key.`,
    blocks: [{ title: "k, v per rank", rows: Array.from({ length: N }, (_, r) => ({ label: `rank ${r}`, rank: r, items: localTokens("ring", N, lens, r).map((t) => item(method, N, lens, t, { hl: r === sel })) })), localIdx: true }],
  });
  steps.push({
    tag: "ret", notes: [tdef, n("q_local = q", `t${S}`), kNote, n("return q_local", `q [B, 1, H, D], k/v [B, ${c}, H, D]`)],
    title: "What each rank computes next",
    desc: "Each rank runs attention of $q$ over its own shard, giving a partial $(o_r, \\ell_r)$.",
    blocks: [{ title: "rank r: (q, k_local, v_local)", rows: Array.from({ length: N }, (_, r) => ({ label: `rank ${r}`, rank: r, items: [{ t: S, own: -1, q: true, hl: r === sel }, { gap: true }, ...localTokens("ring", N, lens, r).map((t) => item(method, N, lens, t, { hl: r === sel }))] })) }],
  });
  return { lens, cu, S, steps };
}

// Insert a gap at each local sequence boundary.
function withLocalGaps(items, lcu) {
  const out = [];
  items.forEach((it, p) => {
    if (p > 0 && lcu.includes(p)) out.push({ gap: true });
    out.push(it);
  });
  return out;
}

// The repo's varlen test driver, abridged to the sharding lines
// (test/test_ring_flash_attn_varlen_func.py and its zig-zag twin).
const driver = (share) => parse(String.raw`
qkv = torch.randn(total_length, 3, nheads, d)  #@t_qkv
dist.broadcast(qkv, src=0)  #@t_qkv
local_cu_seqlens = cu_seqlens // world_size  #@t_qkv
local_max_seqlen = max_seqlen // world_size  #@t_qkv
local_qkv = extract_local(qkv, cu_seqlens, rank, world_size)  #@t_qkv

# lse: one value per query row, from flash-attn  #@t_lse
for lse in extract_lse(lse, cu_seqlens):  #@t_lse
${share}
`);
export const DRIVER_SRC = {
  ring: driver(`    local_lse = lse.chunk(world_size, dim=-1)[rank]  #@t_lse`),
  zigzag: driver(`    local_lse = lse.chunk(2 * world_size, dim=-1)  #@t_lse
    local_lse = torch.cat(  #@t_lse
        [local_lse[rank], local_lse[2 * world_size - 1 - rank]], dim=-1)  #@t_lse`),
};
