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

export const LSE_SRC = parse(String.raw`
def extract_lse(lse, cu_seqlens):
    values = []
    if lse.dim() == 2:  #@d2
        for i in range(len(cu_seqlens) - 1):  #@d2
            start, end = cu_seqlens[i], cu_seqlens[i + 1]  #@d2
            value = lse[:, start:end]  #@d2
            values.append(value)  #@d2
    else:  #@d3
        assert lse.dim() == 3  #@d3
        for i in range(len(cu_seqlens) - 1):  #@d3
            start, end = cu_seqlens[i], cu_seqlens[i + 1]  #@d3
            value = lse[i, :, : end - start]  #@d3
            values.append(value)  #@d3
    return values
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

export function program(method, N, packed, sel, dim3 = false) {
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
      : [n("value = seq0", `[${S}, H, D]`), n("cu_seqlens = torch.tensor", `S = ${S} → [0, ${S}]`)];

    // ---- prep: where cu_seqlens comes from ----
    if (packed) {
      steps.push({
        code: "prep", tag: "lens", title: "Three sequences, three lengths",
        desc: `A batch of ${lens.length} sequences with lengths ${lens.join(", ")}. The classic way to batch them is to <b>pad</b> every one to the longest (${maxLen}): a <code>[${lens.length}, ${maxLen}]</code> tensor with ${lens.length * maxLen - S} of its ${lens.length * maxLen} slots wasted on padding (hatched).`,
        blocks: [{ title: `padded batch  [${lens.length}, ${maxLen}, H, D]`, rows: lens.map((len, i) => ({ label: `seq ${i}`, items: [...Array.from({ length: len }, (_, p) => item(method, N, lens, cu[i] + p, { neutral: true })), ...Array.from({ length: maxLen - len }, () => ({ pad: true }))] })) }],
        prepNotes: [n("seqs = [", `lengths ${lens.join(", ")}`), n("seqlens = torch.tensor", `tensor([${lens.join(", ")}])`)],
      });
      steps.push({
        code: "prep", tag: "pack", title: "Pack: concatenate along the token axis",
        desc: `Varlen kernels skip the padding: concatenate all sequences into one tensor of ${S} tokens, with no batch axis. The sequences now sit back to back, so the kernel needs to be told where each one starts.`,
        blocks: [{ title: `packed value  [${S}, H, D]`, rows: [{ label: "value", items: fullRow(() => ({ neutral: true })) }] }],
        prepNotes: [n("seqlens = torch.tensor", `tensor([${lens.join(", ")}])`), n("value = torch.cat", `[${S}, H, D]: ${lens.map((l, i) => `seq ${i} at ${cu[i]}..${cu[i + 1] - 1}`).join(", ")}`)],
      });
      steps.push({
        code: "prep", tag: "cu", title: "cu_seqlens: the start of every sequence, plus the end",
        desc: `<code>cu_seqlens</code> is the cumulative sum of the lengths with a 0 in front: <code>[0, ${lens.map((_, i) => lens.slice(0, i + 1).join("+")).join(", ")}] = [${cu.join(", ")}]</code>. Entry $i$ is where sequence $i$ starts and entry $i+1$ is where it ends, so sequence $i$ is <code>value[cu_seqlens[i]:cu_seqlens[i+1]]</code>. It has one more entry than there are sequences.`,
        blocks: [{ title: "value with the cu_seqlens boundaries", rows: [{ label: "value", items: fullRow(() => ({ neutral: true }), true) }] }],
        prepNotes: prepFinal,
      });
    } else {
      steps.push({
        code: "prep", tag: "cu", title: "One sequence = a batch of one",
        desc: `With a single sequence of ${S} tokens, <code>cu_seqlens = [0, ${S}]</code>: it starts at 0 and ends at ${S}. The same varlen code runs, and its loop just has one iteration.`,
        blocks: [{ title: `value  [${S}, H, D]`, rows: [{ label: "value", items: fullRow(() => ({ neutral: true }), true) }] }],
        prepNotes: prepFinal,
      });
    }

    const lcu = cu.map((x) => x / N);
    const maxL = Math.max(...lens), lmax = maxL / N, ns = lens.length;
    const mineAll = localTokens(method, N, lens, sel);
    const mineSeq = lens.map((len, i) => mineAll.filter((t) => t >= cu[i] && t < cu[i + 1]));
    const lseShape = dim3 ? `[${ns}, H, ${maxL}]` : `[H, ${S}]`;
    const rlseShape = dim3 ? `[${ns}, H, ${lmax}]` : `[H, ${S / N}]`;
    // a row cut at `marks` boundaries, labelled with `name` (cu_seqlens or local_cu_seqlens)
    const markedRow = (seqs, marks, name, fn = () => ({})) => [
      ...seqs.flatMap((ts, j) => [{ gap: true, mark: marks[j], markName: name }, ...ts.map((t) => item(method, N, lens, t, fn(t, j)))]),
      { gap: true, mark: marks.at(-1), markName: name },
    ];
    const seqToks = lens.map((len, i) => Array.from({ length: len }, (_, p) => cu[i] + p));
    // lse in the kernel's layout: packed [H, total] (one row) or padded [num_seq, H, max] (one row per sequence)
    const lseBlock = (title, seqs, marks, name, w, fn) => (dim3
      ? { title, rows: seqs.map((ts, j) => ({ label: `${name === "cu_seqlens" ? "lse" : "ring_lse"}[${j}]`, items: [...ts.map((t) => item(method, N, lens, t, { lse: true, ...fn(t, j) })), ...Array.from({ length: w - ts.length }, () => ({ pad: true }))] })) }
      : { title, rows: [{ label: name === "cu_seqlens" ? "lse" : "ring_lse", items: markedRow(seqs, marks, name, (t, j) => ({ lse: true, ...fn(t, j) })) }] });
    const dn = {
      qkv: [n("qkv = torch.randn", `[${S}, 3, H, D]: q, k, v of all ${S} tokens, packed`), n("dist.broadcast", "identical on every rank"), n("local_cu_seqlens =", `[${cu.join(", ")}] // ${N} = [${lcu.join(", ")}]`), n("local_max_seqlen =", `${maxL} // ${N} = ${lmax}`)],
      ref: [n("out, lse, _ = flash", `out: [${S}, H, D] · lse: ${lseShape}`)],
      out: [n("local_out = extract_local", `[${S / N}, H, D]: out rows t[${mineAll.join(",")}]`)],
      cut: [n("lse_list = extract_lse(lse", `${lens.map((l) => `[H, ${l}]`).join(", ")}: one piece per sequence`)],
      ring: [n("ring_out, ring_lse, _ =", `on rank ${sel}: ring_out [${S / N}, H, D] · ring_lse ${rlseShape}`)],
      rcut: [n("ring_lse_list = extract_lse", `${lens.map((l) => `[H, ${l / N}]`).join(", ")}`)],
      share: [n(method === "ring" ? "local_lse = lse.chunk" : "[local_lse[rank]", `rank ${sel}: ${mineSeq.map((m) => `t[${toks(m)}]`).join(" | ")}`)],
      cmp: [n('log("out diff"', "≈ 0"), n("lse diff", "≈ 0, per sequence")],
    };
    const qkvDone = n("local_qkv = extract_local", `[${S / N}, 3, H, D] = t[${mineAll.join(",")}]`);
    const drv = (...ks) => ks.flatMap((k) => (k === "qkvDone" ? [qkvDone] : dn[k]));
    steps.push({
      code: "test", tag: "t_qkv", title: "The test: local_qkv = extract_local(qkv, …)",
      desc: `The repo's varlen test starts from one packed <code>qkv</code> of all ${S} tokens, identical on every rank. Each rank calls <code>extract_local</code> to keep only its share, and divides <code>cu_seqlens</code> by $N$ to describe that share: <code>[${lcu.join(", ")}]</code>. Next: step into <code>extract_local</code>, with <code>value = qkv</code>.`,
      blocks: [{ title: `qkv  [${S}, 3, H, D]`, rows: [{ label: "qkv", items: fullRow(() => ({}), true) }] }],
      driverNotes: drv("qkv"),
    });
    steps.push({
      tag: "def", focus: ["def", "init"], title: `extract_local(value, cu_seqlens, rank=${sel}, world_size=${N})`,
      desc: `Every rank calls this on the full packed tensor with its own <code>rank</code>, and gets back only its share. Colours now show the GPU each token will end up on. <code>local_values = []</code> collects the pieces.`,
      blocks: [{ title: `value  [${S}, H, D]`, rows: [{ label: "value", items: fullRow(() => ({}), true) }] }, localBlock()],
      notes: [defNote, n("local_values = []", "starts empty")],
      driverNotes: drv("qkv"),
    });
    lens.forEach((len, i) => {
      const start = cu[i], end = cu[i + 1], c = len / k;
      steps.push({
        tag: "seq", focus: ["loop", "seq"], title: `i = ${i}: start, end = cu_seqlens[${i}], cu_seqlens[${i + 1}]`,
        desc: `<code>start, end = ${start}, ${end}</code>, so <code>value[${start}:${end}]</code> is sequence ${i} (${len} tokens). Each sequence is sharded on its own, so every rank gets a piece of <b>every</b> sequence.`,
        blocks: [{ title: `value[${start}:${end}]  ${shape(len)}`, rows: [{ label: "value", items: fullRow((t) => ({ dim: t < start || t >= end, hl: t >= start && t < end }), true) }] }, localBlock()],
        notes: [defNote, loopNote(i), seNote(i), n("local_value = value[start:end]", `value[${start}:${end}] = sequence ${i}, ${len} tokens`), appendNote()],
        driverNotes: drv("qkv"),
      });
      const chunks = Array.from({ length: k }, (_, j) => Array.from({ length: c }, (_, p) => start + j * c + p));
      const mine = picks(sel);
      steps.push({
        tag: "chunk", focus: ["loop", "chunk"], title: `.chunk(${method === "ring" ? "world_size" : "2 * world_size"}) → ${k} chunks of ${c}`,
        desc: method === "ring"
          ? `<code>value[${start}:${end}].chunk(${N}, dim=0)</code> cuts sequence ${i} into ${N} chunks of ${len}/${N} = ${c} tokens.`
          : `<code>value[${start}:${end}].chunk(${2 * N}, dim=0)</code> cuts sequence ${i} into ${2 * N} chunks of ${len}/${2 * N} = ${c} tokens.`,
        blocks: [{
          title: `value[${start}:${end}].chunk(${k})  →  ${k} × ${shape(c)}`,
          rows: [{ label: `seq ${i}`, items: chunks.flatMap((ch, j) => [...(j ? [{ gap: true }] : []), ...ch.map((t) => item(method, N, lens, t, { hl: mine.includes(j) }))]) }],
          chunkLabels: chunks.map((ch, j) => ({ j, own: method === "ring" ? j : j < N ? j : 2 * N - 1 - j, mine: mine.includes(j) })),
        }, localBlock()],
        driverNotes: drv("qkv"),
        notes: [defNote, loopNote(i), seNote(i), n("local_value = value[start:end]", method === "ring" ? `value[${start}:${end}] (${len} tokens).chunk(${N}) → ${N} × ${c} tokens → [rank=${sel}] = t[${chunks[sel].join(",")}]` : `value[${start}:${end}] (${len} tokens).chunk(${2 * N}) → ${2 * N} × ${c}: ${chunkList(chunks)}`), appendNote()],
      });
      for (let r = 0; r < N; r++) locals[r].push({ seq: i, toks: picks(r).flatMap((j) => chunks[j]) });
      steps.push({
        tag: "pick", focus: ["loop", "pick"], title: method === "ring" ? `[rank] → append` : `[rank], [2 * world_size − 1 − rank] → extend`,
        desc: method === "ring"
          ? `<code>[rank]</code> keeps chunk ${sel} for rank ${sel} (tokens ${chunks[sel].join(", ")}); <code>.detach().clone()</code> makes it an independent copy, and it is appended. Every other rank does the same with its own index.`
          : `Rank ${sel} keeps <code>local_value[${sel}]</code> (tokens ${chunks[mine[0]].join(", ")}) and <code>local_value[${2 * N - 1 - sel}]</code> (tokens ${chunks[mine[1]].join(", ")}), and extends its list with both.`,
        blocks: [localBlock(i)],
        driverNotes: drv("qkv"),
        notes: [defNote, loopNote(i), seNote(i), ...(method === "ring" ? [n("local_value = value[start:end]", `[rank=${sel}] = t[${chunks[sel].join(",")}]`)] : [n("local_value[rank]", `local_value[${sel}] = t[${chunks[mine[0]].join(",")}]`), n("local_value[2 * world_size", `2·${N} − 1 − ${sel} = ${mine[1]} → t[${chunks[mine[1]].join(",")}]`)]), appendNote()],
      });
    });
    steps.push({
      tag: "cat", title: "torch.cat(local_values) → each rank's local tensor",
      desc: `Each rank concatenates its pieces into one tensor of ${S}/${N} = ${S / N} tokens. Every sequence shrank by exactly $N$, so the local boundaries are just <code>cu_seqlens // world_size = [${lcu.join(", ")}]</code>${packed ? ": the attention code passes this local <code>cu_seqlens</code> to the varlen kernel on every rank" : ""}. This is why every length must be divisible by ${method === "ring" ? "world_size" : "2 * world_size"}.${method === "zigzag" ? " Inside each sequence the local order is still increasing (early chunk, then late chunk)." : ""}`,
      blocks: [{
        title: `returned tensor per rank  ${shape(S / N)}`,
        rows: Array.from({ length: N }, (_, r) => ({ label: `rank ${r}`, rank: r, items: withLocalGaps(localTokens(method, N, lens, r).map((t) => item(method, N, lens, t, { hl: r === sel })), lcu) })),
        localIdx: true,
      }],
      notes: [defNote, n("return torch.cat", `[${S / N}, H, D] = t[${localTokens(method, N, lens, sel).join(",")}]  ·  local cu_seqlens = cu_seqlens // ${N} = [${lcu.join(", ")}]`)],
      lcu,
      driverNotes: drv("qkv", "qkvDone"),
    });

    // ---- back in the test: reference out / lse, then cut them the same way ----
    const owner = (t) => ownerOf(method, N, lens, t);
    const perRank = (fn = () => ({})) => Array.from({ length: N }, (_, r) => ({ label: `rank ${r}`, rank: r, items: withLocalGaps(localTokens(method, N, lens, r).map((t) => item(method, N, lens, t, { hl: r === sel, ...fn(t) })), lcu) }));
    steps.push({
      code: "test", tag: "t_ref", title: "Reference: flash-attn on the whole batch → out and lse",
      desc: `To check the ring version, the test first runs plain flash-attn on <b>all</b> of <code>qkv</code> on one GPU. It returns two things per query row (and head): the output row <code>out</code>, and <code>lse</code> $= \\log \\sum_{k \\le t} e^{q_t \\cdot k / \\sqrt d}$, the log of that row's softmax denominator. Ring attention builds lse itself by merging one KV block at a time, so the test checks both. Look at the layouts: <code>out</code> has its tokens on <b>dim 0</b> like <code>qkv</code>; <code>lse</code> has no D and its tokens on the <b>last</b> dim. ${dim3 ? `This is the older flash-attn layout, <code>lse.dim() == 3</code>: padded, one row per sequence, <code>[num_seq, H, max_seqlen] = ${lseShape}</code>.` : `This is the newer flash-attn layout, <code>lse.dim() == 2</code>: packed, <code>[H, total] = ${lseShape}</code>, cut by the same <code>cu_seqlens</code>.`}`,
      blocks: [
        { title: `out  [${S}, H, D] · one row per query token, tokens on dim 0`, rows: [{ label: "out", items: fullRow(() => ({}), true) }] },
        lseBlock(`lse  ${lseShape} · one value per query token (one head shown), tokens on the last dim · round = lse`, seqToks, cu, "cu_seqlens", maxL, () => ({})),
      ],
      driverNotes: drv("qkv", "qkvDone", "ref"),
    });
    steps.push({
      code: "test", tag: "t_out", focus: ["def", "init", "loop", "seq", "chunk", "pick", "cat"], title: "local_out = extract_local(out, …): same function, same rows",
      desc: `<code>out</code> has its tokens on dim 0, exactly like <code>qkv</code>, so the same <code>extract_local</code> cuts it the same way: rank ${sel} gets the output rows of exactly its own query tokens. These are the rows its ring attention should reproduce.`,
      blocks: [{ title: `local_out per rank  [${S / N}, H, D]`, rows: perRank(), localIdx: true }],
      notes: [n("def extract_local", `value = out [${S}, H, D], rank = ${sel}`), n("return torch.cat", `[${S / N}, H, D] = out rows t[${mineAll.join(",")}]`)],
      driverNotes: drv("qkv", "qkvDone", "ref", "out"),
    });
    const lseList = [];
    lens.forEach((len, i) => {
      lseList.push(seqToks[i]);
      steps.push({
        code: "test", tag: "t_cut", lse: true, title: `extract_lse(lse, cu_seqlens) · i = ${i}: ${dim3 ? `lse[${i}, :, :${len}]` : `lse[:, ${cu[i]}:${cu[i + 1]}]`}`,
        desc: i === 0
          ? `Step into <code>extract_lse</code>. It is the lse twin of the loop in <code>extract_local</code>: walk <code>cu_seqlens</code> and cut one piece per sequence, but it <b>does not pick the rank</b>. Its two branches exist only because flash-attn has returned varlen lse in two layouts: ${dim3 ? `here <code>lse.dim() == 3</code>, so sequence $i$ is row $i$, and <code>: end - start</code> drops the padding.` : `here <code>lse.dim() == 2</code>, so sequence $i$ is the columns <code>start:end</code>, just like the tokens.`} Switch the layout in the controls to see the other branch.`
          : `<code>start, end = ${cu[i]}, ${cu[i + 1]}</code>: ${dim3 ? `row ${i}, first ${len} positions` : `columns ${cu[i]}:${cu[i + 1]}`}, sequence ${i}'s ${len} lse values.`,
        blocks: [
          lseBlock(`lse  ${lseShape}`, seqToks, cu, "cu_seqlens", maxL, (t, j) => ({ hl: j === i, dim: j !== i, neutral: true })),
          { title: `lse_list: ${lseList.map((ts) => `[H, ${ts.length}]`).join(", ")}`, rows: lseList.map((ts, j) => ({ label: `[${j}]`, items: ts.map((t) => item(method, N, lens, t, { lse: true, neutral: true, isNew: j === i })) })) },
        ],
        lseNotes: [n("def extract_lse", `lse: ${lseShape}, cu_seqlens = [${cu.join(", ")}]`), n("for i in range", `i = ${i}`, dim3 ? "else:" : undefined), n(dim3 ? "value = lse[i" : "value = lse[:, start", dim3 ? `lse[${i}, :, :${cu[i + 1]} − ${cu[i]}] = lse[${i}, :, :${len}] → [H, ${len}]` : `lse[:, ${cu[i]}:${cu[i + 1]}] → [H, ${len}]`), n(dim3 ? "value = lse[:, start" : "value = lse[i", "(other layout, not taken)")],
        driverNotes: drv("qkv", "qkvDone", "ref", "out", ...(i === ns - 1 ? ["cut"] : [])),
      });
    });
    steps.push({
      code: "test", tag: "t_ring", title: `Ring attention on rank ${sel}'s shard → ring_out, ring_lse`,
      desc: `Each rank runs ${method === "ring" ? "ring" : "zig-zag ring"} attention on <code>local_qkv</code> with <code>local_cu_seqlens = [${lcu.join(", ")}]</code>, and gets results for <b>its own query rows only</b>: <code>ring_out [${S / N}, H, D]</code> and <code>ring_lse ${rlseShape}</code>, in the same layout the kernel uses. ${dim3 ? "Inside, an old-style 3-D lse is flattened to packed for the merges and unflattened before returning (right)." : "Inside, the running lse is <code>[tokens, H, 1]</code> for the merges and is transposed back to <code>[H, tokens]</code> before returning (right)."}`,
      blocks: [lseBlock(`ring_lse on rank ${sel}  ${rlseShape}`, mineSeq, lcu, "local_cu_seqlens", lmax, () => ({ hl: true }))],
      ringNotes: [n("update_out_and_lse", `merges on lse [${S / N}, H, 1]`), dim3 ? n("flatten_varlen_lse(", `[${ns}, H, ${lmax}] → [H, ${S / N}]`) : n("lse.squeeze", `[${S / N}, H, 1] → [H, ${S / N}]`), ...(dim3 ? [n("unflatten_varlen_lse(", `→ [${ns}, H, ${lmax}]`)] : [])],
      driverNotes: drv("qkv", "qkvDone", "ref", "out", "cut", "ring"),
    });
    steps.push({
      code: "test", tag: "t_rcut", lse: true, title: "ring_lse_list = extract_lse(ring_lse, local_cu_seqlens)",
      desc: `The same <code>extract_lse</code>, now with the <b>local</b> boundaries <code>[${lcu.join(", ")}]</code>. This tensor is already rank ${sel}'s share, which is why <code>extract_lse</code> never picks a rank itself.`,
      blocks: [
        lseBlock(`ring_lse  ${rlseShape}`, mineSeq, lcu, "local_cu_seqlens", lmax, () => ({ hl: true })),
        { title: `ring_lse_list: ${mineSeq.map((ts) => `[H, ${ts.length}]`).join(", ")}`, rows: mineSeq.map((ts, j) => ({ label: `[${j}]`, items: ts.map((t) => item(method, N, lens, t, { lse: true, hl: true })) })) },
      ],
      lseNotes: [n("def extract_lse", `ring_lse: ${rlseShape}, cu_seqlens = local [${lcu.join(", ")}]`), n("for i in range", `i = 0..${ns - 1}`, dim3 ? "else:" : undefined), n(dim3 ? "value = lse[i" : "value = lse[:, start", lcu.slice(0, -1).map((c0, j) => (dim3 ? `[${j},:,:${lcu[j + 1] - c0}]` : `[:,${c0}:${lcu[j + 1]}]`)).join(" · ")), n(dim3 ? "value = lse[:, start" : "value = lse[i", "(other layout, not taken)")],
      driverNotes: drv("qkv", "qkvDone", "ref", "out", "cut", "ring", "rcut"),
    });
    steps.push({
      code: "test", tag: "t_share", title: method === "ring" ? "local_lse = lse.chunk(world_size, dim=-1)[rank]" : "local_lse = cat(chunk(2N)[rank], chunk(2N)[2N−1−rank])",
      desc: `Now the test picks rank ${sel}'s share of each reference piece, along <code>dim=-1</code> because that is lse's token axis, with the <b>same rule</b> <code>extract_local</code> used: ${method === "ring" ? `<code>chunk(${N})[${sel}]</code>` : `<code>chunk(${2 * N})</code>, chunks ${sel} and ${2 * N - 1 - sel}`}. Colour = GPU that owns the query row; filled = rank ${sel}.`,
      blocks: [
        { title: `lse_list, coloured by owner: ${lens.map((l) => `[H, ${l}]`).join(", ")}`, rows: seqToks.map((ts, j) => ({ label: `[${j}]`, items: ts.map((t) => item(method, N, lens, t, { lse: true, hl: owner(t) === sel, dim: owner(t) !== sel })) })) },
        { title: `local_lse on rank ${sel}: ${mineSeq.map((ts) => `[H, ${ts.length}]`).join(", ")}`, rows: mineSeq.map((ts, j) => ({ label: `seq ${j}`, rank: sel, items: ts.map((t) => item(method, N, lens, t, { lse: true, hl: true })) })) },
      ],
      driverNotes: drv("qkv", "qkvDone", "ref", "out", "cut", "ring", "rcut", "share"),
    });
    steps.push({
      code: "test", tag: "t_cmp", title: "Compare: out and lse, row for row",
      desc: `Rank ${sel} now holds two versions of the same query rows t[${mineAll.join(",")}]: cut from the one-GPU reference (<code>local_out</code>, <code>local_lse</code>) and computed by the ring (<code>ring_out</code>, <code>ring_lse_list</code>). Same rows, same order, same shapes, so the test just subtracts them. Both diffs ≈ 0 means the ring's KV passing and lse merges are right.`,
      blocks: [
        { title: `rank ${sel}: reference vs ring, per sequence`, rows: [
          { label: "local_lse", rank: sel, items: mineSeq.flatMap((ts, j) => [...(j ? [{ gap: true }] : []), ...ts.map((t) => item(method, N, lens, t, { lse: true, hl: true }))]) },
          { label: "ring_lse", rank: sel, items: mineSeq.flatMap((ts, j) => [...(j ? [{ gap: true }] : []), ...ts.map((t) => item(method, N, lens, t, { lse: true, hl: true }))]) },
          { label: "local_out", rank: sel, items: withLocalGaps(mineAll.map((t) => item(method, N, lens, t, { hl: true })), lcu) },
          { label: "ring_out", rank: sel, items: withLocalGaps(mineAll.map((t) => item(method, N, lens, t, { hl: true })), lcu) },
        ] },
      ],
      driverNotes: drv("qkv", "qkvDone", "ref", "out", "cut", "ring", "rcut", "share", "cmp"),
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
      desc: `One sequence of ${S} tokens on <code>dim=1</code>. Stripe has no varlen version in the repo. Colours show which GPU each token will end up on.`,
      blocks: [{ title: `value  [B, ${S}, H, D]`, rows: [{ label: "value", items: fullRow() }] }],
    });
    steps.push({
      tag: "split", notes: [defNote, n("value = torch.stack", `value.split(${N}, dim=1) → ${G} pieces of ${N} tokens: t[${groups[0].join(",")}], t[${groups[1].join(",")}], …`)],
      title: `value.split(world_size) → ${G} pieces of ${N}`,
      desc: `<code>split(${N})</code> cuts the sequence into pieces of <b>world_size</b> consecutive tokens (not into world_size pieces). Each piece holds one token for every rank.`,
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
      desc: `<code>slicer</code> picks index <code>rank</code> on <code>dim=1</code>. Rank ${sel} gets tokens ${groups.map((g) => g[sel]).join(", ")}: the same as <code>value[:, ${sel}::${N}]</code>. Local position $p$ is token $pN + r$, the fact the shifted causal mask is built on.`,
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
    desc: `Decoding: the KV cache holds ${S} past tokens; the new token $q$ (position ${S}) attends to all of them. Colours show which GPU each KV token will end up on.`,
    blocks: [{ title: `k, v  [B, ${S}, H, D]`, rows: [{ label: "k, v", items: fullRow() }] }, { title: "q  [B, 1, H, D]", rows: [{ label: "q", items: [{ t: S, own: -1, q: true }] }] }],
  });
  steps.push({
    tag: "q", notes: [tdef, n("q_local = q", `the same token (t${S}) on every rank`)],
    title: "q is replicated",
    desc: "Every rank gets the same query token: it is tiny (one token), and every shard of the cache needs it.",
    blocks: [{ title: "q on every rank", rows: Array.from({ length: N }, (_, r) => ({ label: `rank ${r}`, rank: r, items: [{ t: S, own: -1, q: true, hl: r === sel }] })) }],
  });
  steps.push({
    tag: "kv", notes: [tdef, n("q_local = q", `t${S}`), kNote, n("v_local = v.chunk", "same split as k")],
    title: `k.chunk(world_size)[rank] → ${c} tokens each`,
    desc: `The cache is cut into $N=${N}$ contiguous shards. No causal balancing is needed: the single query sees every key, so every shard is equal work.`,
    blocks: [{ title: "k, v per rank", rows: Array.from({ length: N }, (_, r) => ({ label: `rank ${r}`, rank: r, items: localTokens("ring", N, lens, r).map((t) => item(method, N, lens, t, { hl: r === sel })) })), localIdx: true }],
  });
  steps.push({
    tag: "ret", notes: [tdef, n("q_local = q", `t${S}`), kNote, n("return q_local", `q [B, 1, H, D], k/v [B, ${c}, H, D]`)],
    title: "What each rank computes next",
    desc: "Each rank runs attention of $q$ over its own shard, giving a partial $(o_r, \\ell_r)$. The tree-attention tab shows how those are combined.",
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

// The repo's varlen test driver (abridged from test/test_ring_flash_attn_varlen_func.py
// and its zig-zag twin): it shards qkv, runs a one-GPU reference and the ring
// version, then cuts out / lse the same way to compare them.
const driver = (fn, share) => parse(String.raw`
qkv = torch.randn(total_length, 3, nheads, d)  #@t_qkv
dist.broadcast(qkv, src=0)  #@t_qkv
local_cu_seqlens = cu_seqlens // world_size  #@t_qkv
local_max_seqlen = max_seqlen // world_size  #@t_qkv
local_qkv = extract_local(qkv, cu_seqlens, rank, world_size)  #@t_qkv

# reference: plain flash-attn on the whole batch, one GPU
out, lse, _ = flash_attn_varlen_qkvpacked_func(  #@t_ref
    qkv, cu_seqlens, max_seqlen, causal=True, return_attn_probs=True)  #@t_ref
local_out = extract_local(out, cu_seqlens, rank, world_size)  #@t_out
lse_list = extract_lse(lse, cu_seqlens)  #@t_cut

# ring version: this rank's shard only
ring_out, ring_lse, _ = ${fn}(  #@t_ring
    local_qkv, local_cu_seqlens, local_max_seqlen, causal=True, return_attn_probs=True)  #@t_ring
ring_lse_list = extract_lse(ring_lse, local_cu_seqlens)  #@t_rcut

log("out diff", local_out - ring_out)  #@t_cmp
${share}
`);
export const DRIVER_SRC = {
  ring: driver("ring_flash_attn_varlen_qkvpacked_func", `for lse, ring_lse in zip(lse_list, ring_lse_list):  #@t_share t_cmp
    local_lse = lse.chunk(world_size, dim=-1)[rank]  #@t_share
    log("lse diff", local_lse - ring_lse)  #@t_cmp`),
  zigzag: driver("zigzag_ring_flash_attn_varlen_qkvpacked_func", `for i, (lse, ring_lse) in enumerate(zip(lse_list, ring_lse_list)):  #@t_share t_cmp
    local_lse = lse.chunk(2 * world_size, dim=-1)  #@t_share
    local_lse = torch.cat(  #@t_share
        [local_lse[rank], local_lse[2 * world_size - 1 - rank]], dim=-1)  #@t_share
    log(f"lse diff {i}", local_lse - ring_lse)  #@t_cmp`),
};

// Inside ring_flash_attn_varlen_forward: an old-style 3-D lse is flattened to
// the packed layout for merging, and unflattened again before returning.
export const RING_LSE_SRC = parse(String.raw`
block_out, block_lse, _, _ = _flash_attn_varlen_forward(**params)
if block_lse.dim() == 3:  #@d3
    old_lse = True  #@d3
    block_lse = flatten_varlen_lse(block_lse, cu_seqlens=cu_seqlens)  #@d3
out, lse = update_out_and_lse(out, lse, block_out, block_lse)  #@merge
...
if old_lse:  #@d3
    lse = unflatten_varlen_lse(lse, cu_seqlens, max_seqlen)  #@d3
else:  #@d2
    lse = lse.squeeze(dim=-1).transpose(0, 1)  #@d2
return out, lse
`);
