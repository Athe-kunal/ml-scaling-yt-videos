import { useMemo, useState } from "react";
import { T, mono, btn } from "./theme";
import { Tex, TexBlock } from "./tex";
import { ease, useProgress } from "./anim";
import StepShell, { Seg, Stat } from "./StepShell";
import { Canvas, PatchImage, Mat, Op, Row, Label, Bars, heatCss, MM, MMNote, MatMulCard, BlockMMs, DimCtx } from "./draw";
import { SUBJECTS, renderSubject, CLIP_MODELS, CTX, VOCAB, RES, clipToy, fmtN, fmtInt, mulberry } from "./model";
import { dimCol as dimColOf } from "./shapes";
import { useTokenizer, BpePicker, BpeSplit, BpeMerge, BpeIds } from "./BpeScenes";

// CLIP walkthrough: a batch of n (image, caption) pairs -> image ViT and text
// transformer -> projections -> L2 normalise -> n×n logits -> symmetric CE ->
// zero-shot. The "training" slider moves the toy embeddings from init to trained.

const CODE = {
  batch: ["# I[n, 3, 224, 224]  images", "# T[n, 77]  tokenized captions   (pair i ↔ i)"],
  img_cut: ["# I [n, 3, 224, 224] → g×g patches of P×P×3", "# conv1 (next step) does this cut and the projection in one op"],
  img_proj: ["x = conv1(I)   # Conv2d(3, Di, kernel=P, stride=P, bias=False) → [n, Di, g, g]", "x = x.reshape(n, Di, g*g).permute(0, 2, 1)   # [n, g², Di]"],
  img_cls: ["cls = class_embedding + zeros(n, 1, Di)   # learned [Di] → [n, 1, Di]", "x = cat([cls, x], dim=1)   # [n, g²+1, Di]"],
  img_pos: ["x = x + positional_embedding   # learned [g²+1, Di]", "x = ln_pre(x)   # CLIP's extra LayerNorm"],
  img_tf: ["x = transformer(x)   # Li blocks: [n, g²+1, Di] → same shape"],
  img_out: ["I_f = ln_post(x[:, 0, :])   # keep CLS only → [n, Di]"],
  tok_split: ["text = whitespace_clean(basic_clean(text)).lower()", "words = re.findall(pat, text)   # words, single digits, punctuation", "w = ''.join(byte_encoder[b] for b in w.encode('utf-8'))   # bytes → printable chars", "word = tuple(w[:-1]) + (w[-1] + '</w>',)   # d o g</w>"],
  tok_merge: ["while True:", "    bigram = min(pairs, key=lambda p: bpe_ranks.get(p, inf))   # lowest rank first", "    if bigram not in bpe_ranks: break   # nothing left to merge", "    word = merge(word, bigram)   # glue every occurrence"],
  tok_ids: ["ids = [encoder[piece] for piece in pieces]   # vocab of 49,408", "T = [49406] + ids + [49407] + [0] * pad   # → [n, 77]"],
  txt_emb: ["x = token_embedding(T)   # table [49408, Dt] → [n, 77, Dt]", "x = x + positional_embedding   # [77, Dt]"],
  txt_tf: ["x = transformer(x, mask=causal)   # [n, 77, Dt]", "x = ln_final(x)", "T_f = x[arange(n), T.argmax(-1)]   # <eot> has the highest id → [n, Dt]"],
  project: ["I_e = I_f @ W_i   # I_f = ln_post(CLS out) [n,Di]·[Di,E] → [n,E]", "T_e = T_f @ W_t   # T_f = ln_final(<eot> out) [n,Dt]·[Dt,E] → [n,E]"],
  normalize: ["I_e = I_e / I_e.norm(dim=-1, keepdim=True)", "T_e = T_e / T_e.norm(dim=-1, keepdim=True)   # unit length"],
  logits: ["logits = exp(t) * I_e @ T_e.T   # [n,E]·[E,n] → [n,n]", "# t learned, init log(1/0.07), exp(t) clipped at 100"],
  loss_i: ["labels = arange(n)   # the right answer is the diagonal", "loss_i = cross_entropy(logits, labels)   # each row: image → caption"],
  loss_t: ["loss_t = cross_entropy(logits.T, labels)   # each column: caption → image"],
  loss: ["loss = (loss_i + loss_t) / 2"],
  zeroshot: ["W = normalize(encode_text([f'a photo of a {c}' for c in classes]))   # [K, E]", "probs = softmax(100 * normalize(encode_image(x)) @ W.T)   # [1, K]"],
};

const FLOW = [
  { id: "batch", label: "batch of pairs", sub: "n images + n captions", code: CODE.batch },
  {
    group: "image encoder · ViT",
    items: [
      { id: "img_cut", label: "cut into patches", sub: "g² patches of P×P×3 pixels", code: CODE.img_cut },
      { id: "img_proj", label: "project each patch (conv1)", sub: "3P² → Di", code: CODE.img_proj },
      { id: "img_cls", label: "prepend the CLS vector", sub: "g² → g²+1 vectors", code: CODE.img_cls },
      { id: "img_pos", label: "+ positions, ln_pre", sub: "[n, g²+1, Di]", code: CODE.img_pos },
      { id: "img_tf", label: "transformer blocks", sub: "× Li, shape unchanged", code: CODE.img_tf },
      { id: "img_out", label: "keep CLS → ln_post", sub: "I_f [n, Di]", code: CODE.img_out },
    ],
  },
  {
    group: "text encoder · transformer",
    items: [
      { id: "tok_split", label: "split words + </w>", sub: "BPE 1/3 · characters", code: CODE.tok_split },
      { id: "tok_merge", label: "apply learned merges", sub: "BPE 2/3 · lowest rank first", code: CODE.tok_merge },
      { id: "tok_ids", label: "ids + pad to 77", sub: "BPE 3/3 · T [n, 77]", code: CODE.tok_ids },
      { id: "txt_emb", label: "token + pos embedding", sub: "[n, 77, Dt]", code: CODE.txt_emb },
      { id: "txt_tf", label: "causal transformer → EOT", sub: "T_f [n, Dt]", code: CODE.txt_tf },
    ],
  },
  {
    group: "shared embedding space",
    items: [
      { id: "project", label: "linear projections", sub: "CLS → I_e · <eot> → T_e", code: CODE.project },
      { id: "normalize", label: "L2 normalise", sub: "unit vectors", code: CODE.normalize },
      { id: "logits", label: "similarity logits", sub: "[n, n] × exp(t)", code: CODE.logits },
    ],
  },
  {
    group: "symmetric contrastive loss",
    items: [
      { id: "loss_i", label: "image → text CE", sub: "softmax over rows", code: CODE.loss_i },
      { id: "loss_t", label: "text → image CE", sub: "softmax over columns", code: CODE.loss_t },
      { id: "loss", label: "average", sub: "(ℓ_i + ℓ_t) / 2", code: CODE.loss },
    ],
  },
  { id: "zeroshot", label: "zero-shot classify", sub: "prompts become the classifier", code: CODE.zeroshot },
];

const thumbs = SUBJECTS.map((s) => renderSubject(s, RES));
export const PAIR_COL = ["#6EF3DD", "#FF8AC6", "#FFC94D", "#9BE564", "#8DB4FF", "#C79BFF", "#FFB86B", "#FF8A8A"];

// ------------------------------------------------------------ pieces
const SHOW = 48; // bars drawn per vector; the real vector is longer
const More = ({ total }) => <span style={{ fontFamily: mono, fontSize: 13, color: T.dim, whiteSpace: "nowrap" }}>… {total - SHOW} more</span>;

export function Thumb({ i, size = 64, P, ring = true }) {
  return (
    <div style={{ borderRadius: 6, outline: ring ? `2px solid ${PAIR_COL[i]}` : "none", outlineOffset: 1, lineHeight: 0 }}>
      <PatchImage img={thumbs[i]} P={P ?? 224} size={size} grid={P ? 0.7 : 0} />
    </div>
  );
}

function Tok({ s, color = T.soft, bg = T.well }) {
  return <span style={{ fontFamily: mono, fontSize: 14, padding: "2px 6px", borderRadius: 4, border: `1px solid ${color}66`, background: bg, color, whiteSpace: "nowrap" }}>{s}</span>;
}

function Batch({ n }) {
  return (
    <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(150px, 1fr))", gap: 12, padding: 6 }}>
      {SUBJECTS.slice(0, n).map((s, i) => (
        <div key={s.id} style={{ display: "grid", gap: 8, justifyItems: "center", padding: 10, borderRadius: 8, border: `1px solid ${PAIR_COL[i]}55`, background: T.well }}>
          <Thumb i={i} size={110} />
          <span style={{ fontFamily: mono, fontSize: 15, color: PAIR_COL[i], textAlign: "center" }}>"{s.caption}"</span>
          <span style={{ fontFamily: mono, fontSize: 13, color: T.dim }}>pair {i}</span>
        </div>
      ))}
      <div style={{ gridColumn: "1 / -1", fontFamily: mono, fontSize: 15, color: T.soft }}>
        real training: 400M pairs · batches of 32,768
      </div>
    </div>
  );
}

// one image's token sequence: slot 0 = CLS (prepended), then patches 1..g² in
// raster order, drawn from the real patch pixels. stage "in" = entering the
// transformer, "out" = after it, where only slot 0 is kept.
function PatchCrop({ i, k, g, size, dim }) {
  return (
    <Canvas w={size} h={size} deps={[i, k, g, size]} style={{ borderRadius: 2, opacity: dim ? 0.25 : 1, transition: "opacity .4s" }} draw={(x) => {
      const P = RES / g;
      x.imageSmoothingEnabled = false;
      x.drawImage(thumbs[i].canvas, (k % g) * P, Math.floor(k / g) * P, P, P, 0, 0, size, size);
    }} />
  );
}
// a token = a column vector of Di numbers (drawn as 12 cells). Patch tokens get
// their values from conv1(patch); CLS is a learned vector with no pixels behind it.
export function VecCol({ seed, cls, dim, h = 64 }) {
  const r = mulberry(seed * 7919 + 11), cells = 12;
  return (
    <div style={{ width: 22, height: h, display: "grid", gridTemplateRows: `repeat(${cells}, 1fr)`, gap: 1, padding: 1, borderRadius: 3,
      border: `1.5px solid ${cls ? T.accent : T.rule}`, background: T.bg, opacity: dim ? 0.25 : 1, transition: "opacity .4s",
      boxShadow: cls && !dim ? `0 0 0 3px ${T.accent}22` : "none" }}>
      {Array.from({ length: cells }, (_, k) => {
        const v = r();
        return <span key={k} style={{ borderRadius: 1, background: cls ? `hsl(170 70% ${30 + v * 40}%)` : heatCss(0.15 + 0.75 * v) }} />;
      })}
    </div>
  );
}

// one image's token sequence, drawn per stage:
//   cut  — pixel patches 1..g² (no vectors yet)      proj — patch ↓ conv1 ↓ Di-vector
//   cls  — + the CLS vector at index 0 (no pixels)    pos  — + a position vector on every slot
//   tf   — the same g²+1 vectors before → after       out  — only index 0 kept
function TokenSeq({ i = 0, g, Di, P, stage }) {
  const N = g * g, head = Math.min(N, 12);
  const idxs = [...Array.from({ length: head }, (_, k) => k), ...(N > head ? [null, N - 1] : [])];
  const pixels = stage === "cut" || stage === "proj";
  const vecs = stage !== "cut";
  const hasCls = ["cls", "pos", "before", "after", "out"].includes(stage);
  const changed = stage === "after" || stage === "out";
  const PosChip = ({ k }) => <span title="+ positional_embedding[k]" style={{ width: 22, height: 8, borderRadius: 2, background: `hsl(${(k * 47) % 360} 70% 60%)` }} />;
  const col = (k) => {
    if (k === null) return <span key="dots" style={{ fontFamily: mono, fontSize: 15, color: T.dim, alignSelf: "center", padding: "0 4px" }}>…</span>;
    return (
      <div key={k} style={{ display: "grid", gap: 3, justifyItems: "center" }}>
        {pixels && <PatchCrop i={i} k={k} g={g} size={stage === "cut" ? 34 : 22} />}
        {stage === "proj" && <span style={{ fontFamily: mono, fontSize: 13, color: T.accent2, lineHeight: 1 }}>↓</span>}
        {vecs && <VecCol seed={i * 1000 + k + (changed ? 5000 : 0)} dim={stage === "out"} />}
        {stage === "pos" && <PosChip k={k + 1} />}
        <span style={{ fontFamily: mono, fontSize: 12, color: T.dim }}>{stage === "cut" || stage === "proj" ? k : k + 1}</span>
      </div>
    );
  };
  return (
    <div style={{ display: "flex", alignItems: "flex-end", gap: 4, flexWrap: "wrap" }}>
      {vecs && (
        <div style={{ alignSelf: "flex-end", paddingBottom: stage === "pos" ? 27 : 16 }}>
          <span style={{ display: "block", fontFamily: mono, fontSize: 13, color: dimColOf("Di"), writingMode: "vertical-rl", transform: "rotate(180deg)", height: 64, textAlign: "center", borderLeft: `1.5px solid ${dimColOf("Di")}`, paddingLeft: 3 }}>Di = {Di}</span>
        </div>
      )}
      {hasCls && (
        <>
          <div style={{ display: "grid", gap: 3, justifyItems: "center" }}>
            {stage === "cls" && <div style={{ width: 22, height: 22, borderRadius: 2, border: `1px dashed ${T.rule}`, display: "grid", placeItems: "center", fontFamily: mono, fontSize: 8, color: T.dim }}>none</div>}
            {stage === "cls" && <span style={{ fontFamily: mono, fontSize: 13, color: T.accent, lineHeight: 1 }}>⊕</span>}
            <VecCol seed={changed ? 999 : 0} cls />
            {stage === "pos" && <PosChip k={0} />}
            <span style={{ fontFamily: mono, fontSize: 12, color: T.accent, fontWeight: 700 }}>0 CLS</span>
          </div>
          <div style={{ width: 2, alignSelf: "stretch", background: `${T.accent}55`, margin: "0 4px" }} />
        </>
      )}
      {idxs.map(col)}
    </div>
  );
}

const Note = ({ children }) => <div style={{ fontFamily: mono, fontSize: 15, color: T.soft, lineHeight: 1.6, maxWidth: 760 }}>{children}</div>;

function ImgCut({ n, m, g }) {
  const pp = 3 * m.P * m.P;
  return (
    <div style={{ display: "grid", gap: 16, padding: 6 }}>
      <Row gap={10}>{SUBJECTS.slice(0, n).map((s, i) => <Thumb key={s.id} i={i} size={80} P={m.P} />)}</Row>
      <Label>image 0 → {g * g} patches</Label>
      <TokenSeq g={g} Di={m.Di} P={m.P} stage="cut" />
      <Row>
        <Mat shape="n C H W" name="images" rows={RES} cols={RES} color={T.soft} />
        <Op>cut</Op>
        <Mat shape="n N P²C" name="patches" rows={g * g} cols={pp} color={T.soft} />
      </Row>
      <Note>still raw pixels · {pp} numbers per patch</Note>
    </div>
  );
}

function ImgProj({ n, m, g }) {
  const pp = 3 * m.P * m.P;
  return (
    <div style={{ display: "grid", gap: 16, padding: 6 }}>
      <Label color={T.accent2}>every patch → the same linear map → one vector of Di = {m.Di}</Label>
      <TokenSeq g={g} Di={m.Di} P={m.P} stage="proj" />
      <Row>
        <Mat shape="n N P²C" name="patches" rows={g * g} cols={pp} color={T.soft} />
        <Op>×</Op>
        <Mat shape="P²C Di" name="W (conv1)" rows={pp} cols={m.Di} color={T.accent2} />
        <Op>=</Op>
        <Mat shape="n N Di" name="patch tokens" rows={g * g} cols={m.Di} glow />
      </Row>
      <Note>conv1 with kernel = stride = {m.P} · one matrix for every patch</Note>
    </div>
  );
}

function ImgCls({ n, m, g }) {
  return (
    <div style={{ display: "grid", gap: 16, padding: 6 }}>
      <Label color={T.accent}>one learned vector · no pixels</Label>
      <TokenSeq g={g} Di={m.Di} P={m.P} stage="cls" />
      <Row>
        <Mat shape="Di" name="class_embedding" rows={1} cols={m.Di} rl="1" color={T.accent} />
        <Op>cat at index 0</Op>
        <Mat shape="n N Di" name="patch tokens" rows={g * g} cols={m.Di} />
        <Op>=</Op>
        <Mat shape="n T Di" name="x" rows={g * g + 1} cols={m.Di} glow />
      </Row>
      <Note>same width as the patches · same for every image</Note>
    </div>
  );
}

// the position table, row by row: row 0 is CLS's, rows 1…g² are the patches'
function PosRows({ g }) {
  const N = g * g, rows = [0, 1, 2, 3, null, N];
  return (
    <div style={{ display: "grid", gap: 3, padding: "8px 12px", border: `1px solid ${T.rule}`, borderRadius: 8, background: T.well }}>
      <span style={{ fontFamily: mono, fontSize: 13, color: T.dim, marginBottom: 2 }}>positional_embedding</span>
      {rows.map((k) => k === null
        ? <span key="dots" style={{ fontFamily: mono, fontSize: 14, color: T.dim, paddingLeft: 72 }}>⋮</span>
        : (
          <div key={k} style={{ display: "flex", alignItems: "center", gap: 8, fontFamily: mono, fontSize: k === 0 ? 16 : 14 }}>
            <span style={{ width: 64, whiteSpace: "nowrap", textAlign: "right", color: k === 0 ? T.accent : T.dim, fontWeight: k === 0 ? 700 : 400 }}>row {k}</span>
            <span style={{ width: 90, height: 10, borderRadius: 2, background: `hsl(${(k * 47) % 360} 70% 60%)`, outline: k === 0 ? `2px solid ${T.accent}` : "none", outlineOffset: 2 }} />
            <span style={{ color: k === 0 ? T.accent : T.soft, fontWeight: k === 0 ? 700 : 400 }}>→ {k === 0 ? "CLS" : `slot ${k}`}</span>
          </div>
        ))}
    </div>
  );
}

function ImgPos({ n, m, g }) {
  return (
    <div style={{ display: "grid", gap: 16, padding: 6 }}>
      <Label>+ position row k → slot k · <span style={{ color: T.accent }}>row 0 → CLS</span></Label>
      <Row gap={24} style={{ alignItems: "flex-end" }}>
        <TokenSeq g={g} Di={m.Di} P={m.P} stage="pos" />
        <PosRows g={g} />
      </Row>
      <Row>
        <Mat shape="n T Di" name="x" rows={g * g + 1} cols={m.Di} />
        <Op>+</Op>
        <Mat shape="T Di" name="positional_embedding" rows={g * g + 1} cols={m.Di} color={T.accent2} />
        <Op color={T.wire}>→ ln_pre →</Op>
        <Mat shape="n T Di" name="x" rows={g * g + 1} cols={m.Di} glow />
      </Row>
      <Note>ln_pre: CLIP's extra LayerNorm</Note>
    </div>
  );
}

function ImgTf({ n, m, g }) {
  return (
    <div style={{ display: "grid", gap: 14, padding: 6 }}>
      <Label>going in · {g * g + 1} vectors of {m.Di}</Label>
      <TokenSeq g={g} Di={m.Di} P={m.P} stage="before" />
      <div style={{ fontFamily: mono, fontSize: 16, color: T.accent2, paddingLeft: 8 }}>↓ {m.Li} blocks · <b>no mask</b>: everyone sees everyone</div>
      <Label>coming out · still {g * g + 1} vectors of {m.Di}, every one updated</Label>
      <TokenSeq g={g} Di={m.Di} P={m.P} stage="after" />
      <Row>
        <Mat shape="n T Di" name="x" rows={g * g + 1} cols={m.Di} />
        <Op>{m.Li} × block</Op>
        <Mat shape="n T Di" name="x" rows={g * g + 1} cols={m.Di} glow />
      </Row>
    </div>
  );
}

function ImgOut({ n, m, g }) {
  return (
    <div style={{ display: "grid", gap: 16, padding: 6 }}>
      <Label color={T.accent}>keep only index 0 → x[:, 0]</Label>
      <TokenSeq g={g} Di={m.Di} P={m.P} stage="out" />
      <Row>
        <Mat shape="n T Di" name="x" rows={g * g + 1} cols={m.Di} />
        <Op>x[:, 0]</Op>
        <Op color={T.wire}>ln_post</Op>
        <Mat shape="n Di" name="I_f" rows={n} cols={m.Di} glow />
      </Row>
      <Note>CLS = the image summary · patches dropped</Note>
    </div>
  );
}

function TxtEmb({ n, m }) {
  return (
    <div style={{ display: "grid", gap: 16, padding: 6 }}>
      <Row>
        <Mat shape="n L" name="T (ids)" rows={n} cols={CTX} color={T.wire} />
        <Op>lookup rows of</Op>
        <Mat shape="V Dt" name="token_embedding" rows={VOCAB} cols={m.Dt} color={T.accent2} />
        <Op>=</Op>
        <Mat shape="n L Dt" name="x" rows={CTX} cols={m.Dt} />
        <Op>+</Op>
        <Mat shape="L Dt" name="pos_emb" rows={CTX} cols={m.Dt} color={T.accent2} />
      </Row>
      <div style={{ fontFamily: mono, fontSize: 15, color: T.soft }}>a lookup: one row per token</div>
    </div>
  );
}

// every caption's 77 slots, with an arrow where <eot> sits (= argmax of the ids)
function EotPointer({ n, tk }) {
  if (!tk) return null;
  const rows = SUBJECTS.slice(0, n).map((s) => ({ s, ...tk.tokenize(s.caption) }));
  const cell = 7;
  return (
    <div style={{ display: "grid", gap: 6, borderTop: `1px solid ${T.rule}`, paddingTop: 12 }}>
      <Label color={T.wire}>&lt;eot&gt; = the largest id → T.argmax(-1)</Label>
      {rows.map(({ s, ids, len }, i) => (
        <Row key={s.id} gap={10} wrap={false}>
          <Thumb i={i} size={22} />
          <div style={{ display: "flex", gap: 1, position: "relative", paddingTop: 12 }}>
            <span style={{ position: "absolute", top: -1, left: (len - 1) * (cell + 1) - 2, color: T.wire, fontSize: 14, lineHeight: 1 }}>▼</span>
            {ids.map((v, j) => (
              <span key={j} style={{ width: cell, height: 14, borderRadius: 1, background: j === len - 1 ? T.wire : j < len ? PAIR_COL[i] : T.rule, opacity: j < len ? 1 : 0.4 }} />
            ))}
          </div>
          <span style={{ fontFamily: mono, fontSize: 15, color: T.wire, whiteSpace: "nowrap" }}>eot_idx = {len - 1}</span>
        </Row>
      ))}
      <span style={{ fontFamily: mono, fontSize: 15, color: T.soft }}>
        one row per caption → T_f [{n}, Dt]
      </span>
    </div>
  );
}

// the mask that is ADDED to the attention scores: 0 where token i may look at j
// (j ≤ i), −∞ above the diagonal. Zoomed on the real tokens of caption 0 + 3 pads,
// with a small full 77×77 overview.
const MASK_C = { ok: "#2f8f80", eot: "#FFC94D", pad: "#3a4a63", inf: "#5a2a35" };
function CausalMask({ tk }) {
  const cap = SUBJECTS[0].caption;
  const len = tk ? tk.tokenize(cap).len : 7;
  const pieces = tk ? ["<sot>", ...tk.encodeDetailed(cap).flatMap((d) => d.pieces), "<eot>"] : [];
  const K = Math.min(len + 3, 12), cell = 30;
  const name = (j) => (j < len ? (pieces[j] ?? "").replace("</w>", "") : "pad");
  const kind = (i, j) => (j > i ? "inf" : i === len - 1 ? "eot" : i >= len || j >= len ? "pad" : "ok");
  const S = 154;
  return (
    <div style={{ display: "grid", gap: 8 }}>
      <Label>causal mask, added to the scores · caption 0: "{cap}"</Label>
      <Row gap={18} wrap={false} style={{ alignItems: "flex-start" }}>
        <div style={{ display: "grid", gridTemplateColumns: `58px repeat(${K}, ${cell}px)`, gap: 2, alignItems: "center" }}>
          <span style={{ fontFamily: mono, fontSize: 12, color: T.dim, textAlign: "right", paddingRight: 4 }}>query ↓ key →</span>
          {Array.from({ length: K }, (_, j) => (
            <span key={j} style={{ fontFamily: mono, fontSize: 12, color: j === len - 1 ? T.wire : j >= len ? T.dim : T.soft, writingMode: "vertical-rl", transform: "rotate(180deg)", height: 48, textAlign: "left", justifySelf: "center" }}>{name(j)}</span>
          ))}
          {Array.from({ length: K }, (_, i) => [
            <span key={`r${i}`} style={{ fontFamily: mono, fontSize: 13, color: i === len - 1 ? T.wire : i >= len ? T.dim : T.soft, textAlign: "right", paddingRight: 4, fontWeight: i === len - 1 ? 700 : 400 }}>{name(i)}</span>,
            ...Array.from({ length: K }, (_, j) => {
              const k = kind(i, j);
              return (
                <div key={j} style={{
                  height: cell - 6, borderRadius: 3, display: "grid", placeItems: "center", fontFamily: mono, fontSize: 14, fontWeight: 700,
                  background: k === "inf" ? MASK_C.inf : k === "eot" ? `${MASK_C.eot}cc` : k === "pad" ? MASK_C.pad : MASK_C.ok,
                  color: k === "inf" ? "#ff9aa8" : k === "eot" ? T.bg : T.ink,
                }}>{k === "inf" ? "−∞" : "0"}</div>
              );
            }),
          ])}
        </div>
        <div style={{ display: "grid", gap: 6 }}>
          <Canvas w={S} h={S} deps={[len, K]} draw={(x) => {
            const u = S / CTX;
            for (let i = 0; i < CTX; i++) for (let j = 0; j < CTX; j++) {
              const k = kind(i, j);
              x.fillStyle = k === "inf" ? MASK_C.inf : k === "eot" ? MASK_C.eot : k === "pad" ? MASK_C.pad : MASK_C.ok;
              x.fillRect(j * u, i * u, u + 0.4, u + 0.4);
            }
            x.strokeStyle = T.ink; x.lineWidth = 1.5; x.strokeRect(0.75, 0.75, K * u, K * u);
          }} />
          <span style={{ fontFamily: mono, fontSize: 13, color: T.dim }}>full 77 × 77 · white box = zoom</span>
          {[["0 · allowed (j ≤ i)", MASK_C.ok], ["<eot> row: sees whole caption", MASK_C.eot], ["0 · padding (computed, unused)", MASK_C.pad], ["−∞ · future token, blocked", MASK_C.inf]].map(([t, c]) => (
            <span key={t} style={{ display: "flex", gap: 6, alignItems: "center", fontFamily: mono, fontSize: 13, color: T.soft }}><span style={{ width: 12, height: 12, borderRadius: 2, background: c }} />{t}</span>
          ))}
        </div>
      </Row>
      <span style={{ fontFamily: mono, fontSize: 14, color: T.soft, lineHeight: 1.6, maxWidth: 560 }}>
        e^(−∞) = 0 → no peeking ahead
      </span>
    </div>
  );
}

function TxtTf({ n, m, tk }) {
  return (
    <div style={{ display: "grid", gap: 12, padding: 6 }}>
    <Row gap={22} style={{ alignItems: "flex-start" }}>
      <CausalMask tk={tk} />
      <div style={{ display: "grid", gap: 14 }}>
        <Row>
          <Mat shape="n L Dt" name="x" rows={CTX} cols={m.Dt} />
          <Op>{m.Lt} × block · {m.ht} heads</Op>
          <Mat shape="n L Dt" name="x" rows={CTX} cols={m.Dt} />
        </Row>
        <Row>
          <Op>x[arange(n), eot_idx]</Op>
          <Mat shape="n Dt" name="T_f" rows={n} cols={m.Dt} glow color={T.wire} />
        </Row>
        <div style={{ fontFamily: mono, fontSize: 15, color: T.soft, lineHeight: 1.6 }}>no CLS here: &lt;eot&gt; plays that role</div>
      </div>
    </Row>
    <EotPointer n={n} tk={tk} />
    </div>
  );
}

// where I_f and T_f come from: one token of each encoder's final layer
function Origin({ kind, m }) {
  const img = kind === "img";
  return (
    <div style={{ display: "grid", gap: 4, justifyItems: "start", minWidth: 190 }}>
      <Label color={img ? T.accent : T.wire}>{img ? "image encoder, last layer" : "text encoder, last layer"}</Label>
      {img ? (
        <div style={{ display: "flex", gap: 3, alignItems: "flex-end" }}>
          <div style={{ display: "grid", justifyItems: "center", gap: 2 }}><VecCol seed={999} cls h={44} /><span style={{ fontFamily: mono, fontSize: 12, color: T.accent, fontWeight: 700 }}>CLS</span></div>
          {[1, 2, 3, 4].map((k) => <div key={k} style={{ display: "grid", justifyItems: "center", gap: 2 }}><VecCol seed={5000 + k} dim h={44} /><span style={{ fontFamily: mono, fontSize: 12, color: T.dim }}>{k}</span></div>)}
          <span style={{ fontFamily: mono, fontSize: 14, color: T.dim, paddingBottom: 14 }}>…</span>
        </div>
      ) : (
        <div style={{ display: "flex", gap: 3, flexWrap: "wrap", maxWidth: 220 }}>
          {["<sot>", "a", "photo", "of", "a", "dog", "<eot>", "pad"].map((w, k) => (
            <span key={k} style={{ fontFamily: mono, fontSize: 13, padding: "2px 5px", borderRadius: 4, border: `1px solid ${w === "<eot>" ? T.wire : T.rule}`, background: w === "<eot>" ? `${T.wire}33` : T.well, color: w === "<eot>" ? T.wire : T.dim, opacity: w === "<eot>" ? 1 : 0.5, fontWeight: w === "<eot>" ? 700 : 400 }}>{w}</span>
          ))}
        </div>
      )}
      <span style={{ fontFamily: mono, fontSize: 14, color: T.soft }}>
        {img ? <>CLS output → ln_post = <b style={{ color: T.accent }}>I_f</b> [{m.Di}]</> : <>&lt;eot&gt; output → ln_final = <b style={{ color: T.wire }}>T_f</b> [{m.Dt}]</>}
      </span>
    </div>
  );
}

function Project({ n, m }) {
  return (
    <div style={{ display: "grid", gap: 18, padding: 6 }}>
      <Row gap={16}>
        <Origin kind="img" m={m} />
        <Op>→</Op>
        <Mat shape="n Di" name="I_f · CLS outputs" rows={n} cols={m.Di} />
        <Op>×</Op>
        <Mat shape="Di E" name="W_i" rows={m.Di} cols={m.E} color={T.accent2} />
        <Op>=</Op>
        <Mat shape="n E" name="I_e" rows={n} cols={m.E} glow />
      </Row>
      <Row gap={16}>
        <Origin kind="txt" m={m} />
        <Op>→</Op>
        <Mat shape="n Dt" name="T_f · <eot> outputs" rows={n} cols={m.Dt} color={T.wire} />
        <Op>×</Op>
        <Mat shape="Dt E" name="W_t" rows={m.Dt} cols={m.E} color={T.accent2} />
        <Op>=</Op>
        <Mat shape="n E" name="T_e" rows={n} cols={m.E} glow color={T.wire} />
      </Row>
      <div style={{ fontFamily: mono, fontSize: 15, color: T.soft, lineHeight: 1.6 }}>
        both now live in the same {m.E}-dim space
      </div>
    </div>
  );
}

// 2-D picture of what normalisation changes. One image vector, three caption
// vectors with different lengths. Raw dot products reward LENGTH (the long, wrong
// caption wins); after dividing by the norm only the ANGLE is left (the right one wins).
const NV = [
  { id: "dog", label: "a photo of a dog", ang: 40, len: 1.0, col: PAIR_COL[0], right: true },
  { id: "cat", label: "a cat on a rug", ang: 72, len: 1.6, col: PAIR_COL[1] },
  { id: "car", label: "a red car", ang: 90, len: 3.4, col: PAIR_COL[2] },
];
export function Normalize({ m }) {
  const [norm, setNorm] = useState(false);
  const [k, setK] = useState(2.2);
  const [touched, setTouched] = useState(false);
  const p = useProgress(norm ? "n1" : "n0", 900);
  const t = !touched ? 0 : norm ? ease(p) : 1 - ease(p); // 0 = raw lengths, 1 = all on the unit circle
  const rad = (a) => (a * Math.PI) / 180;
  const img = { ang: 30, len: k };
  const vec = (v) => { const L = v.len + (1 - v.len) * t; return [L * Math.cos(rad(v.ang)), L * Math.sin(rad(v.ang))]; };
  const raw = NV.map((v) => img.len * v.len * Math.cos(rad(v.ang - img.ang)));
  const cos = NV.map((v) => Math.cos(rad(v.ang - img.ang)));
  const win = (arr) => arr.indexOf(Math.max(...arr));
  const S = 300, O = S / 2, U = 40; // pixels per unit
  const P = ([x, y]) => [O + x * U, O - y * U];
  const Arrow = ({ v, col, w = 2.5, lab }) => {
    const [x, y] = P(v), dx = x - O, dy = y - O, L = Math.hypot(dx, dy) || 1, ux = dx / L, uy = dy / L;
    return (
      <g>
        <line x1={O} y1={O} x2={x - ux * 7} y2={y - uy * 7} stroke={col} strokeWidth={w} />
        <polygon points={`${x},${y} ${x - ux * 10 - uy * 5},${y - uy * 10 + ux * 5} ${x - ux * 10 + uy * 5},${y - uy * 10 - ux * 5}`} fill={col} />
        {lab && <text x={x + ux * 8} y={y + uy * 8 + 4} fontFamily={mono} fontSize={14} fill={col} textAnchor={ux < 0 ? "end" : "start"}>{lab}</text>}
      </g>
    );
  };
  const iv = (() => { const L = img.len + (1 - img.len) * t; return [L * Math.cos(rad(img.ang)), L * Math.sin(rad(img.ang))]; })();
  const ix = img.len * Math.cos(rad(img.ang)), iy = img.len * Math.sin(rad(img.ang));
  const cell = (on) => ({ padding: "5px 10px", fontFamily: mono, fontSize: 16, textAlign: "right", color: on ? T.bg : T.ink, background: on ? T.accent : "transparent", borderRadius: 4, fontWeight: on ? 700 : 400 });
  return (
    <div style={{ display: "grid", gap: 14, padding: 6 }}>
      <Row gap={10}>
        <button onClick={() => { setTouched(true); setNorm(false); }} style={{ ...btn(!norm, false), height: 28 }}>raw vectors</button>
        <button onClick={() => { setTouched(true); setNorm(true); }} style={{ ...btn(norm, false), height: 28 }}>÷ ‖·‖  normalise</button>
        <label style={{ display: "flex", alignItems: "center", gap: 8, fontFamily: mono, fontSize: 14, color: T.dim }}>
          image length ×
          <input type="range" min={0.5} max={3.5} step={0.1} value={k} onChange={(e) => setK(+e.target.value)} style={{ accentColor: T.accent }} />
          <b style={{ color: T.ink }}>{k.toFixed(1)}</b>
        </label>
        <span style={{ fontFamily: mono, fontSize: 13, color: T.dim }}>2-D sketch</span>
      </Row>
      <Row gap={26} style={{ alignItems: "flex-start" }}>
        <svg width={S} height={S} style={{ background: T.well, borderRadius: 8, flexShrink: 0, overflow: "visible" }}>
          <line x1={0} x2={S} y1={O} y2={O} stroke={T.rule} /><line y1={0} y2={S} x1={O} x2={O} stroke={T.rule} />
          <circle cx={O} cy={O} r={U} fill="none" stroke={T.wire} strokeDasharray="4 4" opacity={0.4 + 0.6 * t} strokeWidth={1.5} />
          <text x={O + U * 0.72} y={O + U + 14} fontFamily={mono} fontSize={13} fill={T.wire} opacity={0.4 + 0.6 * t}>‖v‖ = 1</text>
          {NV.map((v) => <Arrow key={v.id} v={vec(v)} col={v.col} lab={v.id} />)}
          <Arrow v={iv} col={T.ink} w={3.5} lab="image" />
        </svg>
        <div style={{ display: "grid", gap: 14 }}>
          <table style={{ borderCollapse: "separate", borderSpacing: "0 3px" }}>
            <thead>
              <tr style={{ fontFamily: mono, fontSize: 13, color: T.dim, textAlign: "right" }}>
                <th style={{ textAlign: "left", fontWeight: 400 }}>caption</th><th style={{ fontWeight: 400, padding: "0 10px" }}>‖T‖</th>
                <th style={{ fontWeight: 400, padding: "0 10px", color: norm ? T.dim : T.ink }}>raw I·T</th>
                <th style={{ fontWeight: 400, padding: "0 10px", color: norm ? T.ink : T.dim }}>Î·T̂ = cos θ</th>
              </tr>
            </thead>
            <tbody>
              {NV.map((v, i) => (
                <tr key={v.id}>
                  <td style={{ fontFamily: mono, fontSize: 16, color: v.col, paddingRight: 12 }}>"{v.label}"{v.right && <span style={{ color: T.dim, fontSize: 13 }}>  ← true pair</span>}</td>
                  <td style={{ ...cell(false), color: T.soft }}>{v.len.toFixed(1)}</td>
                  <td style={{ ...cell(!norm && win(raw) === i), opacity: norm ? 0.45 : 1 }}>{raw[i].toFixed(2)}</td>
                  <td style={{ ...cell(norm && win(cos) === i), opacity: norm ? 1 : 0.45 }}>{cos[i].toFixed(2)}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <div style={{ fontFamily: mono, fontSize: 16, lineHeight: 1.6, color: T.soft, maxWidth: 440 }}>
            {norm
              ? <>winner: <b style={{ color: T.accent }}>"{NV[win(cos)].label}"</b>, same direction wins</>
              : <>winner: <b style={{ color: T.bad }}>"{NV[win(raw)].label}"</b>, longest vector wins</>}
          </div>
          <div style={{ fontFamily: mono, fontSize: 15, color: T.dim, lineHeight: 1.7, borderTop: `1px solid ${T.rule}`, paddingTop: 10 }}>
            image I = [{ix.toFixed(2)}, {iy.toFixed(2)}]   ‖I‖ = √({ix.toFixed(2)}² + {iy.toFixed(2)}²) = {img.len.toFixed(2)}<br />
            Î = I / ‖I‖ = [{(ix / img.len).toFixed(2)}, {(iy / img.len).toFixed(2)}]
          </div>
        </div>
      </Row>
      <svg width={560} height={40}><Tex x={0} y={4} size={19} color={T.ink} tex={"\\hat{I}\\cdot\\hat{T}=\\frac{I\\cdot T}{\\lVert I\\rVert\\,\\lVert T\\rVert}=\\cos\\theta\\in[-1,1]"} /></svg>
    </div>
  );
}

// The n × n LOGITS matrix, identical on the logits / loss_i / loss_t steps, so the
// numbers never change under you. On the loss steps the selected row (or column)
// is worked out right next to the matrix, aligned cell by cell: L → e^L → p.
const fmtE = (x) => (x >= 1e4 ? x.toExponential(1) : x >= 100 ? x.toFixed(0) : x.toFixed(x >= 10 ? 1 : 2));
function SimMatrix({ n, toy, mode, sel, setSel }) {
  const row = mode === "row", col = mode === "col", loss = row || col;
  const cell = n > 6 ? 46 : 54, k = sel;
  const lo = Math.min(...toy.logits.flat()), hi = Math.max(...toy.logits.flat());
  const shade = (i, j) => (toy.logits[i][j] - lo) / (hi - lo || 1);
  const idx = Array.from({ length: n }, (_, j) => j);
  // the selected line's arithmetic
  const L = row ? toy.logits[k] : toy.logits.map((r) => r[k]);
  const ex = L.map(Math.exp), Z = ex.reduce((a, b) => a + b, 0), P = ex.map((e) => e / Z);
  const C = T.accent, W = T.wire;
  const small = (txt, color = T.soft, bold) => ({ fontFamily: mono, fontSize: 14, color, textAlign: "center", fontWeight: bold ? 700 : 400 });
  const extraCols = col ? " 62px 50px" : row ? " 64px" : "";
  return (
    <div style={{ display: "grid", gridTemplateColumns: `52px repeat(${n}, ${cell}px)${extraCols}`, gap: 3, alignItems: "center" }}>
      {/* header */}
      <span style={{ fontFamily: mono, fontSize: 12, color: T.dim }}>logits L</span>
      {idx.map((j) => (
        <span key={j} onClick={() => col && setSel(j)} style={{ fontFamily: mono, fontSize: 13, color: PAIR_COL[j], textAlign: "center", overflow: "hidden", cursor: col ? "pointer" : "default", opacity: col && j !== k ? 0.5 : 1 }}>T{j} {SUBJECTS[j].id}</span>
      ))}
      {row && <span style={small("", W)}>−log p</span>}
      {col && <><span style={small("", C)}>e^L</span><span style={small("", C)}>p</span></>}
      {/* matrix rows */}
      {idx.map((i) => [
        <div key={`t${i}`} onClick={() => row && setSel(i)} style={{ justifySelf: "center", cursor: row ? "pointer" : "default", opacity: row && i !== k ? 0.5 : 1 }}><Thumb i={i} size={34} /></div>,
        ...idx.map((j) => {
          const picked = !loss || (row && i === k) || (col && j === k);
          return (
            <div key={j} onClick={() => loss && setSel(row ? i : j)} style={{
              height: cell - 14, borderRadius: 4, display: "grid", placeItems: "center", fontFamily: mono, fontSize: 15, fontWeight: i === j ? 700 : 400,
              background: heatCss(shade(i, j)), color: shade(i, j) > 0.45 ? T.bg : T.soft, cursor: loss ? "pointer" : "default",
              outline: i === j ? `2px solid ${C}` : "none", outlineOffset: -2, opacity: picked ? 1 : 0.3, transition: "opacity .3s",
            }}>{toy.logits[i][j].toFixed(1)}</div>
          );
        }),
        row && <span key={`l${i}`} style={small("", W, i === k)}>{toy.lossRow[i].toFixed(2)}</span>,
        col && <span key={`e${i}`} style={small("", i === k ? C : T.soft, i === k)}>{fmtE(ex[i])}</span>,
        col && <span key={`p${i}`} style={small("", i === k ? C : T.soft, i === k)}>{P[i].toFixed(2)}</span>,
      ])}
      {/* row mode: the selected row worked out underneath, column-aligned */}
      {row && [
        <span key="eL" style={{ ...small("", C), textAlign: "right" }}>e^L ↓</span>,
        ...idx.map((j) => <span key={`e${j}`} style={small("", j === k ? C : T.soft, j === k)}>{fmtE(ex[j])}</span>),
        <span key="Z" style={small("", T.ink, true)}>Σ {fmtE(Z)}</span>,
        <span key="pL" style={{ ...small("", C), textAlign: "right" }}>p = e^L/Σ</span>,
        ...idx.map((j) => <span key={`p${j}`} style={small("", j === k ? C : T.soft, j === k)}>{P[j].toFixed(2)}</span>),
        <span key="sum1" style={small("", T.dim)}>sum 1</span>,
      ]}
      {/* col mode: per-column losses + the column's Σ */}
      {col && [
        <span key="lab" style={{ ...small("", W), textAlign: "right" }}>−log p</span>,
        ...idx.map((j) => <span key={j} style={small("", W, j === k)}>{toy.lossCol[j].toFixed(2)}</span>),
        <span key="Z" style={small("", T.ink, true)}>Σ {fmtE(Z)}</span>,
        <span key="s1" style={small("", T.dim)}>sum 1</span>,
      ]}
    </div>
  );
}

// the formula for the selected line, using only numbers visible next to the matrix
function WorkedCE({ n, toy, mode, k }) {
  const row = mode === "row";
  const L = row ? toy.logits[k] : toy.logits.map((r) => r[k]);
  const ex = L.map(Math.exp), Z = ex.reduce((a, b) => a + b, 0), p = ex[k] / Z, l = -Math.log(p);
  const C = "#6EF3DD";
  const name = row ? `\\ell^{\\,\\text{img}}_{${k}}` : `\\ell^{\\,\\text{txt}}_{${k}}`;
  const sumIdx = row ? "j" : "i", Lk = `L_{${k}${k}}`, Lany = row ? `L_{${k}j}` : `L_{i${k}}`;
  const tex = [
    "\\begin{aligned}",
    `${name} &= -\\log \\frac{e^{${Lk}}}{\\sum_{${sumIdx}} e^{${Lany}}} \\\\[6pt]`,
    `&= -\\log \\frac{\\textcolor{${C}}{${fmtE(ex[k])}}}{${fmtE(Z)}} \\\\[6pt]`,
    `&= -\\log \\textcolor{${C}}{${p.toFixed(2)}} = ${l.toFixed(3)}`,
    "\\end{aligned}",
  ].join("\n");
  const all = row ? toy.lossRow : toy.lossCol, mean = row ? toy.li : toy.lt;
  const meanTex = `\\mathcal{L}_{\\text{${row ? "img" : "txt"}}} = \\text{mean} = ${mean.toFixed(3)}`;
  return (
    <div style={{ display: "grid", gap: 8, minWidth: 0 }}>
      <Label color={T.wire}>{row ? `row ${k}: image ${k} must pick caption ${k}` : `column ${k}: caption ${k} must pick image ${k}`} · click another {row ? "row" : "column"}</Label>
      <div style={{ fontFamily: mono, fontSize: 14, color: T.soft, lineHeight: 1.6 }}>
        same logits · worked {row ? "below" : "on the right"}
      </div>
      <TexBlock tex={tex} size={18} />
      <div style={{ fontFamily: mono, fontSize: 14, color: T.dim, lineHeight: 1.5 }}>
        target = the diagonal
      </div>
      <div style={{ fontFamily: mono, fontSize: 14, color: T.dim }}>average of the {row ? "−log p column" : "−log p row"}:</div>
      <TexBlock tex={meanTex} size={16} color={T.soft} />
    </div>
  );
}

function MatrixScene({ n, toy, mode }) {
  const [sel, setSel] = useState(0);
  const k = Math.min(sel, n - 1);
  return (
    <Row gap={22} wrap={false} style={{ alignItems: "flex-start", padding: 6, overflowX: "auto" }}>
      <SimMatrix n={n} toy={toy} mode={mode} sel={k} setSel={setSel} />
      <div style={{ display: "grid", gap: 10, maxWidth: mode === "logits" ? 300 : 440, minWidth: 0 }}>
        {mode === "logits" && <>
          <svg width={300} height={40}><Tex x={0} y={4} size={19} color={T.ink} tex={"L_{ij}=e^{t}\\,\\hat{I}_i\\cdot\\hat{T}_j"} /></svg>
          <span style={{ fontFamily: mono, fontSize: 15, color: T.soft, lineHeight: 1.6 }}>e^t = <b style={{ color: T.wire }}>{toy.scale.toFixed(1)}</b> learned temperature</span>
          <span style={{ fontFamily: mono, fontSize: 15, color: T.soft, lineHeight: 1.6, borderLeft: `2px solid ${T.wire}`, paddingLeft: 8 }}>
            capped at <b style={{ color: T.wire }}>100</b>
          </span>
          <span style={{ fontFamily: mono, fontSize: 15, color: T.soft, lineHeight: 1.6 }}>diagonal = {n} true pairs</span>
        </>}
        {mode === "row" && <WorkedCE n={n} toy={toy} mode="row" k={k} />}
        {mode === "col" && <WorkedCE n={n} toy={toy} mode="col" k={k} />}
      </div>
    </Row>
  );
}

function LossScene({ n, toy, train }) {
  const curve = useMemo(() => Array.from({ length: 41 }, (_, k) => clipToy(n, k / 40).loss), [n]);
  const W = 420, H = 180, mx = Math.max(Math.log(n) * 1.15, ...curve);
  const X = (t) => 40 + t * (W - 60), Y = (v) => 14 + (1 - v / mx) * (H - 40);
  return (
    <div style={{ display: "grid", gap: 14, padding: 6 }}>
      <svg width={620} height={46}><Tex x={0} y={4} size={18} color={T.ink} tex={`\\mathcal{L}=\\tfrac12(\\ell_i+\\ell_t)=\\tfrac12(${toy.li.toFixed(3)}+${toy.lt.toFixed(3)})=${toy.loss.toFixed(3)}`} /></svg>
      <svg width={W} height={H} style={{ background: T.well, borderRadius: 8 }}>
        <line x1={40} x2={W - 20} y1={Y(Math.log(n))} y2={Y(Math.log(n))} stroke={T.bad} strokeDasharray="4 4" />
        <text x={W - 22} y={Y(Math.log(n)) - 5} textAnchor="end" fontFamily={mono} fontSize={13} fill={T.bad}>chance = ln {n} = {Math.log(n).toFixed(2)}</text>
        <polyline points={curve.map((v, k) => `${X(k / 40)},${Y(v)}`).join(" ")} fill="none" stroke={T.accent} strokeWidth={2} />
        <circle cx={X(train)} cy={Y(toy.loss)} r={5} fill={T.wire} />
        <text x={40} y={H - 8} fontFamily={mono} fontSize={13} fill={T.dim}>init</text>
        <text x={W - 20} y={H - 8} textAnchor="end" fontFamily={mono} fontSize={13} fill={T.dim}>trained →</text>
      </svg>
      <div style={{ fontFamily: mono, fontSize: 15, color: T.soft, lineHeight: 1.6 }}>
        drag <b style={{ color: T.wire }}>training</b> ↑
      </div>
    </div>
  );
}

function ZeroShot({ toy, q, setQ }) {
  const sims = toy.allTxt.map((t) => 100 * toy.dot(toy.allImg[q], t));
  const pr = toy.sm(sims);
  const best = pr.indexOf(Math.max(...pr));
  return (
    <Row gap={26} style={{ alignItems: "flex-start", padding: 6 }}>
      <div style={{ display: "grid", gap: 8, justifyItems: "center" }}>
        <Label>query image</Label>
        <Thumb i={q} size={150} />
        <div style={{ display: "flex", gap: 4, flexWrap: "wrap", maxWidth: 170, justifyContent: "center" }}>
          {SUBJECTS.map((s, i) => <button key={s.id} onClick={() => setQ(i)} style={{ fontSize: 19, padding: 2, borderRadius: 5, cursor: "pointer", border: `1px solid ${i === q ? T.accent : T.rule}`, background: T.well }}>{s.emoji}</button>)}
        </div>
      </div>
      <div style={{ display: "grid", gap: 6, minWidth: 360 }}>
        <Label>softmax(100 · Î · Wᵀ) over K = {SUBJECTS.length} prompts · toy values</Label>
        {SUBJECTS.map((s, k) => (
          <div key={s.id} style={{ display: "grid", gridTemplateColumns: "190px 1fr 50px", gap: 8, alignItems: "center", fontFamily: mono, fontSize: 15 }}>
            <span style={{ color: k === best ? T.accent : T.soft }}>"a photo of a {s.id}"</span>
            <div style={{ height: 14, borderRadius: 3, background: T.well }}><div style={{ height: "100%", width: `${pr[k] * 100}%`, borderRadius: 3, background: k === best ? T.accent : T.soft, transition: "width .4s" }} /></div>
            <span style={{ textAlign: "right", color: T.ink }}>{(pr[k] * 100).toFixed(0)}%</span>
          </div>
        ))}
        <span style={{ fontFamily: mono, fontSize: 15, color: T.soft, lineHeight: 1.6, marginTop: 6 }}>text encoder = the classifier</span>
      </div>
    </Row>
  );
}

// ------------------------------------------------------------ view
export default function ClipView() {
  const [mk, setMk] = useState("ViT-B/32");
  const [n, setN] = useState(6);
  const [train, setTrain] = useState(0.7);
  const [q, setQ] = useState(0);
  const tk = useTokenizer();
  const [bpeText, setBpeText] = useState(SUBJECTS[0].caption);
  const [bpeWord, setBpeWord] = useState(4);
  const captions = SUBJECTS.map((s) => s.caption);
  const bpeTok = tk ? tk.tokenize(bpeText) : null, bpeLen = bpeTok ? bpeTok.rawLen : null;
  const m = CLIP_MODELS[mk], g = RES / m.P;
  const toy = useMemo(() => clipToy(n, train), [n, train]);

  const steps = [
    { id: "batch", block: "batch", caption: "a batch of n matching (image, caption) pairs", vars: () => [["n", `${n}`]] },
    { id: "img_cut", block: "img_cut", caption: `cut each image into ${g}×${g} = ${g * g} patches of ${m.P}×${m.P} pixels`, vars: () => [["P", `${m.P}`], ["g²", `${g * g}`]] },
    { id: "img_proj", block: "img_proj", caption: `project every patch to a ${m.Di}-dim vector (conv1)`, vars: () => [["3P²", `${3 * m.P * m.P}`], ["Di", `${m.Di}`]] },
    { id: "img_cls", block: "img_cls", caption: "prepend a learned CLS vector at index 0", vars: () => [["tokens", `${g * g} → ${g * g + 1}`]] },
    { id: "img_pos", block: "img_pos", caption: "add position vectors (row 0 → CLS), then ln_pre", vars: () => [["pos_emb", `[${g * g + 1}, ${m.Di}]`]] },
    { id: "img_tf", block: "img_tf", caption: `${m.Li} transformer blocks: same shape out`, vars: () => [["Li", `${m.Li}`], ["heads", `${m.hi}`]] },
    { id: "img_out", block: "img_out", caption: "keep only the CLS vector → I_f", vars: () => [["I_f", `[${n}, ${m.Di}]`]] },
    { id: "tok_split", block: "tok_split", caption: "BPE starts from characters: split words, mark word ends", vars: () => [["words", tk ? `${tk.encodeDetailed(bpeText).length}` : "…"]] },
    { id: "tok_merge", block: "tok_merge", caption: "replay CLIP's learned merges until none apply", vars: () => [["merges", tk ? `${tk.nMerges.toLocaleString()}` : "…"]] },
    { id: "tok_ids", block: "tok_ids", caption: "look up ids, add <sot>/<eot>, pad to 77", vars: () => [["tokens", bpeLen ? `${bpeLen} / ${CTX}` : "…", bpeTok?.truncated ? "bad" : null], ["vocab", fmtInt(VOCAB)]] },
    { id: "txt_emb", block: "txt_emb", caption: "look up token embeddings, add positions", vars: () => [["Dt", `${m.Dt}`]] },
    { id: "txt_tf", block: "txt_tf", caption: "causal text transformer, then keep the <eot> token", vars: () => [["Lt", `${m.Lt}`], ["heads", `${m.ht}`]] },
    { id: "project", block: "project", caption: `project both into the shared E = ${m.E} space`, vars: () => [["W_i", `[${m.Di}, ${m.E}]`], ["W_t", `[${m.Dt}, ${m.E}]`]] },
    { id: "normalize", block: "normalize", caption: "L2-normalise: every embedding on the unit sphere", vars: () => [["‖·‖", "1"]] },
    { id: "logits", block: "logits", caption: "all n² image-text cosines, scaled by a learned temperature", vars: () => [["exp(t)", toy.scale.toFixed(1)]] },
    { id: "loss_i", block: "loss_i", caption: "rows: each image must pick its own caption", vars: () => [["loss_i", toy.li.toFixed(3)]] },
    { id: "loss_t", block: "loss_t", caption: "columns: each caption must pick its own image", vars: () => [["loss_t", toy.lt.toFixed(3)]] },
    { id: "loss", block: "loss", caption: "the CLIP loss is the average of both directions", vars: () => [["loss", toy.loss.toFixed(3)], ["ln n", Math.log(n).toFixed(2)]] },
    { id: "zeroshot", block: "zeroshot", caption: "zero-shot: class names as prompts, pick the nearest", vars: () => [["K", `${SUBJECTS.length}`]] },
  ];

  const G = {
    n: ["batch of pairs", n], C: ["channels", 3], H: ["image side", RES], W: ["image side", RES], P: ["patch", m.P], "P²C": ["values per patch 3P²", 3 * m.P * m.P], g: ["patches per side 224/P", g], N: ["patches g²", g * g], T: ["image tokens g²+1", g * g + 1],
    Di: ["image width", m.Di], L: ["text context", CTX], V: ["BPE vocab", VOCAB], Dt: ["text width", m.Dt], E: ["shared embed dim", m.E], K: ["classes / prompts", SUBJECTS.length],
  };
  const S = (name, dims, what, io, note) => ({ name, dims, what, io, note });
  const tags = (st) => ({
    batch: [S("I", ["n", "C", "H", "W"], "images", "in"), S("captions", ["n"], "strings", "in")],
    tok_ids: [S("T", ["n", "L"], "integer ids: 49406 <sot> … 49407 <eot>, then 0 padding", "out")],
    normalize: [S("I_e, T_e", ["n", "E"], null, "in"), S("Î_e, T̂_e", ["n", "E"], "unit rows", "out")],
    logits: [S("logits", ["n", "n"], "row = image, column = caption", "out", `e^t = ${toy.scale.toFixed(1)}`)],
    loss_i: [S("softmax(logits, dim=1)", ["n", "n"]), S("loss_i", [], "mean −log of the diagonal", "out")],
    loss_t: [S("softmax(logits, dim=0)", ["n", "n"]), S("loss_t", [], "mean −log of the diagonal", "out")],
    loss: [S("loss", [], "(loss_i + loss_t) / 2", "out")],
    zeroshot: [S("W (prompt embeds)", ["K", "E"]), S("image embed", ["1", "E"]), S("probs", ["K"], null, "out")],
  })[st.id];

  const scene = (st) => {
    const body = {
      batch: <Batch n={n} />,
      img_cut: <ImgCut n={n} m={m} g={g} />,
      img_proj: <ImgProj n={n} m={m} g={g} />,
      img_cls: <ImgCls n={n} m={m} g={g} />,
      img_pos: <ImgPos n={n} m={m} g={g} />,
      img_tf: <ImgTf n={n} m={m} g={g} />,
      img_out: <ImgOut n={n} m={m} g={g} />,
      tok_split: <BpeSplit tk={tk} text={bpeText} />,
      tok_merge: <BpeMerge tk={tk} text={bpeText} wi={bpeWord} setWi={setBpeWord} />,
      tok_ids: <BpeIds tk={tk} text={bpeText} captions={captions} n={n} />,
      txt_emb: <TxtEmb n={n} m={m} />,
      txt_tf: <TxtTf n={n} m={m} tk={tk} />,
      project: <Project n={n} m={m} />,
      normalize: <Normalize m={m} />,
      logits: <MatrixScene n={n} toy={toy} mode="logits" />,
      loss_i: <MatrixScene n={n} toy={toy} mode="row" />,
      loss_t: <MatrixScene n={n} toy={toy} mode="col" />,
      loss: <LossScene n={n} toy={toy} train={train} />,
      zeroshot: <ZeroShot toy={toy} q={q} setQ={setQ} />,
    }[st.id];
    const bpe = st.id.startsWith("tok_");
    return (
      <div style={{ minHeight: 400, overflowX: "auto", display: "grid", gap: 16, padding: bpe ? 6 : 0, alignContent: "start" }}>
        {bpe && <BpePicker n={n} captions={captions} text={bpeText} setText={(t) => { setBpeText(t); setBpeWord(0); }} />}
        {body}
      </div>
    );
  };

  const { Di, Dt, E, hi, ht, Li, Lt } = m, Ti = g * g + 1, pp = 3 * m.P * m.P;
  const eot = tk ? SUBJECTS.slice(0, n).map((s) => tk.tokenize(s.caption).len - 1) : null;
  const matmuls = (st) => {
    const body = {
      batch: <MMNote>inputs: I [{n}, 3, {RES}, {RES}] images · captions → T [{n}, {CTX}] ids (next steps)</MMNote>,
      img_cut: <MMNote>rearrange: [{n}, 3, {RES}, {RES}] → [{n}, {g * g}, {pp}]</MMNote>,
      img_proj: <MM a={["patches", [n, g * g, pp]]} b={["W_conv", [pp, Di]]} c={["tokens", [n, g * g, Di]]} note={`= conv1 (kernel = stride = ${m.P}, bias=False)`} />,
      img_cls: <MMNote>concat: class_embedding [1, 1, {Di}] ‖ tokens [{n}, {g * g}, {Di}] → x [{n}, {Ti}, {Di}]</MMNote>,
      img_pos: <MMNote>add: x [{n}, {Ti}, {Di}] + positional_embedding [{Ti}, {Di}] → ln_pre → [{n}, {Ti}, {Di}]</MMNote>,
      img_tf: <BlockMMs B={n} T={Ti} D={Di} h={hi} d={Di / hi} M={4 * Di} L={Li} act="QuickGELU x·σ(1.702x)" />,
      img_out: <MMNote>index: x[:, 0] of [{n}, {Ti}, {Di}] → ln_post → I_f [{n}, {Di}]</MMNote>,
      tok_split: <MMNote>text only · no tensors yet</MMNote>,
      tok_merge: <MMNote>text only · no tensors yet</MMNote>,
      tok_ids: <MMNote>lookup: T [{n}, {CTX}] (int64)</MMNote>,
      txt_emb: <>
        <MM a={["one_hot(T)", [n, CTX, VOCAB]]} b={["token_embedding", [VOCAB, Dt]]} c={["x", [n, CTX, Dt]]} note="really just a lookup" cost="0 (a gather)" />
        <MMNote>x + positional_embedding [{CTX}, {Dt}] → [{n}, {CTX}, {Dt}]</MMNote>
      </>,
      txt_tf: <>
        <BlockMMs B={n} T={CTX} D={Dt} h={ht} d={Dt / ht} M={4 * Dt} L={Lt} causal act="QuickGELU x·σ(1.702x)" />
        <MMNote>ln_final → gather x[arange({n}), eot_idx{eot ? ` = [${eot.join(", ")}]` : ""}] → T_f [{n}, {Dt}]</MMNote>
      </>,
      project: <>
        <MM a={["I_f", [n, Di]]} b={["W_i", [Di, E]]} c={["I_e", [n, E]]} note="no bias, no MLP head" />
        <MM a={["T_f", [n, Dt]]} b={["W_t", [Dt, E]]} c={["T_e", [n, E]]} />
      </>,
      normalize: <MMNote>row-wise: I_e [{n}, {E}] ÷ ‖I_e‖ [{n}, 1] → [{n}, {E}] · same for T_e</MMNote>,
      logits: <>
        <MM a={["I_e", [n, E]]} b={["T_eᵀ", [E, n]]} c={["cos", [n, n]]} note="cosine of every image–caption pair" />
        <MMNote>logits = e^t · cos, e^t = {toy.scale.toFixed(1)}  (scalar × [{n}, {n}])</MMNote>
      </>,
      loss_i: <MMNote>row softmax: [{n}, {n}] → −log diagonal → mean</MMNote>,
      loss_t: <MMNote>column softmax: [{n}, {n}] → −log diagonal → mean</MMNote>,
      loss: <MMNote>(loss_i + loss_t) / 2 → scalar</MMNote>,
      zeroshot: <>
        <MM a={["Î", [1, E]]} b={["Wᵀ", [E, SUBJECTS.length]]} c={["sims", [1, SUBJECTS.length]]} note="W = normalised text embeddings of the K prompts, built once" />
        <MMNote>× 100 → softmax over K → probs [1, {SUBJECTS.length}]</MMNote>
      </>,
    }[st.id];
    return <MatMulCard>{body}</MatMulCard>;
  };

  return (
    <DimCtx.Provider value={G}>
    <StepShell
      matmuls={matmuls}
      title="CLIP (OpenAI)"
      subtitle="embeddings, similarities and losses are a toy model · shapes are the real configs"
      blocks={FLOW}
      steps={steps}
      scene={scene}
      tags={tags}
      controls={
        <Row gap={18}>
          <Seg label="MODEL" value={mk} options={Object.keys(CLIP_MODELS)} onChange={setMk} />
          <Seg label="BATCH n" value={n} options={[4, 6, 8]} onChange={setN} />
          <label style={{ display: "grid", gap: 4, minWidth: 160 }}>
            <span style={{ fontFamily: mono, fontSize: 13, color: T.dim, letterSpacing: "0.08em", display: "flex", justifyContent: "space-between" }}>TRAINING <b style={{ color: T.wire }}>{Math.round(train * 100)}%</b></span>
            <input type="range" min={0} max={1} step={0.01} value={train} onChange={(e) => setTrain(+e.target.value)} style={{ accentColor: T.wire }} />
          </label>
          <Stat k="loss" v={toy.loss.toFixed(3)} tone={T.wire} />
          <Stat k="exp(t)" v={toy.scale.toFixed(1)} />
        </Row>
      }
    />
    </DimCtx.Provider>
  );
}
