import { useMemo, useState } from "react";
import { T, mono, btn } from "./theme";
import { Tex, TexBlock } from "./tex";
import StepShell, { Seg, Stat } from "./StepShell";
import { Canvas, PatchImage, Mat, Op, Row, Label, heat, heatCss, MM, MMNote, MatMulCard, BlockMMs, DimCtx } from "./draw";
import { SUBJECTS, renderSubject, patchStats, clsRow, SIGLIP_MODELS, S_CTX, S_VOCAB, sigToy, sigmoid } from "./model";
import { dimCol as dimColOf } from "./shapes";
import { PAIR_COL, Thumb, VecCol, Normalize } from "./ClipView";
import { useSpm, SpmPicker, SpmClean, SpmViterbi, SpmIds } from "./SpmScenes";
import { EOS } from "./siglipSpm";
import { RingScene, DEVS } from "./SiglipRing";
import { S2Diff, S2Size, S2Pos, S2Mask, ShapePicker, SHAPES, P2 } from "./SiglipTwo";
import { naflexSize } from "./model";

// SigLIP walkthrough. Same two-tower shape as CLIP, with these differences:
//   image: no CLS token; a MAP head (one learned query) pools the patches
//   text:  SentencePiece unigram, 64 tokens padded with </s>, NO causal mask,
//          pooled at the last slot, then a linear head
//   loss:  L = t·cos + b, each cell its own yes/no: −log σ(z·L), z = ±1
//   scale: the loss splits into blocks → texts travel around a ring of devices

const T0 = 32, B0 = -4; // toy trained values (init: t = 10, b = −10)

const CODE = {
  batch: ["# images [n, 3, R, R], resized (no crop), pixels scaled to [−1, 1]", "# ids [n, 64] tokenized captions   (pair i ↔ i)"],
  img_cut: ["# [n, 3, R, R] → g×g patches of P×P×3", "# the conv in the next step does cut + project in one op"],
  img_proj: ["x = patch_embedding(pixels)   # Conv2d(3, D, kernel=P, stride=P), with bias → [n, D, g, g]", "x = x.flatten(2).transpose(1, 2)   # [n, g², D]"],
  img_pos: ["x = x + position_embedding(arange(g²))   # [g², D], no CLS slot"],
  img_tf: ["x = encoder(x)   # L blocks, no mask: [n, g², D] → same shape", "x = post_layernorm(x)"],
  img_map: ["q = probe.repeat(n, 1, 1)   # learned [1, 1, D]", "h = MultiheadAttention(q, x, x)   # query = probe, keys/values = patches → [n, 1, D]", "h = h + mlp(layernorm(h))", "image_embeds = h[:, 0]   # [n, D]"],
  tok_clean: ["text = text.lower()", "text = remove_punctuation(text)   # string.punctuation", "text = '▁' + re.sub(r'\\s+', ' ', text).strip().replace(' ', '▁')"],
  tok_viterbi: ["pieces = sp_model.encode(text)   # unigram LM: the cut with the highest Σ log p", "# unknown characters → UTF-8 bytes <0xXX>"],
  tok_ids: ["ids = [vocab[p] for p in pieces] + [1]   # </s> = 1", "ids += [1] * (64 - len(ids))   # pad with </s>"],
  txt_emb: ["x = token_embedding(ids)   # table [32000, D] → [n, 64, D]", "x = x + position_embedding(arange(64))   # [64, D]"],
  txt_tf: ["x = encoder(x)   # no causal mask, no padding mask", "x = final_layer_norm(x)"],
  txt_pool: ["pooled = x[:, -1, :]   # the last slot (always </s>)", "text_embeds = head(pooled)   # Linear(D, D)"],
  normalize: ["image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)", "text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)"],
  logits: ["logits = image_embeds @ text_embeds.T * t.exp() + b   # [n, n]", "# learned: t init log 10, b init −10"],
  labels: ["z = 2 * eye(n) - 1   # +1 on the diagonal, −1 elsewhere"],
  cell: ["ell = -logsigmoid(z * logits)   # every cell on its own"],
  loss: ["loss = ell.sum(dim=-1).mean()   # Σ over the row, mean over rows"],
  bias: ["# at init cos ≈ 0, so every logit ≈ b", "# b = −10: σ(−10) ≈ 0 → 'no match' for everyone"],
  vs_clip: ["clip   = cross_entropy(logits, arange(n))   # needs the whole row", "siglip = -logsigmoid(z * logits)   # needs one cell"],
  ring: ["for s in range(D):   # D devices", "    ell += -logsigmoid(z_s * t * img_local @ txt_held.T + b)   # [b, b] block", "    txt_held = send_to_next(txt_held)   # ring permute", "loss = all_reduce(ell.sum()) / n"],
  s2_diff: ["# SigLIP 2 = SigLIP architecture + sigmoid loss, plus:", "#   NaFlex image input, Gemma tokenizer (256k), extra training losses"],
  s2_size: ["H, W = get_image_size_for_max_num_patches(h, w, patch_size=16, max_num_patches=budget)", "# keep aspect ratio, both sides multiples of 16, (H/16)·(W/16) ≤ budget"],
  s2_pos: ["pos = position_embedding.weight.reshape(16, 16, D)   # learned grid", "pos = F.interpolate(pos, size=(gh, gw), mode='bilinear', antialias=True)   # per image"],
  s2_mask: ["x = patch_embedding(pixel_values)   # Linear(16*16*3, D) on [n, budget, 768]", "x = encoder(x + pos, mask=pixel_attention_mask)   # padding never attended", "image_embeds = head(x, pixel_attention_mask)   # MAP head ignores padding too"],
  zeroshot: ["W = normalize(text_model(prompts))   # [K, D]", "p = sigmoid(t * normalize(image_model(x)) @ W.T + b)   # [1, K], each on its own"],
};

const FLOW = [
  { id: "batch", label: "batch of pairs", sub: "n images + n captions", code: CODE.batch },
  {
    group: "image encoder · ViT",
    items: [
      { id: "img_cut", label: "cut into patches", sub: "g² patches of P×P×3", code: CODE.img_cut },
      { id: "img_proj", label: "project each patch", sub: "3P² → D", code: CODE.img_proj },
      { id: "img_pos", label: "+ positions · no CLS", sub: "[n, g², D]", code: CODE.img_pos },
      { id: "img_tf", label: "transformer blocks", sub: "× L, no mask", code: CODE.img_tf },
      { id: "img_map", label: "MAP head", sub: "probe attends → [n, D]", code: CODE.img_map },
    ],
  },
  {
    group: "text encoder · transformer",
    items: [
      { id: "tok_clean", label: "clean + mark ▁", sub: "SentencePiece 1/3", code: CODE.tok_clean },
      { id: "tok_viterbi", label: "best cut (Viterbi)", sub: "SentencePiece 2/3 · unigram", code: CODE.tok_viterbi },
      { id: "tok_ids", label: "ids + pad to 64", sub: "SentencePiece 3/3", code: CODE.tok_ids },
      { id: "txt_emb", label: "token + pos embedding", sub: "[n, 64, D]", code: CODE.txt_emb },
      { id: "txt_tf", label: "transformer · no mask", sub: "everyone sees everyone", code: CODE.txt_tf },
      { id: "txt_pool", label: "last slot → head", sub: "[n, D]", code: CODE.txt_pool },
    ],
  },
  {
    group: "shared embedding space",
    items: [
      { id: "normalize", label: "L2 normalise", sub: "unit vectors", code: CODE.normalize },
      { id: "logits", label: "logits t·cos + b", sub: "[n, n]", code: CODE.logits },
    ],
  },
  {
    group: "sigmoid loss",
    items: [
      { id: "labels", label: "labels ±1", sub: "diagonal +1, rest −1", code: CODE.labels },
      { id: "cell", label: "each cell: −log σ(z·L)", sub: "n² yes/no questions", code: CODE.cell },
      { id: "loss", label: "sum ÷ n", sub: "one scalar", code: CODE.loss },
      { id: "bias", label: "why b = −10", sub: "n positives vs n² − n negatives", code: CODE.bias },
      { id: "vs_clip", label: "vs CLIP softmax", sub: "same logits", code: CODE.vs_clip },
    ],
  },
  {
    group: "chunked across devices",
    items: [{ id: "ring", label: "ring of devices", sub: "texts travel, images stay", code: CODE.ring }],
  },
  { id: "zeroshot", label: "zero-shot classify", sub: "independent sigmoids", code: CODE.zeroshot },
  {
    group: "SigLIP 2 · what changes",
    items: [
      { id: "s2_diff", label: "SigLIP vs SigLIP 2", sub: "same core, new input", code: CODE.s2_diff },
      { id: "s2_size", label: "keep shape, fit budget", sub: "NaFlex 1/3", code: CODE.s2_size },
      { id: "s2_pos", label: "resize position grid", sub: "NaFlex 2/3", code: CODE.s2_pos },
      { id: "s2_mask", label: "pad + mask the batch", sub: "NaFlex 3/3", code: CODE.s2_mask },
    ],
  },
];

// ------------------------------------------------------------ image tower
const Note = ({ children }) => <div style={{ fontFamily: mono, fontSize: 15, color: T.soft, lineHeight: 1.6, maxWidth: 760 }}>{children}</div>;
const imgOf = (i, res) => renderSubject(SUBJECTS[i], res);

function Batch({ n }) {
  return (
    <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(150px, 1fr))", gap: 12, padding: 6 }}>
      {SUBJECTS.slice(0, n).map((s, i) => (
        <div key={s.id} style={{ display: "grid", gap: 8, justifyItems: "center", padding: 10, borderRadius: 8, border: `1px solid ${PAIR_COL[i]}55`, background: T.well }}>
          <Thumb i={i} size={100} />
          <span style={{ fontFamily: mono, fontSize: 15, color: PAIR_COL[i], textAlign: "center" }}>"{s.caption}"</span>
        </div>
      ))}
      <div style={{ gridColumn: "1 / -1", fontFamily: mono, fontSize: 15, color: T.soft }}>
        real training: batches of 32,768 pairs
      </div>
    </div>
  );
}

function Crop({ i, k, g, P, res, size }) {
  return (
    <Canvas w={size} h={size} deps={[i, k, g, P, res, size]} style={{ borderRadius: 2 }} draw={(x) => {
      const img = imgOf(i, res);
      x.imageSmoothingEnabled = false;
      x.drawImage(img.canvas, (k % g) * P, Math.floor(k / g) * P, P, P, 0, 0, size, size);
    }} />
  );
}

// one image's tokens: slot k = patch k. There is no extra slot.
const PosChip = ({ k }) => <span style={{ width: 22, height: 8, borderRadius: 2, background: `hsl(${(k * 47) % 360} 70% 60%)` }} />;
function ImgSeq({ g, m, stage }) {
  const N = g * g, head = 12, idxs = [...Array.from({ length: head }, (_, k) => k), null, N - 1];
  const pixels = stage === "cut" || stage === "proj", vecs = stage !== "cut";
  return (
    <div style={{ display: "flex", alignItems: "flex-end", gap: 4, flexWrap: "wrap" }}>
      {vecs && (
        <div style={{ alignSelf: "flex-end", paddingBottom: stage === "pos" ? 27 : 16 }}>
          <span style={{ display: "block", fontFamily: mono, fontSize: 13, color: dimColOf("D"), writingMode: "vertical-rl", transform: "rotate(180deg)", height: 64, textAlign: "center", borderLeft: `1.5px solid ${dimColOf("D")}`, paddingLeft: 3 }}>D = {m.D}</span>
        </div>
      )}
      {idxs.map((k) => k === null
        ? <span key="dots" style={{ fontFamily: mono, fontSize: 15, color: T.dim, alignSelf: "center", padding: "0 4px" }}>…</span>
        : (
          <div key={k} style={{ display: "grid", gap: 3, justifyItems: "center" }}>
            {pixels && <Crop i={0} k={k} g={g} P={m.P} res={m.res} size={stage === "cut" ? 34 : 22} />}
            {stage === "proj" && <span style={{ fontFamily: mono, fontSize: 13, color: T.accent2, lineHeight: 1 }}>↓</span>}
            {vecs && <VecCol seed={k + (stage === "after" ? 5000 : 0)} />}
            {stage === "pos" && <PosChip k={k} />}
            <span style={{ fontFamily: mono, fontSize: 12, color: T.dim }}>{k}</span>
          </div>
        ))}
    </div>
  );
}

function ImgCut({ n, m, g }) {
  const pp = 3 * m.P * m.P;
  return (
    <div style={{ display: "grid", gap: 16, padding: 6 }}>
      <Row gap={10}>{SUBJECTS.slice(0, n).map((s, i) => <div key={s.id} style={{ outline: `2px solid ${PAIR_COL[i]}`, borderRadius: 4, lineHeight: 0 }}><PatchImage img={imgOf(i, m.res)} P={m.P} size={84} grid={0.6} /></div>)}</Row>
      <Label>image 0 · {m.res}×{m.res} → {g}×{g} = {g * g} patches</Label>
      {m.res % m.P > 0 && <Note>{m.res} = {g} × {m.P} + {m.res % m.P} · the last {m.res % m.P} pixel rows and columns are never seen</Note>}
      <ImgSeq g={g} m={m} stage="cut" />
      <Row>
        <Mat shape="n C H W" name="images" rows={m.res} cols={m.res} color={T.soft} />
        <Op>cut</Op>
        <Mat shape="n N P²C" name="patches" rows={g * g} cols={pp} color={T.soft} />
      </Row>
    </div>
  );
}

function ImgProj({ m, g }) {
  const pp = 3 * m.P * m.P;
  return (
    <div style={{ display: "grid", gap: 16, padding: 6 }}>
      <Label color={T.accent2}>every patch → the same linear map → D = {m.D}</Label>
      <ImgSeq g={g} m={m} stage="proj" />
      <Row>
        <Mat shape="n N P²C" name="patches" rows={g * g} cols={pp} color={T.soft} />
        <Op>×</Op>
        <Mat shape="P²C D" name="W (conv)" rows={pp} cols={m.D} color={T.accent2} />
        <Op>+ bias =</Op>
        <Mat shape="n N D" name="patch tokens" rows={g * g} cols={m.D} glow />
      </Row>
    </div>
  );
}

function PosRows({ N }) {
  const rows = [0, 1, 2, null, N - 1];
  return (
    <div style={{ display: "grid", gap: 3, padding: "8px 12px", border: `1px solid ${T.rule}`, borderRadius: 8, background: T.well }}>
      <span style={{ fontFamily: mono, fontSize: 13, color: T.dim, marginBottom: 2 }}>position_embedding [{N}, D]</span>
      {rows.map((k) => k === null
        ? <span key="dots" style={{ fontFamily: mono, fontSize: 14, color: T.dim, paddingLeft: 82 }}>⋮</span>
        : (
          <div key={k} style={{ display: "flex", alignItems: "center", gap: 8, fontFamily: mono, fontSize: 14 }}>
            <span style={{ width: 74, whiteSpace: "nowrap", textAlign: "right", color: T.dim }}>row {k}</span>
            <span style={{ width: 90, height: 10, borderRadius: 2, background: `hsl(${(k * 47) % 360} 70% 60%)` }} />
            <span style={{ color: T.soft }}>→ patch {k}</span>
          </div>
        ))}
    </div>
  );
}

function ImgPos({ m, g }) {
  const N = g * g;
  return (
    <div style={{ display: "grid", gap: 16, padding: 6 }}>
      <Label>+ position row k → patch k · <span style={{ color: T.accent2 }}>no CLS slot</span></Label>
      <Row gap={24} style={{ alignItems: "flex-end" }}>
        <ImgSeq g={g} m={m} stage="pos" />
        <PosRows N={N} />
      </Row>
      <Row>
        <Mat shape="n N D" name="x" rows={N} cols={m.D} />
        <Op>+</Op>
        <Mat shape="N D" name="position_embedding" rows={N} cols={m.D} color={T.accent2} />
        <Op>=</Op>
        <Mat shape="n N D" name="x" rows={N} cols={m.D} glow />
      </Row>
      <Note>{N} tokens, not {N + 1} · CLIP: {N + 1}</Note>
    </div>
  );
}

function ImgTf({ m, g }) {
  const N = g * g;
  return (
    <div style={{ display: "grid", gap: 14, padding: 6 }}>
      <Label>going in · {N} vectors of {m.D}</Label>
      <ImgSeq g={g} m={m} stage="before" />
      <div style={{ fontFamily: mono, fontSize: 16, color: T.accent2, paddingLeft: 8 }}>↓ {m.L} blocks · <b>no mask</b>: everyone sees everyone</div>
      <Label>coming out · still {N} vectors, every one updated</Label>
      <ImgSeq g={g} m={m} stage="after" />
      <Row>
        <Mat shape="n N D" name="x" rows={N} cols={m.D} />
        <Op>{m.L} × block</Op>
        <Op color={T.wire}>post_layernorm</Op>
        <Mat shape="n N D" name="x" rows={N} cols={m.D} glow />
      </Row>
    </div>
  );
}

// MAP = multihead attention pooling: ONE learned query (the probe) attends over
// all patch outputs. It is not a token: it never went through the transformer.
function ImgMap({ n, m, g }) {
  const img = imgOf(0, m.res);
  const w = useMemo(() => clsRow("object", patchStats(img, m.P).feats, g, 0.8), [img, m.P, g]);
  return (
    <div style={{ display: "grid", gap: 16, padding: 6 }}>
      <Row gap={22} style={{ alignItems: "center" }}>
        <div style={{ display: "grid", gap: 4, justifyItems: "center" }}>
          <VecCol seed={777} cls h={80} />
          <span style={{ fontFamily: mono, fontSize: 14, color: T.accent, fontWeight: 700 }}>probe</span>
          <span style={{ fontFamily: mono, fontSize: 13, color: T.dim }}>learned [1, D]</span>
        </div>
        <Op>query →</Op>
        <div style={{ display: "grid", gap: 4, justifyItems: "center" }}>
          <PatchImage img={img} P={m.P} size={200} grid={0.25} overlay={w} />
          <span style={{ fontFamily: mono, fontSize: 13, color: T.dim }}>keys/values: all {g * g} patch outputs · toy weights</span>
        </div>
        <Op>Σ weights × v →</Op>
        <div style={{ display: "grid", gap: 4, justifyItems: "center" }}>
          <VecCol seed={778} h={80} />
          <span style={{ fontFamily: mono, fontSize: 14, color: T.ink, fontWeight: 700 }}>[1, D]</span>
        </div>
      </Row>
      <Row>
        <Mat shape="n N D" name="x" rows={g * g} cols={m.D} />
        <Op>attend(probe, x, x)</Op>
        <Mat shape="n 1 D" name="h" rows={1} cols={m.D} rl="1" />
        <Op>+ MLP(LN(h))</Op>
        <Op>[:, 0]</Op>
        <Mat shape="n D" name="image_embeds" rows={n} cols={m.D} glow />
      </Row>
      <Note>a learned question asked of every patch · CLIP used a CLS token instead</Note>
    </div>
  );
}

// ------------------------------------------------------------ text tower
function TxtEmb({ m }) {
  return (
    <div style={{ display: "grid", gap: 16, padding: 6 }}>
      <Row>
        <Mat shape="n L" name="ids" rows={8} cols={S_CTX} color={T.wire} />
        <Op>lookup rows of</Op>
        <Mat shape="V D" name="token_embedding" rows={S_VOCAB} cols={m.D} color={T.accent2} />
        <Op>=</Op>
        <Mat shape="n L D" name="x" rows={S_CTX} cols={m.D} />
        <Op>+</Op>
        <Mat shape="L D" name="position_embedding" rows={S_CTX} cols={m.D} color={T.accent2} />
      </Row>
      <Note>a lookup: one row per token</Note>
    </div>
  );
}

const MC = { ok: "#2f8f80", pad: "#3a6a73", inf: "#5a2a35" };
function MaskThumb({ ctx, causal, label }) {
  const S = 130;
  return (
    <div style={{ display: "grid", gap: 4, justifyItems: "start" }}>
      <Canvas w={S} h={S} deps={[ctx, causal]} draw={(x) => {
        const u = S / ctx;
        for (let i = 0; i < ctx; i++) for (let j = 0; j < ctx; j++) {
          x.fillStyle = causal && j > i ? MC.inf : MC.ok;
          x.fillRect(j * u, i * u, u + 0.4, u + 0.4);
        }
      }} />
      <span style={{ fontFamily: mono, fontSize: 14, color: causal ? T.dim : T.accent, fontWeight: causal ? 400 : 700 }}>{label}</span>
    </div>
  );
}

function TxtTf({ m, tk }) {
  const cap = SUBJECTS[0].caption;
  const toks = tk ? tk.encode(cap).toks.map((t) => t.piece) : [];
  const len = toks.length + 1, K = Math.min(len + 3, 12), cell = 30;
  const name = (j) => (j < len - 1 ? toks[j] : "</s>");
  return (
    <div style={{ display: "grid", gap: 14, padding: 6 }}>
      <Label>mask added to the scores · caption 0: "{cap}"</Label>
      <Row gap={26} wrap={false} style={{ alignItems: "flex-start" }}>
        <div style={{ display: "grid", gridTemplateColumns: `58px repeat(${K}, ${cell}px)`, gap: 2, alignItems: "center" }}>
          <span style={{ fontFamily: mono, fontSize: 12, color: T.dim, textAlign: "right", paddingRight: 4 }}>query ↓ key →</span>
          {Array.from({ length: K }, (_, j) => (
            <span key={j} style={{ fontFamily: mono, fontSize: 12, color: j >= len - 1 ? T.wire : T.soft, writingMode: "vertical-rl", transform: "rotate(180deg)", height: 48, justifySelf: "center" }}>{name(j)}</span>
          ))}
          {Array.from({ length: K }, (_, i) => [
            <span key={`r${i}`} style={{ fontFamily: mono, fontSize: 13, color: i >= len - 1 ? T.wire : T.soft, textAlign: "right", paddingRight: 4 }}>{name(i)}</span>,
            ...Array.from({ length: K }, (_, j) => (
              <div key={j} style={{ height: cell - 6, borderRadius: 3, display: "grid", placeItems: "center", fontFamily: mono, fontSize: 14, fontWeight: 700, background: i >= len - 1 || j >= len - 1 ? MC.pad : MC.ok, color: T.ink }}>0</div>
            )),
          ])}
        </div>
        <div style={{ display: "grid", gap: 14 }}>
          <Row gap={18} wrap={false}>
            <MaskThumb ctx={77} causal label="CLIP · causal" />
            <MaskThumb ctx={S_CTX} label="SigLIP · no mask" />
          </Row>
          <Row>
            <Mat shape="n L D" name="x" rows={S_CTX} cols={m.D} />
            <Op>{m.L} × block · {m.h} heads</Op>
            <Mat shape="n L D" name="x" rows={S_CTX} cols={m.D} glow />
          </Row>
        </div>
      </Row>
      <Note>all zeros: every token, padding included, sees every token</Note>
    </div>
  );
}

function TxtPool({ n, m, tk }) {
  if (!tk) return null;
  const cell = 7;
  return (
    <div style={{ display: "grid", gap: 16, padding: 6 }}>
      <Label color={T.wire}>take slot 63, the last one, for every caption</Label>
      <div style={{ display: "grid", gap: 6 }}>
        {SUBJECTS.slice(0, n).map((s, i) => {
          const { ids, len } = tk.tokenize(s.caption);
          return (
            <Row key={s.id} gap={10} wrap={false}>
              <Thumb i={i} size={22} />
              <div style={{ display: "flex", gap: 1, position: "relative", paddingTop: 12 }}>
                <span style={{ position: "absolute", top: -1, right: -2, color: T.wire, fontSize: 14, lineHeight: 1 }}>▼</span>
                {ids.map((v, j) => <span key={j} style={{ width: cell, height: 14, borderRadius: 1, background: j === S_CTX - 1 ? T.wire : j < len - 1 ? PAIR_COL[i] : `${T.wire}55` }} />)}
              </div>
              <span style={{ fontFamily: mono, fontSize: 14, color: T.dim, whiteSpace: "nowrap" }}>{len - 1} pieces</span>
            </Row>
          );
        })}
      </div>
      <Row>
        <Mat shape="n L D" name="x" rows={S_CTX} cols={m.D} />
        <Op>x[:, −1]</Op>
        <Mat shape="n D" name="pooled" rows={n} cols={m.D} color={T.wire} />
        <Op>× head</Op>
        <Mat shape="D D" name="W_head" rows={m.D} cols={m.D} color={T.accent2} />
        <Op>=</Op>
        <Mat shape="n D" name="text_embeds" rows={n} cols={m.D} glow color={T.wire} />
      </Row>
      <Note>a padding &lt;/s&gt; · no mask, so it has read the whole caption</Note>
    </div>
  );
}

// ------------------------------------------------------------ the n × n matrix
// The same logits on every loss step; a second line in each cell adds what
// the step introduces (z, then ℓ). Diagonal = true pairs.
function SigMatrix({ n, toy, mode, sel, setSel }) {
  const cell = n > 6 ? 54 : 62, idx = Array.from({ length: n }, (_, j) => j);
  const lo = Math.min(...toy.logits.flat()), hi = Math.max(...toy.logits.flat());
  const shade = (i, j) => (toy.logits[i][j] - lo) / (hi - lo || 1);
  const sub = mode === "labels" || mode === "cell" || mode === "loss";
  const sum = mode === "loss";
  return (
    <div style={{ display: "grid", gridTemplateColumns: `40px repeat(${n}, ${cell}px)${sum ? " 60px" : ""}`, gap: 3, alignItems: "center" }}>
      <span style={{ fontFamily: mono, fontSize: 12, color: T.dim }}>L</span>
      {idx.map((j) => <span key={j} style={{ fontFamily: mono, fontSize: 13, color: PAIR_COL[j], textAlign: "center", overflow: "hidden" }}>{SUBJECTS[j].id}</span>)}
      {sum && <span style={{ fontFamily: mono, fontSize: 13, color: T.wire, textAlign: "center" }}>Σ row</span>}
      {idx.map((i) => [
        <div key={`t${i}`} style={{ justifySelf: "center" }}><Thumb i={i} size={32} /></div>,
        ...idx.map((j) => {
          const on = sel && sel[0] === i && sel[1] === j, d = i === j, s = shade(i, j);
          const [r_, g_, b_] = heat(s), light = 0.299 * r_ + 0.587 * g_ + 0.114 * b_ > 150; // text colour from the cell's brightness
          return (
            <div key={j} onClick={() => setSel && setSel([i, j])} style={{
              height: sub ? cell - 6 : cell - 14, borderRadius: 4, display: "grid", placeItems: "center", alignContent: "center", gap: 1, fontFamily: mono,
              background: heatCss(s), color: light ? T.bg : T.ink, cursor: setSel ? "pointer" : "default",
              outline: on ? `3px solid ${T.ink}` : d ? `2px solid ${T.accent}` : "none", outlineOffset: -2,
            }}>
              <span style={{ fontSize: 15, fontWeight: d ? 700 : 400 }}>{toy.logits[i][j].toFixed(1)}</span>
              {mode === "labels" && <span style={{ fontSize: 12, fontWeight: 700, color: d ? "#0b5e50" : "#7a1f2c", background: d ? "#9ff5e3" : "#ffb3be", borderRadius: 3, padding: "0 3px" }}>{d ? "+1" : "−1"}</span>}
              {(mode === "cell" || mode === "loss") && <span style={{ fontSize: 12, color: light ? "#1a2230" : T.soft }}>ℓ {toy.ell[i][j].toFixed(2)}</span>}
            </div>
          );
        }),
        sum && <span key={`s${i}`} style={{ fontFamily: mono, fontSize: 15, color: T.wire, textAlign: "right" }}>{toy.rowSum[i].toFixed(2)}</span>,
      ])}
    </div>
  );
}

// σ(x) with every cell of the matrix dropped on it: positives want x → +∞
// (σ → 1), negatives want x → −∞ (σ → 0)
function SigmoidPlot({ toy, sel, xb, w = 300, h = 170 }) {
  const X0 = -8, X1 = 8, X = (x) => 30 + ((x - X0) / (X1 - X0)) * (w - 40), Y = (y) => 10 + (1 - y) * (h - 40);
  const pts = Array.from({ length: 81 }, (_, k) => X0 + (k / 80) * (X1 - X0));
  return (
    <svg width={w} height={h} style={{ background: T.well, borderRadius: 8 }}>
      <line x1={X(0)} x2={X(0)} y1={Y(1)} y2={Y(0)} stroke={T.rule} />
      <line x1={X(X0)} x2={X(X1)} y1={Y(0.5)} y2={Y(0.5)} stroke={T.rule} strokeDasharray="3 4" />
      <polyline points={pts.map((x) => `${X(x)},${Y(sigmoid(x))}`).join(" ")} fill="none" stroke={T.soft} strokeWidth={2} />
      {toy && toy.logits.flatMap((r, i) => r.map((l, j) => {
        const on = sel && sel[0] === i && sel[1] === j, c = i === j ? T.accent : T.bad, lx = Math.max(X0, Math.min(X1, l));
        return <circle key={`${i}-${j}`} cx={X(lx)} cy={Y(sigmoid(l))} r={on ? 7 : 4} fill={c} stroke={on ? T.ink : "none"} strokeWidth={2} opacity={on ? 1 : 0.75} />;
      }))}
      {xb != null && <circle cx={X(Math.max(X0, xb))} cy={Y(sigmoid(xb))} r={7} fill={T.wire} />}
      <text x={X(X1)} y={h - 8} textAnchor="end" fontFamily={mono} fontSize={13} fill={T.dim}>L →</text>
      <text x={X(X0)} y={h - 8} fontFamily={mono} fontSize={13} fill={T.dim}>σ(L)</text>
      <text x={X(0) + 4} y={Y(1) + 12} fontFamily={mono} fontSize={13} fill={T.dim}>1</text>
    </svg>
  );
}

function CellSide({ toy, sel }) {
  const [i, j] = sel, z = toy.z[i][j], L = toy.logits[i][j], zl = z * L, p = sigmoid(zl), ell = -Math.log(p);
  const pos = i === j, C = pos ? "#6EF3DD" : "#FF8A8A";
  const tex = [
    "\\begin{aligned}",
    `\\ell_{${i}${j}} &= -\\log \\sigma(z_{${i}${j}}\\,L_{${i}${j}}) \\\\[6pt]`,
    `&= -\\log \\sigma(\\textcolor{${C}}{${z > 0 ? "+1" : "-1"}} \\times ${L.toFixed(2)}) \\\\[6pt]`,
    `&= -\\log ${p.toFixed(3)} = ${ell.toFixed(3)}`,
    "\\end{aligned}",
  ].join("\n");
  return (
    <div style={{ display: "grid", gap: 10, minWidth: 0 }}>
      <Label color={C}>{pos ? `image ${i} + caption ${j}: a true pair` : `image ${i} + caption ${j}: not a pair`} · click any cell</Label>
      <TexBlock tex={tex} size={18} />
      <svg width={300} height={34}><Tex x={0} y={2} size={16} color={T.soft} tex={"\\sigma(x)=\\frac{1}{1+e^{-x}}"} /></svg>
      <SigmoidPlot toy={toy} sel={sel} />
      <span style={{ fontFamily: mono, fontSize: 14, color: T.soft }}><b style={{ color: T.accent }}>●</b> pairs want σ → 1 · <b style={{ color: T.bad }}>●</b> others want σ → 0</span>
    </div>
  );
}

function LossSide({ n, toy }) {
  const tex = [
    "\\begin{aligned}",
    `\\mathcal{L} &= \\frac{1}{n}\\sum_i \\sum_j \\ell_{ij} \\\\[6pt]`,
    `&= \\frac{\\textcolor{#6EF3DD}{${toy.pos.toFixed(2)}} + \\textcolor{#FF8A8A}{${toy.neg.toFixed(2)}}}{${n}} \\\\[6pt]`,
    `&= ${toy.loss.toFixed(3)}`,
    "\\end{aligned}",
  ].join("\n");
  return (
    <div style={{ display: "grid", gap: 10 }}>
      <TexBlock tex={tex} size={19} />
      <span style={{ fontFamily: mono, fontSize: 15, color: T.soft, lineHeight: 1.7 }}>
        <b style={{ color: T.accent }}>{n}</b> positives (diagonal)<br />
        <b style={{ color: T.bad }}>{n * n - n}</b> negatives<br />
        no softmax · no second direction
      </span>
    </div>
  );
}

function MatrixScene({ n, toy, mode }) {
  const [sel, setSel] = useState([0, 0]);
  const s = [Math.min(sel[0], n - 1), Math.min(sel[1], n - 1)];
  return (
    <Row gap={24} wrap={false} style={{ alignItems: "flex-start", padding: 6, overflowX: "auto" }}>
      <SigMatrix n={n} toy={toy} mode={mode} sel={mode === "cell" ? s : null} setSel={mode === "cell" ? setSel : null} />
      <div style={{ display: "grid", gap: 10, minWidth: 0, maxWidth: 380 }}>
        {mode === "logits" && <>
          <svg width={300} height={40}><Tex x={0} y={4} size={20} color={T.ink} tex={"L_{ij}=t\\,\\hat{I}_i\\cdot\\hat{T}_j+b"} /></svg>
          <span style={{ fontFamily: mono, fontSize: 16, color: T.soft, lineHeight: 1.8 }}>
            t = <b style={{ color: T.wire }}>{toy.t}</b> · b = <b style={{ color: T.wire }}>{toy.b}</b> <span style={{ color: T.dim, fontSize: 14 }}>toy values</span><br />
            both learned · start at t = 10, b = −10
          </span>
        </>}
        {mode === "labels" && <>
          <svg width={300} height={66}><Tex x={0} y={4} size={20} color={T.ink} tex={"z_{ij}=\\begin{cases}+1 & i=j\\\\-1 & i\\neq j\\end{cases}"} h={60} /></svg>
          <span style={{ fontFamily: mono, fontSize: 16, color: T.soft, lineHeight: 1.8 }}>
            <b style={{ color: T.accent }}>+1</b>: "these match"<br />
            <b style={{ color: T.bad }}>−1</b>: "these don't"<br />
            n² separate yes/no questions
          </span>
        </>}
        {mode === "cell" && <CellSide toy={toy} sel={s} />}
        {mode === "loss" && <LossSide n={n} toy={toy} />}
      </div>
    </Row>
  );
}

// at init cos ≈ 0 so every logit ≈ b. Per image: one positive, n − 1 negatives.
function BiasScene() {
  const [b, setB] = useState(-10);
  const [nn, setNn] = useState(32768);
  const posL = -Math.log(sigmoid(b)), negL = -Math.log(sigmoid(-b)), negT = (nn - 1) * negL, tot = posL + negT;
  const fmt = (x) => (x >= 1000 ? x.toLocaleString("en-US", { maximumFractionDigits: 0 }) : x >= 1 ? x.toFixed(2) : x.toExponential(1));
  const bar = (v, c) => <div style={{ height: 20, width: `${Math.max(0.5, (v / tot) * 100)}%`, background: c, borderRadius: 3, transition: "width .3s" }} />;
  return (
    <div style={{ display: "grid", gap: 16, padding: 6 }}>
      <Row gap={22}>
        <label style={{ display: "flex", alignItems: "center", gap: 10, fontFamily: mono, fontSize: 16, color: T.soft }}>
          b
          <input type="range" min={-15} max={0} step={0.5} value={b} onChange={(e) => setB(+e.target.value)} style={{ accentColor: T.wire, width: 200 }} />
          <b style={{ color: T.wire, minWidth: 50 }}>{b.toFixed(1)}</b>
        </label>
        <Seg label="BATCH" value={nn} options={[8, 32768]} onChange={setNn} fmt={(v) => v.toLocaleString("en-US")} />
      </Row>
      <Row gap={30} style={{ alignItems: "flex-start" }}>
        <div style={{ display: "grid", gap: 6 }}>
          <Label>at the start: cos ≈ 0 → every L ≈ b</Label>
          <SigmoidPlot xb={b} />
          <span style={{ fontFamily: mono, fontSize: 15, color: T.soft }}>σ(b) = {sigmoid(b) < 0.001 ? sigmoid(b).toExponential(1) : sigmoid(b).toFixed(3)} · "probably no match"</span>
        </div>
        <div style={{ display: "grid", gap: 10, minWidth: 340, flex: 1 }}>
          <Label>one image's loss at the start</Label>
          <div style={{ display: "grid", gridTemplateColumns: "200px 1fr 90px", gap: 8, alignItems: "center", fontFamily: mono, fontSize: 15 }}>
            <span style={{ color: T.accent }}>1 positive</span>{bar(posL, T.accent)}<span style={{ textAlign: "right", color: T.ink }}>{fmt(posL)}</span>
            <span style={{ color: T.bad }}>{(nn - 1).toLocaleString("en-US")} negatives</span>{bar(negT, T.bad)}<span style={{ textAlign: "right", color: T.ink }}>{fmt(negT)}</span>
          </div>
          <span style={{ fontFamily: mono, fontSize: 17, color: T.ink }}>total {fmt(tot)}</span>
          <span style={{ fontFamily: mono, fontSize: 15, color: T.soft, lineHeight: 1.7, borderTop: `1px solid ${T.rule}`, paddingTop: 8 }}>
            {b > -3
              ? <>negatives drown the signal: <b style={{ color: T.bad }}>{((negT / tot) * 100).toFixed(1)}%</b> of the loss</>
              : <>start from "no match" · the loss is about the <b style={{ color: T.accent }}>positives</b></>}
          </span>
        </div>
      </Row>
    </div>
  );
}

// the same logits, CLIP's way and SigLIP's way, for one row
function VsClip({ n, toy }) {
  const [k, setK] = useState(0);
  const row = toy.logits[k];
  const clipL = row.map((l) => l - toy.b); // CLIP: t·cos, no b (adding b would change nothing)
  const m = Math.max(...clipL), e = clipL.map((x) => Math.exp(x - m)), Z = e.reduce((a, c) => a + c, 0), sm = e.map((x) => x / Z);
  const sg = row.map(sigmoid), sgSum = sg.reduce((a, c) => a + c, 0);
  const Col = ({ title, vals, sumv, tex, c }) => (
    <div style={{ display: "grid", gap: 6, minWidth: 300, flex: 1 }}>
      <Label color={c}>{title}</Label>
      <div style={{ minHeight: 56, display: "flex", alignItems: "center" }}><TexBlock tex={tex} size={18} /></div>
      {vals.map((v, j) => (
        <div key={j} style={{ display: "grid", gridTemplateColumns: "70px 1fr 50px", gap: 8, alignItems: "center", fontFamily: mono, fontSize: 15 }}>
          <span style={{ color: PAIR_COL[j] }}>{SUBJECTS[j].id}</span>
          <div style={{ height: 14, borderRadius: 3, background: T.well }}><div style={{ height: "100%", width: `${v * 100}%`, borderRadius: 3, background: j === k ? c : T.soft, transition: "width .3s" }} /></div>
          <span style={{ textAlign: "right", color: j === k ? c : T.soft, fontWeight: j === k ? 700 : 400 }}>{v.toFixed(2)}</span>
        </div>
      ))}
      <span style={{ fontFamily: mono, fontSize: 16, color: T.ink, borderTop: `1px solid ${T.rule}`, paddingTop: 6 }}>sum = <b style={{ color: c }}>{sumv.toFixed(2)}</b></span>
    </div>
  );
  return (
    <div style={{ display: "grid", gap: 14, padding: 6 }}>
      <Row gap={8}>
        <Label>row = image</Label>
        {Array.from({ length: n }, (_, i) => <button key={i} onClick={() => setK(i)} style={{ ...btn(i === k, false), height: 30, padding: "0 6px" }}><Thumb i={i} size={22} ring={false} /></button>)}
      </Row>
      <Row gap={36} style={{ alignItems: "flex-start" }}>
        <Col title="CLIP · softmax over the row" vals={sm} sumv={1} c={T.wire} tex={"p_j=\\frac{e^{L_{kj}}}{\\sum_{j'} e^{L_{kj'}}}"} />
        <Col title="SigLIP · sigmoid per cell" vals={sg} sumv={sgSum} c={T.accent} tex={"p_j=\\sigma(L_{kj})"} />
      </Row>
      <Note>softmax: one cell needs the <b style={{ color: T.wire }}>whole row</b> · sigmoid: one cell needs <b style={{ color: T.accent }}>one cell</b></Note>
    </div>
  );
}

function ZeroShot({ toy, q, setQ }) {
  const s = toy.allTxt.map((t) => toy.t * toy.dot(toy.allImg[q], t) + toy.b);
  const p = s.map(sigmoid), sum = p.reduce((a, c) => a + c, 0), best = p.indexOf(Math.max(...p));
  return (
    <Row gap={26} style={{ alignItems: "flex-start", padding: 6 }}>
      <div style={{ display: "grid", gap: 8, justifyItems: "center" }}>
        <Label>query image</Label>
        <Thumb i={q} size={150} />
        <div style={{ display: "flex", gap: 4, flexWrap: "wrap", maxWidth: 170, justifyContent: "center" }}>
          {SUBJECTS.map((x, i) => <button key={x.id} onClick={() => setQ(i)} style={{ fontSize: 19, padding: 2, borderRadius: 5, cursor: "pointer", border: `1px solid ${i === q ? T.accent : T.rule}`, background: T.well }}>{x.emoji}</button>)}
        </div>
      </div>
      <div style={{ display: "grid", gap: 6, minWidth: 380 }}>
        <Label>σ(t · Î·W + b) · each prompt on its own · toy values</Label>
        {SUBJECTS.map((x, k) => (
          <div key={x.id} style={{ display: "grid", gridTemplateColumns: "190px 1fr 50px", gap: 8, alignItems: "center", fontFamily: mono, fontSize: 15 }}>
            <span style={{ color: k === best ? T.accent : T.soft }}>"a photo of a {x.id}"</span>
            <div style={{ height: 14, borderRadius: 3, background: T.well }}><div style={{ height: "100%", width: `${p[k] * 100}%`, borderRadius: 3, background: k === best ? T.accent : T.soft, transition: "width .4s" }} /></div>
            <span style={{ textAlign: "right", color: T.ink }}>{(p[k] * 100).toFixed(0)}%</span>
          </div>
        ))}
        <span style={{ fontFamily: mono, fontSize: 16, color: T.soft, marginTop: 6 }}>sum = <b style={{ color: T.wire }}>{(sum * 100).toFixed(0)}%</b> · not a distribution</span>
      </div>
    </Row>
  );
}

// ------------------------------------------------------------ view
export default function SiglipView() {
  const [mk, setMk] = useState("B/16");
  const [n, setN] = useState(8);
  const [q, setQ] = useState(0);
  const tk = useSpm();
  const [text, setText] = useState("A Goldendoodle, doing Photosynthesis!");
  const [wi, setWi] = useState(null);
  const [shape, setShape] = useState(SHAPES[1]);
  const [budget, setBudget] = useState(256);
  const nf = naflexSize(shape.h, shape.w, P2, budget);
  const captions = SUBJECTS.map((s) => s.caption);
  const m = SIGLIP_MODELS[mk], g = Math.floor(m.res / m.P), N = g * g; // conv padding="valid": leftover pixels dropped
  const toy = useMemo(() => sigToy(n, T0, B0), [n]);
  const tok = tk ? tk.tokenize(text) : null;

  const ringSteps = [
    ...Array.from({ length: DEVS }, (_, s) => ({ id: `ring${s}`, block: "ring", caption: s === 0 ? "each GPU: its own images × its own captions" : `captions move one GPU over · step ${s}`, vars: () => [["block", `[${n / DEVS}, ${n / DEVS}]`], ["step", `${s} / ${DEVS - 1}`]] })),
    { id: "ringsum", block: "ring", caption: "add up every GPU's sum: the same loss", vars: () => [["loss", toy.loss.toFixed(3)]] },
  ];
  const steps = [
    { id: "batch", block: "batch", caption: "a batch of n matching (image, caption) pairs", vars: () => [["n", `${n}`]] },
    { id: "img_cut", block: "img_cut", caption: `cut each image into ${g}×${g} = ${N} patches of ${m.P}×${m.P}`, vars: () => [["P", `${m.P}`], ["N", `${N}`]] },
    { id: "img_proj", block: "img_proj", caption: `project every patch to a ${m.D}-dim vector`, vars: () => [["3P²", `${3 * m.P * m.P}`], ["D", `${m.D}`]] },
    { id: "img_pos", block: "img_pos", caption: "add a position to every patch · no CLS token", vars: () => [["tokens", `${N}`]] },
    { id: "img_tf", block: "img_tf", caption: `${m.L} transformer blocks: same shape out`, vars: () => [["L", `${m.L}`], ["heads", `${m.h}`]] },
    { id: "img_map", block: "img_map", caption: "one learned query attends to all patches → one vector", vars: () => [["out", `[${n}, ${m.D}]`]] },
    { id: "tok_clean", block: "tok_clean", caption: "clean the text, mark each word start with ▁", vars: () => [["chars", tk ? `${[...tk.marked(text)].length}` : "…"]] },
    { id: "tok_viterbi", block: "tok_viterbi", caption: "pick the cut into vocab pieces with the highest total score", vars: () => [["vocab", "32,000"]] },
    { id: "tok_ids", block: "tok_ids", caption: "look up ids, add </s>, pad with </s> to 64", vars: () => [["tokens", tok ? `${tok.rawLen} / ${S_CTX}` : "…", tok?.truncated ? "bad" : null]] },
    { id: "txt_emb", block: "txt_emb", caption: "look up token embeddings, add positions", vars: () => [["D", `${m.D}`]] },
    { id: "txt_tf", block: "txt_tf", caption: "text transformer with no mask: every token sees every token", vars: () => [["L", `${m.L}`], ["heads", `${m.h}`]] },
    { id: "txt_pool", block: "txt_pool", caption: "keep the last slot, then a linear head", vars: () => [["out", `[${n}, ${m.D}]`]] },
    { id: "normalize", block: "normalize", caption: "L2-normalise: every embedding on the unit sphere", vars: () => [["‖·‖", "1"]] },
    { id: "logits", block: "logits", caption: "all n² cosines, scaled by t and shifted by b", vars: () => [["t", `${toy.t}`], ["b", `${toy.b}`]] },
    { id: "labels", block: "labels", caption: "label every cell: +1 for a true pair, −1 for the rest", vars: () => [["+1", `${n}`], ["−1", `${n * n - n}`]] },
    { id: "cell", block: "cell", caption: "each cell is its own yes/no question: −log σ(z·L)", vars: () => [["cells", `${n * n}`]] },
    { id: "loss", block: "loss", caption: "add all n² cell losses, divide by n", vars: () => [["loss", toy.loss.toFixed(3)]] },
    { id: "bias", block: "bias", caption: "why b starts at −10: almost every pair is a negative", vars: () => [["b₀", "−10"], ["t₀", "10"]] },
    { id: "vs_clip", block: "vs_clip", caption: "same logits: CLIP needs the whole row, SigLIP one cell", vars: () => [] },
    ...ringSteps,
    { id: "zeroshot", block: "zeroshot", caption: "zero-shot: each prompt gets its own probability", vars: () => [["K", `${SUBJECTS.length}`]] },
    { id: "s2_diff", block: "s2_diff", caption: "SigLIP 2: same model and loss · new image input, tokenizer, training", vars: () => [] },
    { id: "s2_size", block: "s2_size", caption: "SigLIP 2: keep the aspect ratio, resize to fit a token budget", vars: () => [["grid", `${nf.gh}×${nf.gw}`], ["patches", `${nf.N} / ${budget}`]] },
    { id: "s2_pos", block: "s2_pos", caption: "SigLIP 2: one learned 16×16 position grid, resized to each image", vars: () => [["grid", `16×16 → ${nf.gh}×${nf.gw}`]] },
    { id: "s2_mask", block: "s2_mask", caption: "SigLIP 2: pad every image to the budget, mask the padding", vars: () => [["budget", `${budget}`]] },
  ];

  const G = {
    n: ["batch of pairs", n], C: ["channels", 3], H: ["image side", m.res], W: ["image side", m.res], P: ["patch", m.P], "P²C": ["values per patch 3P²", 3 * m.P * m.P],
    g: ["patches per side", g], N: ["patches g² (= tokens)", N], D: ["width (both towers)", m.D], L: ["text context", S_CTX], V: ["SentencePiece vocab", S_VOCAB], M: ["MLP width", m.M], K: ["classes / prompts", SUBJECTS.length],
  };
  const S = (name, dims, what, io) => ({ name, dims, what, io });
  const tags = (st) => ({
    batch: [S("pixels", ["n", "C", "H", "W"], null, "in"), S("captions", ["n"], null, "in")],
    tok_ids: [S("ids", ["n", "L"], null, "out")],
    normalize: [S("image_embeds, text_embeds", ["n", "D"], null, "in"), S("unit rows", ["n", "D"], null, "out")],
    logits: [S("logits", ["n", "n"], null, "out")],
    labels: [S("z", ["n", "n"], null, "out")],
    cell: [S("logits ⊙ z", ["n", "n"], null, "in"), S("ℓ", ["n", "n"], null, "out")],
    loss: [S("ℓ", ["n", "n"], null, "in"), S("loss", [], null, "out")],
    zeroshot: [S("W (prompt embeds)", ["K", "D"]), S("image embed", ["1", "D"]), S("p", ["K"], null, "out")],
  })[st.id];

  const scene = (st) => {
    const ring = st.id.startsWith("ring");
    const body = ring ? <RingScene n={n} toy={toy} s={st.id === "ringsum" ? DEVS - 1 : +st.id.slice(4)} final={st.id === "ringsum"} /> : {
      batch: <Batch n={n} />,
      img_cut: <ImgCut n={n} m={m} g={g} />,
      img_proj: <ImgProj m={m} g={g} />,
      img_pos: <ImgPos m={m} g={g} />,
      img_tf: <ImgTf m={m} g={g} />,
      img_map: <ImgMap n={n} m={m} g={g} />,
      tok_clean: <SpmClean tk={tk} text={text} />,
      tok_viterbi: <SpmViterbi tk={tk} text={text} wi={wi} setWi={setWi} />,
      tok_ids: <SpmIds tk={tk} text={text} captions={captions} n={n} />,
      txt_emb: <TxtEmb m={m} />,
      txt_tf: <TxtTf m={m} tk={tk} />,
      txt_pool: <TxtPool n={n} m={m} tk={tk} />,
      normalize: <Normalize m={m} />,
      logits: <MatrixScene n={n} toy={toy} mode="logits" />,
      labels: <MatrixScene n={n} toy={toy} mode="labels" />,
      cell: <MatrixScene n={n} toy={toy} mode="cell" />,
      loss: <MatrixScene n={n} toy={toy} mode="loss" />,
      bias: <BiasScene />,
      vs_clip: <VsClip n={n} toy={toy} />,
      zeroshot: <ZeroShot toy={toy} q={q} setQ={setQ} />,
      s2_diff: <S2Diff />,
      s2_size: <S2Size shape={shape} budget={budget} />,
      s2_pos: <S2Pos shape={shape} budget={budget} m={m} />,
      s2_mask: <S2Mask budget={budget} m={m} />,
    }[st.id];
    const nfStep = ["s2_size", "s2_pos", "s2_mask"].includes(st.id);
    const tokStep = st.id.startsWith("tok_");
    return (
      <div style={{ minHeight: 400, overflowX: "auto", display: "grid", gap: 16, padding: tokStep ? 6 : 0, alignContent: "start" }}>
        {nfStep && <ShapePicker shape={shape} setShape={setShape} budget={budget} setBudget={setBudget} noImage={st.id === "s2_mask"} />}
        {tokStep && <SpmPicker n={n} captions={captions} text={text} setText={(t) => { setText(t); setWi(null); }} />}
        {body}
      </div>
    );
  };

  const { D, L, h, M } = m, pp = 3 * m.P * m.P, b = n / DEVS, dh = D / h;
  const matmuls = (st) => {
    const ring = st.id.startsWith("ring");
    const body = ring ? (st.id === "ringsum"
      ? <MMNote>all-reduce: {DEVS} scalars → Σ ÷ {n} = loss · nothing [{n}, {n}] ever exists</MMNote>
      : <>
        <MM a={["img_local", [b, D]]} b={["txt_heldᵀ", [D, b]]} c={["block", [b, b]]} times={DEVS} note={`one per GPU, in parallel · then −logsigmoid(z · (t·block + b))`} />
        {+st.id.slice(4) < DEVS - 1 && <MMNote>send txt_held [{b}, {D}] → next GPU (ring permute)</MMNote>}
      </>) : {
      batch: <MMNote>inputs: pixels [{n}, 3, {m.res}, {m.res}] · captions → ids [{n}, {S_CTX}] (next steps)</MMNote>,
      img_cut: <MMNote>rearrange: [{n}, 3, {m.res}, {m.res}] → [{n}, {N}, {pp}]</MMNote>,
      img_proj: <MM a={["patches", [n, N, pp]]} b={["W_conv", [pp, D]]} c={["tokens", [n, N, D]]} note={`= Conv2d, kernel = stride = ${m.P}, + bias`} />,
      img_pos: <MMNote>add: x [{n}, {N}, {D}] + position_embedding [{N}, {D}] → [{n}, {N}, {D}]</MMNote>,
      img_tf: <><BlockMMs B={n} T={N} D={D} h={h} d={dh} M={M} L={L} act="GELU (tanh approximation)" /><MMNote>post_layernorm → [{n}, {N}, {D}]</MMNote></>,
      img_map: <>
        <MM a={["probe", [1, D]]} b={["W_q", [D, D]]} c={["q", [1, D]]} note="the same q for every image" />
        <MM a={["x", [n, N, D]]} b={["W_kv", [D, 2 * D]]} c={["k, v", [n, N, 2 * D]]} />
        <MM a={["q", [n, h, 1, dh]]} b={["kᵀ", [n, h, dh, N]]} c={["S", [n, h, 1, N]]} note={`÷ √${dh}, softmax over the ${N} patches`} />
        <MM a={["A", [n, h, 1, N]]} b={["v", [n, h, N, dh]]} c={["o", [n, h, 1, dh]]} note={`concat heads → [${n}, 1, ${D}] → out_proj`} />
        <MM a={["LN(h)", [n, 1, D]]} b={["W_1", [D, M]]} c={["u", [n, 1, M]]} note="GELU" />
        <MM a={["u", [n, 1, M]]} b={["W_2", [M, D]]} c={["mlp", [n, 1, D]]} note={`h + mlp → [:, 0] → image_embeds [${n}, ${D}]`} />
      </>,
      tok_clean: <MMNote>text only · no tensors yet</MMNote>,
      tok_viterbi: <MMNote>text only · no tensors yet</MMNote>,
      tok_ids: <MMNote>ids [{n}, {S_CTX}] (int64) · &lt;/s&gt; = {EOS}</MMNote>,
      txt_emb: <>
        <MM a={["one_hot(ids)", [n, S_CTX, S_VOCAB]]} b={["token_embedding", [S_VOCAB, D]]} c={["x", [n, S_CTX, D]]} note="really just a lookup" cost="0 (a gather)" />
        <MMNote>x + position_embedding [{S_CTX}, {D}] → [{n}, {S_CTX}, {D}]</MMNote>
      </>,
      txt_tf: <><BlockMMs B={n} T={S_CTX} D={D} h={h} d={dh} M={M} L={L} act="GELU (tanh approximation)" /><MMNote>final_layer_norm → [{n}, {S_CTX}, {D}]</MMNote></>,
      txt_pool: <>
        <MMNote>index: x[:, −1] of [{n}, {S_CTX}, {D}] → pooled [{n}, {D}]</MMNote>
        <MM a={["pooled", [n, D]]} b={["W_head", [D, D]]} c={["text_embeds", [n, D]]} note="+ bias" />
      </>,
      normalize: <MMNote>row-wise: [{n}, {D}] ÷ ‖·‖ [{n}, 1] → [{n}, {D}] · both towers</MMNote>,
      logits: <>
        <MM a={["Î", [n, D]]} b={["T̂ᵀ", [D, n]]} c={["cos", [n, n]]} />
        <MMNote>logits = t · cos + b  (scalars, t = {toy.t}, b = {toy.b})</MMNote>
      </>,
      labels: <MMNote>z = 2 · eye({n}) − 1 → [{n}, {n}]</MMNote>,
      cell: <MMNote>elementwise: −logsigmoid(z ⊙ logits) → ℓ [{n}, {n}]</MMNote>,
      loss: <MMNote>ℓ.sum(dim=−1) → [{n}] → mean → scalar</MMNote>,
      bias: <MMNote>at init: cos ≈ 0 → logits ≈ b in every cell</MMNote>,
      vs_clip: <MMNote>CLIP: softmax over [{n}] per row · SigLIP: σ per cell, no Σ</MMNote>,
      zeroshot: <>
        <MM a={["Î", [1, D]]} b={["Wᵀ", [D, SUBJECTS.length]]} c={["cos", [1, SUBJECTS.length]]} note="W = normalised text embeddings of the K prompts" />
        <MMNote>σ(t · cos + b) → p [1, {SUBJECTS.length}], no softmax</MMNote>
      </>,
      s2_diff: <MMNote>SigLIP 2 changes the image input; every matmul after it is the same</MMNote>,
      s2_size: <MMNote>resize: [3, {shape.h}, {shape.w}] → [3, {nf.H}, {nf.W}] → {nf.gh}×{nf.gw} = {nf.N} patches of 768</MMNote>,
      s2_pos: <MMNote>interpolate: [16, 16, {D}] → [{nf.gh}, {nf.gw}, {D}] → [{nf.N}, {D}], padded to [{budget}, {D}]</MMNote>,
      s2_mask: <>
        <MM a={["pixel_values", [n, budget, 768]]} b={["W_patch", [768, D]]} c={["x", [n, budget, D]]} note="a Linear, not a Conv: patches arrive already cut" />
        <MMNote>+ positions [{n}, {budget}, {D}] · spatial_shapes [{n}, 2] · pixel_attention_mask [{n}, {budget}] → scores + (−∞ on padding keys)</MMNote>
      </>,
    }[st.id];
    return <MatMulCard>{body}</MatMulCard>;
  };

  return (
    <DimCtx.Provider value={G}>
      <StepShell
        matmuls={matmuls}
        title="SigLIP"
        subtitle="embeddings, similarities and losses are a toy model · shapes and tokenizer are real"
        blocks={FLOW}
        steps={steps}
        scene={scene}
        tags={tags}
        controls={
          <Row gap={18}>
            <Seg label="MODEL" value={mk} options={Object.keys(SIGLIP_MODELS)} onChange={setMk} />
            <Seg label="BATCH n" value={n} options={[4, 8]} onChange={setN} />
            <Stat k="loss" v={toy.loss.toFixed(3)} tone={T.wire} />
            <Stat k="t · b" v={`${toy.t} · ${toy.b}`} />
          </Row>
        }
      />
    </DimCtx.Provider>
  );
}
