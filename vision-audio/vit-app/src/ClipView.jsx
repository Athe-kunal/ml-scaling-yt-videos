import { useMemo, useState } from "react";
import { T, mono } from "./theme";
import { Tex } from "./tex";
import { ease, useProgress } from "./anim";
import StepShell, { Seg, Stat } from "./StepShell";
import { Canvas, PatchImage, Mat, Op, Row, Label, Bars, heatCss, MM, MMNote, MatMulCard, BlockMMs, DimCtx } from "./draw";
import { SUBJECTS, renderSubject, CLIP_MODELS, CTX, VOCAB, RES, clipToy, fmtN, fmtInt } from "./model";
import { useTokenizer, BpePicker, BpeSplit, BpeMerge, BpeIds } from "./BpeScenes";

// CLIP walkthrough: a batch of n (image, caption) pairs -> image ViT and text
// transformer -> projections -> L2 normalise -> n×n logits -> symmetric CE ->
// zero-shot. The "training" slider moves the toy embeddings from init to trained.

const CODE = {
  batch: ["# I[n, 3, 224, 224]  images", "# T[n, 77]  tokenized captions   (pair i ↔ i)"],
  img_patch: ["x = conv1(I)   # [n,3,224,224] → [n,Di,g,g]  kernel = stride = P", "x = x.reshape(n, Di, g*g).permute(0, 2, 1)   # [n, g², Di]", "x = cat([class_emb, x], 1) + pos_emb   # [n, g²+1, Di]", "x = ln_pre(x)"],
  img_tf: ["x = transformer(x)   # Li blocks, [n, g²+1, Di]", "I_f = ln_post(x[:, 0, :])   # CLS token → [n, Di]"],
  tok_split: ["text = whitespace_clean(basic_clean(text)).lower()", "words = re.findall(pat, text)   # words, numbers, punctuation", "word = tuple(w[:-1]) + (w[-1] + '</w>',)   # d o g</w>"],
  tok_merge: ["while True:", "    bigram = min(pairs, key=lambda p: bpe_ranks.get(p, inf))   # lowest rank first", "    if bigram not in bpe_ranks: break   # nothing left to merge", "    word = merge(word, bigram)   # glue every occurrence"],
  tok_ids: ["ids = [encoder[piece] for piece in pieces]   # vocab of 49,408", "T = [49406] + ids + [49407] + [0] * pad   # → [n, 77]"],
  txt_emb: ["x = token_embedding(T)   # table [49408, Dt] → [n, 77, Dt]", "x = x + positional_embedding   # [77, Dt]"],
  txt_tf: ["x = transformer(x, mask=causal)   # [n, 77, Dt]", "x = ln_final(x)", "T_f = x[arange(n), T.argmax(-1)]   # <eot> has the highest id → [n, Dt]"],
  project: ["I_e = I_f @ W_i   # [n,Di]·[Di,E] → [n,E]", "T_e = T_f @ W_t   # [n,Dt]·[Dt,E] → [n,E]"],
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
      { id: "img_patch", label: "patches + CLS + pos", sub: "[n, g²+1, Di]", code: CODE.img_patch },
      { id: "img_tf", label: "transformer → CLS", sub: "I_f [n, Di]", code: CODE.img_tf },
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
      { id: "project", label: "linear projections", sub: "Di, Dt → E", code: CODE.project },
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
const PAIR_COL = ["#6EF3DD", "#FF8AC6", "#FFC94D", "#9BE564", "#8DB4FF", "#C79BFF", "#FFB86B", "#FF8A8A"];

// ------------------------------------------------------------ pieces
const SHOW = 48; // bars drawn per vector; the real vector is longer
const More = ({ total }) => <span style={{ fontFamily: mono, fontSize: 10.5, color: T.dim, whiteSpace: "nowrap" }}>… {total - SHOW} more</span>;

function Thumb({ i, size = 64, P, ring = true }) {
  return (
    <div style={{ borderRadius: 6, outline: ring ? `2px solid ${PAIR_COL[i]}` : "none", outlineOffset: 1, lineHeight: 0 }}>
      <PatchImage img={thumbs[i]} P={P ?? 224} size={size} grid={P ? 0.7 : 0} />
    </div>
  );
}

function Tok({ s, color = T.soft, bg = T.well }) {
  return <span style={{ fontFamily: mono, fontSize: 11, padding: "2px 6px", borderRadius: 4, border: `1px solid ${color}66`, background: bg, color, whiteSpace: "nowrap" }}>{s}</span>;
}

function Batch({ n }) {
  return (
    <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(150px, 1fr))", gap: 12, padding: 6 }}>
      {SUBJECTS.slice(0, n).map((s, i) => (
        <div key={s.id} style={{ display: "grid", gap: 8, justifyItems: "center", padding: 10, borderRadius: 8, border: `1px solid ${PAIR_COL[i]}55`, background: T.well }}>
          <Thumb i={i} size={110} />
          <span style={{ fontFamily: mono, fontSize: 12, color: PAIR_COL[i], textAlign: "center" }}>"{s.caption}"</span>
          <span style={{ fontFamily: mono, fontSize: 10, color: T.dim }}>pair {i}</span>
        </div>
      ))}
      <div style={{ gridColumn: "1 / -1", fontFamily: mono, fontSize: 12, color: T.soft }}>
        CLIP trained on 400M such pairs scraped from the web · batch n = 32,768 in the paper (here n = {n})
      </div>
    </div>
  );
}

function ImgPatch({ n, m, g }) {
  const show = Math.min(g * g, 30);
  return (
    <div style={{ display: "grid", gap: 16, padding: 6 }}>
      <Row gap={10}>{SUBJECTS.slice(0, n).map((s, i) => <Thumb key={s.id} i={i} size={80} P={m.P} />)}</Row>
      <Row gap={4} wrap>
        <Tok s="CLS" color={T.bg} bg={T.accent} />
        {Array.from({ length: show }, (_, k) => <span key={k} style={{ width: 12, height: 22, borderRadius: 2, background: heatCss(0.25 + 0.5 * ((k * 37) % 11) / 11) }} />)}
        {g * g > show && <span style={{ fontFamily: mono, fontSize: 11, color: T.dim }}>… {g * g - show} more</span>}
      </Row>
      <Row>
        <Mat shape="n C H W" name="images" rows={RES} cols={RES} color={T.soft} />
        <Op>conv P={m.P}</Op>
        <Mat shape="n N Di" name="patch tokens" rows={g * g} cols={m.Di} />
        <Op>+ CLS + pos</Op>
        <Mat shape="n T Di" name="x" rows={g * g + 1} cols={m.Di} glow />
        <Op color={T.wire}>ln_pre</Op>
      </Row>
      <div style={{ fontFamily: mono, fontSize: 12, color: T.soft }}>same recipe as the ViT tab · CLIP adds an extra LayerNorm (ln_pre) before the blocks</div>
    </div>
  );
}

function ImgTf({ n, m, g, toy }) {
  return (
    <div style={{ display: "grid", gap: 16, padding: 6 }}>
      <Row>
        <Mat shape="n T Di" name="x" rows={g * g + 1} cols={m.Di} />
        <Op>{m.Li} × block</Op>
        <Mat shape="n T Di" name="x" rows={g * g + 1} cols={m.Di} />
        <Op>x[:, 0] → ln_post</Op>
        <Mat shape="n Di" name="I_f" rows={n} cols={m.Di} glow />
      </Row>
      <Label>one CLS vector per image · I_f[i] has Di = {m.Di} floats · first {SHOW} shown</Label>
      <div style={{ display: "grid", gap: 6 }}>
        {SUBJECTS.slice(0, n).map((s, i) => (
          <Row key={s.id} gap={10} wrap={false}>
            <Thumb i={i} size={32} />
            <Bars v={toy.allImg[i].slice(0, SHOW).map((x, k) => x * (1 + 0.3 * Math.sin(k)))} w={300} h={30} color={PAIR_COL[i]} />
            <More total={m.Di} />
          </Row>
        ))}
      </div>
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
        <Mat shape="n L Dt" name="x" rows={n * CTX} cols={m.Dt} />
        <Op>+</Op>
        <Mat shape="L Dt" name="pos_emb" rows={CTX} cols={m.Dt} color={T.accent2} />
      </Row>
      <div style={{ fontFamily: mono, fontSize: 12, color: T.soft }}>an embedding lookup is a matrix product with one-hot rows: {fmtN(VOCAB * m.Dt)} parameters in the table</div>
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
      <Label color={T.wire}>eot_idx = T.argmax(-1): the largest id (49407) sits at a different column in every row</Label>
      {rows.map(({ s, ids, len }, i) => (
        <Row key={s.id} gap={10} wrap={false}>
          <Thumb i={i} size={22} />
          <div style={{ display: "flex", gap: 1, position: "relative", paddingTop: 12 }}>
            <span style={{ position: "absolute", top: -1, left: (len - 1) * (cell + 1) - 2, color: T.wire, fontSize: 11, lineHeight: 1 }}>▼</span>
            {ids.map((v, j) => (
              <span key={j} style={{ width: cell, height: 14, borderRadius: 1, background: j === len - 1 ? T.wire : v ? PAIR_COL[i] : T.rule, opacity: v ? 1 : 0.4 }} />
            ))}
          </div>
          <span style={{ fontFamily: mono, fontSize: 12, color: T.wire, whiteSpace: "nowrap" }}>eot_idx = {len - 1}</span>
        </Row>
      ))}
      <span style={{ fontFamily: mono, fontSize: 12, color: T.soft }}>
        eot_idx = [{rows.map((r) => r.len - 1).join(", ")}] → T_f = x[arange({n}), eot_idx] picks one [Dt] row per caption → [{n}, Dt]
      </span>
    </div>
  );
}

function TxtTf({ n, m, tk }) {
  const len = tk ? tk.tokenize(SUBJECTS[0].caption).len : 7, S = 230;
  return (
    <div style={{ display: "grid", gap: 12, padding: 6 }}>
    <Row gap={22} style={{ alignItems: "flex-start" }}>
      <div style={{ display: "grid", gap: 6 }}>
        <Label>causal mask [77 × 77] · caption 0</Label>
        <Canvas w={S} h={S} deps={[len]} draw={(x) => {
          const u = S / CTX;
          for (let i = 0; i < CTX; i++) for (let j = 0; j <= i; j++) {
            x.fillStyle = i < len && j < len ? (i === len - 1 ? T.wire : `${T.accent}aa`) : "#232c3a";
            x.fillRect(j * u, i * u, u + 0.3, u + 0.3);
          }
        }} />
        <span style={{ fontFamily: mono, fontSize: 10.5, color: T.dim, maxWidth: S }}>token i sees tokens ≤ i only · the <b style={{ color: T.wire }}>&lt;eot&gt; row</b> sees the whole caption</span>
      </div>
      <div style={{ display: "grid", gap: 14 }}>
        <Row>
          <Mat shape="n L Dt" name="x" rows={n * CTX} cols={m.Dt} />
          <Op>{m.Lt} × block · {m.ht} heads</Op>
          <Mat shape="n L Dt" name="x" rows={n * CTX} cols={m.Dt} />
        </Row>
        <Row>
          <Op>x[arange(n), eot_idx]</Op>
          <Mat shape="n Dt" name="T_f" rows={n} cols={m.Dt} glow color={T.wire} />
        </Row>
        <div style={{ fontFamily: mono, fontSize: 12, color: T.soft, lineHeight: 1.6 }}>the text side has no CLS token:<br />the &lt;eot&gt; position plays that role</div>
      </div>
    </Row>
    <EotPointer n={n} tk={tk} />
    </div>
  );
}

function Project({ n, m }) {
  return (
    <div style={{ display: "grid", gap: 18, padding: 6 }}>
      <Row>
        <Mat shape="n Di" name="I_f" rows={n} cols={m.Di} />
        <Op>×</Op>
        <Mat shape="Di E" name="W_i" rows={m.Di} cols={m.E} color={T.accent2} />
        <Op>=</Op>
        <Mat shape="n E" name="I_e" rows={n} cols={m.E} glow />
      </Row>
      <Row>
        <Mat shape="n Dt" name="T_f" rows={n} cols={m.Dt} color={T.wire} />
        <Op>×</Op>
        <Mat shape="Dt E" name="W_t" rows={m.Dt} cols={m.E} color={T.accent2} />
        <Op>=</Op>
        <Mat shape="n E" name="T_e" rows={n} cols={m.E} glow color={T.wire} />
      </Row>
      <div style={{ fontFamily: mono, fontSize: 12, color: T.soft }}>two encoders with different widths now live in the same {m.E}-dim space</div>
    </div>
  );
}

function Normalize({ n, m, toy }) {
  const p = useProgress("norm", 1200);
  const rows = Math.min(n, 3);
  return (
    <div style={{ display: "grid", gap: 14, padding: 6 }}>
      <Label>projected I_e[i] has E = {m.E} floats · first {SHOW} shown</Label>
      {Array.from({ length: rows }, (_, i) => {
        const r = toy.rawNorm(i, "img"), s = 1 + (r - 1) * (1 - ease(p));
        return (
          <Row key={i} gap={12} wrap={false}>
            <Thumb i={i} size={34} />
            <Bars v={toy.img[i].slice(0, SHOW).map((x) => x * s)} w={260} h={44} color={PAIR_COL[i]} />
            <More total={m.E} />
            <span style={{ fontFamily: mono, fontSize: 13, color: s < 1.05 ? T.accent : T.soft, minWidth: 110 }}>‖I_e‖ = {s.toFixed(2)}</span>
          </Row>
        );
      })}
      <svg width={540} height={44}><Tex x={0} y={4} size={16} color={T.ink} tex={"\\hat{u}=\\frac{u}{\\lVert u\\rVert_2}\\quad\\Rightarrow\\quad \\hat{I}_i\\cdot\\hat{T}_j=\\cos\\theta_{ij}\\in[-1,1]"} /></svg>
      <div style={{ fontFamily: mono, fontSize: 12, color: T.soft }}>after this, a dot product is a cosine similarity: only direction matters</div>
    </div>
  );
}

// the n × n matrix in three modes: raw logits, row softmax, column softmax
function SimMatrix({ n, toy, mode }) {
  const cell = n > 6 ? 50 : 62;
  const val = (i, j) => (mode === "logits" ? toy.logits[i][j] : mode === "row" ? toy.pRow[i][j] : toy.pCol[j][i]);
  const shade = (i, j) => (mode === "logits" ? (toy.cos[i][j] + 0.2) / 1.2 : val(i, j));
  return (
    <div style={{ display: "grid", gridTemplateColumns: `52px repeat(${n}, ${cell}px) ${mode === "row" ? "70px" : ""}`, gap: 3, alignItems: "center" }}>
      <span />
      {SUBJECTS.slice(0, n).map((s, j) => <span key={s.id} style={{ fontFamily: mono, fontSize: 10, color: PAIR_COL[j], textAlign: "center", overflow: "hidden" }}>T{j} {s.id}</span>)}
      {mode === "row" && <span style={{ fontFamily: mono, fontSize: 10, color: T.wire, textAlign: "center" }}>−log p</span>}
      {SUBJECTS.slice(0, n).map((s, i) => [
        <div key={`t${i}`} style={{ justifySelf: "center" }}><Thumb i={i} size={36} /></div>,
        ...SUBJECTS.slice(0, n).map((_, j) => {
          const v = val(i, j), diag = i === j;
          return (
            <div key={j} style={{
              height: cell - 14, borderRadius: 4, display: "grid", placeItems: "center", fontFamily: mono, fontSize: 11, fontWeight: diag ? 700 : 400,
              background: heatCss(shade(i, j)), color: shade(i, j) > 0.45 ? T.bg : T.soft,
              outline: diag ? `2px solid ${T.accent}` : "none", outlineOffset: -2, transition: "background .3s",
            }}>
              {mode === "logits" ? v.toFixed(1) : v < 0.005 ? "0" : v.toFixed(2)}
            </div>
          );
        }),
        mode === "row" && <span key={`l${i}`} style={{ fontFamily: mono, fontSize: 12, color: T.wire, textAlign: "center" }}>{toy.lossRow[i].toFixed(2)}</span>,
      ])}
      {mode === "col" && [
        <span key="lab" style={{ fontFamily: mono, fontSize: 10, color: T.wire }}>−log p</span>,
        ...SUBJECTS.slice(0, n).map((_, j) => <span key={j} style={{ fontFamily: mono, fontSize: 12, color: T.wire, textAlign: "center" }}>{toy.lossCol[j].toFixed(2)}</span>),
      ]}
    </div>
  );
}

function MatrixScene({ n, toy, mode }) {
  return (
    <Row gap={26} style={{ alignItems: "flex-start", padding: 6 }}>
      <SimMatrix n={n} toy={toy} mode={mode} />
      <div style={{ display: "grid", gap: 10, maxWidth: 300 }}>
        {mode === "logits" && <>
          <svg width={300} height={40}><Tex x={0} y={4} size={16} color={T.ink} tex={"L_{ij}=e^{t}\\,\\hat{I}_i\\cdot\\hat{T}_j"} /></svg>
          <span style={{ fontFamily: mono, fontSize: 12, color: T.soft, lineHeight: 1.6 }}>e^t = <b style={{ color: T.wire }}>{toy.scale.toFixed(1)}</b> (init 1/0.07 = 14.3, learned). Without it cosines in [−1, 1] give a flat softmax.</span>
          <span style={{ fontFamily: mono, fontSize: 12, color: T.soft, lineHeight: 1.6, borderLeft: `2px solid ${T.wire}`, paddingLeft: 8 }}>
            s = min(e^t, <b style={{ color: T.wire }}>100</b>): once e^t reaches 100 it is clipped, and t gets zero gradient, so it stops learning. This keeps training stable (§2.5).
          </span>
          <span style={{ fontFamily: mono, fontSize: 12, color: T.soft, lineHeight: 1.6 }}>diagonal = the {n} true pairs · off-diagonal = {n * n - n} negatives, free from the batch</span>
        </>}
        {mode === "row" && <>
          <svg width={300} height={44}><Tex x={0} y={4} size={15} color={T.ink} tex={"\\ell_i=\\frac1n\\sum_i -\\log\\frac{e^{L_{ii}}}{\\sum_j e^{L_{ij}}}"} /></svg>
          <span style={{ fontFamily: mono, fontSize: 12, color: T.soft, lineHeight: 1.6 }}>each image picks among the {n} captions; every row sums to 1</span>
          <Stat k="loss_i" v={toy.li.toFixed(3)} tone={T.wire} />
        </>}
        {mode === "col" && <>
          <svg width={300} height={44}><Tex x={0} y={4} size={15} color={T.ink} tex={"\\ell_t=\\frac1n\\sum_j -\\log\\frac{e^{L_{jj}}}{\\sum_i e^{L_{ij}}}"} /></svg>
          <span style={{ fontFamily: mono, fontSize: 12, color: T.soft, lineHeight: 1.6 }}>each caption picks among the {n} images; every column sums to 1</span>
          <Stat k="loss_t" v={toy.lt.toFixed(3)} tone={T.wire} />
        </>}
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
        <text x={W - 22} y={Y(Math.log(n)) - 5} textAnchor="end" fontFamily={mono} fontSize={10} fill={T.bad}>chance = ln {n} = {Math.log(n).toFixed(2)}</text>
        <polyline points={curve.map((v, k) => `${X(k / 40)},${Y(v)}`).join(" ")} fill="none" stroke={T.accent} strokeWidth={2} />
        <circle cx={X(train)} cy={Y(toy.loss)} r={5} fill={T.wire} />
        <text x={40} y={H - 8} fontFamily={mono} fontSize={10} fill={T.dim}>init</text>
        <text x={W - 20} y={H - 8} textAnchor="end" fontFamily={mono} fontSize={10} fill={T.dim}>trained →</text>
      </svg>
      <div style={{ fontFamily: mono, fontSize: 12, color: T.soft, lineHeight: 1.6 }}>
        drag <b style={{ color: T.wire }}>training</b> in the header: the diagonal lights up, loss falls from ln n toward 0<br />
        bigger n = more negatives per positive = a harder, more informative task
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
          {SUBJECTS.map((s, i) => <button key={s.id} onClick={() => setQ(i)} style={{ fontSize: 16, padding: 2, borderRadius: 5, cursor: "pointer", border: `1px solid ${i === q ? T.accent : T.rule}`, background: T.well }}>{s.emoji}</button>)}
        </div>
      </div>
      <div style={{ display: "grid", gap: 6, minWidth: 360 }}>
        <Label>softmax(100 · Î · Wᵀ) over K = {SUBJECTS.length} prompts</Label>
        {SUBJECTS.map((s, k) => (
          <div key={s.id} style={{ display: "grid", gridTemplateColumns: "190px 1fr 50px", gap: 8, alignItems: "center", fontFamily: mono, fontSize: 12 }}>
            <span style={{ color: k === best ? T.accent : T.soft }}>"a photo of a {s.id}"</span>
            <div style={{ height: 14, borderRadius: 3, background: T.well }}><div style={{ height: "100%", width: `${pr[k] * 100}%`, borderRadius: 3, background: k === best ? T.accent : T.soft, transition: "width .4s" }} /></div>
            <span style={{ textAlign: "right", color: T.ink }}>{(pr[k] * 100).toFixed(0)}%</span>
          </div>
        ))}
        <span style={{ fontFamily: mono, fontSize: 12, color: T.soft, lineHeight: 1.6, marginTop: 6 }}>no training on these classes: the text encoder writes the classifier weights W [K, E]</span>
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
  const bpeLen = tk ? tk.tokenize(bpeText).len : null;
  const m = CLIP_MODELS[mk], g = RES / m.P;
  const toy = useMemo(() => clipToy(n, train), [n, train]);

  const steps = [
    { id: "batch", block: "batch", caption: "a batch of n matching (image, caption) pairs", vars: () => [["n", `${n}`]] },
    { id: "img_patch", block: "img_patch", caption: `images → ${g}×${g} patches + CLS → [n, ${g * g + 1}, ${m.Di}]`, vars: () => [["P", `${m.P}`], ["tokens", `${g * g + 1}`]] },
    { id: "img_tf", block: "img_tf", caption: "image transformer, then keep the CLS token", vars: () => [["Li", `${m.Li}`], ["heads", `${m.hi}`]] },
    { id: "tok_split", block: "tok_split", caption: "BPE starts from characters: split words, mark word ends", vars: () => [["words", tk ? `${tk.encodeDetailed(bpeText).length}` : "…"]] },
    { id: "tok_merge", block: "tok_merge", caption: "replay CLIP's learned merges until none apply", vars: () => [["merges", tk ? `${tk.nMerges.toLocaleString()}` : "…"]] },
    { id: "tok_ids", block: "tok_ids", caption: "look up ids, add <sot>/<eot>, pad to 77", vars: () => [["tokens", bpeLen ? `${bpeLen} / ${CTX}` : "…", bpeLen > CTX ? "bad" : null], ["vocab", fmtInt(VOCAB)]] },
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
    n: ["batch of pairs", n], C: ["channels", 3], H: ["image side", RES], W: ["image side", RES], P: ["patch", m.P], g: ["patches per side 224/P", g], N: ["patches g²", g * g], T: ["image tokens g²+1", g * g + 1],
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
    zeroshot: [S("W (prompt embeds)", ["K", "E"]), S("image embed", ["E"]), S("probs", ["K"], null, "out")],
  })[st.id];

  const scene = (st) => {
    const body = {
      batch: <Batch n={n} />,
      img_patch: <ImgPatch n={n} m={m} g={g} />,
      img_tf: <ImgTf n={n} m={m} g={g} toy={toy} />,
      tok_split: <BpeSplit tk={tk} text={bpeText} />,
      tok_merge: <BpeMerge tk={tk} text={bpeText} wi={bpeWord} setWi={setBpeWord} />,
      tok_ids: <BpeIds tk={tk} text={bpeText} captions={captions} n={n} />,
      txt_emb: <TxtEmb n={n} m={m} />,
      txt_tf: <TxtTf n={n} m={m} tk={tk} />,
      project: <Project n={n} m={m} />,
      normalize: <Normalize n={n} m={m} toy={toy} />,
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
      img_patch: <>
        <MMNote>cut each image into {g}×{g} patches of {m.P}×{m.P}×3: [{n}, 3, {RES}, {RES}] → [{n}, {g * g}, {pp}]</MMNote>
        <MM a={["patches", [n, g * g, pp]]} b={["W_conv", [pp, Di]]} c={["tokens", [n, g * g, Di]]} note={`= conv1 (kernel = stride = ${m.P}, bias=False) · concat CLS → [${n}, ${Ti}, ${Di}] · + pos_emb [${Ti}, ${Di}] · ln_pre`} />
      </>,
      img_tf: <>
        <BlockMMs B={n} T={Ti} D={Di} h={hi} d={Di / hi} M={4 * Di} L={Li} />
        <MMNote>x[:, 0] (CLS) → ln_post → I_f [{n}, {Di}]</MMNote>
      </>,
      tok_split: <MMNote>string processing only, no tensors yet</MMNote>,
      tok_merge: <MMNote>string processing only, no tensors yet</MMNote>,
      tok_ids: <MMNote>dictionary lookup: pieces → ids · T [{n}, {CTX}] (int64)</MMNote>,
      txt_emb: <>
        <MM a={["one_hot(T)", [n, CTX, VOCAB]]} b={["token_embedding", [VOCAB, Dt]]} c={["x", [n, CTX, Dt]]} note="what the lookup computes; in code it is an index (embedding[T]), no real multiply" />
        <MMNote>x + positional_embedding [{CTX}, {Dt}] (broadcast over n) → [{n}, {CTX}, {Dt}]</MMNote>
      </>,
      txt_tf: <>
        <BlockMMs B={n} T={CTX} D={Dt} h={ht} d={Dt / ht} M={4 * Dt} L={Lt} causal />
        <MMNote>ln_final → gather x[arange({n}), eot_idx{eot ? ` = [${eot.join(", ")}]` : ""}] → T_f [{n}, {Dt}]</MMNote>
      </>,
      project: <>
        <MM a={["I_f", [n, Di]]} b={["W_i", [Di, E]]} c={["I_e", [n, E]]} note="no bias, no MLP head" />
        <MM a={["T_f", [n, Dt]]} b={["W_t", [Dt, E]]} c={["T_e", [n, E]]} />
      </>,
      normalize: <MMNote>row-wise: I_e [{n}, {E}] ÷ ‖I_e‖ [{n}, 1] (broadcast) → [{n}, {E}] · same for T_e</MMNote>,
      logits: <>
        <MM a={["I_e", [n, E]]} b={["T_eᵀ", [E, n]]} c={["cos", [n, n]]} note={`row i · column j = cos(image i, text j) · ${n * n} pairs, ${n} positives`} />
        <MMNote>logits = s · cos with s = min(e^t, 100) = {Math.min(toy.scale, 100).toFixed(1)}  (scalar × [{n}, {n}])</MMNote>
      </>,
      loss_i: <MMNote>softmax over dim 1 (each row) of logits [{n}, {n}] → take diagonal → −log → mean → scalar</MMNote>,
      loss_t: <MMNote>logits_per_text = logits_per_imageᵀ [{n}, {n}] → softmax each row (= each column of the original) → −log diag → mean</MMNote>,
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
      subtitle="Learning Transferable Visual Models From Natural Language Supervision · Radford et al. 2021 · embeddings are a toy model; shapes are the real configs"
      blocks={FLOW}
      steps={steps}
      scene={scene}
      tags={tags}
      controls={
        <Row gap={18}>
          <Seg label="MODEL" value={mk} options={Object.keys(CLIP_MODELS)} onChange={setMk} />
          <Seg label="BATCH n" value={n} options={[4, 6, 8]} onChange={setN} />
          <label style={{ display: "grid", gap: 4, minWidth: 160 }}>
            <span style={{ fontFamily: mono, fontSize: 10, color: T.dim, letterSpacing: "0.08em", display: "flex", justifyContent: "space-between" }}>TRAINING <b style={{ color: T.wire }}>{Math.round(train * 100)}%</b></span>
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
