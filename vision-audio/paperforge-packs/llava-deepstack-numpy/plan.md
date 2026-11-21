# LLaVA → DeepStack: one walk-through of the vision → language bridge

Both problems use the same toy model and the same conventions as the ViT pack. The CLIP-ViT block and a LLaMA decoder block (RMSNorm, RoPE, SwiGLU, causal) are **given**. You write only the part between them.

```
image ─ViT─▶ hs[-2][:,1:] ─MLP─▶ N tokens ─splice at <image>─▶ LM layers ─▶ logits
                                                                ▲  ▲  ▲
image_hires ─4 tiles─ViT─stitch─MLP─dilate─▶ 4 stacks ── += at vis_pos (DeepStack)
```

| | LLaVA-1.5 (01) | DeepStack (02) |
|---|---|---|
| ViT feature | penultimate hidden state, no `ln_post`, CLS dropped | same, for the global view **and** each high-res tile |
| Connector | `Linear → GELU → Linear` | same MLP for both streams |
| Into the LM | replace `<image>` by N tokens (T → T − 1 + N) | same sequence |
| Extra detail | — | 4 stacks of N tokens, `H[vis] +=` before layers `l_start, l_start+n, …` |
| Context cost | N | **still N** (4× the visual detail at no extra length) |

## Problems

1. **LLaVA bridge** (hard): which ViT layer, the projector, the splice, positions and labels.
   · *Kills: last-layer features, applying `ln_post`, keeping CLS, LLaVA-1's linear connector, the image as a prefix, pre-splice positions, an unshifted loss, supervising image slots.*
2. **DeepStack** (hard): tiles → stitched grid → projector → Sampling2D → Algorithm 1.
   · *Kills: stack = tile (quadrant), tiles stitched column-major, wrong stack order, adding after the layer, summing everything into the input, adding to the first N positions, ignoring `n`, last-layer high-res features.*

Start with 01. 02 asks you to paste your 01 bridge.

Sources: LLaVA-1.5 is `paper.pdf`; LLaVA-1 and DeepStack are in `refs/`. `select_layer = -2` and `select_feature = "patch"` come from the LLaVA repo's config, not the paper text.
