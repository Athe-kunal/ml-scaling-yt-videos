# ViT → CLIP → SigLIP, barebones in NumPy

There are three files in one style, meant to be read side by side and walked through line by line. Every weight is an explicit argument. LayerNorm has no gain or bias, and there are no biases inside the blocks, so what remains is exactly what makes each model different.

| | ViT (01) | CLIP (02) | SigLIP (03) |
|---|---|---|---|
| Patch embed | `@ W_patch + bias` | `@ W_patch` (no bias) | `@ W_patch + bias` |
| CLS token | yes | yes | **no** |
| Before blocks | — | **`ln_pre`** | — |
| Image pooling | LN(CLS) | ln_post(CLS) | LN → **MAP head** (learned probe) |
| Activation | GELU (tanh) | **QuickGELU** | GELU (tanh) |
| Text mask | — | **causal** | **none** |
| Text pooling | — | **[EOS]** (argmax id) | **last position** |
| Output | class logits | joint embedding | joint embedding |
| Loss | — | symmetric **softmax** CE over rows and columns, $e^{t}$ | per-pair **sigmoid**, $e^{t'}$ and **bias $b$**, ÷N |

## Problems

1. **ViT forward** (medium): patchify, CLS, positions, pre-LN MHA blocks, the head. This is the shared base.
   · *Kills: strip "patches" from a direct reshape, a broken head split/merge, a softmax over the wrong axis, post-LN blocks, mean pooling.*
2. **CLIP** (hard): the image tower (ViT plus `ln_pre`/`ln_post` and a projection), the causal text tower pooled at [EOS], and the symmetric InfoNCE loss.
   · *Kills: no `ln_pre`, last-token or mean text pooling, no causal mask, one CE direction or the same axis twice, raw dot products, an unstable log-softmax.*
3. **SigLIP** (hard): the MAP head, the no-CLS image tower, the bidirectional last-token text tower, and the sigmoid loss of Alg. 1.
   · *Kills: mean pooling or all-tokens-as-queries in MAP, CLIP habits (CLS pooling, causal mask, [EOS] pooling), a softmax loss, dividing by N², 0/1 labels, a naive `log_sigmoid`, adding the bias before the temperature.*

Sources: SigLIP is `paper.pdf`; ViT and CLIP are in `refs/`. The SigLIP tower details (MAP head, last-token text, no mask) come from the released big_vision models, not the paper.
