---
id: 03-siglip-forward-and-loss
title: "SigLIP in NumPy: MAP pooling, last-token text, and the sigmoid loss"
difficulty: hard
runtime: python
timeout_s: 30
entrypoint: siglip_loss
paper_refs: ["SigLIP §3", "SigLIP Alg. 1", "SigLIP §3.2"]
---

## Context

SigLIP (Zhai et al. 2023) keeps CLIP's two-tower design and replaces the **loss**. CLIP's softmax normalises each row and column over the whole batch. SigLIP instead turns every one of the $N^2$ (image, text) pairs into an **independent binary classification**: is this a true pair or not? (Alg. 1)

```
t = exp(t_prime)
zimg = l2_normalize(img_emb);  ztxt = l2_normalize(txt_emb)
logits = dot(zimg, ztxt.T) * t + b
labels = 2 * eye(n) - ones(n)          # +1 on the diagonal, -1 elsewhere
l = -sum(log_sigmoid(labels * logits)) / n
```

With $N$ positives and $N^2-N$ negatives, the loss would start out dominated by the negatives. The paper adds a **learnable bias $b$** and initialises $t'=\log 10$, $b=-10$ (§3.2), so that at the start every pair is predicted "negative" with high confidence. Because no term normalises over the batch, the loss is a plain sum of per-pair terms. That is what makes its chunked, all-gather-free implementation possible (Fig. 1).

The towers in the **released** SigLIP models (big_vision code; the paper only says "ViT + text transformer") differ from CLIP's in two places:

- **Image: no class token.** After the blocks and a final LayerNorm, a **MAP head** (multihead attention pooling) does the pooling. A single **learned probe** vector is the only query, attending over all patch tokens. A residual MLP follows: `y = MHA(q=probe, kv=x); y = y + MLP(LN(y))`.
- **Text: no causal mask.** The text transformer is bidirectional, and the embedding is read at the **last position**, which has attended to every token.

## Task

The conventions are those of problem 01: LN without gain or bias (eps 1e-6), no biases inside blocks or attention, pre-LN blocks, and tanh-GELU.

```python
map_head(x, num_heads, probe, Wq, Wk, Wv, Wo, W1, W2)                                   -> (B, D)
siglip_image_encoder(image, patch_size, num_heads, W_patch, patch_bias, pos_embed,
                     encoder_weights, map_weights, W_proj)                              -> (B, E)
siglip_text_encoder(tokens, num_heads, token_embed, pos_embed, encoder_weights, W_proj) -> (B, E)
siglip_loss(image_emb, text_emb, t_prime, b)                                            -> (loss, logits)
```

| arg | shape |
|---|---|
| `probe` | `(D,)`. It goes through `Wq` like any query. |
| `pos_embed` (image) | `(N, D)`, with no CLS row |
| `map_weights` | dict with `probe, Wq, Wk, Wv, Wo, W1, W2` |
| image pipeline | `patches @ W_patch + patch_bias + pos_embed → blocks → LN → map_head → @ W_proj` |
| text pipeline | `token_embed[tokens] + pos_embed[:L] → blocks (no mask) → LN → x[:, -1] → @ W_proj` |

## Constraints

- NumPy only, vectorised.
- `log_sigmoid` must be stable for $|x|$ in the thousands, with no `inf` and no `nan`.

## What the tests check

- `test_map_head_matches_reference`: three configs, including a single token.
- `test_map_head_is_permutation_invariant_and_uses_the_probe`
- `test_map_head_is_not_mean_pooling`
- `test_image_encoder_matches_reference`, `test_text_encoder_matches_reference`: against loop-based references.
- `test_text_last_position_sees_the_first_token`
- `test_loss_matches_reference`: against an explicit double loop over all $N^2$ pairs, including the paper's init.
- `test_loss_has_no_global_normalisation`: adding pair $N$ adds exactly the new row and column of binary terms.
- `test_loss_is_symmetric_and_stable`
- `test_init_bias_makes_negatives_cheap`
