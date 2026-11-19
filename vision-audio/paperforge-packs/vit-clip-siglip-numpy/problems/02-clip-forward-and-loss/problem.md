---
id: 02-clip-forward-and-loss
title: "CLIP in NumPy: both towers and the symmetric contrastive loss"
difficulty: hard
runtime: python
timeout_s: 30
entrypoint: clip_loss
paper_refs: ["CLIP §2.3", "CLIP Fig. 3", "CLIP §2.4"]
---

## Context

CLIP (Radford et al. 2021) learns an image encoder and a text encoder jointly. In a batch of $N$ (image, text) pairs, each image must pick out its own caption among the $N$ captions, and each caption must pick out its own image (Fig. 3):

```
I_e = l2_normalize(I_f @ W_i);  T_e = l2_normalize(T_f @ W_t)
logits = I_e @ T_e.T * exp(t)
loss = (CE(logits, arange(n), axis=0) + CE(logits, arange(n), axis=1)) / 2
```

The two towers (§2.4) each change one thing compared with a plain transformer:

- **Image: CLIP-ViT.** The same ViT as problem 01, except that the patch embedding has **no bias**, there is an extra LayerNorm **`ln_pre`** on `[CLS; patches] + pos` before the first block, and the output is `ln_post(CLS) @ W_proj` (a projection into the joint space, not a classifier). The activation is **QuickGELU**, $x\,\sigma(1.702x)$.
- **Text: a GPT-2-style transformer** with a **causal mask**. *"The activations of the highest layer of the transformer at the [EOS] token are treated as the feature representation of the text which is layer normalized and then linearly projected."* [EOS] has the largest token id, so its position is `tokens.argmax(-1)`. Captions are padded with 0 after [EOS].

## Task

Implement three functions, using the conventions of problem 01: LN without gain or bias (eps 1e-6), no biases inside the blocks, and pre-LN blocks with QuickGELU.

```python
clip_image_encoder(image, patch_size, num_heads, W_patch, cls_token, pos_embed, encoder_weights, W_proj) -> (B, E)
clip_text_encoder(tokens, num_heads, token_embed, pos_embed, encoder_weights, W_proj)                 -> (B, E)
clip_loss(image_emb, text_emb)                                                                         -> (loss, logits)
```

| arg | shape |
|---|---|
| `W_patch` | `(P·P·C, D)` (no bias) |
| `pos_embed` (image) | `(N+1, D)` |
| `tokens` | int `(B, L)`, `[SOS] w … w [EOS] 0 0 …` |
| `token_embed`, `pos_embed` (text) | `(V, D)`, `(L_max, D)`; use the first `L` rows |
| `W_proj` | `(D, E)` |

`clip_loss` returns the scalar loss and `logits_per_image` `(N, N)`, where row $i$ is image $i$ scored against every text.

**The logit scale is computed inside `clip_loss`, not passed in.** CLIP learns the temperature $\tau$ as a log-parameterised scalar (§2.3: *"directly optimized during training as a log-parameterized multiplicative scalar"*). It stores $t = \log(1/\tau)$, initialised from $\tau = 0.07$ (`TEMPERATURE`, §2.5), and multiplies the cosine similarities by $e^{t}$, clipped at `MAX_SCALE = 100`. Work out the value of `logit_scale` and of the multiplier, and use them.

## Constraints

- NumPy only, vectorised (no Python loop over batch rows, tokens or heads).
- Use a numerically stable log-softmax (subtract the max).

## What the tests check

- `test_image_encoder_matches_reference`: two configs, against a loop-based reference.
- `test_text_encoder_matches_reference`: three configs, with captions of different lengths in one batch.
- `test_text_ignores_everything_after_eos`: random junk after [EOS] changes nothing.
- `test_text_depends_on_every_token_before_eos`
- `test_loss_matches_reference`: the logits and loss against an explicit double-loop CE in both directions.
- `test_logit_scale_is_computed_not_passed`: the signature is `(image_emb, text_emb)`, and identical unit vectors give exactly the multiplier.
- `test_loss_is_symmetric_and_scale_invariant`: swapping the two modalities, and rescaling individual embedding rows, leave the loss unchanged.
- `test_loss_single_pair_is_zero`
