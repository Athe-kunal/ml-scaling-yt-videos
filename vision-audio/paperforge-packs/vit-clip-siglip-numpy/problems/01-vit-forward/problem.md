---
id: 01-vit-forward
title: "ViT forward pass in NumPy"
difficulty: medium
runtime: python
timeout_s: 30
entrypoint: vit_forward
paper_refs: ["ViT §3.1", "ViT Eq. 1-4"]
---

## Context

CLIP's image tower is a ViT (CLIP §2.4), and so is SigLIP's. This problem is the shared base: a barebones ViT forward pass in pure NumPy. ViT §3.1, Eq. 1–4:

$$z_0 = [x_{\text{class}};\, x_p^1 E;\, \dots;\, x_p^N E] + E_{pos},\qquad z'_\ell = \text{MSA}(\text{LN}(z_{\ell-1})) + z_{\ell-1},$$
$$z_\ell = \text{MLP}(\text{LN}(z'_\ell)) + z'_\ell,\qquad y = \text{LN}(z_L^0).$$

"Barebones" means that LayerNorm has **no learnable gain or bias**, and the attention and MLP matrices have **no biases**. Every weight is passed in explicitly. Problems 02 and 03 keep the same conventions, so the three files can be read side by side.

## Task

```python
vit_forward(image, patch_size, num_heads, W_patch, patch_bias,
            cls_token, pos_embed, encoder_weights, W_head) -> (B, num_classes)
```

| arg | shape |
|---|---|
| `image` | `(B, H, W, C)` channels-last. If `H` or `W` is not divisible by `P`, crop to the largest multiple. |
| `W_patch`, `patch_bias` | `(P·P·C, D)`, `(D,)` |
| `cls_token` | `(D,)` |
| `pos_embed` | `(N+1, D)`, where index 0 is for CLS and `N = (H//P)·(W//P)` |
| `encoder_weights` | list of dicts with `Wq, Wk, Wv, Wo: (D, D)`, `W1: (D, 4D)`, `W2: (4D, D)` |
| `W_head` | `(D, num_classes)` |

- **Patches** are taken in row-major order over the patch grid. Each patch is flattened in `(row, col, channel)` order.
- **Block** (pre-LN): `y = x + MHA(LN(x)) @ Wo`, then `out = y + GELU(LN(y) @ W1) @ W2`. MHA splits `D` into `num_heads` contiguous slices of size `D/num_heads`, uses scaled dot-product attention with a softmax over keys, and concatenates the heads back.
- **LayerNorm**: `(x - mean) / sqrt(var + 1e-6)` over the last axis, with population variance.
- **GELU**: tanh approximation.
- **Head**: `LN(z_L[:, 0]) @ W_head`.

Suggested helpers (not tested individually): `softmax`, `layer_norm`, `gelu`, `vit_encoder_block`, `classification_head`.

## Constraints

NumPy only, vectorised over the batch and heads. Do not modify the inputs.

## What the tests check

- `test_matches_reference`: the output matches a loop-based reference (per image, per patch, per head), over three configs including a non-square image, one channel, and a crop.
- `test_patch_pixel_order`: the pixel layout inside a patch must match `(row, col, channel)`.
- `test_images_in_a_batch_are_independent`
- `test_does_not_mutate_inputs`
