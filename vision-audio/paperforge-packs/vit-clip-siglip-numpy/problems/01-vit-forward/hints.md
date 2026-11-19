Patchify in two reshapes: `(B, H, W, C) → (B, hp, P, wp, P, C)`, then bring the two grid axes forward with `transpose(0, 1, 3, 2, 4, 5)`, then reshape to `(B, hp·wp, P·P·C)`. Reshaping directly to `(B, N, P·P·C)` produces horizontal strips, not square patches.
---
Heads: `t.reshape(B, N, H, dh).transpose(0, 2, 1, 3)` gives `(B, H, N, dh)`. The scores are `Q @ K.transpose(0, 1, 3, 2) / sqrt(dh)`, with the softmax on the last axis (keys). Undo the split with `transpose(0, 2, 1, 3).reshape(B, N, D)`.
---
CLS goes at index 0 *before* adding `pos_embed`, which has `N+1` rows. The head reads `x[:, 0]`, applies LN, then multiplies by `W_head`.
