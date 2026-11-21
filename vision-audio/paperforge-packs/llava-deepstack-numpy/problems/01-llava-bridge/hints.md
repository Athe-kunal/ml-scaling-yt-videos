Run the tower and keep a list: `hs = [ln_pre_output]`, then append after every block. LLaVA's feature is `hs[-2]`, which is the output of the second-to-last block. Slice `[:, 1:]` to drop CLS. CLIP's `ln_post` and `W_proj` are never used.
---
Splice row by row: find `p = flatnonzero(ids[b] == image_token_id)[0]`, then `concat(embed[ids[b,:p]], image_tokens[b], embed[ids[b,p+1:]])`. Build the labels the same way, with `np.full(N, -100)` in the middle, so every text label stays with its token.
---
Positions are `arange(L)` over the **spliced** length, not the original ids. For the loss, use `logits[:, :-1]` against `labels[:, 1:]`, a stable log-softmax, and a mean over the targets that are not −100.
