Image tower = problem 01 with three edits: drop `patch_bias`, add `x = layer_norm(x)` right after `+ pos_embed`, and replace the classifier with `layer_norm(x[:, 0]) @ W_proj`.
---
Causal mask: `np.triu(np.full((L, L), -np.inf), k=1)`, added to the scores *before* the softmax. Pooling: `x[np.arange(B), tokens.argmax(-1)]`. With a causal mask, [EOS] is the first position that has seen the whole caption, and nothing after it can leak in.
---
For row $i$ of `logits`, `log_softmax(logits, axis=1)[i, i]` is "image $i$ picks text $i$". `log_softmax(logits, axis=0)[j, j]` is "text $j$ picks image $j$" (the softmax runs down a column). Take the mean of each, then average the two.
---
`logit_scale = np.log(1 / TEMPERATURE)` ≈ 2.659 is the stored, learned parameter. The multiplier is `min(np.exp(logit_scale), MAX_SCALE)` = 1/0.07 ≈ 14.29. Why store the log? It keeps the scale positive, and gradient steps on $t$ change the scale multiplicatively.
