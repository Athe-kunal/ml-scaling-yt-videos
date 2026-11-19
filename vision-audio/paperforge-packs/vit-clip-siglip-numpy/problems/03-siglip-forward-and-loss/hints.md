Write one `multihead_attention(q_in, kv_in, ...)` that accepts different query and key/value inputs. The encoder block calls it with `(h, h)`. The MAP head calls it with `q_in = probe` broadcast to `(B, 1, D)` and `kv_in = x`.
---
`log_sigmoid(x) = -log(1 + e^{-x}) = -np.logaddexp(0, -x)`. The naive form overflows when $x \ll 0$, which is exactly the case for a very confident wrong prediction.
---
Labels: `2*np.eye(n) - 1`. The loss is `-np.sum(log_sigmoid(labels * logits)) / n`: a sum over all $N^2$ pairs, divided by $N$ (not $N^2$). The bias goes *after* the temperature: `cos * exp(t') + b`.
