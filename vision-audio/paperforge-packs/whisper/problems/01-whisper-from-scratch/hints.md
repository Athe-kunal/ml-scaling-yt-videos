**Front end.** `torch.stft(center=True)` reflect-pads 200 samples on each side, so a 480000-sample input gives `480000 // 160 + 1 = 3001` frames. Whisper keeps `stft[..., :-1]`. Then square the magnitude: it's a *power* spectrum, even though the paper says "log-magnitude". The clamp uses `log_spec.max()`, a single scalar over the whole tensor.
---
**Sinusoids.** `inv = exp(-log(10000) / (C//2 - 1) * arange(C//2))`, so the first channel has timescale 1 and the last has exactly 10000. `scaled = arange(L)[:, None] * inv[None, :]`, then `cat([sin, cos], dim=1)`.
**Key bias.** A key bias adds `q · b_k` to every score in a query's row. That is a constant per row, and softmax cancels it. So Whisper drops it and saves the parameters, but the raw `qk` *would* change.
---
**Attention.** `q.view(B, T, H, -1).permute(0, 2, 1, 3)` (same for k and v). `qk = (q * s) @ (k * s).transpose(-1, -2)` with `s = (D // H) ** -0.25`. Then `qk = qk + mask[:Tq, :Tq]`, `w = qk.softmax(-1)`, `out = (w @ v).permute(0, 2, 1, 3).flatten(2)`.
**Mask.** `torch.empty(n, n).fill_(-np.inf).triu_(1)` and `self.register_buffer("mask", mask, persistent=False)`. **Logits:** `x @ self.token_embedding.weight.T`.
---
**Greedy.** `logits = model.decoder(tokens, feats)[:, -1]`. Compute `finished = tokens[:, -1] == eot` *before* appending. Then `sum_logprobs += logprobs[arange(B), nxt] * (~finished)` and `nxt[finished] = eot`. The first EOT *is* scored, because the row wasn't finished yet when it was chosen.
