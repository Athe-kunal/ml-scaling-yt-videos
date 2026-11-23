Write the full sequence first, then slice: inputs are `seq[:-1]`, targets are `seq[1:]`. Target `j` is `seq[j+1]`. Where does the previous text sit in `seq`, so which target indices hold it?
---
With a context, `seq = [PREV, p0 … p_{n-1}, SOT, LANG, TASK, …]`, so the previous text is at `seq[1 : 1+n]`, which is targets `0 … n-1`. Only those are masked. `SOT` (target `n`) and `LANG` (target `n+1`) are trained.
---
`−log softmax(z)[y] = logsumexp(z) − z[y]`. Use `torch.logsumexp(logits, -1)` and `logits.gather(-1, targets[..., None])`. Multiply by the mask, sum over everything, and divide by `mask.sum()`, not by `B` and not per row.
---
`whisper_loss` is `nll_loss(model(mel, input_ids), target_ids, loss_mask)`. Don't wrap the encoder in `no_grad` or `.detach()` it: the text loss is the encoder's only training signal.
