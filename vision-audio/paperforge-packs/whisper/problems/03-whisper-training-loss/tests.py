import math

import torch

from solution import ModelDimensions, Whisper, make_training_example, nll_loss, whisper_loss

# toy vocabulary: 0-7 text, then the special tokens
SOT, EN, TRANSCRIBE, EOT, PREV = 8, 9, 10, 11, 12
SOT_SEQ = [SOT, EN, TRANSCRIBE]
DIMS = ModelDimensions(n_mels=6, n_audio_ctx=5, n_audio_state=8, n_audio_head=2, n_audio_layer=2,
                       n_vocab=13, n_text_ctx=12, n_text_state=8, n_text_head=2, n_text_layer=2)


def _model(seed=0):
    torch.manual_seed(seed)
    m = Whisper(DIMS).double()
    g = torch.Generator().manual_seed(seed + 1)
    with torch.no_grad():
        for name, p in m.named_parameters():
            p.copy_(torch.randn(p.shape, generator=g, dtype=p.dtype) * (0.3 if p.ndim > 1 else 0.5)
                    + (1.0 if name.endswith("ln.weight") or "_ln.weight" in name or "ln_post.weight" in name else 0.0))
    return m


def _mel(B=1, seed=7):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(B, DIMS.n_mels, 2 * DIMS.n_audio_ctx, generator=g, dtype=torch.float64)


def _ref_nll(logits, targets, mask):
    """Loop reference: -log softmax at the target, summed over counted tokens, / number of counted tokens."""
    tot, n = 0.0, 0
    B, T, V = logits.shape
    for b in range(B):
        for t in range(T):
            if mask[b, t]:
                row = logits[b, t].double()
                m = row.max().item()
                lse = m + math.log(sum(math.exp(v - m) for v in row.tolist()))
                tot += lse - row[targets[b, t]].item()
                n += 1
    return tot / n


# ============================================================ the multitask training sequence
def test_example_layout_with_previous_text():
    inp, tgt, mask = make_training_example([3, 4], SOT_SEQ, [5, 6, 7], EOT, PREV)
    seq = [PREV, 3, 4, SOT, EN, TRANSCRIBE, 5, 6, 7, EOT]
    assert inp.tolist() == seq[:-1], f"input_ids should be the sequence without its last token, got {inp.tolist()}"
    assert tgt.tolist() == seq[1:], f"target_ids should be the sequence shifted left by one, got {tgt.tolist()}"
    want = [False, False] + [True] * 7
    assert mask.tolist() == want, f"only the two previous-text targets are masked; got {mask.tolist()}"
    assert mask.dtype == torch.bool and inp.dtype == torch.long and tgt.dtype == torch.long


def test_no_startofprev_without_previous_text():
    inp, tgt, mask = make_training_example([], SOT_SEQ, [5, 6], EOT, PREV)
    assert inp.tolist() == [SOT, EN, TRANSCRIBE, 5, 6], f"no context → the sequence starts at <|startoftranscript|>, got {inp.tolist()}"
    assert mask.all(), "with no previous text every target is trained"


def test_language_token_is_a_trained_target():
    inp, tgt, mask = make_training_example([3, 4, 5], SOT_SEQ, [6], EOT, PREV)
    i = inp.tolist().index(SOT)
    assert tgt[i].item() == EN, "the position holding <|startoftranscript|> must predict the language token"
    assert mask[i], ("the language token is trained (that's how Whisper does language ID); "
                     "only the previous-context TEXT is masked")
    assert mask[i - 1], "the token after the context (<|startoftranscript|>) is not previous text either"


def test_shift_by_one_and_eot_is_only_a_target():
    inp, tgt, mask = make_training_example([1], SOT_SEQ, [5, 6, 7], EOT, PREV)
    assert torch.equal(inp[1:], tgt[:-1]), "target_ids[i] must equal input_ids[i+1]"
    assert tgt[-1].item() == EOT and mask[-1], "the last target is <|endoftext|> and it is trained"
    assert EOT not in inp.tolist(), "<|endoftext|> is predicted, never fed in"


# ============================================================ the masked NLL
def test_nll_matches_loop_reference():
    g = torch.Generator().manual_seed(1)
    logits = torch.randn(3, 6, 9, generator=g, dtype=torch.float64) * 2
    targets = torch.randint(0, 9, (3, 6), generator=g)
    mask = torch.rand(3, 6, generator=g) > 0.35
    got = nll_loss(logits, targets, mask)
    ref = _ref_nll(logits, targets, mask)
    assert got.dim() == 0, "nll_loss must return a scalar"
    assert abs(got.item() - ref) < 1e-9, f"got {got.item():.6f}, reference {ref:.6f}"


def test_nll_averages_over_tokens_not_sequences():
    g = torch.Generator().manual_seed(2)
    logits = torch.randn(2, 5, 7, generator=g, dtype=torch.float64) * 3
    targets = torch.randint(0, 7, (2, 5), generator=g)
    mask = torch.tensor([[1, 0, 0, 0, 0], [1, 1, 1, 1, 1]], dtype=torch.bool)  # 1 token vs 5 tokens
    got = nll_loss(logits, targets, mask).item()
    ref = _ref_nll(logits, targets, mask)
    per_seq = 0.5 * (_ref_nll(logits[:1], targets[:1], mask[:1]) + _ref_nll(logits[1:], targets[1:], mask[1:]))
    assert abs(got - ref) < 1e-9, (f"got {got:.5f}; token-level mean is {ref:.5f} "
                                   f"(mean of per-sequence means would be {per_seq:.5f})")


def test_masked_targets_do_not_matter():
    g = torch.Generator().manual_seed(3)
    logits = torch.randn(2, 6, 8, generator=g, dtype=torch.float64)
    targets = torch.randint(0, 8, (2, 6), generator=g)
    mask = torch.tensor([[0, 0, 1, 1, 1, 1], [0, 1, 1, 1, 0, 0]], dtype=torch.bool)
    a = nll_loss(logits, targets, mask)
    t2, l2 = targets.clone(), logits.clone()
    t2[~mask] = (t2[~mask] + 3) % 8
    l2[~mask] = l2[~mask] * 50 - 7
    b = nll_loss(l2, t2, mask)
    assert torch.allclose(a, b), "changing masked positions (their targets or logits) must not change the loss"


def test_nll_is_stable_for_huge_logits():
    logits = torch.tensor([[[1e4, 0.0, -1e4], [0.0, 1e4, 2e4]]], dtype=torch.float64)
    targets = torch.tensor([[1, 0]])
    mask = torch.ones(1, 2, dtype=torch.bool)
    got = nll_loss(logits, targets, mask)
    assert torch.isfinite(got), "use logsumexp (or subtract the max): log(softmax) underflows to log(0) = -inf"
    assert abs(got.item() - 1.5e4) < 1e-6, f"expected (1e4 + 2e4)/2 = 15000, got {got.item()}"


def test_uniform_logits_give_log_vocab():
    V = 51865
    got = nll_loss(torch.zeros(1, 3, V, dtype=torch.float64), torch.tensor([[0, 5, 9]]), torch.ones(1, 3, dtype=torch.bool))
    assert abs(got.item() - math.log(V)) < 1e-9, f"a uniform guess costs log V = {math.log(V):.4f} nats, got {got.item():.4f}"


def test_gradient_is_softmax_minus_onehot_over_count():
    g = torch.Generator().manual_seed(4)
    logits = torch.randn(2, 4, 6, generator=g, dtype=torch.float64, requires_grad=True)
    targets = torch.randint(0, 6, (2, 4), generator=g)
    mask = torch.tensor([[1, 1, 0, 1], [0, 1, 1, 0]], dtype=torch.bool)
    nll_loss(logits, targets, mask).backward()
    want = (logits.detach().softmax(-1) - torch.nn.functional.one_hot(targets, 6)) * mask[..., None] / mask.sum()
    assert torch.allclose(logits.grad, want, atol=1e-12), "d loss / d logits = (softmax − one_hot) · mask / n_counted"


# ============================================================ teacher forcing through Whisper
def _batch():
    a = make_training_example([3, 4], SOT_SEQ, [5, 6, 7], EOT, PREV)   # length 9
    return [x.unsqueeze(0) for x in a]


def test_teacher_forcing_equals_stepwise_scoring():
    model = _model().eval()
    mel = _mel()
    inp, tgt, mask = _batch()
    got = whisper_loss(model, mel, inp, tgt, mask)
    tot, n = 0.0, 0
    with torch.no_grad():
        for i in range(inp.shape[1]):                       # feed the true prefix, score the next token
            logits = model(mel, inp[:, : i + 1])[:, -1]
            if mask[0, i]:
                tot += (torch.logsumexp(logits[0], -1) - logits[0, tgt[0, i]]).item()
                n += 1
    assert abs(got.item() - tot / n) < 1e-9, (f"one teacher-forced pass {got.item():.6f} must equal step-by-step "
                                              f"scoring {tot / n:.6f}; position i predicts token i+1")


def test_text_loss_reaches_the_encoder():
    model = _model().train()
    inp, tgt, mask = _batch()
    whisper_loss(model, _mel(), inp, tgt, mask).backward()
    for name in ["encoder.conv1.weight", "encoder.blocks.0.attn.query.weight", "decoder.blocks.0.cross_attn.key.weight"]:
        p = dict(model.named_parameters())[name]
        assert p.grad is not None and p.grad.abs().sum() > 0, f"{name} got no gradient: the text loss must train the audio path"


def test_padded_batch_is_token_weighted():
    model = _model().eval()
    mel = _mel(2)
    a = make_training_example([3, 4], SOT_SEQ, [5, 6, 7], EOT, PREV)   # 9 positions, 7 counted
    b = make_training_example([], SOT_SEQ, [5], EOT, PREV)               # 4 positions, 4 counted
    T = len(a[0])
    pad = lambda x, v: torch.cat([x, torch.full((T - len(x),), v, dtype=x.dtype)])
    inp = torch.stack([a[0], pad(b[0], EOT)])
    tgt = torch.stack([a[1], pad(b[1], EOT)])
    msk = torch.stack([a[2], pad(b[2], False)])
    got = whisper_loss(model, mel, inp, tgt, msk).item()
    la = whisper_loss(model, mel[:1], a[0][None], a[1][None], a[2][None]).item()
    lb = whisper_loss(model, mel[1:], b[0][None], b[1][None], b[2][None]).item()
    want = (7 * la + 4 * lb) / 11
    assert abs(got - want) < 1e-9, (f"padded batch {got:.6f} must equal the token-weighted mean {want:.6f} of the "
                                    "unpadded losses; padding must be masked and causal attention keeps it out of the real rows")
