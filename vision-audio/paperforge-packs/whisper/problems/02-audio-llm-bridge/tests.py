import math

import pytest
import torch
import torch.nn.functional as F

from solution import (
    AuTEncoder,
    Qwen2AudioEncoder,
    Qwen2AudioProjector,
    TinyLM,
    aut_output_lengths,
    expand_audio_placeholders,
    greedy_generate,
    merge_audio_embeddings,
    qwen2_audio_generate,
    qwen2_audio_output_lengths,
    qwen3_omni_generate,
    sinusoids,
)

torch.set_default_dtype(torch.float64)
AUDIO, EOS, VOCAB, H_LM = 30, 31, 32, 24


def _randomize(m, seed, std=0.3):
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for name, p in m.named_parameters():
            if "embed_positions" in name:
                continue
            base = 1.0 if (p.ndim == 1 and ("norm" in name or "ln" in name)) else 0.0
            p.copy_(base + torch.randn(p.shape, generator=g) * (std if p.ndim > 1 else 0.2))
    return m.eval()


def _q2(seed=0):
    torch.manual_seed(seed)
    return _randomize(Qwen2AudioEncoder(n_mels=8, d=16, n_head=2, n_layer=2, max_source_positions=12), seed)


def _aut(seed=0, n_window_infer=200):
    torch.manual_seed(seed)
    return _randomize(AuTEncoder(n_mels=16, d=16, n_head=2, n_layer=2, output_dim=H_LM, downsample_hidden_size=4,
                                 n_window=50, n_window_infer=n_window_infer, max_source_positions=20), seed)


def _lm(seed=0):
    torch.manual_seed(seed)
    return _randomize(TinyLM(VOCAB, H_LM, 2, 2), seed, std=0.5)


# ============================================================ independent references
def _ln(x, m):
    mu, var = x.mean(-1, keepdim=True), x.var(-1, keepdim=True, unbiased=False)
    return (x - mu) / torch.sqrt(var + m.eps) * m.weight + m.bias


def _ref_attn(x, a, n_keys):
    """x: (T, D). Every query attends to keys 0..n_keys-1 only. Per-head loop."""
    T, D = x.shape
    H = a.n_head
    dh = D // H
    q, k, v = x @ a.q_proj.weight.T + a.q_proj.bias, x @ a.k_proj.weight.T, x @ a.v_proj.weight.T + a.v_proj.bias
    if a.k_proj.bias is not None:
        k = k + a.k_proj.bias
    out = torch.zeros(T, D)
    for h in range(H):
        s = slice(h * dh, (h + 1) * dh)
        sc = q[:, s] @ k[:n_keys, s].T / math.sqrt(dh)
        out[:, s] = sc.softmax(-1) @ v[:n_keys, s]
    return out @ a.out_proj.weight.T + a.out_proj.bias


def _ref_layer(x, L, n_keys):
    x = x + _ref_attn(_ln(x, L.self_attn_layer_norm), L.self_attn, n_keys)
    h = _ln(x, L.final_layer_norm)
    return x + F.gelu(h @ L.fc1.weight.T + L.fc1.bias) @ L.fc2.weight.T + L.fc2.bias


def _conv_len(n, times):
    x = torch.zeros(1, 1, n)
    for _ in range(times):
        x = F.conv1d(x, torch.zeros(1, 1, 3), stride=2, padding=1)
    return x.shape[-1]


def _ref_q2_encoder(enc, mel, n_frames):
    """mel: (n_mels, 2*P). Explicit: convs, positions, attention over REAL keys only, pair-average, LN."""
    x = F.gelu(F.conv1d(mel[None], enc.conv1.weight, enc.conv1.bias, padding=1))
    x = F.gelu(F.conv1d(x, enc.conv2.weight, enc.conv2.bias, stride=2, padding=1))[0].T
    x = x + sinusoids(x.shape[0], x.shape[1])
    n_keys = _conv_len(n_frames, 1)
    for L in enc.layers:
        x = _ref_layer(x, L, n_keys)
    T = x.shape[0] // 2
    x = torch.stack([(x[2 * i] + x[2 * i + 1]) / 2 for i in range(T)])
    return _ln(x, enc.layer_norm)


def _ref_aut_encoder(enc, mels):
    out = []
    for mel in mels:
        toks = []
        for s in range(0, mel.shape[1], 100):
            piece = mel[:, s:s + 100]
            n_real = piece.shape[1]
            piece = F.pad(piece, (0, 100 - n_real))[None, None]          # zero-pad the chunk to 100 frames
            y = F.gelu(F.conv2d(piece, enc.conv2d1.weight, enc.conv2d1.bias, stride=2, padding=1))
            y = F.gelu(F.conv2d(y, enc.conv2d2.weight, enc.conv2d2.bias, stride=2, padding=1))
            y = F.gelu(F.conv2d(y, enc.conv2d3.weight, enc.conv2d3.bias, stride=2, padding=1))[0]  # (c, f, t)
            c, fdim, t = y.shape
            flat = torch.stack([torch.cat([y[ci, fi, ti:ti + 1] for ci in range(c) for fi in range(fdim)])
                                for ti in range(t)])                       # (t, c*f) channel-major
            z = flat @ enc.conv_out.weight.T + sinusoids(t, enc.conv_out.weight.shape[0])
            toks.append(z[: _conv_len(n_real, 3)])
        x = torch.cat(toks)
        W = 13 * (enc.n_window_infer // 100)
        windows = []
        for s in range(0, x.shape[0], W):
            w = x[s:s + W]
            for L in enc.layers:
                w = _ref_layer(w, L, w.shape[0])
            windows.append(w)
        x = _ln(torch.cat(windows), enc.ln_post)
        x = F.gelu(x @ enc.proj1.weight.T + enc.proj1.bias) @ enc.proj2.weight.T + enc.proj2.bias
        out.append(x)
    return torch.cat(out)


# ============================================================ Qwen2-Audio
def test_qwen2_output_lengths_match_conv_arithmetic():
    lens = torch.tensor([3000, 2999, 1000, 37, 8, 5, 4, 3])
    a, b = qwen2_audio_output_lengths(lens)
    for n, x, y in zip(lens.tolist(), a.tolist(), b.tolist()):
        exp_a = _conv_len(n, 1)
        exp_b = F.avg_pool1d(torch.zeros(1, 1, exp_a), 2, 2).shape[-1]
        assert (x, y) == (exp_a, exp_b), (
            f"mel {n}: got (after_conv, after_pool)=({x},{y}), expected ({exp_a},{exp_b}). conv2 is k3/s2/p1, the pool is "
            "k2/s2 with no padding (30 s: 3000 -> 1500 -> 750 tokens, 40 ms each)")


@pytest.mark.parametrize("lens,seed", [([24, 24], 0), ([17, 9], 1), ([5], 2)])
def test_qwen2_encoder_matches_reference(lens, seed):
    enc = _q2(seed)
    g = torch.Generator().manual_seed(seed + 50)
    mel = torch.randn(len(lens), 8, 24, generator=g)
    with torch.no_grad():
        out = enc(mel, torch.tensor(lens))
    assert out.shape == (len(lens), 6, 16), (
        f"shape {tuple(out.shape)}: 24 frames -> conv2 -> 12 -> pool -> 6 (pool over TIME, not over channels)")
    for b, n in enumerate(lens):
        ref = _ref_q2_encoder(enc, mel[b], n)
        err = (out[b] - ref).abs().max().item()
        assert err < 1e-9, (
            f"audio {b} (len {n}) max err {err:.2e}. Order: GELU(conv1) -> GELU(conv2) -> + positions -> layers with "
            "PADDED KEYS masked -> AvgPool1d(2) over time -> LayerNorm (after the pool)")


def test_qwen2_padding_frames_cannot_leak_into_real_tokens():
    enc = _q2(3)
    g = torch.Generator().manual_seed(7)
    n = 9
    mel = torch.randn(1, 8, 24, generator=g)
    mel2 = mel.clone()
    mel2[..., n + 3:] += 5.0 * torch.randn(8, 24 - n - 3, generator=g)   # far beyond the conv receptive field
    with torch.no_grad():
        a, b = enc(mel, torch.tensor([n])), enc(mel2, torch.tensor([n]))
    _, k = qwen2_audio_output_lengths(torch.tensor([n]))
    k = int(k)
    assert torch.allclose(a[0, :k], b[0, :k], atol=1e-12), (
        "changing PADDING frames changed the real audio tokens: the encoder always sees 30 s of input, so the padded "
        "positions must be masked out as attention keys")


# ============================================================ Qwen3-Omni AuT
def test_aut_output_lengths_match_chunked_conv_arithmetic():
    lens = torch.tensor([100, 200, 250, 130, 99, 1, 37, 801, 3000])
    got = aut_output_lengths(lens, 50).tolist()
    for n, x in zip(lens.tolist(), got):
        exp = sum(_conv_len(min(100, n - s), 3) for s in range(0, n, 100))
        assert x == exp, (
            f"mel {n}: got {x}, expected {exp}. AuT cuts audio into 100-frame (1 s) chunks; each FULL chunk gives "
            "100 -> 50 -> 25 -> 13 tokens and the tail chunk goes through the same three stride-2 convs (12.5 Hz)")


@pytest.mark.parametrize("lens,seed,infer", [([250, 130], 0, 200), ([300], 1, 100), ([100, 40, 260], 2, 200)])
def test_aut_encoder_matches_reference(lens, seed, infer):
    enc = _aut(seed, infer)
    g = torch.Generator().manual_seed(seed + 80)
    mels = [torch.randn(16, n, generator=g) for n in lens]
    with torch.no_grad():
        out = enc(mels)
    n_tok = int(aut_output_lengths(torch.tensor(lens)).sum())
    assert out.shape == (n_tok, H_LM), (
        f"shape {tuple(out.shape)}, expected ({n_tok}, {H_LM}): PACKED tokens of all audios, already projected to the "
        "LLM width; padded positions of short chunks removed")
    ref = _ref_aut_encoder(enc, mels)
    err = (out - ref).abs().max().item()
    assert err < 1e-9, (
        f"max err {err:.2e}. Per 100-frame chunk: 3x GELU(Conv2d s2) -> (t, c*f) channel-major -> conv_out -> + "
        "sinusoids restarting at 0 IN EACH CHUNK -> keep valid; then windowed attention, ln_post, proj1, GELU, proj2")


def test_aut_attention_is_windowed_and_never_crosses_audios():
    enc = _aut(4, n_window_infer=200)              # window = 2 chunks = 26 tokens
    g = torch.Generator().manual_seed(5)
    a, b = torch.randn(16, 400, generator=g), torch.randn(16, 150, generator=g)
    with torch.no_grad():
        base = enc([a, b])
        a2 = a.clone()
        a2[:, 200:] += 3.0                         # change only window 2 of audio a
        alt = enc([a2, b])
    assert torch.allclose(base[:26], alt[:26], atol=1e-12), (
        "tokens in the FIRST 8 s window changed when only the second window's audio changed: AuT attention is "
        "block-diagonal over windows of 13 * (n_window_infer / 100) tokens")
    assert not torch.allclose(base[26:52], alt[26:52]), "the changed window itself must change"
    assert torch.allclose(base[52:], alt[52:], atol=1e-12), (
        "the SECOND audio changed when only the first audio changed: windows never span two audios")


# ============================================================ the bridge
def test_expand_audio_placeholders():
    ids = [1, 2, 29, AUDIO, 28, 3, 29, AUDIO, 28]
    out = expand_audio_placeholders(ids, AUDIO, [3, 2])
    assert out == [1, 2, 29, AUDIO, AUDIO, AUDIO, 28, 3, 29, AUDIO, AUDIO, 28], (
        f"got {out}: the k-th <|AUDIO|> becomes n_tokens[k] copies, everything else unchanged")
    with pytest.raises(ValueError):
        expand_audio_placeholders([1, AUDIO], AUDIO, [3, 2])


def test_merge_scatters_audio_in_reading_order_and_checks_count():
    g = torch.Generator().manual_seed(9)
    ids = torch.tensor([[1, AUDIO, AUDIO, 2, AUDIO], [AUDIO, 3, 4, AUDIO, 5]])
    text = torch.randn(2, 5, 6, generator=g)
    feats = torch.arange(24, dtype=torch.float64).view(4, 6)
    keep = text.clone()
    pos = [(0, 1), (0, 2), (0, 4), (1, 0), (1, 3)]
    with pytest.raises(ValueError):
        merge_audio_embeddings(text, ids, AUDIO, feats)       # 5 placeholders, 4 features
    feats5 = torch.arange(30, dtype=torch.float64).view(5, 6)
    out = merge_audio_embeddings(text, ids, AUDIO, feats5)
    assert torch.equal(text, keep), "do not modify inputs_embeds in place (it may be the embedding output)"
    for i, (b, t) in enumerate(pos):
        assert torch.equal(out[b, t], feats5[i]), (
            f"placeholder #{i} at (batch {b}, pos {t}) should hold feature row {i}: fill in row-major reading order")
    m = ids != AUDIO
    assert torch.equal(out[m], text[m]), "text positions must keep their token embeddings"


class _SpyLM(torch.nn.Module):
    def __init__(self, lm):
        super().__init__()
        self.lm, self.embed_tokens, self.seen = lm, lm.embed_tokens, []

    def forward(self, x):
        self.seen.append(x.detach().clone())
        return self.lm(x)


def test_greedy_generate_keeps_the_audio_prefix():
    lm = _lm(1)
    g = torch.Generator().manual_seed(11)
    prefix = torch.randn(1, 7, H_LM, generator=g) * 2
    spy = _SpyLM(lm)
    out = greedy_generate(spy, prefix, 5, EOS)
    assert 1 <= len(out) <= 5
    for step, x in enumerate(spy.seen):
        assert x.shape[1] == 7 + step, f"step {step}: LM input length {x.shape[1]}, expected {7 + step}"
        assert torch.equal(x[:, :7], prefix), (
            f"step {step}: the first 7 input embeddings are no longer the merged prefix. The audio exists ONLY as "
            "embeddings, so never rebuild inputs from token ids (placeholders would become the <|AUDIO|> embedding)")
        if step:
            exp = lm.embed_tokens(torch.tensor([out[:step]]))
            assert torch.allclose(x[:, 7:], exp), "generated tokens are appended as their token embeddings"
    # reference greedy
    x, ref = prefix, []
    with torch.no_grad():
        for _ in range(5):
            t = int(lm(x)[0, -1].argmax())
            ref.append(t)
            if t == EOS:
                break
            x = torch.cat([x, lm.embed_tokens(torch.tensor([[t]]))], 1)
    assert out == ref, f"tokens {out} != reference {ref} (argmax of the LAST position; stop after emitting EOS)"


# ============================================================ end to end
def test_qwen2_audio_pipeline_end_to_end():
    enc, lm = _q2(5), _lm(6)
    torch.manual_seed(7)
    proj = _randomize(Qwen2AudioProjector(16, H_LM), 7)
    g = torch.Generator().manual_seed(12)
    mel, n = torch.randn(8, 24, generator=g), 15
    prompt = [1, 29, AUDIO, 28, 4, 5]
    spy = _SpyLM(lm)
    out = qwen2_audio_generate(enc, proj, spy, mel, n, prompt, AUDIO, 4, EOS)
    k = _conv_len(n, 1)
    k = (k - 2) // 2 + 1
    with torch.no_grad():
        feats = _ref_q2_encoder(enc, mel, n)[:k] @ proj.linear.weight.T + proj.linear.bias
        ids = [1, 29] + [AUDIO] * k + [28, 4, 5]
        emb = lm.embed_tokens(torch.tensor([ids]))
        emb[0, 2:2 + k] = feats
    first = spy.seen[0]
    assert first.shape == emb.shape, (
        f"LM prefix length {first.shape[1]}, expected {emb.shape[1]}: {k} audio tokens for {n} real frames (drop the "
        "encoder outputs that come from padding)")
    assert torch.allclose(first, emb, atol=1e-9), (
        f"max err {(first - emb).abs().max():.2e}: prefix = text embeddings with the projected, REAL audio tokens "
        "scattered into the <|AUDIO|> slots")
    assert len(out) >= 1


def test_qwen3_omni_pipeline_end_to_end():
    enc, lm = _aut(8), _lm(9)
    g = torch.Generator().manual_seed(13)
    mels = [torch.randn(16, 230, generator=g), torch.randn(16, 60, generator=g)]
    prompt = [1, 29, AUDIO, 28, 29, AUDIO, 28, 6]
    spy = _SpyLM(lm)
    qwen3_omni_generate(enc, spy, mels, prompt, AUDIO, 3, EOS)
    with torch.no_grad():
        feats = _ref_aut_encoder(enc, mels)
    n1 = sum(_conv_len(min(100, 230 - s), 3) for s in range(0, 230, 100))
    n2 = _conv_len(60, 3)
    ids = [1, 29] + [AUDIO] * n1 + [28, 29] + [AUDIO] * n2 + [28, 6]
    emb = lm.embed_tokens(torch.tensor([ids])).detach().clone()
    emb[0, 2:2 + n1] = feats[:n1]
    emb[0, 4 + n1:4 + n1 + n2] = feats[n1:]
    first = spy.seen[0]
    assert first.shape == emb.shape, f"LM prefix length {first.shape[1]}, expected {emb.shape[1]} ({n1} + {n2} audio tokens)"
    assert torch.allclose(first, emb, atol=1e-9), (
        f"max err {(first - emb).abs().max():.2e}: AuT output is already in the Thinker's width (no extra projector); "
        "audio 1's tokens fill the first placeholder run, audio 2's the second")
