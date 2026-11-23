import math
from pathlib import Path

import numpy as np
import pytest
import torch

from solution import (
    ModelDimensions,
    Whisper,
    greedy_decode,
    log_mel_spectrogram,
    pad_or_trim,
    sinusoids,
)

HERE = Path(__file__).parent
MEL80 = torch.from_numpy(np.load(HERE / "files" / "mel_filters.npz")["mel_80"]).double()
N_FFT, HOP = 400, 160

DIMS = ModelDimensions(n_mels=6, n_audio_ctx=5, n_audio_state=8, n_audio_head=2, n_audio_layer=2,
                       n_vocab=13, n_text_ctx=7, n_text_state=8, n_text_head=2, n_text_layer=2)


def _model(dims=DIMS, seed=0):
    torch.manual_seed(seed)
    m = Whisper(dims).double()
    g = torch.Generator().manual_seed(seed + 1)
    with torch.no_grad():
        for name, p in m.named_parameters():
            # random LN gains/biases too, so a missing or misplaced LayerNorm can't hide
            p.copy_(torch.randn(p.shape, generator=g, dtype=p.dtype) * (0.3 if p.ndim > 1 else 0.5)
                    + (1.0 if name.endswith("ln.weight") or "_ln.weight" in name or "ln_post.weight" in name else 0.0))
    return m.eval()


# ============================================================ independent references (loops, numpy)
def _ref_log_mel(audio: np.ndarray, filters: np.ndarray) -> np.ndarray:
    """Explicit framing + rfft. torch.stft(center=True) reflect-pads n_fft//2 on both sides."""
    x = np.pad(audio, (N_FFT // 2, N_FFT // 2), mode="reflect")
    n_frames = 1 + (len(x) - N_FFT) // HOP
    win = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(N_FFT) / N_FFT)  # periodic Hann
    frames = np.stack([x[i * HOP: i * HOP + N_FFT] * win for i in range(n_frames)], axis=1)
    power = np.abs(np.fft.rfft(frames, axis=0)) ** 2
    power = power[:, :-1]
    mel = filters @ power
    log = np.log10(np.maximum(mel, 1e-10))
    log = np.maximum(log, log.max() - 8.0)
    return (log + 4.0) / 4.0


def _ref_sinusoids(length, channels):
    half = channels // 2
    out = np.zeros((length, channels))
    for t in range(length):
        for i in range(half):
            inv = math.exp(-i * math.log(10000) / (half - 1))
            out[t, i] = math.sin(t * inv)
            out[t, half + i] = math.cos(t * inv)
    return out


def _ln(x, w, b, eps=1e-5):
    mu = x.mean(-1, keepdims=True)
    var = x.var(-1, keepdims=True)
    return (x - mu) / np.sqrt(var + eps) * w + b


def _gelu(x):
    from math import erf
    return x * 0.5 * (1 + np.vectorize(erf)(x / math.sqrt(2)))


def _lin(x, sd, p, bias=True):
    y = x @ sd[p + ".weight"].T
    return y + sd[p + ".bias"] if bias else y


def _ref_mha(x, xa, sd, p, H, causal):
    """One sequence. x: (Tq, D), xa: (Tk, D). Returns (out, raw scores (H, Tq, Tk))."""
    src = x if xa is None else xa
    q, k, v = _lin(x, sd, p + ".query"), _lin(src, sd, p + ".key", bias=False), _lin(src, sd, p + ".value")
    Tq, D = q.shape
    dh = D // H
    out = np.zeros((Tq, D))
    scores = np.zeros((H, Tq, len(src)))
    for h in range(H):
        s = slice(h * dh, (h + 1) * dh)
        for i in range(Tq):
            for j in range(len(src)):
                scores[h, i, j] = q[i, s] @ k[j, s] / math.sqrt(dh)
                if causal and j > i:
                    scores[h, i, j] = -np.inf
            w = np.exp(scores[h, i] - scores[h, i].max())
            w /= w.sum()
            out[i, s] = w @ v[:, s]
    return _lin(out, sd, p + ".out"), scores


def _ref_block(x, xa, sd, p, H, causal):
    x = x + _ref_mha(_ln(x, sd[p + ".attn_ln.weight"], sd[p + ".attn_ln.bias"]), None, sd, p + ".attn", H, causal)[0]
    if xa is not None:
        h = _ln(x, sd[p + ".cross_attn_ln.weight"], sd[p + ".cross_attn_ln.bias"])
        x = x + _ref_mha(h, xa, sd, p + ".cross_attn", H, False)[0]
    h = _ln(x, sd[p + ".mlp_ln.weight"], sd[p + ".mlp_ln.bias"])
    return x + _lin(_gelu(_lin(h, sd, p + ".mlp.0")), sd, p + ".mlp.2")


def _ref_conv1d(x, w, b, stride):
    """x: (C_in, T), w: (C_out, C_in, 3), padding 1."""
    xp = np.pad(x, ((0, 0), (1, 1)))
    T_out = (x.shape[1] + 2 - 3) // stride + 1
    out = np.zeros((w.shape[0], T_out))
    for t in range(T_out):
        out[:, t] = np.einsum("oik,ik->o", w, xp[:, t * stride: t * stride + 3]) + b
    return out


def _sd(m):
    return {k: v.detach().numpy().astype(np.float64) for k, v in m.state_dict().items()}


def _ref_encoder(mel, sd, dims):
    x = _gelu(_ref_conv1d(mel, sd["encoder.conv1.weight"], sd["encoder.conv1.bias"], 1))
    x = _gelu(_ref_conv1d(x, sd["encoder.conv2.weight"], sd["encoder.conv2.bias"], 2)).T
    x = x + _ref_sinusoids(dims.n_audio_ctx, dims.n_audio_state)
    for i in range(dims.n_audio_layer):
        x = _ref_block(x, None, sd, f"encoder.blocks.{i}", dims.n_audio_head, causal=False)
    return _ln(x, sd["encoder.ln_post.weight"], sd["encoder.ln_post.bias"])


def _ref_decoder(tokens, xa, sd, dims):
    E = sd["decoder.token_embedding.weight"]
    x = E[tokens] + sd["decoder.positional_embedding"][:len(tokens)]
    for i in range(dims.n_text_layer):
        x = _ref_block(x, xa, sd, f"decoder.blocks.{i}", dims.n_text_head, causal=True)
    x = _ln(x, sd["decoder.ln.weight"], sd["decoder.ln.bias"])
    return x @ E.T


# ============================================================ audio.py
def test_pad_or_trim_pads_right_and_trims_tail():
    a = torch.arange(1, 6, dtype=torch.float64)
    p = pad_or_trim(a, 8)
    assert p.shape == (8,) and torch.equal(p[:5], a) and torch.all(p[5:] == 0), (
        f"got {p.tolist()}: zero-pad on the RIGHT (the audio must start at t=0, timestamps are relative to it)")
    t = pad_or_trim(a, 3)
    assert torch.equal(t, a[:3]), f"got {t.tolist()}: trim keeps the FIRST `length` samples"
    b = torch.randn(2, 10, dtype=torch.float64)
    assert pad_or_trim(b, 12).shape == (2, 12), "pad/trim acts on the last axis only"
    assert pad_or_trim(torch.zeros(400000)).shape == (480000,), "default length is 30 s * 16 kHz = 480000"


def test_thirty_seconds_give_exactly_3000_frames():
    audio = torch.randn(480000, dtype=torch.float64) * 0.1
    mel = log_mel_spectrogram(audio, MEL80)
    assert mel.shape == (80, 3000), (
        f"shape {tuple(mel.shape)}. center=True STFT gives 480000/160 + 1 = 3001 frames; Whisper drops the LAST "
        "frame (stft[..., :-1]) so 30 s = 3000 frames, which the stride-2 conv turns into 1500 tokens (20 ms each)")


@pytest.mark.parametrize("n,seed", [(16000, 0), (8160, 1), (3999, 2)])
def test_log_mel_matches_explicit_reference(n, seed):
    r = np.random.default_rng(seed)
    t = np.arange(n) / 16000
    audio = 0.3 * np.sin(2 * np.pi * 440 * t) + 0.05 * r.normal(size=n)
    audio[: n // 3] *= 1e-4  # a quiet part, so the dynamic-range clamp is active
    out = log_mel_spectrogram(torch.from_numpy(audio), MEL80).numpy()
    ref = _ref_log_mel(audio, MEL80.numpy())
    assert out.shape == ref.shape, f"shape {out.shape} != {ref.shape} (did you drop the last STFT frame?)"
    err = np.abs(out - ref)
    assert err.max() < 1e-8, (
        f"max err {err.max():.2e} at mel/frame {np.unravel_index(err.argmax(), err.shape)}. Recipe: POWER spectrum "
        "|STFT|^2 -> filters @ power -> log10(clamp(., 1e-10)) -> max(., global_max - 8) -> (x + 4) / 4")


def test_dynamic_range_clamp_is_global_not_per_frame():
    n = 32000
    audio = np.zeros(n)
    audio[n // 2:] = 0.5 * np.sin(2 * np.pi * 1000 * np.arange(n // 2) / 16000)
    audio[: n // 2] = 1e-7 * np.random.default_rng(3).normal(size=n // 2)
    out = log_mel_spectrogram(torch.from_numpy(audio), MEL80).numpy()
    floor = (out.max() * 4 - 4 - 8 + 4) / 4  # = out.max() - 2
    assert out.min() >= floor - 1e-9, f"min {out.min():.3f} below global floor {floor:.3f}"
    quiet = out[:, : out.shape[1] // 2 - 5]
    assert np.allclose(quiet, floor, atol=1e-9), (
        "the quiet half should be clamped flat at (global max - 8 + 4)/4. If quiet frames still vary, your clamp uses a "
        "per-frame or per-channel max: Whisper uses ONE max over the whole spectrogram")


# ============================================================ model.py: encoder
@pytest.mark.parametrize("length,channels", [(5, 8), (11, 6), (3, 4)])
def test_sinusoids_match_reference(length, channels):
    out = sinusoids(length, channels).double().numpy()
    ref = _ref_sinusoids(length, channels)
    assert out.shape == ref.shape, f"shape {out.shape} != {ref.shape}"
    assert np.allclose(out, ref, atol=1e-6), (
        f"max err {np.abs(out - ref).max():.2e}. Whisper: [sin | cos] CONCATENATED (all sins, then all cosines), "
        "timescales exp(-i * log(10000) / (C/2 - 1)) for i in 0..C/2-1, so the last channel pair has period 2*pi*10000")


def test_state_dict_names_and_shapes_match_whisper():
    m = _model()
    sd = m.state_dict()
    D, M, V = 8, 6, 13
    expect = {
        "encoder.conv1.weight": (D, M, 3), "encoder.conv1.bias": (D,),
        "encoder.conv2.weight": (D, D, 3), "encoder.conv2.bias": (D,),
        "encoder.positional_embedding": (5, D),
        "encoder.ln_post.weight": (D,), "encoder.ln_post.bias": (D,),
        "decoder.token_embedding.weight": (V, D), "decoder.positional_embedding": (7, D),
        "decoder.ln.weight": (D,), "decoder.ln.bias": (D,),
    }
    for side, cross in (("encoder", False), ("decoder", True)):
        for i in range(2):
            attns = ["attn", "cross_attn"] if cross else ["attn"]
            for a in attns:
                p = f"{side}.blocks.{i}.{a}"
                for n in ("query", "value", "out"):
                    expect[f"{p}.{n}.weight"] = (D, D)
                    expect[f"{p}.{n}.bias"] = (D,)
                expect[f"{p}.key.weight"] = (D, D)
                expect[f"{side}.blocks.{i}.{a}_ln.weight"] = (D,)
                expect[f"{side}.blocks.{i}.{a}_ln.bias"] = (D,)
            p = f"{side}.blocks.{i}"
            expect.update({f"{p}.mlp.0.weight": (4 * D, D), f"{p}.mlp.0.bias": (4 * D,),
                           f"{p}.mlp.2.weight": (D, 4 * D), f"{p}.mlp.2.bias": (D,),
                           f"{p}.mlp_ln.weight": (D,), f"{p}.mlp_ln.bias": (D,)})
    got = {k: tuple(v.shape) for k, v in sd.items()}
    missing, extra = sorted(set(expect) - set(got)), sorted(set(got) - set(expect))
    assert not missing and not extra, (
        f"missing {missing[:6]}, unexpected {extra[:6]}. Whisper's names: query/key/value/out, key has NO bias, "
        "the decoder has NO separate output head (logits use token_embedding.weight), the causal mask is a "
        "non-persistent buffer")
    bad = {k: (got[k], expect[k]) for k in expect if got[k] != expect[k]}
    assert not bad, f"shape mismatch (got, expected): {bad}"


def test_attention_returns_raw_scaled_scores():
    m = _model()
    attn = m.encoder.blocks[0].attn
    x = torch.randn(1, 4, 8, dtype=torch.float64, generator=torch.Generator().manual_seed(4))
    out, qk = attn(x)
    sd = {"a." + k: v for k, v in _sd(attn).items()}
    ref_out, ref_qk = _ref_mha(x[0].numpy(), None, sd, "a", 2, causal=False)
    assert qk.shape == (1, 2, 4, 4), f"qk shape {tuple(qk.shape)}, expected (B, heads, Tq, Tk)"
    assert np.allclose(qk[0].numpy(), ref_qk, atol=1e-9), (
        f"scores max err {np.abs(qk[0].numpy() - ref_qk).max():.2e}. The returned qk must be q.k / sqrt(d_head) "
        "(Whisper scales q and k each by d_head^-1/4). These raw scores feed word-timestamp alignment, so a key "
        "bias (softmax-invisible) or a wrong scale shows up here")
    assert np.allclose(out[0].detach().numpy(), ref_out, atol=1e-9), (
        f"output max err {np.abs(out[0].detach().numpy() - ref_out).max():.2e}")


@pytest.mark.parametrize("B,seed", [(1, 0), (3, 5)])
def test_encoder_matches_reference(B, seed):
    m = _model(seed=seed)
    mel = torch.randn(B, 6, 10, dtype=torch.float64, generator=torch.Generator().manual_seed(seed + 9))
    out = m.encoder(mel).detach().numpy()
    sd = _sd(m)
    assert out.shape == (B, 5, 8), (
        f"shape {out.shape}: 10 mel frames -> conv1 (stride 1) -> conv2 (stride 2) -> 5 audio tokens")
    for b in range(B):
        ref = _ref_encoder(mel[b].numpy(), sd, DIMS)
        assert np.allclose(out[b], ref, atol=1e-8), (
            f"batch {b} max err {np.abs(out[b] - ref).max():.2e}. Encoder: GELU(conv1) -> GELU(conv2, stride 2) -> "
            "transpose to (T, D) -> + sinusoids -> pre-LN blocks -> ln_post")


def test_encoder_rejects_wrong_length():
    m = _model()
    with pytest.raises(Exception):
        m.encoder(torch.randn(1, 6, 12, dtype=torch.float64))


# ============================================================ model.py: decoder
@pytest.mark.parametrize("T,seed", [(1, 0), (4, 1), (7, 2)])
def test_decoder_matches_reference(T, seed):
    m = _model(seed=seed)
    g = torch.Generator().manual_seed(seed + 20)
    xa = torch.randn(2, 5, 8, dtype=torch.float64, generator=g)
    tokens = torch.randint(0, 13, (2, T), generator=g)
    out = m.decoder(tokens, xa).detach().numpy()
    assert out.shape == (2, T, 13), f"logits shape {out.shape}, expected (B, T, n_vocab)"
    sd = _sd(m)
    for b in range(2):
        ref = _ref_decoder(tokens[b].numpy(), xa[b].numpy(), sd, DIMS)
        assert np.allclose(out[b], ref, atol=1e-8), (
            f"batch {b} max err {np.abs(out[b] - ref).max():.2e}. Block = x + selfattn(LN x, causal) -> "
            "x + crossattn(LN x, audio, NO mask) -> x + MLP(LN x); then final ln; logits = x @ token_embedding.T")


def test_decoder_is_causal_but_sees_all_audio():
    m = _model(seed=3)
    g = torch.Generator().manual_seed(30)
    xa = torch.randn(1, 5, 8, dtype=torch.float64, generator=g)
    tok = torch.randint(0, 13, (1, 6), generator=g)
    base = m.decoder(tok, xa).detach()
    tok2 = tok.clone()
    tok2[0, 4] = (tok2[0, 4] + 1) % 13
    alt = m.decoder(tok2, xa).detach()
    assert torch.allclose(base[0, :4], alt[0, :4], atol=1e-12), "changing token 4 changed logits at positions < 4: no causal mask?"
    assert not torch.allclose(base[0, 4], alt[0, 4]), "changing token 4 must change ITS OWN logits (diagonal must be visible)"
    xa2 = xa.clone()
    xa2[0, -1] += 1.0
    alt2 = m.decoder(tok, xa2).detach()
    assert not torch.allclose(base[0, 0], alt2[0, 0]), (
        "changing the LAST audio token didn't change the FIRST text position: cross-attention must not be masked")


def test_logits_use_the_tied_embedding():
    m = _model(seed=4)
    xa = torch.randn(1, 5, 8, dtype=torch.float64)
    tok = torch.tensor([[1, 2, 3]])
    base = m.decoder(tok, xa).detach()
    with torch.no_grad():
        m.decoder.token_embedding.weight[7] += 1.0   # token 7 is not in the input
    alt = m.decoder(tok, xa).detach()
    diff = (alt - base).abs().amax(dim=(0, 1))
    assert diff[7] > 1e-6 and diff[torch.arange(13) != 7].max() < 1e-12, (
        "editing embedding row 7 (a token NOT in the input) must change exactly logit 7: the output layer IS the "
        "token embedding (tied, Press & Wolf 2017)")


def test_whisper_forward_is_decoder_of_encoder():
    m = _model(seed=5)
    mel = torch.randn(2, 6, 10, dtype=torch.float64)
    tok = torch.tensor([[1, 2], [3, 4]])
    a = m(mel, tok)
    b = m.decoder(tok, m.encoder(mel))
    assert torch.allclose(a, b), "Whisper.forward(mel, tokens) = decoder(tokens, encoder(mel))"


# ============================================================ decoding
class _FakeEncoder:
    def __init__(self):
        self.calls = 0

    def __call__(self, mel):
        self.calls += 1
        return torch.zeros(mel.shape[0], 2, 4, dtype=torch.float64)


class _ScriptedDecoder:
    """Logits for the NEXT token depend only on the current length; row 0 wants EOT at step 2, row 1 at step 4."""
    V, EOT = 6, 5

    def __init__(self):
        self.lengths = []

    def __call__(self, tokens, xa):
        B, T = tokens.shape
        self.lengths.append(T)
        logits = torch.full((B, T, self.V), -3.0, dtype=torch.float64)
        for b in range(B):
            for t in range(T):
                step = t - 1  # number of generated tokens so far, given a 2-token prompt
                want = self.EOT if step >= (1 if b == 0 else 3) else (step % 4) + 1
                logits[b, t, want] = 2.0 + 0.1 * t
                logits[b, t, 0] = -10.0 * (t + 1)   # position-dependent junk at early positions
        return logits


class _FakeModel:
    def __init__(self):
        self.encoder, self.decoder = _FakeEncoder(), _ScriptedDecoder()


def test_greedy_decode_scripted():
    fm = _FakeModel()
    mel = torch.zeros(2, 3, 4)
    tokens, slp = greedy_decode(fm, mel, sot_sequence=(0, 0), eot=5, max_new_tokens=10)
    assert fm.encoder.calls == 1, f"encoder called {fm.encoder.calls} times: encode the audio ONCE, reuse the features"
    tokens = tokens.tolist()
    assert tokens[0] == [0, 0, 1, 5, 5, 5], (
        f"row 0 got {tokens[0]}. Use the logits at the LAST position; once a row emits EOT it keeps emitting EOT; "
        "stop when ALL rows have emitted EOT")
    assert tokens[1] == [0, 0, 1, 2, 3, 5], f"row 1 got {tokens[1]}"

    def lp(t):
        row = torch.full((6,), -3.0, dtype=torch.float64)
        row[0] = -10.0 * (t + 1)
        return row
    exp = []
    for b, n_tok in ((0, 2), (1, 4)):
        s = 0.0
        for i in range(n_tok):
            t = 1 + i
            row = lp(t)
            step = t - 1
            want = 5 if step >= (1 if b == 0 else 3) else (step % 4) + 1
            row[want] = 2.0 + 0.1 * t
            s += float(torch.log_softmax(row, 0)[want])
        exp.append(s)
    assert np.allclose(slp.tolist(), exp, atol=1e-9), (
        f"sum_logprobs {slp.tolist()} != {exp}. Add log_softmax(logits)[chosen] for every token up to AND INCLUDING "
        "the first EOT, and nothing after it (a finished row's padding EOTs are not scored)")


def test_greedy_decode_matches_full_model_reference():
    dims = ModelDimensions(n_mels=6, n_audio_ctx=5, n_audio_state=8, n_audio_head=2, n_audio_layer=1,
                           n_vocab=13, n_text_ctx=12, n_text_state=8, n_text_head=2, n_text_layer=2)
    m = _model(dims, seed=7).requires_grad_(False)
    mel = torch.randn(2, 6, 10, dtype=torch.float64, generator=torch.Generator().manual_seed(8))
    sot = (10, 11)
    tokens, slp = greedy_decode(m, mel, sot, eot=12, max_new_tokens=6)
    for b in range(2):
        xa = m.encoder(mel[b:b + 1])
        seq, s = list(sot), 0.0
        for _ in range(6):
            if seq[-1] == 12:
                seq.append(12)
                continue
            lg = m.decoder(torch.tensor([seq]), xa)[0, -1]
            nt = int(lg.argmax())
            s += float(torch.log_softmax(lg, -1)[nt])
            seq.append(nt)
        got = tokens[b].tolist()
        assert got == seq[: len(got)], f"row {b}: {got} vs reference {seq}"
        assert len(got) == len(seq) or all(t[-1] == 12 for t in tokens.tolist()), (
            f"stopped early at length {len(got)} although not every row had emitted EOT")
        assert abs(float(slp[b]) - s) < 1e-8, f"row {b}: sum_logprobs {float(slp[b])} vs reference {s}"
