"""Whisper from scratch: audio.py + model.py + greedy decoding (with and without KV cache)."""
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

# ---------------------------------------------------------------- audio.py constants
SAMPLE_RATE = 16000
N_FFT = 400            # 25 ms window
HOP_LENGTH = 160       # 10 ms hop
CHUNK_LENGTH = 30
N_SAMPLES = CHUNK_LENGTH * SAMPLE_RATE   # 480000 samples in a 30 s window
N_FRAMES = N_SAMPLES // HOP_LENGTH       # 3000 mel frames in a 30 s window


# ================================================================ audio.py
def pad_or_trim(audio: torch.Tensor, length: int = N_SAMPLES) -> torch.Tensor:
    """Make the LAST axis exactly `length` samples long."""
    n = audio.shape[-1]
    if n > length:
        return audio[..., :length]                 # keep the FIRST `length` samples
    if n < length:
        return F.pad(audio, (0, length - n))       # zero-pad on the RIGHT
    return audio


def log_mel_spectrogram(audio: torch.Tensor, filters: torch.Tensor) -> torch.Tensor:
    window = torch.hann_window(N_FFT, dtype=audio.dtype)
    stft = torch.stft(audio, N_FFT, HOP_LENGTH, window=window, return_complex=True)

    power = stft[..., :-1].abs() ** 2                         # drop last frame, |X|^2
    mel = filters @ power                                     # (..., 80, 3000)
    log_spec = torch.clamp(mel, min=1e-10).log10()
    log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)  # global max, not per frame
    return (log_spec + 4.0) / 4.0


# ================================================================ model.py
@dataclass
class ModelDimensions:
    n_mels: int
    n_audio_ctx: int      # encoder tokens = mel frames // 2   (1500 for real Whisper)
    n_audio_state: int
    n_audio_head: int
    n_audio_layer: int
    n_vocab: int
    n_text_ctx: int       # max decoder length (448 for real Whisper)
    n_text_state: int
    n_text_head: int
    n_text_layer: int


def sinusoids(length: int, channels: int, max_timescale: float = 10000) -> torch.Tensor:
    assert channels % 2 == 0
    half = channels // 2
    log_inc = np.log(max_timescale) / (half - 1)
    inv = torch.exp(-log_inc * torch.arange(half, dtype=torch.float64))
    t = torch.arange(length, dtype=torch.float64)[:, None] * inv[None, :]
    out = torch.cat([torch.sin(t), torch.cos(t)], dim=1)
    return out.to(torch.get_default_dtype())


class MultiHeadAttention(nn.Module):
    def __init__(self, n_state: int, n_head: int):
        super().__init__()
        self.n_head = n_head
        self.query = nn.Linear(n_state, n_state)
        self.key = nn.Linear(n_state, n_state, bias=False)   # bias would cancel in softmax
        self.value = nn.Linear(n_state, n_state)
        self.out = nn.Linear(n_state, n_state)

    def forward(self, x, xa=None, mask=None, past=None):
        """x: (B, Tq, D).  xa: (B, Tk, D) for cross-attention, else None.
        mask: (n_ctx, n_ctx) additive mask or None.
        past: optional dict, filled/read in place ({} on first use) -> the KV cache.

        Returns (out: (B, Tq, D), qk: (B, H, Tq, Tk) pre-softmax scores, detached).
        """
        q = self.query(x)
        B, Tq, D = q.shape
        H = self.n_head
        Dh = D // H
        split = lambda t: t.view(B, t.shape[1], H, Dh).permute(0, 2, 1, 3)   # (B, H, T, Dh)

        offset = 0                                       # absolute position of first query
        if xa is None:                                   # self-attention: append to cache
            k, v = split(self.key(x)), split(self.value(x))
            if past is not None:
                if "k" in past:
                    offset = past["k"].shape[2]
                    k = torch.cat([past["k"], k], dim=2)
                    v = torch.cat([past["v"], v], dim=2)
                past["k"], past["v"] = k, v
        else:                                            # cross-attention: compute once, reuse
            if past is not None and "k" in past:
                k, v = past["k"], past["v"]
            else:
                k, v = split(self.key(xa)), split(self.value(xa))
                if past is not None:
                    past["k"], past["v"] = k, v

        q = split(q)
        scale = Dh ** -0.25                              # q and k each scaled -> q.k / sqrt(Dh)
        qk = (q * scale) @ (k * scale).transpose(-1, -2)          # (B, H, Tq, Tk)
        if mask is not None:
            qk = qk + mask[offset:offset + Tq, :k.shape[2]]       # offset=0 -> mask[:Tq, :Tq]
        w = F.softmax(qk, dim=-1)
        out = (w @ v).permute(0, 2, 1, 3).reshape(B, Tq, D)
        return self.out(out), qk.detach()


class ResidualAttentionBlock(nn.Module):
    def __init__(self, n_state: int, n_head: int, cross_attention: bool = False):
        super().__init__()
        self.attn = MultiHeadAttention(n_state, n_head)
        self.attn_ln = nn.LayerNorm(n_state)
        self.cross_attn = MultiHeadAttention(n_state, n_head) if cross_attention else None
        self.cross_attn_ln = nn.LayerNorm(n_state) if cross_attention else None
        n_mlp = n_state * 4
        self.mlp = nn.Sequential(nn.Linear(n_state, n_mlp), nn.GELU(), nn.Linear(n_mlp, n_state))
        self.mlp_ln = nn.LayerNorm(n_state)

    def forward(self, x, xa=None, mask=None, past=None):
        # past: None, or {"self": {}, "cross": {}} for this block
        x = x + self.attn(self.attn_ln(x), mask=mask,
                          past=None if past is None else past["self"])[0]
        if self.cross_attn is not None:
            x = x + self.cross_attn(self.cross_attn_ln(x), xa,          # no mask: sees all audio
                                    past=None if past is None else past["cross"])[0]
        x = x + self.mlp(self.mlp_ln(x))
        return x


class AudioEncoder(nn.Module):
    def __init__(self, n_mels, n_ctx, n_state, n_head, n_layer):
        super().__init__()
        self.conv1 = nn.Conv1d(n_mels, n_state, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(n_state, n_state, kernel_size=3, stride=2, padding=1)
        self.register_buffer("positional_embedding", sinusoids(n_ctx, n_state))
        self.blocks = nn.ModuleList([ResidualAttentionBlock(n_state, n_head) for _ in range(n_layer)])
        self.ln_post = nn.LayerNorm(n_state)

    def forward(self, mel):
        """mel: (B, n_mels, 2 * n_ctx)  ->  (B, n_ctx, n_state)"""
        x = F.gelu(self.conv1(mel))
        x = F.gelu(self.conv2(x))
        x = x.permute(0, 2, 1)                                   # (B, T, D)
        assert x.shape[1:] == self.positional_embedding.shape, "incorrect audio shape"
        x = x + self.positional_embedding
        for block in self.blocks:
            x = block(x)
        return self.ln_post(x)


class TextDecoder(nn.Module):
    def __init__(self, n_vocab, n_ctx, n_state, n_head, n_layer):
        super().__init__()
        self.token_embedding = nn.Embedding(n_vocab, n_state)
        self.positional_embedding = nn.Parameter(torch.empty(n_ctx, n_state).normal_(0, 0.02))
        self.blocks = nn.ModuleList(
            [ResidualAttentionBlock(n_state, n_head, cross_attention=True) for _ in range(n_layer)]
        )
        self.ln = nn.LayerNorm(n_state)
        mask = torch.empty(n_ctx, n_ctx).fill_(-np.inf).triu_(1)   # -inf strictly above diagonal
        self.register_buffer("mask", mask, persistent=False)

    def forward(self, tokens, xa, past=None):
        """tokens: long (B, T). With a cache, T is the number of NEW tokens.
        past: None, or a list with one {"self": {}, "cross": {}} dict per block."""
        offset = past[0]["self"]["k"].shape[2] if (past and "k" in past[0]["self"]) else 0
        T = tokens.shape[-1]
        x = self.token_embedding(tokens) + self.positional_embedding[offset:offset + T]
        for i, block in enumerate(self.blocks):
            x = block(x, xa, mask=self.mask, past=None if past is None else past[i])
        x = self.ln(x)
        return x @ self.token_embedding.weight.T                  # tied output head


class Whisper(nn.Module):
    def __init__(self, dims: ModelDimensions):
        super().__init__()
        self.dims = dims
        self.encoder = AudioEncoder(dims.n_mels, dims.n_audio_ctx, dims.n_audio_state,
                                    dims.n_audio_head, dims.n_audio_layer)
        self.decoder = TextDecoder(dims.n_vocab, dims.n_text_ctx, dims.n_text_state,
                                   dims.n_text_head, dims.n_text_layer)

    def forward(self, mel, tokens):
        return self.decoder(tokens, self.encoder(mel))


# ================================================================ decoding.py
@torch.no_grad()
def greedy_decode(model: Whisper, mel: torch.Tensor, sot_sequence, eot: int, max_new_tokens: int):
    """No cache: re-runs the whole decoder on all tokens every step."""
    B = mel.shape[0]
    tokens = torch.tensor([list(sot_sequence)] * B, dtype=torch.long)
    audio_features = model.encoder(mel)                           # encode ONCE
    sum_logprobs = torch.zeros(B, dtype=audio_features.dtype)

    for _ in range(max_new_tokens):
        logits = model.decoder(tokens, audio_features)[:, -1]     # LAST position only
        logprobs = F.log_softmax(logits, dim=-1)
        nxt = logits.argmax(dim=-1)
        lp = logprobs.gather(1, nxt[:, None])[:, 0]

        finished = tokens[:, -1] == eot                           # ended BEFORE this step
        sum_logprobs = sum_logprobs + torch.where(finished, torch.zeros_like(lp), lp)
        nxt = torch.where(finished, torch.full_like(nxt, eot), nxt)
        tokens = torch.cat([tokens, nxt[:, None]], dim=1)

        if (tokens[:, -1] == eot).all():
            break
    return tokens, sum_logprobs


@torch.no_grad()
def greedy_decode_cached(model: Whisper, mel: torch.Tensor, sot_sequence, eot: int, max_new_tokens: int):
    """Same result as greedy_decode, but feeds only the newest token after the first step."""
    B = mel.shape[0]
    tokens = torch.tensor([list(sot_sequence)] * B, dtype=torch.long)
    audio_features = model.encoder(mel)
    sum_logprobs = torch.zeros(B, dtype=audio_features.dtype)
    past = [{"self": {}, "cross": {}} for _ in model.decoder.blocks]
    inp = tokens                                                  # step 0: the whole prompt

    for _ in range(max_new_tokens):
        logits = model.decoder(inp, audio_features, past=past)[:, -1]
        logprobs = F.log_softmax(logits, dim=-1)
        nxt = logits.argmax(dim=-1)
        lp = logprobs.gather(1, nxt[:, None])[:, 0]

        finished = tokens[:, -1] == eot
        sum_logprobs = sum_logprobs + torch.where(finished, torch.zeros_like(lp), lp)
        nxt = torch.where(finished, torch.full_like(nxt, eot), nxt)
        tokens = torch.cat([tokens, nxt[:, None]], dim=1)
        inp = nxt[:, None]                                        # later steps: only the new token

        if (tokens[:, -1] == eot).all():
            break
    return tokens, sum_logprobs


if __name__ == "__main__":
    t = torch.arange(32000, dtype=torch.float64) / SAMPLE_RATE
    audio = pad_or_trim(0.5 * torch.sin(2 * torch.pi * 440 * t))
    try:
        filters = torch.from_numpy(np.load("files/mel_filters.npz")["mel_80"]).double()
    except FileNotFoundError:
        filters = torch.rand(80, N_FFT // 2 + 1, dtype=torch.float64)
    mel = log_mel_spectrogram(audio, filters)
    print("mel", tuple(mel.shape), f"range [{mel.min():.2f}, {mel.max():.2f}]")

    dims = ModelDimensions(80, 1500, 32, 4, 2, 60, 16, 32, 4, 2)
    model = Whisper(dims).double().eval()
    print("audio features", tuple(model.encoder(mel[None]).shape))

    t1, l1 = greedy_decode(model, mel[None], (50, 51, 52), eot=59, max_new_tokens=8)
    t2, l2 = greedy_decode_cached(model, mel[None], (50, 51, 52), eot=59, max_new_tokens=8)
    print("tokens", t1.tolist(), "sum_logprobs", l1.tolist())
    assert torch.equal(t1, t2) and torch.allclose(l1, l2), "cache changed the result!"
    print("cached == uncached: OK")