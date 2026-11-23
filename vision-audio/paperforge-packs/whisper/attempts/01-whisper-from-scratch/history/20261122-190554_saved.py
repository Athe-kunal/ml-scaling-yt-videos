"""Whisper from scratch: audio.py + model.py + a greedy decode loop."""
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

# ---------------------------------------------------------------- audio.py constants (given)
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

    power = stft[..., :-1].abs() ** 2                       # drop last frame, |X|^2
    mel = filters @ power                                   # (..., 80, 3000)
    log_spec = torch.clamp(mel, min=1e-10).log10()
    log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)  # global max, not per frame
    return (log_spec + 4.0) / 4.0


# ================================================================ model.py
@dataclass
class ModelDimensions:
    n_mels: int
    n_audio_ctx: int
    n_audio_state: int
    n_audio_head: int
    n_audio_layer: int
    n_vocab: int
    n_text_ctx: int
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
        self.key = nn.Linear(n_state, n_state, bias=False)
        self.value = nn.Linear(n_state, n_state)
        self.out = nn.Linear(n_state, n_state)

    def forward(self, x, xa=None, mask=None):
        q = self.query(x)
        src = x if xa is None else xa
        k = self.key(src)
        v = self.value(src)
        B, Tq, D = q.shape
        Tk = k.shape[1]
        H = self.n_head

        q = q.view(B, Tq, H, D // H).permute(0, 2, 1, 3)   # (B, H, Tq, d)
        k = k.view(B, Tk, H, D // H).permute(0, 2, 1, 3)
        v = v.view(B, Tk, H, D // H).permute(0, 2, 1, 3)

        scale = (D // H) ** -0.25
        qk = (q * scale) @ (k * scale).transpose(-1, -2)    # (B, H, Tq, Tk)
        if mask is not None:
            qk = qk + mask[:Tq, :Tq]
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

    def forward(self, x, xa=None, mask=None):
        x = x + self.attn(self.attn_ln(x), mask=mask)[0]
        if self.cross_attn is not None:
            x = x + self.cross_attn(self.cross_attn_ln(x), xa)[0]   # no mask: sees all audio
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

    def forward(self, tokens, xa):
        T = tokens.shape[-1]
        x = self.token_embedding(tokens) + self.positional_embedding[:T]
        for block in self.blocks:
            x = block(x, xa, mask=self.mask)
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


# ================================================================ decoding.py (greedy, no KV cache)
@torch.no_grad()
def greedy_decode(model: Whisper, mel: torch.Tensor, sot_sequence, eot: int, max_new_tokens: int):
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