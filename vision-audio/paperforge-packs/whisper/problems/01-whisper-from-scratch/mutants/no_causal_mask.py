"""Decoder self-attention without the causal mask."""
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
N_SAMPLES = CHUNK_LENGTH * SAMPLE_RATE   # 480000
N_FRAMES = N_SAMPLES // HOP_LENGTH       # 3000


# ================================================================ audio.py
def pad_or_trim(audio: torch.Tensor, length: int = N_SAMPLES) -> torch.Tensor:
    """Zero-pad on the RIGHT or cut the last axis to exactly `length` samples."""
    n = audio.shape[-1]
    if n > length:
        return audio[..., :length]
    if n < length:
        return F.pad(audio, (0, length - n))
    return audio


def log_mel_spectrogram(audio: torch.Tensor, filters: torch.Tensor) -> torch.Tensor:
    """audio: (..., n_samples) at 16 kHz.  filters: (n_mels, N_FFT//2 + 1).

    Returns (..., n_mels, n_samples // HOP_LENGTH).
    """
    window = torch.hann_window(N_FFT, dtype=audio.dtype)
    stft = torch.stft(audio, N_FFT, HOP_LENGTH, window=window, return_complex=True)
    # center=True gives n//hop + 1 frames; Whisper drops the last one -> exactly 3000 for 30 s
    magnitudes = stft[..., :-1].abs() ** 2                      # POWER spectrum
    mel_spec = filters.to(magnitudes.dtype) @ magnitudes
    log_spec = torch.clamp(mel_spec, min=1e-10).log10()
    # dynamic range: 8 orders of magnitude (80 dB) below the loudest bin of the WHOLE input
    log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
    return (log_spec + 4.0) / 4.0                               # roughly [-1, 1]


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
    """(length, channels): first half sin, second half cos (concatenated, not interleaved)."""
    assert channels % 2 == 0
    log_timescale_increment = np.log(max_timescale) / (channels // 2 - 1)
    inv_timescales = torch.exp(-log_timescale_increment * torch.arange(channels // 2))
    scaled_time = torch.arange(length)[:, None] * inv_timescales[None, :]
    return torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1)


class MultiHeadAttention(nn.Module):
    def __init__(self, n_state: int, n_head: int):
        super().__init__()
        self.n_head = n_head
        self.query = nn.Linear(n_state, n_state)
        self.key = nn.Linear(n_state, n_state, bias=False)   # no key bias
        self.value = nn.Linear(n_state, n_state)
        self.out = nn.Linear(n_state, n_state)

    def forward(self, x, xa=None, mask=None):
        """x: (B, Tq, D). xa: (B, Tk, D) for cross-attention, else None.
        mask: (n_ctx, n_ctx) additive mask or None.  Returns (out (B,Tq,D), qk (B,H,Tq,Tk))."""
        q = self.query(x)
        src = x if xa is None else xa
        k = self.key(src)
        v = self.value(src)
        B, Tq, D = q.shape
        H = self.n_head
        scale = (D // H) ** -0.25
        q = q.view(B, Tq, H, -1).permute(0, 2, 1, 3)
        k = k.view(B, k.shape[1], H, -1).permute(0, 2, 1, 3)
        v = v.view(B, v.shape[1], H, -1).permute(0, 2, 1, 3)
        qk = (q * scale) @ (k * scale).transpose(-1, -2)          # = q k^T / sqrt(d_head)
        if mask is not None:
            qk = qk + mask[:Tq, :Tq]
        w = qk.softmax(dim=-1)
        out = (w @ v).permute(0, 2, 1, 3).flatten(start_dim=2)
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
            x = x + self.cross_attn(self.cross_attn_ln(x), xa)[0]   # no mask on cross-attention
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
        """mel: (B, n_mels, 2 * n_ctx) -> (B, n_ctx, n_state)"""
        x = F.gelu(self.conv1(mel))
        x = F.gelu(self.conv2(x))                 # stride 2: 3000 frames -> 1500 tokens
        x = x.permute(0, 2, 1)
        assert x.shape[1:] == self.positional_embedding.shape, "incorrect audio shape"
        x = x + self.positional_embedding.to(x.dtype)
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
        mask = torch.empty(n_ctx, n_ctx).fill_(-np.inf).triu_(1)
        self.register_buffer("mask", mask, persistent=False)

    def forward(self, tokens, xa):
        """tokens: long (B, T <= n_ctx).  xa: (B, n_audio_ctx, n_state).  -> logits (B, T, n_vocab)"""
        T = tokens.shape[-1]
        x = self.token_embedding(tokens) + self.positional_embedding[:T]
        x = x.to(xa.dtype)
        for block in self.blocks:
            x = block(x, xa)
        x = self.ln(x)
        return x @ self.token_embedding.weight.to(x.dtype).T     # tied output embedding


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


# ================================================================ decoding.py (greedy, no cache)
@torch.no_grad()
def greedy_decode(model: Whisper, mel: torch.Tensor, sot_sequence, eot: int, max_new_tokens: int):
    """Returns (tokens (B, len(sot_sequence) + n_steps) long, sum_logprobs (B,))."""
    B = mel.shape[0]
    audio_features = model.encoder(mel)                       # encode ONCE
    tokens = torch.tensor([list(sot_sequence)] * B, dtype=torch.long)
    sum_logprobs = torch.zeros(B, dtype=audio_features.dtype)
    for _ in range(max_new_tokens):
        logits = model.decoder(tokens, audio_features)[:, -1]   # last position predicts the next token
        next_tokens = logits.argmax(dim=-1)
        logprobs = F.log_softmax(logits, dim=-1)
        finished = tokens[:, -1] == eot
        sum_logprobs += logprobs[torch.arange(B), next_tokens] * (~finished)
        next_tokens[finished] = eot                              # finished rows keep emitting EOT
        tokens = torch.cat([tokens, next_tokens[:, None]], dim=-1)
        if (tokens[:, -1] == eot).all():
            break
    return tokens, sum_logprobs
