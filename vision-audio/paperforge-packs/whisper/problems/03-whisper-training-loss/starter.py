"""Whisper's training loss: the multitask sequence, teacher forcing, and the masked NLL."""
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

# ================================================================ GIVEN: the model from problem 01 (don't change)
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
            x = block(x, xa, mask=self.mask)
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


# ================================================================ YOUR TASK
def make_training_example(prev_ids, sot_sequence, text_ids, eot: int, sot_prev: int):
    """-> (input_ids, target_ids, loss_mask), each 1-D long/long/bool of length len(sequence) - 1."""
    raise NotImplementedError


def nll_loss(logits: torch.Tensor, target_ids: torch.Tensor, loss_mask: torch.Tensor) -> torch.Tensor:
    """logits (B, T, V), target_ids (B, T), loss_mask (B, T) -> scalar mean NLL over counted tokens."""
    raise NotImplementedError


def whisper_loss(model: "Whisper", mel: torch.Tensor, input_ids: torch.Tensor,
                 target_ids: torch.Tensor, loss_mask: torch.Tensor) -> torch.Tensor:
    """Teacher-forced masked NLL of a Whisper model on (B, T) token batches."""
    raise NotImplementedError


if __name__ == "__main__":
    SOT, EN, TRANSCRIBE, EOT, PREV = 8, 9, 10, 11, 12
    print(make_training_example([3, 4], [SOT, EN, TRANSCRIBE], [5, 6, 7], EOT, PREV))
