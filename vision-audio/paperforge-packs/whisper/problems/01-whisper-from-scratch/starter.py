"""Whisper from scratch: audio.py + model.py + a greedy decode loop.

Read this file TOP-DOWN. Level 0 is the whole model in a few lines; every later level implements one
box of the level above. Cmd/Ctrl+click a name to jump to it (Ctrl+- jumps back).

  L0  Whisper (given) = AudioEncoder + TextDecoder           greedy_decode(model, mel, ...)
  L1  AudioEncoder: conv stem → + sinusoids → blocks → ln_post
      TextDecoder:  tokens + learned pos → blocks(causal, cross to audio) → ln → tied logits
  L2  ResidualAttentionBlock: self-attn → (cross-attn) → MLP, pre-LN
  L3  MultiHeadAttention, sinusoids
  L4  input front end (audio.py): pad_or_trim → log_mel_spectrogram   (produces the `mel` of L0)

Fill in every `# ===== YOUR CODE =====` block. Everything else (names, shapes, glue) is given and
matches openai/whisper, so your state_dict() keys are the real checkpoint keys.
"""
from __future__ import annotations

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


# ================================================================ L0 · the architecture: model.py Whisper + decoding.py (greedy, no KV cache)
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


@torch.no_grad()
def greedy_decode(model: Whisper, mel: torch.Tensor, sot_sequence, eot: int, max_new_tokens: int):
    """mel: (B, n_mels, frames).  sot_sequence: e.g. (<|startoftranscript|>, <|en|>, <|transcribe|>).

    Returns (tokens: long (B, len(sot_sequence) + n_steps), sum_logprobs: (B,)).
    """
    B = mel.shape[0]
    tokens = torch.tensor([list(sot_sequence)] * B, dtype=torch.long)
    # ===== YOUR CODE =====
    # Encode the audio ONCE (model.encoder). Then loop up to max_new_tokens (model.decoder each step):
    #   - logits for the next token (which position?), argmax
    #   - add the chosen token's log-prob to sum_logprobs, but not for rows that had already finished
    #   - finished rows keep emitting eot
    #   - append; stop when every row's last token is eot
    raise NotImplementedError
    # =====================
    return tokens, sum_logprobs


# ================================================================ L1 · encoder and decoder
class AudioEncoder(nn.Module):
    def __init__(self, n_mels, n_ctx, n_state, n_head, n_layer):
        super().__init__()
        # ===== YOUR CODE =====
        # self.conv1, self.conv2: Conv1d, kernel 3, padding 1. One of them has stride 2.
        raise NotImplementedError
        # =====================
        self.register_buffer("positional_embedding", sinusoids(n_ctx, n_state))
        self.blocks = nn.ModuleList([ResidualAttentionBlock(n_state, n_head) for _ in range(n_layer)])
        self.ln_post = nn.LayerNorm(n_state)

    def forward(self, mel):
        """mel: (B, n_mels, 2 * n_ctx)  ->  (B, n_ctx, n_state)"""
        # ===== YOUR CODE =====
        # conv stem (with GELUs) -> (B, T, D) -> assert T matches n_ctx -> + positions -> blocks -> ?
        raise NotImplementedError
        # =====================


class TextDecoder(nn.Module):
    def __init__(self, n_vocab, n_ctx, n_state, n_head, n_layer):
        super().__init__()
        self.token_embedding = nn.Embedding(n_vocab, n_state)
        self.positional_embedding = nn.Parameter(torch.empty(n_ctx, n_state).normal_(0, 0.02))  # LEARNED
        self.blocks = nn.ModuleList(
            [ResidualAttentionBlock(n_state, n_head, cross_attention=True) for _ in range(n_layer)]
        )
        self.ln = nn.LayerNorm(n_state)
        # ===== YOUR CODE =====
        # mask: (n_ctx, n_ctx) additive causal mask. Register it as a NON-persistent buffer named "mask"
        # (it is not in the checkpoint).
        raise NotImplementedError
        # =====================

    def forward(self, tokens, xa):
        """tokens: long (B, T <= n_ctx).   xa: encoder output (B, n_audio_ctx, n_state).
        Returns logits (B, T, n_vocab)."""
        # ===== YOUR CODE =====
        # embed + positions -> blocks (with xa and the mask) -> final LN -> logits.
        # There is no output-head parameter: where do the logits come from?
        raise NotImplementedError
        # =====================


# ================================================================ L2 · one transformer block
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
        # ===== YOUR CODE =====
        # Pre-LN residual sub-layers: self-attention (with mask), cross-attention (only if
        # self.cross_attn exists; which mask?), MLP.  Order matters.
        raise NotImplementedError
        # =====================


# ================================================================ L3 · attention and positions
class MultiHeadAttention(nn.Module):
    def __init__(self, n_state: int, n_head: int):
        super().__init__()
        self.n_head = n_head
        # ===== YOUR CODE =====
        # Four Linear(n_state, n_state) layers named query, key, value, out.
        # One of them has no bias in Whisper. Which one, and why is that harmless?
        raise NotImplementedError
        # =====================

    def forward(self, x, xa=None, mask=None):
        """x: (B, Tq, D).  xa: (B, Tk, D) for cross-attention, else None (self-attention).
        mask: (n_ctx, n_ctx) ADDITIVE mask (0 / -inf) or None.

        Returns (out: (B, Tq, D), qk: (B, H, Tq, Tk) the pre-softmax scores, detached).
        """
        q = self.query(x)
        src = x if xa is None else xa
        k = self.key(src)
        v = self.value(src)
        B, Tq, D = q.shape
        H = self.n_head

        # ===== YOUR CODE =====
        # Split into heads -> (B, H, T, D/H).
        # Scores: Whisper multiplies q AND k each by a scale so that q.k ends up / sqrt(D/H).
        # Add the mask (sliced to the current length), softmax over keys, weight v, merge heads.
        raise NotImplementedError
        # =====================
        return self.out(out), qk.detach()


def sinusoids(length: int, channels: int, max_timescale: float = 10000) -> torch.Tensor:
    """Fixed positional embedding for the encoder.  Returns (length, channels)."""
    assert channels % 2 == 0
    # ===== YOUR CODE =====
    # channels // 2 timescales, geometrically spaced from 1 to max_timescale (inclusive at both ends).
    # Output layout: first half sin, second half cos.
    raise NotImplementedError
    # =====================


# ================================================================ L4 · input front end: audio.py
def pad_or_trim(audio: torch.Tensor, length: int = N_SAMPLES) -> torch.Tensor:
    """Make the LAST axis exactly `length` samples long."""
    n = audio.shape[-1]
    # ===== YOUR CODE =====
    # Too long  -> keep which samples?
    # Too short -> zero-pad on which side? (timestamps are measured from the start of the window)
    raise NotImplementedError
    # =====================


def log_mel_spectrogram(audio: torch.Tensor, filters: torch.Tensor) -> torch.Tensor:
    """audio: (..., n_samples) at 16 kHz.   filters: (n_mels, N_FFT // 2 + 1) = (80, 201).

    Returns (..., n_mels, n_samples // HOP_LENGTH)  -> (80, 3000) for 30 s.
    """
    window = torch.hann_window(N_FFT, dtype=audio.dtype)
    stft = torch.stft(audio, N_FFT, HOP_LENGTH, window=window, return_complex=True)
    # stft: (..., 201, n_samples // 160 + 1) complex.  center=True -> one frame too many.

    # ===== YOUR CODE =====
    # 1. spectrum: which frames, and magnitude or power?
    # 2. project onto the mel filters
    # 3. log (which base?), with a floor of 1e-10 inside the log
    # 4. dynamic-range clamp: nothing more than 8 below the max (max over what?)
    # 5. rescale to roughly [-1, 1]
    raise NotImplementedError
    # =====================


if __name__ == "__main__":
    # scratch: a 2-second tone through your front end and a random tiny model
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
    toks, lp = greedy_decode(model, mel[None], (50, 51, 52), eot=59, max_new_tokens=8)
    print("tokens", toks.tolist(), "sum_logprobs", lp.tolist())
