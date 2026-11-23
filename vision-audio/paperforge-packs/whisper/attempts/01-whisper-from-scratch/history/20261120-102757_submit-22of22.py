"""Whisper from scratch: audio.py + model.py + a greedy decode loop.

Fill in every `# ===== YOUR CODE =====` block. Everything else (names, shapes, glue) is given and
matches openai/whisper, so your state_dict() keys are the real checkpoint keys.
"""
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
    n = audio.shape[-1]
    # ===== YOUR CODE =====
    # Too long  -> keep which samples?
    # Too short -> zero-pad on which side? (timestamps are measured from the start of the window)
    # (batch_size, n_samples)
    n = audio.shape[-1]
    if n > length:
        return audio[...,:length]
    if n < length:
        return F.pad(audio, (0, length - n))
    return audio


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
    power = stft[..., :-1].abs() ** 2
    mel = filters @ power
    log_spec = torch.clamp(mel, min=1e-10).log10()
    log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
    return (log_spec + 4.0) / 4.0

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
    """Fixed positional embedding for the encoder.  Returns (length, channels)."""
    assert channels % 2 == 0
    # ===== YOUR CODE =====
    # channels // 2 timescales, geometrically spaced from 1 to max_timescale (inclusive at both ends).
    # Output layout: first half sin, second half cos.
    assert channels % 2 == 0
    half = channels // 2
    log_inc = np.log(max_timescale) / (half - 1)
    inv = torch.exp(-log_inc * torch.arange(half, dtype=torch.float64))
    t = torch.arange(length, dtype=torch.float64)[:,None] @ inv[None,:]
    out = torch.cat([torch.sin(t), torch.cos(t)], dim=1)
    return out.to(torch.get_default_dtype())

class MultiHeadAttention(nn.Module):
    def __init__(self, n_state: int, n_head: int):
        super().__init__()
        self.n_head = n_head
        # ===== YOUR CODE =====
        # Four Linear(n_state, n_state) layers named query, key, value, out.
        # One of them has no bias in Whisper. Which one, and why is that harmless?
        self.query = nn.Linear(n_state, n_state)
        self.key = nn.Linear(n_state, n_state, bias=False)
        self.value = nn.Linear(n_state, n_state)
        self.out = nn.Linear(n_state, n_state)

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
        Dh = D // H
        split = lambda t: t.view(B, t.shape[1], H, Dh).permute(0, 2, 1, 3)   # (B, H, T, Dh)
        q, k, v = split(q), split(k), split(v)
        scale = (D // H) ** -0.25
        qk = (q * scale) @ (k * scale).transpose(-1, -2)    # (B, H, Tq, Tk)
        if mask is not None:
            qk = qk + mask[:Tq, :Tq]
        w = F.softmax(qk,dim=-1)
        out = (w @ v).permute(0,2,1,3).reshape(B,Tq,D)
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
        # ===== YOUR CODE =====
        # Pre-LN residual sub-layers: self-attention (with mask), cross-attention (only if
        # self.cross_attn exists; which mask?), MLP.  Order matters.
        x = x + self.attn(self.attn_ln(x), mask=mask)[0]
        if self.cross_attn is not None:
            x = x + self.cross_attn(self.cross_attn_ln(x), xa)[0]
        x = x + self.mlp(self.mlp_ln(x))
        return x


class AudioEncoder(nn.Module):
    def __init__(self, n_mels, n_ctx, n_state, n_head, n_layer):
        super().__init__()
        # ===== YOUR CODE =====
        # self.conv1, self.conv2: Conv1d, kernel 3, padding 1. One of them has stride 2.
        self.conv1 = nn.Conv1d(n_mels, n_state, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(n_state, n_state, kernel_size=3, stride=2, padding=1)
        # =====================
        self.register_buffer("positional_embedding", sinusoids(n_ctx, n_state))
        self.blocks = nn.ModuleList([ResidualAttentionBlock(n_state, n_head) for _ in range(n_layer)])
        self.ln_post = nn.LayerNorm(n_state)

    def forward(self, mel):
        """mel: (B, n_mels, 2 * n_ctx)  ->  (B, n_ctx, n_state)"""
        # ===== YOUR CODE =====
        # conv stem (with GELUs) -> (B, T, D) -> assert T matches n_ctx -> + positions -> blocks -> ?
        x = F.gelu(self.conv1(mel))
        x = F.gelu(self.conv2(x))
        x = x.permute(0,2,1) #(B,T,D)
        assert x.shape[1:] == self.positional_embedding.shape, "incorrect audio shape"
        x = x + self.positional_embedding
        for block in self.blocks:
            x = block(x)
        return self.ln_post(x)

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
        # (it is not in the checkpoint)-np.inf..triu_(1)
        mask = torch.empty(n_ctx, n_ctx).fill_(-np.inf).triu_(1)
        self.register_buffer("mask", mask, persistent=False)

    def forward(self, tokens, xa):
        """tokens: long (B, T <= n_ctx).   xa: encoder output (B, n_audio_ctx, n_state).
        Returns logits (B, T, n_vocab)."""
        # ===== YOUR CODE =====
        # embed + positions -> blocks (with xa and the mask) -> final LN -> logits.
        # There is no output-head parameter: where do the logits come from?
        T = tokens.shape[-1]
        x = self.token_embedding(tokens) + self.positional_embedding[:T]
        for block in self.blocks:
            x = block(x, xa,mask=self.mask)
        x = self.ln(x)
        return x @ self.token_embedding.weight.T


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
    """mel: (B, n_mels, frames).  sot_sequence: e.g. (<|startoftranscript|>, <|en|>, <|transcribe|>).

    Returns (tokens: long (B, len(sot_sequence) + n_steps), sum_logprobs: (B,)).
    """
    B = mel.shape[0]
    tokens = torch.tensor([list(sot_sequence)] * B, dtype=torch.long)
    # ===== YOUR CODE =====
    # Encode the audio. Then loop up to max_new_tokens:
    #   - logits for the next token (which position?), argmax
    #   - add the chosen token's log-prob to sum_logprobs, but not for rows that had already finished
    #   - finished rows keep emitting eot
    #   - append; stop when every row's last token is eot
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