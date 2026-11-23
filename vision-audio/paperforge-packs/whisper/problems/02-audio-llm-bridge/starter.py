"""Qwen2-Audio and Qwen3-Omni: how an audio encoder feeds a decoder-only LLM.

Read this file TOP-DOWN. Level 0 is the whole architecture in a few lines; every later level
implements one box of the level above. Cmd/Ctrl+click a name to jump to it (Ctrl+- jumps back).

  L0  qwen2_audio_generate                       qwen3_omni_generate
        ├─ Qwen2AudioEncoder        (L2a)         ├─ AuTEncoder               (L2b)
        ├─ Qwen2AudioProjector      (given)       │    (output already packed, LLM width)
        ├─ qwen2_audio_output_lengths (L2a)       ├─ aut_output_lengths       (L2b)
        └─ the bridge (L1) ───────────────────────┴─ the bridge (L1)
  L1  expand_audio_placeholders → merge_audio_embeddings → greedy_generate
  L2a Qwen2AudioEncoder.forward  → qwen2_audio_output_lengths     (Whisper encoder + pool)
  L2b AuTEncoder.forward         → aut_output_lengths             (chunked AuT)
  L3  given: sinusoids, AudioAttention, AudioEncoderLayer, TinyLM

Fill in every `# ===== YOUR CODE =====` block (9 of them).
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


# ================================================================ L0 · the architecture: end-to-end pipelines
@torch.no_grad()
def qwen2_audio_generate(encoder, projector, lm, mel, mel_len, prompt_ids, audio_token_id, max_new_tokens, eos_id):
    """mel: (n_mels, 2*max_source_positions) padded. prompt_ids: list with ONE <|AUDIO|>."""
    # ===== YOUR CODE =====
    # 1. feats = projector(encoder(mel[None], ...))          -> (1, P//2, d_text)   [Qwen2AudioEncoder]
    # 2. n = how many of those are REAL audio tokens?         [qwen2_audio_output_lengths]
    # 3. ids = prompt with <|AUDIO|> expanded to n copies     [expand_audio_placeholders]
    # 4. embeds = lm.embed_tokens(ids), audio rows swapped in [merge_audio_embeddings]
    # 5. decode from the embeddings                           [greedy_generate]
    raise NotImplementedError
    # =====================


@torch.no_grad()
def qwen3_omni_generate(encoder, lm, mels, prompt_ids, audio_token_id, max_new_tokens, eos_id):
    """mels: list of (n_mels, L_i). prompt_ids: one <|AUDIO|> per audio, in order."""
    # ===== YOUR CODE =====
    # Same 5 steps as qwen2_audio_generate, with AuTEncoder for step 1 and aut_output_lengths for step 2.
    # What is different (how many audios? padded?), and which step is no longer needed?
    raise NotImplementedError
    # =====================


# ================================================================ L1 · the bridge into the LLM (used by both pipelines)
def expand_audio_placeholders(input_ids: list, audio_token_id: int, n_tokens: list) -> list:
    """Processor step: the k-th single <|AUDIO|> becomes n_tokens[k] copies."""
    # ===== YOUR CODE =====
    # Raise ValueError if the number of placeholders != len(n_tokens).
    raise NotImplementedError
    # =====================


def merge_audio_embeddings(inputs_embeds, input_ids, audio_token_id, audio_features):
    """inputs_embeds: (B, T, H) text embeddings.  audio_features: PACKED (N, H) in reading order.
    Returns a NEW tensor where the N placeholder positions hold the audio features."""
    # ===== YOUR CODE =====
    # Raise ValueError if #placeholders != #features. Do not modify inputs_embeds in place.
    raise NotImplementedError
    # =====================


@torch.no_grad()
def greedy_generate(lm: TinyLM, inputs_embeds, max_new_tokens: int, eos_id: int) -> list:
    """Batch 1. The audio lives only in the prefix EMBEDDINGS, so we must keep extending the embeddings,
    never re-embed input_ids (the placeholders would turn back into the <|AUDIO|> embedding)."""
    # ===== YOUR CODE =====
    # lm(x) -> logits (1, T, vocab). Return the list of generated ids (EOS included if emitted).
    raise NotImplementedError
    # =====================


# ================================================================ L2a · Qwen2-Audio encoder side
class Qwen2AudioEncoder(nn.Module):
    """Whisper-large-v3 encoder + AvgPool1d(2, 2) + LayerNorm.  40 ms per output token."""

    def __init__(self, n_mels, d, n_head, n_layer, max_source_positions):
        super().__init__()
        self.max_source_positions = max_source_positions
        self.conv1 = nn.Conv1d(n_mels, d, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(d, d, kernel_size=3, stride=2, padding=1)
        self.embed_positions = nn.Embedding(max_source_positions, d)
        self.embed_positions.weight.data.copy_(sinusoids(max_source_positions, d))
        self.layers = nn.ModuleList([AudioEncoderLayer(d, n_head) for _ in range(n_layer)])
        self.avg_pooler = nn.AvgPool1d(2, stride=2)
        self.layer_norm = nn.LayerNorm(d)

    def forward(self, input_features, feature_lens):
        """input_features: (B, n_mels, 2*max_source_positions) ALWAYS padded to 30 s.
        feature_lens: (B,) number of real mel frames.   Returns (B, max_source_positions // 2, d)."""
        assert input_features.shape[-1] == 2 * self.max_source_positions
        x = F.gelu(self.conv1(input_features))
        x = F.gelu(self.conv2(x)).permute(0, 2, 1)
        x = x + self.embed_positions.weight.to(x.dtype)
        # x: (B, max_source_positions, d).  Every audio was padded to 30 s.
        # ===== YOUR CODE =====
        # 1. Build an additive attention mask (broadcastable to (B, H, T, T)) that hides the PADDED
        #    positions as KEYS. How many of the T positions are real for each audio?
        # 2. Run self.layers with that mask.
        # 3. self.avg_pooler (over which axis?) and self.layer_norm (before or after the pool?).
        raise NotImplementedError
        # =====================


def qwen2_audio_output_lengths(mel_lens: torch.Tensor):
    """Valid lengths after conv2 (stride 2) and after the stride-2 average pool."""
    # ===== YOUR CODE =====
    # conv2: kernel 3, stride 2, padding 1.  avg pool: kernel 2, stride 2, no padding.
    # (3000 mel frames -> 1500 -> 750 tokens of 40 ms)
    raise NotImplementedError
    # =====================


class Qwen2AudioProjector(nn.Module):
    def __init__(self, d_audio, d_text):
        super().__init__()
        self.linear = nn.Linear(d_audio, d_text)

    def forward(self, x):
        return self.linear(x)


# ================================================================ L2b · Qwen3-Omni AuT encoder
class AuTEncoder(nn.Module):
    """Qwen3-Omni audio encoder: chunked Conv2d stem (8x in time -> 12.5 Hz), windowed attention, projection."""

    def __init__(self, n_mels, d, n_head, n_layer, output_dim, downsample_hidden_size,
                 n_window=50, n_window_infer=800, max_source_positions=1500):
        super().__init__()
        self.n_window, self.n_window_infer = n_window, n_window_infer
        c = downsample_hidden_size
        self.conv2d1 = nn.Conv2d(1, c, 3, 2, padding=1)
        self.conv2d2 = nn.Conv2d(c, c, 3, 2, padding=1)
        self.conv2d3 = nn.Conv2d(c, c, 3, 2, padding=1)
        f = (((n_mels + 1) // 2 + 1) // 2 + 1) // 2
        self.conv_out = nn.Linear(c * f, d, bias=False)
        self.register_buffer("positional_embedding", sinusoids(max_source_positions, d), persistent=False)
        self.layers = nn.ModuleList([AudioEncoderLayer(d, n_head, key_bias=True) for _ in range(n_layer)])
        self.ln_post = nn.LayerNorm(d)
        self.proj1 = nn.Linear(d, d)
        self.proj2 = nn.Linear(d, output_dim)

    def forward(self, features: list) -> torch.Tensor:
        """features: list of (n_mels, L_i) un-padded mel spectrograms.
        Returns PACKED tokens (sum_i aut_output_lengths(L_i), output_dim), audio after audio."""
        chunk = 2 * self.n_window
        lens = torch.tensor([f.shape[1] for f in features])
        # 1. cut every audio into 1 s chunks (the last one may be shorter); pad chunks to a common length
        chunks, chunk_lens = [], []
        for f in features:
            for s in range(0, f.shape[1], chunk):
                piece = f[:, s:s + chunk]
                chunks.append(piece.T)
                chunk_lens.append(piece.shape[1])
        padded = nn.utils.rnn.pad_sequence(chunks, batch_first=True).transpose(1, 2)   # (N, n_mels, Tc)
        # 2. conv stem on each chunk independently: (N, 1, mel, time) -> (N, c, f, t)
        x = padded.unsqueeze(1)
        x = F.gelu(self.conv2d1(x))
        x = F.gelu(self.conv2d2(x))
        x = F.gelu(self.conv2d3(x))
        N, c, f, t = x.shape
        # ===== YOUR CODE =====
        # 3. (N, c, f, t) -> (N, t, c*f) with the channel index OUTER (c-major), then self.conv_out -> (N, t, d)
        # 4. add self.positional_embedding: where do positions start inside each chunk?
        # 5. drop the positions that came from chunk padding (use aut_output_lengths on chunk_lens),
        #    pack all audios into one (total, d) sequence
        # 6. attention windows: 13 * (n_window_infer // chunk) tokens each, counted per audio
        #    (an audio's last window may be shorter; a window never spans two audios).
        #    Build a (total, total) additive block-diagonal mask and run self.layers on x[None].
        # 7. ln_post -> proj1 -> GELU -> proj2   (output is already the LLM's width)
        raise NotImplementedError
        # =====================


def aut_output_lengths(mel_lens: torch.Tensor, n_window: int = 50) -> torch.Tensor:
    """Tokens per audio: every full 100-frame chunk gives 13 tokens; the tail chunk goes through 3 stride-2 convs."""
    # ===== YOUR CODE =====
    # Audio is cut into chunks of 2*n_window = 100 frames. Each chunk goes through THREE convs with
    # kernel 3, stride 2, padding 1 (along time). A full chunk: 100 -> 50 -> 25 -> 13.
    # Must work elementwise on a LongTensor (no Python loops over audios).
    raise NotImplementedError
    # =====================


# ================================================================ L3 · given building blocks (read, don't change)
def sinusoids(length: int, channels: int, max_timescale: float = 10000) -> torch.Tensor:
    """Whisper's sinusoids: [sin | cos], (length, channels)."""
    inc = np.log(max_timescale) / (channels // 2 - 1)
    inv = torch.exp(-inc * torch.arange(channels // 2, dtype=torch.float64))
    t = torch.arange(length, dtype=torch.float64)[:, None] * inv[None, :]
    return torch.cat([torch.sin(t), torch.cos(t)], dim=1)


class AudioAttention(nn.Module):
    """Whisper-style attention (HF names). Whisper/Qwen2-Audio: k_proj has no bias. AuT: it does."""

    def __init__(self, d: int, n_head: int, key_bias: bool = False):
        super().__init__()
        self.n_head = n_head
        self.q_proj = nn.Linear(d, d)
        self.k_proj = nn.Linear(d, d, bias=key_bias)
        self.v_proj = nn.Linear(d, d)
        self.out_proj = nn.Linear(d, d)

    def forward(self, x, mask=None):
        """x: (B, T, D). mask: additive, broadcastable to (B, H, T, T), or None."""
        B, T, D = x.shape
        H = self.n_head
        q = self.q_proj(x).view(B, T, H, -1).transpose(1, 2)
        k = self.k_proj(x).view(B, T, H, -1).transpose(1, 2)
        v = self.v_proj(x).view(B, T, H, -1).transpose(1, 2)
        s = q @ k.transpose(-1, -2) / math.sqrt(D // H)
        if mask is not None:
            s = s + mask
        out = (s.softmax(-1) @ v).transpose(1, 2).reshape(B, T, D)
        return self.out_proj(out)


class AudioEncoderLayer(nn.Module):
    """Pre-LN Whisper encoder layer (HF names)."""

    def __init__(self, d: int, n_head: int, key_bias: bool = False):
        super().__init__()
        self.self_attn = AudioAttention(d, n_head, key_bias)
        self.self_attn_layer_norm = nn.LayerNorm(d)
        self.fc1 = nn.Linear(d, 4 * d)
        self.fc2 = nn.Linear(4 * d, d)
        self.final_layer_norm = nn.LayerNorm(d)

    def forward(self, x, mask=None):
        x = x + self.self_attn(self.self_attn_layer_norm(x), mask)
        return x + self.fc2(F.gelu(self.fc1(self.final_layer_norm(x))))


class TinyLM(nn.Module):
    """A decoder-only LLM that eats `inputs_embeds` (stands in for Qwen-7B / the Qwen3-Omni Thinker).

    Pre-RMSNorm causal blocks with 1-D RoPE over positions 0..T-1, untied lm_head.
    """

    def __init__(self, vocab: int, d: int, n_head: int, n_layer: int):
        super().__init__()
        self.n_head = n_head
        self.embed_tokens = nn.Embedding(vocab, d)
        self.layers = nn.ModuleList()
        for _ in range(n_layer):
            self.layers.append(nn.ModuleDict(dict(
                ln1=nn.RMSNorm(d), qkv=nn.Linear(d, 3 * d, bias=False), o=nn.Linear(d, d, bias=False),
                ln2=nn.RMSNorm(d), up=nn.Linear(d, 4 * d, bias=False), down=nn.Linear(4 * d, d, bias=False))))
        self.norm = nn.RMSNorm(d)
        self.lm_head = nn.Linear(d, vocab, bias=False)

    def _rope(self, x):
        T, dh = x.shape[-2], x.shape[-1]
        inv = 1.0 / (10000 ** (torch.arange(0, dh, 2, dtype=x.dtype) / dh))
        f = torch.arange(T, dtype=x.dtype)[:, None] * inv[None]
        cos, sin = torch.cat([f, f], -1).cos(), torch.cat([f, f], -1).sin()
        x1, x2 = x[..., : dh // 2], x[..., dh // 2:]
        return x * cos + torch.cat([-x2, x1], -1) * sin

    def forward(self, inputs_embeds):
        """inputs_embeds: (B, T, d) -> logits (B, T, vocab)."""
        x = inputs_embeds
        B, T, D = x.shape
        H = self.n_head
        causal = torch.full((T, T), float("-inf"), dtype=x.dtype).triu(1)
        for L in self.layers:
            q, k, v = L["qkv"](L["ln1"](x)).view(B, T, 3, H, -1).permute(2, 0, 3, 1, 4)
            q, k = self._rope(q), self._rope(k)
            a = ((q @ k.transpose(-1, -2)) / math.sqrt(D // H) + causal).softmax(-1) @ v
            x = x + L["o"](a.transpose(1, 2).reshape(B, T, D))
            x = x + L["down"](F.silu(L["up"](L["ln2"](x))))
        return self.lm_head(self.norm(x))


if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(0)
    AUDIO, EOS = 30, 31
    lm = TinyLM(32, 24, 2, 2)
    print("Qwen2-Audio lengths for 3000 / 1000 frames:", qwen2_audio_output_lengths(torch.tensor([3000, 1000])))
    q2 = Qwen2AudioEncoder(8, 16, 2, 2, max_source_positions=12)
    mel = torch.randn(8, 24)
    print("Qwen2-Audio ->", qwen2_audio_generate(q2, Qwen2AudioProjector(16, 24), lm, mel, 15,
                                                 [1, 29, AUDIO, 28, 4], AUDIO, 5, EOS))
    print("AuT lengths for 100 / 250 / 3000 frames:", aut_output_lengths(torch.tensor([100, 250, 3000])))
    aut = AuTEncoder(16, 16, 2, 2, output_dim=24, downsample_hidden_size=4, n_window_infer=200, max_source_positions=20)
    print("Qwen3-Omni ->", qwen3_omni_generate(aut, lm, [torch.randn(16, 250), torch.randn(16, 60)],
                                               [1, 29, AUDIO, 28, 29, AUDIO, 28, 6], AUDIO, 5, EOS))
