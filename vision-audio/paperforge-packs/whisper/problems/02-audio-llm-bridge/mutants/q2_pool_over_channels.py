"""Pools over the channel axis instead of time."""
import math

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


# ================================================================ given: shared building blocks
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


# ================================================================ Qwen2-Audio
def qwen2_audio_output_lengths(mel_lens: torch.Tensor):
    """Valid lengths after conv2 (stride 2) and after the stride-2 average pool."""
    after_conv = (mel_lens - 1) // 2 + 1
    after_pool = (after_conv - 2) // 2 + 1
    return after_conv, after_pool


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
        after_conv, _ = qwen2_audio_output_lengths(feature_lens)
        T = x.shape[1]
        pad = torch.arange(T)[None, :] >= after_conv[:, None]                  # (B, T) keys that are padding
        mask = torch.zeros(pad.shape, dtype=x.dtype).masked_fill(pad, float("-inf"))[:, None, None, :]
        for layer in self.layers:
            x = layer(x, mask)
        x = self.avg_pooler(x)
        return self.layer_norm(x)                                             # LN after the pool


class Qwen2AudioProjector(nn.Module):
    def __init__(self, d_audio, d_text):
        super().__init__()
        self.linear = nn.Linear(d_audio, d_text)

    def forward(self, x):
        return self.linear(x)


# ================================================================ Qwen3-Omni AuT encoder
def aut_output_lengths(mel_lens: torch.Tensor, n_window: int = 50) -> torch.Tensor:
    """Tokens per audio: every full 100-frame chunk gives 13 tokens; the tail chunk goes through 3 stride-2 convs."""
    chunk = 2 * n_window
    leave = mel_lens % chunk
    feat = (leave - 1) // 2 + 1
    return ((feat - 1) // 2 + 1 - 1) // 2 + 1 + (mel_lens // chunk) * 13


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
        x = self.conv_out(x.permute(0, 3, 1, 2).reshape(N, t, c * f))                   # channel-major (c, f)
        # 3. positions RESTART at 0 inside every chunk
        x = x + self.positional_embedding[:t].to(x.dtype)
        # 4. drop the padded positions of short chunks, pack everything into one sequence
        valid = aut_output_lengths(torch.tensor(chunk_lens), self.n_window)
        keep = torch.arange(t)[None, :] < valid[:, None]
        x = x[keep]                                                                     # (total, d)
        # 5. block-diagonal attention: windows of 13 * (n_window_infer / chunk) tokens, never across audios
        window = 13 * (self.n_window_infer // chunk)
        seg = []
        for n in aut_output_lengths(lens, self.n_window).tolist():
            seg += [window] * (n // window) + ([n % window] if n % window else [])
        ids = torch.repeat_interleave(torch.arange(len(seg)), torch.tensor(seg))
        mask = torch.zeros(len(ids), len(ids), dtype=x.dtype).masked_fill(ids[:, None] != ids[None, :], float("-inf"))
        x = x[None]
        for layer in self.layers:
            x = layer(x, mask)
        x = self.ln_post(x[0])
        return self.proj2(F.gelu(self.proj1(x)))


# ================================================================ the bridge into the LLM
def expand_audio_placeholders(input_ids: list, audio_token_id: int, n_tokens: list) -> list:
    """Processor step: the k-th single <|AUDIO|> becomes n_tokens[k] copies."""
    out, k = [], 0
    for t in input_ids:
        if t == audio_token_id:
            out += [audio_token_id] * n_tokens[k]
            k += 1
        else:
            out.append(t)
    if k != len(n_tokens):
        raise ValueError(f"{k} audio placeholders for {len(n_tokens)} audios")
    return out


def merge_audio_embeddings(inputs_embeds, input_ids, audio_token_id, audio_features):
    """inputs_embeds: (B, T, H) text embeddings.  audio_features: PACKED (N, H) in reading order.
    Returns a NEW tensor where the N placeholder positions hold the audio features."""
    mask = input_ids == audio_token_id
    if int(mask.sum()) != audio_features.shape[0]:
        raise ValueError(f"tokens {int(mask.sum())} != features {audio_features.shape[0]}")
    return inputs_embeds.masked_scatter(mask[..., None], audio_features.to(inputs_embeds.dtype))


@torch.no_grad()
def greedy_generate(lm: TinyLM, inputs_embeds, max_new_tokens: int, eos_id: int) -> list:
    """Batch 1. The audio lives only in the prefix EMBEDDINGS, so we must keep extending the embeddings,
    never re-embed input_ids (the placeholders would turn back into the <|AUDIO|> embedding)."""
    x = inputs_embeds
    out = []
    for _ in range(max_new_tokens):
        nxt = int(lm(x)[0, -1].argmax())
        out.append(nxt)
        if nxt == eos_id:
            break
        x = torch.cat([x, lm.embed_tokens(torch.tensor([[nxt]]))], dim=1)
    return out


# ================================================================ end-to-end pipelines
@torch.no_grad()
def qwen2_audio_generate(encoder, projector, lm, mel, mel_len, prompt_ids, audio_token_id, max_new_tokens, eos_id):
    """mel: (n_mels, 2*max_source_positions) padded. prompt_ids: list with ONE <|AUDIO|>."""
    feats = projector(encoder(mel[None], torch.tensor([mel_len])))[0]       # (P, H)
    _, n = qwen2_audio_output_lengths(torch.tensor([mel_len]))
    feats = feats[: int(n)]                                                 # keep only real audio tokens
    ids = torch.tensor([expand_audio_placeholders(prompt_ids, audio_token_id, [int(n)])])
    embeds = merge_audio_embeddings(lm.embed_tokens(ids), ids, audio_token_id, feats)
    return greedy_generate(lm, embeds, max_new_tokens, eos_id)


@torch.no_grad()
def qwen3_omni_generate(encoder, lm, mels, prompt_ids, audio_token_id, max_new_tokens, eos_id):
    """mels: list of (n_mels, L_i). prompt_ids: one <|AUDIO|> per audio, in order."""
    feats = encoder(mels)                                                   # already in LLM width
    n = aut_output_lengths(torch.tensor([m.shape[1] for m in mels]), encoder.n_window).tolist()
    ids = torch.tensor([expand_audio_placeholders(prompt_ids, audio_token_id, n)])
    embeds = merge_audio_embeddings(lm.embed_tokens(ids), ids, audio_token_id, feats)
    return greedy_generate(lm, embeds, max_new_tokens, eos_id)
