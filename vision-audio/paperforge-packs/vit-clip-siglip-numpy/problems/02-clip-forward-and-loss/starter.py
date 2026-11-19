import math

import numpy as np


def softmax(x, axis=-1):
    """Numerically stable softmax along `axis`."""
    raise NotImplementedError


def log_softmax(x, axis=-1):
    """Numerically stable log-softmax along `axis`."""
    raise NotImplementedError


def layer_norm(x, eps=1e-6):
    """LayerNorm over the last axis, no learnable gain/bias, population variance."""
    raise NotImplementedError


def quick_gelu(x):
    """CLIP's activation: x * sigmoid(1.702 x)."""
    raise NotImplementedError


def l2_normalize(x, axis=-1):
    """Divide each vector along `axis` by its L2 norm."""
    raise NotImplementedError


def encoder_block(x, num_heads, blk, mask=None):
    """Pre-LN transformer block (see statement)."""
    raise NotImplementedError


def clip_image_encoder(image, patch_size, num_heads, W_patch, cls_token, pos_embed, encoder_weights, W_proj):
    """image: (B, H, W, C) -> image embedding (B, E)"""
    raise NotImplementedError


def clip_text_encoder(tokens, num_heads, token_embed, pos_embed, encoder_weights, W_proj):
    """tokens: int (B, L) -> text embedding (B, E)"""
    raise NotImplementedError


TEMPERATURE = 0.07     # tau at initialisation (CLIP §2.5)
MAX_SCALE = 100.0      # the logit multiplier is clipped at 100


def clip_loss(image_emb, text_emb):
    """Compute logit_scale from TEMPERATURE inside. Returns (loss: float, logits_per_image: (N, N))."""
    raise NotImplementedError
