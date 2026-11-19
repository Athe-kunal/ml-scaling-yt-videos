import math

import numpy as np
import pytest

from solution import clip_image_encoder, clip_loss, clip_text_encoder

V, CTX = 20, 9
EOS, SOS, PAD = V - 1, V - 2, 0


def _blocks(r, D, layers):
    bl = []
    for _ in range(layers):
        b = {k: r.normal(0, D ** -0.5, (D, D)) for k in ("Wq", "Wk", "Wv", "Wo")}
        b["W1"] = r.normal(0, D ** -0.5, (D, 4 * D))
        b["W2"] = r.normal(0, (4 * D) ** -0.5, (4 * D, D))
        bl.append(b)
    return bl


# ---- loop-based reference --------------------------------------------------------------------
def _ln(v, eps=1e-6):
    return (v - v.mean()) / math.sqrt(v.var() + eps)


def _qg(v):
    return v * (1 / (1 + np.exp(-1.702 * v)))


def _ref_block(x, heads, b, causal):
    N, D = x.shape
    dh = D // heads
    h = np.stack([_ln(t) for t in x])
    Q, K, Vv = h @ b["Wq"], h @ b["Wk"], h @ b["Wv"]
    out = np.zeros_like(x)
    for hd in range(heads):
        s = slice(hd * dh, (hd + 1) * dh)
        for i in range(N):
            keys = range(i + 1) if causal else range(N)
            sc = np.array([Q[i, s] @ K[j, s] / math.sqrt(dh) for j in keys])
            w = np.exp(sc - sc.max())
            w /= w.sum()
            out[i, s] = sum(wj * Vv[j, s] for wj, j in zip(w, keys))
    y = x + out @ b["Wo"]
    return y + _qg(np.stack([_ln(t) for t in y]) @ b["W1"]) @ b["W2"]


def _ref_image(p, P, heads):
    outs = []
    for img in p["image"]:
        hp, wp = img.shape[0] // P, img.shape[1] // P
        x = np.stack([img[i * P:(i + 1) * P, j * P:(j + 1) * P].reshape(-1) for i in range(hp) for j in range(wp)])
        x = np.vstack([p["cls"], x @ p["W_patch"]]) + p["pos"]
        x = np.stack([_ln(t) for t in x])                         # ln_pre
        for b in p["blocks"]:
            x = _ref_block(x, heads, b, causal=False)
        outs.append(_ln(x[0]) @ p["W_proj"])
    return np.stack(outs)


def _ref_text(p, tokens, heads):
    outs = []
    for row in tokens:
        x = p["tok"][row] + p["pos"][:len(row)]
        for b in p["blocks"]:
            x = _ref_block(x, heads, b, causal=True)
        e = list(row).index(EOS)
        outs.append(_ln(x[e]) @ p["W_proj"])
    return np.stack(outs)


def _ref_loss(I, T, s):
    I = I / np.sqrt((I ** 2).sum(1, keepdims=True))
    T = T / np.sqrt((T ** 2).sum(1, keepdims=True))
    n = len(I)
    L = math.exp(s) * I @ T.T
    li = -sum(L[i, i] - math.log(sum(math.exp(L[i, j]) for j in range(n))) for i in range(n)) / n
    lt = -sum(L[j, j] - math.log(sum(math.exp(L[i, j]) for i in range(n))) for j in range(n)) / n
    return (li + lt) / 2, L


# ---- fixtures --------------------------------------------------------------------------------
IMG_CFGS = [(2, 8, 8, 3, 4, 8, 2, 2, 6), (3, 12, 8, 1, 4, 12, 3, 1, 5)]   # (B,H,W,C,P,D,heads,layers,E)
TXT_CFGS = [(8, 2, 2, 6), (12, 3, 1, 5), (6, 1, 3, 4)]                     # (D, heads, layers, E)


def _img_params(cfg, seed):
    B, H, W, C, P, D, heads, layers, E = cfg
    r = np.random.default_rng(seed)
    N = (H // P) * (W // P)
    return dict(image=r.normal(size=(B, H, W, C)), W_patch=r.normal(0, (P * P * C) ** -0.5, (P * P * C, D)),
                cls=r.normal(size=D), pos=r.normal(0, 0.5, (N + 1, D)), blocks=_blocks(r, D, layers),
                W_proj=r.normal(0, D ** -0.5, (D, E)))


def _txt_params(cfg, seed):
    D, heads, layers, E = cfg
    r = np.random.default_rng(seed)
    return dict(tok=r.normal(size=(V, D)), pos=r.normal(0, 0.5, (CTX, D)), blocks=_blocks(r, D, layers),
                W_proj=r.normal(0, D ** -0.5, (D, E)))


def _captions(seed, lengths):
    r = np.random.default_rng(seed)
    rows = []
    for k in lengths:
        row = [SOS] + list(r.integers(1, SOS, k)) + [EOS]
        rows.append(row + [PAD] * (CTX - len(row)))
    return np.array(rows)


def _img(p, cfg):
    return clip_image_encoder(p["image"], cfg[4], cfg[6], p["W_patch"], p["cls"], p["pos"], p["blocks"], p["W_proj"])


def _txt(p, tokens, cfg):
    return clip_text_encoder(tokens, cfg[1], p["tok"], p["pos"], p["blocks"], p["W_proj"])


# ---- image tower -----------------------------------------------------------------------------
@pytest.mark.parametrize("cfg", IMG_CFGS)
def test_image_encoder_matches_reference(cfg):
    p = _img_params(cfg, 0)
    out, ref = np.asarray(_img(p, cfg)), _ref_image(p, cfg[4], cfg[6])
    assert out.shape == ref.shape, f"shape {out.shape} != {ref.shape}"
    assert np.allclose(out, ref, atol=1e-9), (
        f"max err {np.abs(out - ref).max():.2e}. CLIP-ViT: patch @ W_patch (no bias) -> [CLS] + patches -> + pos -> "
        f"ln_pre -> blocks (QuickGELU) -> ln_post(CLS) -> @ W_proj"
    )


# ---- text tower ------------------------------------------------------------------------------
@pytest.mark.parametrize("cfg", TXT_CFGS)
def test_text_encoder_matches_reference(cfg):
    p = _txt_params(cfg, 1)
    tokens = _captions(2, [1, 4, 7])
    out, ref = np.asarray(_txt(p, tokens, cfg)), _ref_text(p, tokens, cfg[1])
    assert out.shape == ref.shape, f"shape {out.shape} != {ref.shape}"
    assert np.allclose(out, ref, atol=1e-9), (
        f"max err {np.abs(out - ref).max():.2e}. CLIP text: causal mask, ln_final, read the [EOS] position "
        f"(argmax of the token ids), @ W_proj"
    )


@pytest.mark.parametrize("cfg", TXT_CFGS)
def test_text_ignores_everything_after_eos(cfg):
    p = _txt_params(cfg, 3)
    tokens = _captions(4, [2, 5, 1])
    junk = tokens.copy()
    r = np.random.default_rng(5)
    for row in junk:
        e = list(row).index(EOS)
        row[e + 1:] = r.integers(1, SOS, CTX - e - 1)
    a, b = np.asarray(_txt(p, tokens, cfg)), np.asarray(_txt(p, junk, cfg))
    assert np.allclose(a, b, atol=1e-12), (
        "tokens after [EOS] changed the text embedding: you need BOTH a causal mask and pooling at [EOS]"
    )


def test_text_depends_on_every_token_before_eos():
    cfg = TXT_CFGS[0]
    p = _txt_params(cfg, 6)
    tokens = _captions(7, [5])
    base = np.asarray(_txt(p, tokens, cfg))
    for pos in range(1, 6):
        t2 = tokens.copy()
        t2[0, pos] = 1 + t2[0, pos] % (SOS - 2)
        assert np.abs(np.asarray(_txt(p, t2, cfg)) - base).max() > 1e-8, f"token {pos} has no influence"


# ---- loss ------------------------------------------------------------------------------------
SCALE = 1 / 0.07          # exp(log(1/0.07)), below the clip at 100


@pytest.mark.parametrize("n,e", [(6, 5), (5, 3), (9, 8), (2, 2)])
def test_loss_matches_reference(n, e):
    r = np.random.default_rng(n)
    I, T = r.normal(size=(n, e)), r.normal(size=(n, e))
    loss, logits = clip_loss(I, T)
    ref, L = _ref_loss(I, T, math.log(SCALE))
    assert np.allclose(logits, L, atol=1e-10), (
        f"logits[0, 0] = {logits[0, 0]:.4f}, expected {L[0, 0]:.4f}. The logits are exp(logit_scale) * cos(image_i, text_j), "
        f"where logit_scale = log(1/0.07), so the multiplier is 1/0.07 ≈ 14.29 (not 0.07, not 2.66, not 1)"
    )
    assert abs(float(loss) - ref) < 1e-10, (
        f"loss {float(loss):.6f} != {ref:.6f}: mean of CE over rows (image->text) and CE over columns (text->image)"
    )


def test_logit_scale_is_computed_not_passed():
    import inspect
    params = list(inspect.signature(clip_loss).parameters)
    assert params == ["image_emb", "text_emb"], (
        f"clip_loss{tuple(params)}: compute the logit scale inside the function from tau = 0.07, don't take it as input"
    )
    I = np.eye(3)
    _, logits = clip_loss(I, I)
    assert abs(logits[0, 0] - SCALE) < 1e-10, f"identical unit vectors should give logit 1/0.07 = {SCALE:.4f}, got {logits[0, 0]:.4f}"


def test_loss_is_symmetric_and_scale_invariant():
    r = np.random.default_rng(11)
    I, T = r.normal(size=(7, 4)), r.normal(size=(7, 4))
    a, _ = clip_loss(I, T)
    b, _ = clip_loss(T, I)
    assert abs(float(a) - float(b)) < 1e-12, "swapping images and texts must not change the CLIP loss"
    c, _ = clip_loss(I * r.uniform(0.1, 20, (7, 1)), T)
    assert abs(float(a) - float(c)) < 1e-12, "rescaling embeddings changed the loss: L2-normalise each row"


def test_loss_single_pair_is_zero():
    r = np.random.default_rng(12)
    loss, _ = clip_loss(r.normal(size=(1, 4)), r.normal(size=(1, 4)))
    assert abs(float(loss)) < 1e-12, "N=1 has only one candidate, so the loss is 0"
