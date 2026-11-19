import math

import numpy as np
import pytest

from solution import vit_forward

# (B, H, W, C, P, D, heads, layers, classes)
CONFIGS = [
    (2, 8, 8, 3, 4, 8, 2, 2, 5),
    (3, 12, 8, 1, 4, 12, 3, 1, 4),    # non-square image, single channel
    (1, 10, 10, 3, 4, 8, 4, 3, 3),    # 10 is not divisible by 4: crop to 8x8
]


def _params(cfg, seed):
    B, H, W, C, P, D, heads, layers, K = cfg
    r = np.random.default_rng(seed)
    N = (H // P) * (W // P)
    blocks = [{k: r.normal(0, D ** -0.5, (D, D)) for k in ("Wq", "Wk", "Wv", "Wo")} for _ in range(layers)]
    for b in blocks:
        b["W1"] = r.normal(0, D ** -0.5, (D, 4 * D))
        b["W2"] = r.normal(0, (4 * D) ** -0.5, (4 * D, D))
    return dict(
        image=r.normal(size=(B, H, W, C)),
        W_patch=r.normal(0, (P * P * C) ** -0.5, (P * P * C, D)),
        patch_bias=r.normal(0, 0.1, D),
        cls_token=r.normal(size=D),
        pos_embed=r.normal(0, 0.5, (N + 1, D)),
        blocks=blocks,
        W_head=r.normal(0, D ** -0.5, (D, K)),
    )


# ---- independent, loop-based reference -------------------------------------------------------
def _ln(v, eps=1e-6):
    return (v - v.mean()) / math.sqrt(v.var() + eps)


def _gelu(v):
    return 0.5 * v * (1 + np.tanh(math.sqrt(2 / math.pi) * (v + 0.044715 * v ** 3)))


def _ref_block(x, heads, b):            # x: (N, D), one image
    N, D = x.shape
    dh = D // heads
    h = np.stack([_ln(t) for t in x])
    Q, K, V = h @ b["Wq"], h @ b["Wk"], h @ b["Wv"]
    out = np.zeros_like(x)
    for hd in range(heads):
        s = slice(hd * dh, (hd + 1) * dh)
        sc = Q[:, s] @ K[:, s].T / math.sqrt(dh)
        sc = np.exp(sc - sc.max(1, keepdims=True))
        out[:, s] = (sc / sc.sum(1, keepdims=True)) @ V[:, s]
    y = x + out @ b["Wo"]
    return y + _gelu(np.stack([_ln(t) for t in y]) @ b["W1"]) @ b["W2"]


def _ref_forward(p, P, heads):
    outs = []
    for img in p["image"]:
        hp, wp = img.shape[0] // P, img.shape[1] // P
        toks = [img[i * P:(i + 1) * P, j * P:(j + 1) * P, :].reshape(-1) for i in range(hp) for j in range(wp)]
        x = np.stack(toks) @ p["W_patch"] + p["patch_bias"]
        x = np.vstack([p["cls_token"], x]) + p["pos_embed"]
        for b in p["blocks"]:
            x = _ref_block(x, heads, b)
        outs.append(_ln(x[0]) @ p["W_head"])
    return np.stack(outs)


def _run(p, P, heads):
    return vit_forward(p["image"], P, heads, p["W_patch"], p["patch_bias"], p["cls_token"],
                       p["pos_embed"], p["blocks"], p["W_head"])


@pytest.mark.parametrize("cfg", CONFIGS)
def test_matches_reference(cfg):
    p = _params(cfg, seed=0)
    out = np.asarray(_run(p, cfg[4], cfg[6]))
    ref = _ref_forward(p, cfg[4], cfg[6])
    assert out.shape == ref.shape, f"output shape {out.shape} != {ref.shape}"
    assert np.allclose(out, ref, atol=1e-9), (
        f"max err {np.abs(out - ref).max():.2e}. Check: patch order (row-major patches, each flattened as "
        f"(row, col, channel)), CLS at position 0, pre-LN blocks, attention scaling, head = LN(CLS) @ W_head."
    )


def test_patch_pixel_order():
    cfg = CONFIGS[0]
    p = _params(cfg, seed=1)
    out = np.asarray(_run(p, cfg[4], cfg[6]))
    ref = _ref_forward(p, cfg[4], cfg[6])
    # Swapping two pixels INSIDE one patch must matter (W_patch is position-specific).
    p2 = dict(p, image=p["image"].copy())
    p2["image"][:, 0, 1, :], p2["image"][:, 1, 0, :] = p["image"][:, 1, 0, :], p["image"][:, 0, 1, :]
    assert not np.allclose(_run(p2, cfg[4], cfg[6]), out), "pixel layout within a patch is ignored"
    assert np.allclose(_run(p2, cfg[4], cfg[6]), _ref_forward(p2, cfg[4], cfg[6]), atol=1e-9), (
        "a patch must be flattened as (row, col, channel): reshape to (B, hp, P, wp, P, C) then move the two "
        "patch-grid axes in front of the within-patch axes"
    )
    assert np.allclose(out, ref, atol=1e-9)


def test_images_in_a_batch_are_independent():
    cfg = CONFIGS[0]
    p = _params(cfg, seed=2)
    a = np.asarray(_run(p, cfg[4], cfg[6]))
    p2 = dict(p, image=p["image"].copy())
    p2["image"][1] = np.random.default_rng(9).normal(size=p["image"][1].shape)
    b = np.asarray(_run(p2, cfg[4], cfg[6]))
    assert np.allclose(a[0], b[0], atol=1e-12), "changing image 1 changed image 0's logits: mixing across the batch axis"
    assert not np.allclose(a[1], b[1])


def test_does_not_mutate_inputs():
    cfg = CONFIGS[1]
    p = _params(cfg, seed=3)
    img = p["image"].copy()
    _run(p, cfg[4], cfg[6])
    assert np.array_equal(img, p["image"]), "vit_forward modified the input image"
