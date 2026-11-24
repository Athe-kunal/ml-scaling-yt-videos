"""Every formula shown in the lecture, checked numerically on the toy example."""
import numpy as np
import toy
from toy import W, H_MTP, H_NTP, p, q, x


def close(a, b, tol=1e-5):
    return np.allclose(a, b, atol=tol)


def test_ce_logit_gradient_is_p_minus_y():
    z0 = W @ H_MTP
    f = lambda z: float(-(x * (z - np.log(np.exp(z).sum()))).sum())
    assert close(toy.fd_grad(f, z0), p - x)


def test_hard_gradient():
    g = toy.fd_grad(lambda h: toy.ce(x, h), H_MTP)
    assert close(g, toy.g_hard)


def test_decomposition_is_exact():
    assert close(toy.g_hard, toy.g_kl + toy.g_res, 1e-12)


def test_kl_gradient_wrt_mtp_hidden():
    g = toy.fd_grad(lambda h: toy.kl(q, h), H_MTP)
    assert close(g, toy.g_kl)


def test_residual_is_ntp_gradient_at_aligned_state():
    g = toy.fd_grad(lambda h: toy.ce(x, h), H_NTP)
    assert close(g, toy.g_res)


def test_centroid_picture():
    assert close(toy.g_hard, toy.mu_p - toy.w_y, 1e-12)
    assert close(toy.g_kl, toy.mu_p - toy.mu_q, 1e-12)
    assert close(toy.g_res, toy.mu_q - toy.w_y, 1e-12)


def test_straight_through_forward_is_unchanged():
    assert toy.augmented_loss(H_MTP, H_NTP, H_MTP) == toy.ce(x, H_NTP)


def test_straight_through_backward():
    # d/dh_mtp of the augmented NTP loss (sg held constant) = -g_res
    g_mtp = toy.fd_grad(lambda h: toy.augmented_loss(h, H_NTP, H_MTP), H_MTP)
    assert close(g_mtp, -toy.g_res)
    # NTP state still gets its ordinary gradient
    g_ntp = toy.fd_grad(lambda h: toy.augmented_loss(H_MTP, h, H_MTP), H_NTP)
    assert close(g_ntp, toy.g_res)
    # total at the MTP state: hard CE + routed = KL gradient
    total = toy.fd_grad(lambda h: toy.ce(x, h) + toy.augmented_loss(h, H_NTP, H_MTP), H_MTP)
    assert close(total, toy.g_kl)


def test_noise_has_zero_mean():
    mean = sum(toy.PI[i] * toy.g_res_for(i) for i in range(4))
    assert close(mean, toy.ntp_error, 1e-12)


def test_scaling_factor():
    lam, K, T = 1.0, 3, 2048
    for k in range(1, K + 1):
        N0, Nk = T, T - k
        s = lam * N0 / (K * Nk)
        # routed gradient -(s/N0) g_res must equal -(lam/(K Nk)) g_res
        assert abs(s / N0 - lam / (K * Nk)) < 1e-15


def test_no_negative_zero():
    assert toy.f2(-0.0001) == "0.00"


def test_rounded_numbers_on_screen_add_up():
    """The lecture shows 2 decimals; the shown numbers must add up exactly too."""
    r = lambda v: np.round(v, 2)
    assert close(r(toy.g_kl) + r(toy.g_res), r(toy.g_hard), 1e-9)
    assert close(r(toy.mu_p) - r(toy.mu_q), r(toy.g_kl), 1e-9)
    assert close(r(toy.mu_q) - r(toy.w_y), r(toy.g_res), 1e-9)
    assert close(r(toy.mu_p) - r(toy.w_y), r(toy.g_hard), 1e-9)
    assert close(r(toy.g_hard) - r(toy.g_res), r(toy.g_kl), 1e-9)
    assert abs(r(toy.p).sum() - 1) < 1e-9 and abs(r(toy.q).sum() - 1) < 1e-9
