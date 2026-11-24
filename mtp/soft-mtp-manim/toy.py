"""The one toy example every lesson uses: V = 4 tokens, d = 2 hidden dims.

Position t = 3, MTP head k = 1. The MTP head h3^(1) and the aligned NTP state
h4^(0) both predict the same token x5 = "cat". With d = 2 every gradient is an
arrow in the plane, so g_hard = g_KL + g_res can be drawn tip to tail.
"""
import numpy as np

VOCAB = ["the", "cat", "sat", "mat"]
W = np.array([[2.0, 0.4],     # the
              [0.2, 2.0],     # cat
              [-1.9, 0.5],    # sat
              [-0.3, -1.9]])  # mat      shared output head W  [V=4, d=2]
H_MTP = np.array([0.9, 0.2])  # h_3^(1)  (MTP head 1 at t = 3)
H_NTP = np.array([0.3, 0.95]) # h_4^(0)  (NTP state at t + k = 4)
TARGET = 1                    # x_5 = "cat"
PI = np.array([0.22, 0.70, 0.06, 0.02])  # "true" distribution of x_5 (lesson 4)


def softmax(z):
    e = np.exp(z - z.max())
    return e / e.sum()


def onehot(i, n=4):
    v = np.zeros(n)
    v[i] = 1.0
    return v


def ce(y, h):
    """CE(y, softmax(W h)) for a one-hot or soft target y."""
    z = W @ h
    return float(-(y * (z - np.log(np.exp(z).sum()))).sum())


def kl(q, h):
    """KL(q || softmax(W h)) with q held fixed (stop-gradient teacher)."""
    p = softmax(W @ h)
    return float((q * (np.log(q) - np.log(p))).sum())


z_p = W @ H_MTP
z_q = W @ H_NTP
p = softmax(z_p)              # p_3^(1)
q = softmax(z_q)              # q_3^(1) = p_4^(0)
x = onehot(TARGET)            # x_5
mu_p = W.T @ p                # probability-weighted average of W's rows
mu_q = W.T @ q
w_y = W[TARGET]               # = W^T x

g_hard = W.T @ (p - x)
g_kl = W.T @ (p - q)
g_res = W.T @ (q - x)
mu_pi = W.T @ PI
ntp_error = W.T @ (q - PI)    # bias part of g_res


def g_res_for(i):
    """g_res if the sampled target had been token i (noise part changes)."""
    return W.T @ (q - onehot(i))


def fd_grad(f, h, eps=1e-6):
    """Central finite differences: the check that every formula on screen is right."""
    g = np.zeros_like(h)
    for j in range(len(h)):
        e = np.zeros_like(h)
        e[j] = eps
        g[j] = (f(h + e) - f(h - e)) / (2 * eps)
    return g


def augmented_loss(h_mtp, h_ntp, h_const):
    """NTP CE at t+k on h~ = h_ntp + (sg(h_mtp) - h_mtp); sg is the constant h_const."""
    h_tilde = h_ntp + (h_const - h_mtp)
    return ce(x, h_tilde)


def f2(v, n=2):
    """Format a number for the screen (no -0.00)."""
    v = round(float(v), n)
    return f"{0.0 if v == 0 else v:.{n}f}"


def vec(v, n=2):
    return "(" + ",\\ ".join(f2(a, n) for a in v) + ")"
