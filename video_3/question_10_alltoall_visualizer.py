"""Visualizer for Question 10 ("Fun with AllToAll"), part 2, from
https://jax-ml.github.io/scaling-book/sharding/

Setup: an N x N matrix A[I_X, J] is sharded row-wise across D devices arranged
in a single-directional ring (each device can only send to its one forward
neighbor over ICI). Device 0's local shard is itself split into D pieces, one
per destination device — labeled A, B, C, D, ... (piece k is bound for device
k). Unlike an AllGather, where every device eventually needs a copy of every
shard (so each piece circulates the *entire* ring), an AllToAll piece only
needs to travel as far as its destination and then stops — so the pieces
bound for nearby devices "get off the bus" early.

Book's four-device example: device 0 holds [A, B, C, D].
  Hop 1: device 0 -> device 1 forwards B, C, D (keeps A — already home).
  Hop 2: device 1 -> device 2 forwards C, D (keeps B — now home).
  Hop 3: device 2 -> device 3 forwards D (keeps C — now home).
  D arrives at device 3 — now home.
This yields N^2(D-1)/(2D) scalars per link (~N^2/2 for large D), half of the
AllGather cost N^2(D-1)/D (~N^2) — because each piece only rides part of the
ring instead of the whole thing.
"""

import math

import matplotlib.pyplot as plt
import streamlit as st
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch

st.set_page_config(page_title="Question 10 — Fun with AllToAll", page_icon="⇄", layout="wide")

# ------------------------------------------------------------------ theme ---
BG = "#0A0E14"
PANEL_BG = "#111826"
CARD_BG = "#161F2E"
INK = "#EDF1F7"
INK_SOFT = "#8E9BAF"
RULE = "#26314A"
ACCENT = "#5EEAD4"
ACCENT2 = "#F472B6"
COMM_COLOR = "#FBBF24"
IDLE_EDGE = "#2B3648"
SHADOW = "#00000055"
FIG_DPI = 210
LETTER_COLORS = ["#5EEAD4", "#F472B6", "#FBBF24", "#818CF8", "#34D399", "#FB923C", "#38BDF8", "#F87171"]

plt.rcParams["font.family"] = "monospace"
plt.rcParams["axes.edgecolor"] = RULE
plt.rcParams["savefig.facecolor"] = BG

st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans+Condensed:wght@600;700&family=IBM+Plex+Sans:wght@400;500&display=swap');
    html, body, [class*="css"] {{ font-family: 'IBM Plex Sans', sans-serif; -webkit-font-smoothing: antialiased; }}
    .stApp {{
        background:
            radial-gradient(1100px 550px at 12% -8%, rgba(94,234,212,0.07), transparent),
            radial-gradient(900px 500px at 100% 0%, rgba(244,114,182,0.05), transparent),
            {BG};
        color: {INK};
    }}
    .block-container {{ padding-top: 2.2rem; max-width: 1300px; }}
    h1, h2, h3 {{ font-family: 'IBM Plex Sans Condensed', sans-serif !important; letter-spacing: 0.02em; }}

    .hero-title {{
        font-family: 'IBM Plex Sans Condensed', sans-serif; font-weight: 700;
        font-size: clamp(26px, 3vw, 38px); letter-spacing: 0.01em; margin: 0;
        background: linear-gradient(100deg, {INK} 30%, {ACCENT});
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }}
    .hero-sub {{ font-family: 'IBM Plex Mono', monospace; font-size: 13px; color: {INK_SOFT}; margin-top: 6px; }}

    .case-box, .step-desc, .eq-box {{
        margin-top: 14px; padding: 16px 20px; border-radius: 14px; border: 1px solid {RULE};
        background: linear-gradient(180deg, {CARD_BG}, {PANEL_BG});
        box-shadow: 0 1px 0 rgba(255,255,255,0.04) inset, 0 10px 28px -12px {SHADOW};
    }}
    .case-box {{ font-family: 'IBM Plex Mono', monospace; font-size: 14px; color: {INK}; line-height: 1.55; }}
    .case-box b {{ color: {ACCENT}; }}
    .step-desc {{ font-family: 'IBM Plex Sans', sans-serif; font-size: 14.5px; color: {INK}; line-height: 1.6; }}
    .step-desc b {{ color: {ACCENT}; }}
    .eq-box .katex {{ color: {INK} !important; font-size: 1.05em; }}
    .eq-box [data-testid="stElementContainer"] {{ margin-bottom: 0 !important; }}

    .comm-tag {{
        display: inline-block; font-family: 'IBM Plex Mono', monospace; font-size: 10.5px;
        font-weight: 600; letter-spacing: 0.09em; text-transform: uppercase;
        padding: 4px 10px; border-radius: 999px; margin-bottom: 10px;
        background: rgba(251,191,36,0.12); color: {COMM_COLOR}; border: 1px solid rgba(251,191,36,0.35);
    }}
    .comm-none {{ background: rgba(94,234,212,0.10); color: {ACCENT}; border-color: rgba(94,234,212,0.32); }}

    .mono-table {{ width: 100%; border-collapse: collapse; font-family: 'IBM Plex Mono', monospace;
                    font-size: 12.5px; color: {INK}; margin-top: 12px; }}
    .mono-table th, .mono-table td {{ padding: 8px 12px; border-bottom: 1px solid {RULE}; text-align: left; }}
    .mono-table th {{ color: {INK_SOFT}; font-weight: 600; text-transform: uppercase; font-size: 10.5px; letter-spacing: 0.05em; }}

    .winner-badge {{
        display: inline-block; font-family: 'IBM Plex Mono', monospace; font-size: 11px; font-weight: 700;
        letter-spacing: 0.06em; text-transform: uppercase; padding: 3px 10px; border-radius: 999px;
        background: rgba(94,234,212,0.14); color: {ACCENT}; border: 1px solid rgba(94,234,212,0.4);
    }}

    div[data-testid="stButton"] button {{
        font-family: 'IBM Plex Mono', monospace; font-weight: 500; border-radius: 10px;
        border: 1px solid {RULE}; background: {CARD_BG}; transition: border-color 0.15s ease, color 0.15s ease;
    }}
    div[data-testid="stButton"] button:hover:not(:disabled) {{ border-color: {ACCENT}; color: {ACCENT}; }}
    div[data-testid="stButton"] button:disabled {{ opacity: 0.35; }}
    div[data-testid="stImage"] img {{ border-radius: 16px; }}
    code {{ color: {ACCENT}; }}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<p class="hero-title">Question 10 — Fun with AllToAll (part 2)</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="hero-sub">Single-directional ICI ring — why AllToAll only ships each shard as far as it '
    'needs to go, instead of all the way around like AllGather. Based on jax-ml.github.io/scaling-book/sharding/</p>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="case-box"><b>A[I_X, J]</b> is an N&times;N matrix sharded row-wise across D devices in a ring, '
    "single-directional ICI (each device can only send to its one forward neighbor). Device 0's local shard is "
    "itself split into D pieces — one per destination device, labeled A, B, C, D, ... (piece <b>k</b> is bound for "
    "device <b>k</b>). Step through the hops below to watch where each piece is.</div>",
    unsafe_allow_html=True,
)


# --------------------------------------------------------------- formulas ---
def render_equations(lines):
    st.markdown('<div class="eq-box">', unsafe_allow_html=True)
    st.latex(r"\begin{aligned}" + r"\\[4pt] ".join(lines) + r"\end{aligned}")
    st.markdown("</div>", unsafe_allow_html=True)


def fmt_num(x):
    if x >= 1e9:
        return f"{x / 1e9:.2f}B"
    if x >= 1e6:
        return f"{x / 1e6:.2f}M"
    if x >= 1e3:
        return f"{x / 1e3:.2f}K"
    return f"{x:.1f}"


def step_nav(key, n_steps):
    if key not in st.session_state:
        st.session_state[key] = 0
    st.session_state[key] = max(0, min(st.session_state[key], n_steps - 1))
    c1, c2, c3 = st.columns([1, 3, 1])
    with c1:
        if st.button("◀ Previous", key=key + "_prev", disabled=st.session_state[key] == 0, use_container_width=True):
            st.session_state[key] -= 1
    with c3:
        if st.button("Next ▶", key=key + "_next", disabled=st.session_state[key] == n_steps - 1, use_container_width=True):
            st.session_state[key] += 1
    st.session_state[key] = max(0, min(st.session_state[key], n_steps - 1))
    with c2:
        dots = "&nbsp;".join(
            f'<span style="color:{ACCENT};">●</span>' if i == st.session_state[key] else f'<span style="color:{RULE};">●</span>'
            for i in range(n_steps)
        )
        st.markdown(
            f'<p style="text-align:center; font-size:16px; margin-top:8px;">{dots}</p>'
            f'<p style="text-align:center; font-family:\'IBM Plex Mono\',monospace; font-size:11px; color:{INK_SOFT};">'
            f'step {st.session_state[key] + 1} / {n_steps}</p>',
            unsafe_allow_html=True,
        )
    return st.session_state[key]


def rrect(ax, x0, y0, w, h, r=None, **kw):
    if r is None:
        r = max(min(w, h) * 0.14, 0.02)
    ax.add_patch(FancyBboxPatch((x0, y0), w, h, boxstyle=f"round,pad=0,rounding_size={r}", **kw))


# --------------------------------------------------------------- ring viz ---
def device_pos(i, D, R):
    angle = math.pi / 2 - i * (2 * math.pi / D)  # start at top, go clockwise
    return R * math.cos(angle), R * math.sin(angle)


def render_ring(D, holder, active_edge, letters, hop_bytes):
    R = 3.2
    r_dev = max(0.32, min(0.62, R * math.sin(math.pi / D) * 0.62))
    fig, ax = plt.subplots(figsize=(6.6, 6.6), dpi=FIG_DPI, facecolor=BG)
    ax.set_facecolor(BG)

    positions = [device_pos(i, D, R) for i in range(D)]

    # Ring edges — full physical ring drawn faint; the forward chain (0->1->...->D-1)
    # is the only direction ever used in this single-directional example.
    for i in range(D):
        j = (i + 1) % D
        x0, y0 = positions[i]
        x1, y1 = positions[j]
        dx, dy = x1 - x0, y1 - y0
        dist = math.hypot(dx, dy)
        ux, uy = dx / dist, dy / dist
        sx, sy = x0 + ux * r_dev, y0 + uy * r_dev
        ex, ey = x1 - ux * r_dev, y1 - uy * r_dev
        is_forward_chain = i < D - 1
        is_active = active_edge is not None and i == active_edge
        if is_active:
            color, lw, alpha = COMM_COLOR, 2.8, 1.0
        elif is_forward_chain:
            color, lw, alpha = "#3A4B63", 1.6, 0.85
        else:
            color, lw, alpha = IDLE_EDGE, 1.2, 0.45
        style = "-" if is_forward_chain else (0, (3, 3))
        arrow = FancyArrowPatch(
            (sx, sy), (ex, ey), arrowstyle="-|>", mutation_scale=16 if is_active else 11,
            color=color, linewidth=lw, alpha=alpha, linestyle=style,
            connectionstyle="arc3,rad=0.12", zorder=4 if is_active else 2,
        )
        ax.add_patch(arrow)
        if is_active and hop_bytes is not None:
            mx, my = (sx + ex) / 2, (sy + ey) / 2
            nx, ny = -uy, ux  # outward normal-ish offset for label placement
            ax.text(mx + nx * 0.55, my + ny * 0.55, f"{fmt_num(hop_bytes)} scalars",
                    ha="center", va="center", fontsize=8.6, color=COMM_COLOR,
                    fontweight="bold", fontfamily="monospace", zorder=5)

    # Devices + the tokens (pieces) each one currently holds.
    for i in range(D):
        x, y = positions[i]
        held = [k for k in range(D) if holder[k] == i]
        is_active_device = active_edge is not None and i in (active_edge, active_edge + 1)
        ring_color = COMM_COLOR if is_active_device else RULE
        ax.add_patch(Circle((x, y), r_dev, facecolor=CARD_BG, edgecolor=ring_color,
                             linewidth=2.2 if is_active_device else 1.3, zorder=6))
        ax.text(x, y - r_dev - 0.28, f"device {i}", ha="center", va="top", fontsize=9.5,
                color=INK_SOFT, fontfamily="monospace", fontweight="bold", zorder=6)

        n_held = max(len(held), 1)
        chip = min(0.30, r_dev * 1.15 / n_held)
        total_w = len(held) * chip + max(len(held) - 1, 0) * (chip * 0.18)
        start_x = x - total_w / 2
        for idx, k in enumerate(held):
            cx = start_x + idx * (chip * 1.18) + chip / 2
            color = LETTER_COLORS[k % len(LETTER_COLORS)]
            home = " (home)" if holder[k] == k else ""
            rrect(ax, cx - chip / 2, y - chip / 2, chip, chip, r=chip * 0.22,
                  facecolor=color, edgecolor="#08131A" if not home else "#08131A",
                  linewidth=1.4 if home else 0, zorder=8)
            ax.text(cx, y, letters[k], ha="center", va="center", fontsize=9.5 if chip > 0.22 else 7.5,
                    color="#08131A", fontweight="bold", fontfamily="monospace", zorder=9)

    pad = 1.15
    ax.set_xlim(-R - pad, R + pad)
    ax.set_ylim(-R - pad, R + pad)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.tight_layout()
    return fig


# ------------------------------------------------------------------- app ---
c1, c2 = st.columns(2)
with c1:
    N = st.number_input("N (matrix is N × N)", 256, 65536, 4096, step=256, key="N10")
with c2:
    D = st.slider("D (devices in the ring)", 4, 8, 4, key="D10")

letters = [chr(ord("A") + k) for k in range(D)]
piece_size = (N * N) / (D * D)  # elements per piece — each of device 0's D sub-shards

render_equations([
    r"\text{AllGather: each shard rides the whole ring} \Rightarrow \dfrac{N^2(D-1)}{D}\ \text{scalars / link}",
    r"\text{AllToAll: piece }k\text{ only rides }k\text{ hops} \Rightarrow \dfrac{N^2(D-1)}{2D}\ \text{scalars / link}",
])

# Simulate the D-1 hops: holder[k] = current device index of piece k (target device k).
history = [[0] * D]
holder = [0] * D
for hop in range(1, D):
    holder = [h + 1 if h < k else h for h, k in zip(holder, range(D))]
    history.append(list(holder))

step = step_nav("q10_step", D)
holder_now = history[step]
active_edge = step - 1 if step >= 1 else None

if step == 0:
    desc = (
        f"<b>Initial state.</b> Device 0 holds all {D} pieces of its local shard — "
        f"{', '.join(letters)} — one per destination device. Nothing has moved yet."
    )
    comm_label = None
    hop_bytes = None
else:
    moving = [letters[k] for k in range(D) if k >= step]
    just_arrived = letters[step - 1]
    hop_bytes = piece_size * (D - step)
    desc = (
        f"<b>Hop {step}: device {step - 1} → device {step}.</b> "
        f"Forwards {', '.join(moving)} onward. Shard <b>{just_arrived}</b> stops here — "
        f"device {step - 1} is its final home, so it never needs to move again."
    )
    comm_label = f"hop {step} of {D - 1}"

tag_html = (
    f'<span class="comm-tag">{comm_label}</span><br>' if comm_label
    else '<span class="comm-tag comm-none">initial</span><br>'
)
st.markdown(f'<div class="step-desc">{tag_html}{desc}</div>', unsafe_allow_html=True)

st.pyplot(render_ring(D, holder_now, active_edge, letters, hop_bytes), use_container_width=False)

st.markdown(
    f'<p style="font-family:\'IBM Plex Mono\',monospace; font-size:12px; color:{INK_SOFT};">'
    f'each piece is N&sup2;/D&sup2; = {fmt_num(piece_size)} scalars &nbsp;&nbsp;|&nbsp;&nbsp; '
    + (f"this hop moves {fmt_num(hop_bytes)} scalars ({D - step} piece{'s' if D - step != 1 else ''} still in flight)"
       if hop_bytes is not None else "no data has moved yet")
    + "</p>",
    unsafe_allow_html=True,
)

# ------------------------------------------------------------------ totals ---
allgather_total = N * N * (D - 1) / D
alltoall_total = N * N * (D - 1) / (2 * D)
st.markdown(
    f'<table class="mono-table"><tr><th>collective</th><th>scalars / link</th><th>vs. N&sup2;</th></tr>'
    f'<tr><td>AllGather (part 1)</td><td>{fmt_num(allgather_total)}</td><td>{allgather_total / (N * N):.3f} × N&sup2;</td></tr>'
    f'<tr><td>AllToAll (part 2)</td><td>{fmt_num(alltoall_total)}</td><td>{alltoall_total / (N * N):.3f} × N&sup2;</td></tr>'
    f"</table>",
    unsafe_allow_html=True,
)
st.markdown(
    f'<p style="margin-top:10px;"><span class="winner-badge">AllToAll moves {allgather_total / alltoall_total:.1f}× less</span>'
    f'&nbsp;&nbsp;<span style="font-family:\'IBM Plex Mono\',monospace; font-size:12.5px; color:{INK_SOFT};">'
    "per link than AllGather here — because each piece only rides as far as its own destination (a triangular "
    "1+2+&hellip;+(D&minus;1) sum) instead of circulating the entire ring D&minus;1 times over. The diagram above "
    "only tracks device 0's own pieces for clarity — every device is doing the identical thing with its own "
    "pieces at the same time, and by the ring's symmetry every link ends up carrying this same total.</span></p>",
    unsafe_allow_html=True,
)
