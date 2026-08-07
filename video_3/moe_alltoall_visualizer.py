"""Mixture-of-Experts routing/combine, and how AllToAll implements it across devices.

Part 1 — MoE on one conceptual device: a router picks top-k experts per token
(gate weights via softmax), each chosen expert runs its MLP, and the outputs are
scattered back to each token's original position and combined with a **gate-weighted
sum** — never concatenation, since every expert emits the same d_model width.

Part 2 — Expert parallelism: tokens live sharded across devices by sequence position,
experts live sharded across devices one-per-device. The router's decision doesn't move
any data by itself — AllToAll is the communication primitive that actually ships each
token to the device holding its assigned expert (dispatch), and ships the results back
(combine), matching the AllToAll primitive from jax-ml.github.io/scaling-book/sharding/:
[A, B_X] -> [A_X, B], cost V / (4 . W_ici).
"""

import colorsys
import math
import random

import matplotlib.pyplot as plt
import streamlit as st
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch

st.set_page_config(page_title="MoE + AllToAll", page_icon="⇄", layout="wide")

# ------------------------------------------------------------------ theme ---
BG = "#0A0E14"
PANEL_BG = "#111826"
CARD_BG = "#161F2E"
DEVICE_BG = "#1A2436"
INK = "#EDF1F7"
INK_SOFT = "#8E9BAF"
RULE = "#26314A"
ACCENT = "#5EEAD4"
AXIS_COLOR = {"x": "#5EEAD4", "y": "#F472B6"}
COMM_COLOR = "#FBBF24"
INVALID_COLOR = "#FB7185"
SHADOW = "#00000055"
FIG_DPI = 220

plt.rcParams["font.family"] = "monospace"
plt.rcParams["text.antialiased"] = True
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
    .block-container {{ padding-top: 2.2rem; max-width: 1400px; }}
    section[data-testid="stSidebar"] {{ background: {PANEL_BG}; border-right: 1px solid {RULE}; }}
    h1, h2, h3 {{ font-family: 'IBM Plex Sans Condensed', sans-serif !important; letter-spacing: 0.02em; }}

    .hero-title {{
        font-family: 'IBM Plex Sans Condensed', sans-serif; font-weight: 700;
        font-size: clamp(26px, 3vw, 38px); letter-spacing: 0.01em; margin: 0;
        background: linear-gradient(100deg, {INK} 30%, {ACCENT});
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }}
    .hero-sub {{ font-family: 'IBM Plex Mono', monospace; font-size: 13px; color: {INK_SOFT}; margin-top: 6px; }}

    .case-box {{
        margin-top: 14px; padding: 16px 20px; border-radius: 14px; border: 1px solid {RULE};
        background: linear-gradient(180deg, {CARD_BG}, {PANEL_BG});
        box-shadow: 0 1px 0 rgba(255,255,255,0.04) inset, 0 10px 28px -12px {SHADOW};
        font-family: 'IBM Plex Mono', monospace; font-size: 14px; color: {INK}; line-height: 1.55;
    }}
    .case-box b {{ color: {ACCENT}; }}

    .step-desc {{
        margin-top: 14px; padding: 16px 20px; border-radius: 14px; border: 1px solid {RULE};
        background: linear-gradient(180deg, {CARD_BG}, {PANEL_BG});
        box-shadow: 0 1px 0 rgba(255,255,255,0.04) inset, 0 10px 28px -14px {SHADOW};
        font-family: 'IBM Plex Sans', sans-serif; font-size: 14.5px; color: {INK}; line-height: 1.6;
    }}
    .step-desc b {{ color: {ACCENT}; }}

    .comm-tag {{
        display: inline-block; font-family: 'IBM Plex Mono', monospace; font-size: 10.5px;
        font-weight: 600; letter-spacing: 0.09em; text-transform: uppercase;
        padding: 4px 10px; border-radius: 999px; margin-bottom: 10px;
    }}
    .comm-none {{ background: rgba(94,234,212,0.10); color: {ACCENT}; border: 1px solid rgba(94,234,212,0.32); }}
    .comm-yes {{ background: rgba(251,191,36,0.12); color: {COMM_COLOR}; border: 1px solid rgba(251,191,36,0.35); }}
    .comm-invalid {{ background: rgba(251,113,133,0.12); color: {INVALID_COLOR}; border: 1px solid rgba(251,113,133,0.38); }}

    div[data-testid="stButton"] button {{
        font-family: 'IBM Plex Mono', monospace; font-weight: 500; border-radius: 10px;
        border: 1px solid {RULE}; background: {CARD_BG}; transition: border-color 0.15s ease, color 0.15s ease;
    }}
    div[data-testid="stButton"] button:hover:not(:disabled) {{ border-color: {ACCENT}; color: {ACCENT}; }}
    div[data-testid="stButton"] button:disabled {{ opacity: 0.35; }}

    .stTabs [data-baseweb="tab-list"] {{ gap: 6px; border-bottom: 1px solid {RULE}; }}
    .stTabs [data-baseweb="tab"] {{
        font-family: 'IBM Plex Mono', monospace; font-size: 12.5px; color: {INK_SOFT};
        padding: 10px 18px; border-radius: 10px 10px 0 0; background: transparent;
    }}
    .stTabs [aria-selected="true"] {{
        color: {ACCENT} !important; background: {CARD_BG};
        box-shadow: inset 0 -2px 0 {ACCENT};
    }}
    div[data-testid="stImage"] img {{ border-radius: 16px; }}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<p class="hero-title">Mixture of Experts &rarr; AllToAll</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="hero-sub">How token routing and expert combine work, and how AllToAll implements '
    'them across real devices.</p>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="case-box"><b>Short answer to "concat or reduce?"</b> — neither, exactly. Each token\'s '
    "final output is a <b>scatter</b> back to its original sequence position, combined across its chosen "
    "experts with a <b>gate-weighted sum</b>. Concatenation isn't even an option: every expert's MLP outputs "
    "the same d_model width as the input, precisely so the outputs <i>can</i> be summed into one vector.</div>",
    unsafe_allow_html=True,
)

# --------------------------------------------------------------- helpers ---
def chunk_color(k, n):
    hue = (k / max(n, 1)) * 0.80 + 0.03
    return colorsys.hls_to_rgb(hue % 1.0, 0.60, 0.62)


def to_hex(rgb):
    return "#{:02x}{:02x}{:02x}".format(*(round(c * 255) for c in rgb))


def color_legend(colors, prefix, note=""):
    swatches = "".join(
        f'<span style="display:inline-flex; align-items:center; gap:5px; margin-right:14px;">'
        f'<span style="width:11px; height:11px; border-radius:3px; background:{to_hex(c)}; display:inline-block;"></span>'
        f'<span style="color:{INK_SOFT};">{prefix}{i}</span></span>'
        for i, c in enumerate(colors)
    )
    note_html = f' <span style="color:{INK_SOFT}; opacity:0.75;">— {note}</span>' if note else ""
    st.markdown(
        f'<div style="font-family:\'IBM Plex Mono\',monospace; font-size:11.5px; '
        f'margin: 10px 0 2px; display:flex; flex-wrap:wrap; align-items:center;">{swatches}{note_html}</div>',
        unsafe_allow_html=True,
    )


def rrect(ax, x0, y0, w, h, r=None, shadow=False, **kw):
    if r is None:
        r = max(min(w, h) * 0.16, 0.02)
    r = min(r, min(w, h) / 2 - 1e-4) if min(w, h) > 2e-4 else 0
    if shadow:
        ax.add_patch(FancyBboxPatch(
            (x0 + 0.03, y0 - 0.03), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
            linewidth=0, facecolor="black", alpha=0.22, zorder=kw.get("zorder", 2) - 0.2,
        ))
    ax.add_patch(FancyBboxPatch((x0, y0), w, h, boxstyle=f"round,pad=0,rounding_size={r}", **kw))


def draw_operator(ax, cx, cy, symbol, r=0.14, fontsize=11):
    ax.add_patch(Circle((cx, cy), r, facecolor=CARD_BG, edgecolor=RULE, linewidth=1.1, zorder=6))
    ax.text(cx, cy, symbol, ha="center", va="center", fontsize=fontsize, color=INK_SOFT, zorder=7)


def curve(ax, p0, p1, color, lw=1.4, alpha=0.85, rad=0.25, arrow=True, z=3):
    style = "-|>" if arrow else "-"
    ax.add_patch(FancyArrowPatch(
        p0, p1, connectionstyle=f"arc3,rad={rad}", arrowstyle=style,
        color=color, linewidth=lw, alpha=alpha, mutation_scale=10, zorder=z,
    ))


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


def comm_tag(comm):
    if comm is None:
        return '<span class="comm-tag comm-none">no communication</span>'
    return f'<span class="comm-tag comm-yes">{comm}</span>'


def route_tokens(n_tokens, n_experts, top_k, seed):
    """routing[t] = [(expert_id, gate_weight), ...] length top_k, weights sum to 1."""
    rng = random.Random(seed)
    routing = []
    for _ in range(n_tokens):
        chosen = rng.sample(range(n_experts), min(top_k, n_experts))
        logits = [rng.uniform(0.0, 2.2) for _ in chosen]
        m = max(logits)
        exps = [math.exp(l - m) for l in logits]
        s = sum(exps)
        weights = [e / s for e in exps]
        routing.append(list(zip(chosen, weights)))
    return routing


# =========================================================== TOKEN CHIP ===
def draw_chip(ax, x, y, w, h, label, fill, border, alpha=0.95, text_color=None, fontsize=8.5, shadow=True, lw=2.0):
    rrect(ax, x, y, w, h, facecolor=fill, edgecolor=border, linewidth=lw, alpha=alpha, zorder=3, shadow=shadow)
    ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=fontsize,
            color=text_color or INK, fontweight="bold", fontfamily="monospace", zorder=4)


# ======================================================= PART 1: routing ===
def render_routing_step(step, tokens, n_experts, expert_colors, top_k):
    n = len(tokens)
    tw, th, gap = 0.62, 0.55, 0.28
    total_w = (n - 1) * (tw + gap) + tw if n > 1 else tw

    buckets = {e: [] for e in range(n_experts)}
    for i, t in enumerate(tokens):
        for eid, wgt in t:
            buckets[eid].append((i, wgt))
    max_count = max(1, max((len(b) for b in buckets.values()), default=1))
    slot_h = 0.62

    tok_x = [i * (tw + gap) for i in range(n)]
    eb_w, eb_h = 0.85, 0.6
    slot_w = total_w / n_experts if n_experts else total_w
    expert_x = [e * slot_w + slot_w / 2 - eb_w / 2 for e in range(n_experts)]

    fig_w = max(7.5, total_w * 1.05 + 2.4)
    if step in (1, 2):
        fig_h = max(4.4, 1.1 + max_count * slot_h + 0.9)
    elif step == 3:
        max_terms = max((len(t) for t in tokens), default=1)
        fig_h = 4.6 + max_terms * 0.32
    else:
        fig_h = 4.4
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=FIG_DPI, facecolor=BG)
    ax.set_facecolor(BG)

    y_tok = 3.35
    y_router = 2.35
    y_expert = 0.55

    def token_center(i):
        return tok_x[i] + tw / 2, y_tok + th / 2

    def expert_top(e):
        return expert_x[e] + eb_w / 2, y_expert + eb_h

    # ---- step 0: tokens + router + routing arrows ----
    if step == 0:
        for e in range(n_experts):
            rrect(ax, expert_x[e], y_expert, eb_w, eb_h, facecolor="none",
                  edgecolor=expert_colors[e], linewidth=1.6, alpha=0.9, zorder=2)
            ax.text(expert_x[e] + eb_w / 2, y_expert + eb_h / 2, f"E{e}", ha="center", va="center",
                    fontsize=9.5, color=expert_colors[e], fontweight="bold", fontfamily="monospace", zorder=3)

        for i, t in enumerate(tokens):
            tc = token_center(i)
            for eid, wgt in t:
                curve(ax, (tc[0], y_tok), expert_top(eid), expert_colors[eid], lw=1.1 + 2.2 * wgt,
                      alpha=0.55 + 0.35 * wgt, rad=0.15 if eid >= i else -0.15)

        rrect(ax, -0.15, y_router, total_w + 0.3, 0.55, facecolor=PANEL_BG, edgecolor=RULE, linewidth=1.2, zorder=1)
        ax.text(total_w / 2, y_router + 0.275, "router — softmax(W·x), pick top-{}".format(min(top_k, n_experts)),
                ha="center", va="center", fontsize=8.6, color=INK_SOFT, fontfamily="monospace", zorder=2)

        for i, t in enumerate(tokens):
            x, y = tok_x[i], y_tok
            rrect(ax, x, y, tw, th, facecolor=CARD_BG, edgecolor=RULE, linewidth=1.3, zorder=3, shadow=True)
            ax.text(x + tw / 2, y + th / 2, f"t{i}", ha="center", va="center", fontsize=8.5,
                    color=INK, fontweight="bold", fontfamily="monospace", zorder=4)
        ax.text(-0.35, y_tok + th / 2, "tokens", ha="right", va="center", fontsize=9, color=INK_SOFT, fontfamily="monospace")

    # ---- step 1: dispatch — grouped into expert buckets ----
    elif step == 1:
        bucket_h = max_count * slot_h + 0.3
        base_y = 0.35
        for e in range(n_experts):
            bx = expert_x[e]
            rrect(ax, bx - 0.05, base_y, eb_w + 0.1, bucket_h, facecolor=PANEL_BG, edgecolor=expert_colors[e],
                  linewidth=1.4, alpha=0.35, zorder=1)
            ax.text(bx + eb_w / 2, base_y + bucket_h + 0.15, f"E{e} bucket", ha="center", va="bottom", fontsize=8.5,
                    color=expert_colors[e], fontweight="bold", fontfamily="monospace", zorder=2)
            for j, (i, wgt) in enumerate(buckets[e]):
                cy = base_y + 0.2 + j * slot_h
                draw_chip(ax, bx, cy, eb_w, 0.5, f"t{i}", CARD_BG, expert_colors[e], fontsize=8)
        ax.set_xlim(-1.1, total_w + 0.3)
        ax.set_ylim(-0.3, base_y + bucket_h + 0.7)

    # ---- step 2: local expert MLP compute ----
    elif step == 2:
        bucket_h = max_count * slot_h + 0.3
        base_y = 0.35
        header_h = 0.5
        for e in range(n_experts):
            bx = expert_x[e]
            rrect(ax, bx - 0.05, base_y, eb_w + 0.1, bucket_h, facecolor=expert_colors[e], edgecolor=expert_colors[e],
                  linewidth=1.4, alpha=0.14, zorder=1)
            rrect(ax, bx - 0.05, base_y + bucket_h + 0.15, eb_w + 0.1, header_h, facecolor=expert_colors[e],
                  edgecolor="none", alpha=0.85, zorder=2)
            ax.text(bx + eb_w / 2, base_y + bucket_h + 0.15 + header_h / 2, f"E{e} MLP", ha="center", va="center",
                    fontsize=8.2, color="#08131A", fontweight="bold", fontfamily="monospace", zorder=3)
            for j, (i, wgt) in enumerate(buckets[e]):
                cy = base_y + 0.2 + j * slot_h
                draw_chip(ax, bx, cy, eb_w, 0.5, f"o{i}", expert_colors[e], "none", alpha=0.85, text_color="#08131A")
        ax.set_xlim(-1.1, total_w + 0.3)
        ax.set_ylim(-0.3, base_y + bucket_h + header_h + 0.5)

    # ---- step 3: combine — gate-weighted sum, scatter back ----
    elif step == 3:
        y_expert3 = 0.4
        y_out = y_expert3 + eb_h + 1.5
        max_terms = max((len(t) for t in tokens), default=1)

        for i, t in enumerate(tokens):
            x = tok_x[i]
            for eid, wgt in t:
                curve(ax, (expert_x[eid] + eb_w / 2, y_expert3 + eb_h),
                      (x + tw / 2, y_out), expert_colors[eid], lw=1.1 + 2.2 * wgt,
                      alpha=0.5 + 0.4 * wgt, rad=0.15 if eid >= i else -0.15)
            rrect(ax, x, y_out, tw, th, facecolor=ACCENT, edgecolor="none", alpha=0.92, zorder=3, shadow=True)
            ax.text(x + tw / 2, y_out + th / 2, f"t{i}", ha="center", va="center", fontsize=8.5,
                    color="#08131A", fontweight="bold", fontfamily="monospace", zorder=4)
            for k, (eid, wgt) in enumerate(t):
                ax.text(x + tw / 2, y_out + th + 0.1 + k * 0.28, f"{wgt:.2f}·E{eid}",
                        ha="center", va="bottom", fontsize=6.4, color=expert_colors[eid],
                        fontfamily="monospace", fontweight="bold", zorder=4)

        for e in range(n_experts):
            rrect(ax, expert_x[e], y_expert3, eb_w, eb_h, facecolor="none",
                  edgecolor=expert_colors[e], linewidth=1.6, alpha=0.9, zorder=2)
            ax.text(expert_x[e] + eb_w / 2, y_expert3 + eb_h / 2, f"E{e}", ha="center", va="center",
                    fontsize=9.5, color=expert_colors[e], fontweight="bold", fontfamily="monospace", zorder=3)

        ax.text(-0.35, y_out + th / 2, "output\n(same width\nas input)", ha="right", va="center",
                fontsize=7.6, color=ACCENT, fontfamily="monospace")

        # explicit contrast callout, vertically centered on the whole figure
        panel_h = y_out + th + max_terms * 0.28 + 0.3 - y_expert3
        cx0 = total_w + 0.5
        rrect(ax, cx0, y_expert3, 1.9, panel_h, facecolor=PANEL_BG, edgecolor=RULE, linewidth=1.1, zorder=1)
        ax.text(cx0 + 0.95, y_expert3 + panel_h * 0.85, "combine rule", ha="center", va="center", fontsize=8,
                color=INK_SOFT, fontweight="bold", fontfamily="monospace", zorder=2)
        ax.text(cx0 + 0.95, y_expert3 + panel_h * 0.58, "✓ weighted SUM\nsame width out", ha="center", va="center",
                fontsize=7.6, color=ACCENT, fontfamily="monospace", zorder=2)
        ax.text(cx0 + 0.95, y_expert3 + panel_h * 0.28, "✗ concatenate\nwould double width", ha="center", va="center",
                fontsize=7.6, color=INVALID_COLOR, fontfamily="monospace", zorder=2)

        ax.set_xlim(-1.1, total_w + 2.7)
        ax.set_ylim(-0.2, y_out + th + max_terms * 0.28 + 0.5)

    if step == 0:
        ax.set_xlim(-1.1, total_w + 0.3)
        ax.set_ylim(-0.5, 4.2)

    ax.set_aspect("equal")
    ax.axis("off")
    fig.tight_layout()
    return fig


# ============================================== PART 2: AllToAll devices ===
def draw_chip_split(ax, x, y, w, h, label, fill, color_top, color_bottom, fontsize=7.3):
    """A chip tagged with 2 small color tabs — used when a token has 2 distinct destinations."""
    rrect(ax, x, y, w, h, facecolor=fill, edgecolor=RULE, linewidth=1.0, alpha=0.95, zorder=3, shadow=True)
    tab_h = h * 0.16
    tab_margin = w * 0.14
    tab_w = w - 2 * tab_margin
    rrect(ax, x + tab_margin, y + h - tab_h - 0.04, tab_w, tab_h, r=tab_h * 0.45,
          facecolor=color_top, edgecolor="none", zorder=4)
    rrect(ax, x + tab_margin, y + 0.04, tab_w, tab_h, r=tab_h * 0.45,
          facecolor=color_bottom, edgecolor="none", zorder=4)
    ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=fontsize,
            color=INK, fontweight="bold", fontfamily="monospace", zorder=5)


def render_device_grid(step, n_devices, tokens_per_device, origin, routing, device_colors, ncols):
    n_tokens = n_devices * tokens_per_device
    nrows = (n_devices + ncols - 1) // ncols
    chip_w, chip_h, gap = 0.55, 0.5, 0.15
    max_per_row = max(2, tokens_per_device)

    def slots_for(d):
        # step 0/4: one entry per token that lives here (its full route attached).
        # step 1/2: one entry per (token, expert) slot this device currently computes.
        # step 3: one entry per (token, expert) slot whose token originated here.
        if step in (0, 4):
            return [(t, routing[t]) for t in range(n_tokens) if origin[t] == d]
        if step == 3:
            return [(t, eid, wgt) for t in range(n_tokens) for eid, wgt in routing[t] if origin[t] == d]
        return [(t, eid, wgt) for t in range(n_tokens) for eid, wgt in routing[t] if eid == d]

    max_shown = max((len(slots_for(d)) for d in range(n_devices)), default=0)
    max_rows = max(1, -(-max_shown // max_per_row))  # ceil div
    card_h = max_rows * (chip_h + gap) + 0.55
    card_w = max_per_row * (chip_w + gap) + 0.3
    row0_y = card_h - 0.6 - chip_h

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2.7, nrows * (card_h + 1.1)), dpi=FIG_DPI,
                              squeeze=False, facecolor=BG)

    for idx in range(nrows * ncols):
        r, c = divmod(idx, ncols)
        ax = axes[r][c]
        ax.set_facecolor(BG)
        if idx >= n_devices:
            ax.axis("off")
            continue
        d = idx
        shown = slots_for(d)

        rrect(ax, -0.15, -0.25, card_w, card_h, r=0.12, facecolor=PANEL_BG, edgecolor=RULE,
              linewidth=1.0, zorder=0, shadow=True)

        for j, item in enumerate(shown):
            cx = (j % max_per_row) * (chip_w + gap)
            cy = row0_y - (j // max_per_row) * (chip_h + gap)

            if step == 0:
                t, route = item
                if len(route) == 1:
                    draw_chip(ax, cx, cy, chip_w, chip_h, f"t{t}", device_colors[d],
                              device_colors[route[0][0]], fontsize=7.3, lw=2.0)
                else:
                    draw_chip_split(ax, cx, cy, chip_w, chip_h, f"t{t}", device_colors[d],
                                     device_colors[route[0][0]], device_colors[route[1][0]])
            elif step == 4:
                t, route = item
                draw_chip(ax, cx, cy, chip_w, chip_h, f"t{t}", ACCENT, "none",
                          fontsize=7.3, text_color="#08131A", lw=2.0)
            else:
                t, eid, wgt = item
                if step == 1:
                    fill, border = device_colors[origin[t]], device_colors[d]
                elif step == 2:
                    fill, border = device_colors[d], "none"
                else:  # step 3
                    fill, border = device_colors[eid], device_colors[d]
                label = f"t{t}" if step != 2 else f"o{t}"
                text_color = "#08131A" if step == 2 else INK
                draw_chip(ax, cx, cy, chip_w, chip_h, label, fill, border, fontsize=7.3,
                          alpha=0.95 if step != 1 else 0.9, text_color=text_color, lw=2.0)

        title = f"device {d}"
        subtitle = {
            0: f"owns Expert {d} · seq shard {d}",
            1: "received via AllToAll",
            2: f"Expert {d} MLP (local)",
            3: "results via AllToAll",
            4: "restored to original order",
        }[step]
        ax.set_title(f"{title}\n{subtitle}", fontsize=8.3, color=INK_SOFT, fontfamily="monospace", pad=8)

        ax.set_xlim(-0.3, card_w - 0.15)
        ax.set_ylim(-0.45, card_h - 0.05)
        ax.set_aspect("equal")
        ax.axis("off")

    fig.subplots_adjust(wspace=0.3, hspace=0.55)
    fig.tight_layout()
    return fig


# ===========================================================================
tab1, tab2 = st.tabs(["1 — MoE routing & combine", "2 — AllToAll across devices"])

# ------------------------------------------------------------- PART 1 ---
with tab1:
    st.markdown(
        '<div class="case-box">One conceptual device, no sharding yet — just the MoE computation itself: '
        "how tokens pick experts, and how expert outputs get put back together.</div>",
        unsafe_allow_html=True,
    )
    c1, c2, c3 = st.columns(3)
    with c1:
        n_tokens = st.slider("Number of tokens", 3, 8, 6, key="p1_tokens")
    with c2:
        n_experts = st.slider("Number of experts", 2, 6, 4, key="p1_experts")
    with c3:
        top_k = st.radio("Top-k routing", [1, 2], index=1, horizontal=True, key="p1_topk")

    if "p1_seed" not in st.session_state:
        st.session_state["p1_seed"] = 0
    if st.button("🔀 re-route tokens", key="p1_reroute"):
        st.session_state["p1_seed"] += 1

    tokens = route_tokens(n_tokens, n_experts, top_k, st.session_state["p1_seed"])
    expert_colors = [chunk_color(e, n_experts) for e in range(n_experts)]

    steps_p1 = [
        dict(title="Tokens & router", comm=None,
             desc=f"Each token computes gate scores over all {n_experts} experts (a softmax over a small "
                  f"linear projection) and keeps its top-{top_k}. Line thickness below shows the gate weight; "
                  "line color shows which expert it's headed to."),
        dict(title="Dispatch — grouped by expert", comm=None,
             desc="Conceptually, tokens are gathered into per-expert buckets. A token picked by 2 experts "
                  "(top-2) appears in 2 buckets — it isn't split or duplicated in data, just referenced twice."),
        dict(title="Expert MLP forward", comm=None,
             desc="Each expert runs its own MLP <b>locally</b> on the tokens in its bucket. Every expert's MLP "
                  "has the exact same input/output width (d_model) — that's what makes the next step possible."),
        dict(title="Combine — gate-weighted sum, scatter back", comm=None,
             desc="For every token, its expert output(s) are scaled by their gate weight(s) and <b>summed</b> "
                  "into one vector, then placed back at the token's original sequence position. Not concatenation — "
                  "concatenation would change the width; summation keeps it at d_model."),
    ]
    step_p1 = step_nav("p1_step", len(steps_p1))
    s = steps_p1[step_p1]
    st.markdown(f'<div class="step-desc">{comm_tag(s["comm"])}<br><b>{s["title"]}</b><br>{s["desc"]}</div>', unsafe_allow_html=True)

    color_legend(expert_colors, "E", "expert color key")
    fig_p1 = render_routing_step(step_p1, tokens, n_experts, expert_colors, top_k)
    st.pyplot(fig_p1, use_container_width=True)
    plt.close(fig_p1)

# ------------------------------------------------------------- PART 2 ---
with tab2:
    st.markdown(
        '<div class="case-box">Now spread across real devices: each device holds a slice of the token '
        "sequence <i>and</i> owns exactly one expert. The router's decision doesn't move any data — "
        "<b>AllToAll</b> is what actually ships each token to the device holding its assigned expert, and ships "
        "the result back.</div>",
        unsafe_allow_html=True,
    )
    c1, c2, c3 = st.columns(3)
    with c1:
        n_devices = st.slider("Number of devices (= number of experts, 1 per device)", 2, 6, 4, key="p2_devices")
    with c2:
        tokens_per_device = st.slider("Tokens per device (sequence shard size)", 1, 3, 2, key="p2_tpd")
    with c3:
        top_k2 = st.radio("Top-k routing", [1, 2], index=0, horizontal=True, key="p2_topk")

    if "p2_seed" not in st.session_state:
        st.session_state["p2_seed"] = 0
    if st.button("🔀 re-route tokens", key="p2_reroute"):
        st.session_state["p2_seed"] += 1

    n_tokens2 = n_devices * tokens_per_device
    routing2 = route_tokens(n_tokens2, n_devices, top_k2, st.session_state["p2_seed"])
    origin = [t // tokens_per_device for t in range(n_tokens2)]
    device_colors = [chunk_color(d, n_devices) for d in range(n_devices)]
    ncols2 = min(n_devices, 4)

    if top_k2 == 1:
        dispatch_desc = ("Every device sends each token to the device that owns its assigned expert, and receives "
                          "the tokens routed to <i>its own</i> expert from every other device — all in one collective "
                          "exchange. Chip fill still shows each token's origin device.")
        combine_local_desc = ("Back home, each device takes the one expert result it received for each of its "
                               "tokens (top-1, so no summing needed here — just a gate-weight scale) and drops it "
                               "back into its original sequence position.")
    else:
        dispatch_desc = ("With top-2, <b>each token is sent twice</b> — once to each of its 2 assigned experts' "
                          "devices — roughly doubling the bytes moved through AllToAll versus top-1. A token's two "
                          "copies (split-border chips before dispatch) land on two different devices.")
        combine_local_desc = ("Back home, each device now has <b>2 results per token</b> — one from each assigned "
                               "expert — and does the gate-weighted <b>sum</b> across them (same rule as Part 1) "
                               "before dropping the combined result back into its original sequence position.")

    steps_p2 = [
        dict(title="Initial — tokens sharded by sequence, experts sharded by device", comm=None,
             desc="Every device holds a contiguous slice of the token sequence (its data-parallel shard) and "
                  "owns exactly one expert. A token's chip is filled with its <b>home device</b>'s color and "
                  "outlined in its assigned expert's color(s)" +
                  (" — a two-tone border means it's headed to 2 different devices." if top_k2 == 2 else
                   " — most tokens need to go elsewhere.")),
        dict(title="Dispatch AllToAll", comm="AllToAll", desc=dispatch_desc),
        dict(title="Local expert MLP forward", comm=None,
             desc="No communication here — each device just runs its expert's MLP on whatever tokens it's "
                  "currently holding, regardless of where they came from."),
        dict(title="Combine AllToAll", comm="AllToAll",
             desc="The computed outputs are shipped back to each token's <b>origin</b> device — the exact "
                  "reverse shuffle of the dispatch step."),
        dict(title="Local combine — weighted sum, restore order", comm=None, desc=combine_local_desc),
    ]
    step_p2 = step_nav("p2_step", len(steps_p2))
    s = steps_p2[step_p2]
    st.markdown(f'<div class="step-desc">{comm_tag(s["comm"])}<br><b>{s["title"]}</b><br>{s["desc"]}</div>', unsafe_allow_html=True)

    color_legend(device_colors, "device ", "= Expert N, since device N owns Expert N")
    chip_key = {
        0: "chip fill = home device (this panel) &nbsp;|&nbsp; tab/border color(s) = assigned expert device(s), i.e. where it's headed",
        1: "chip fill = origin device (where it came from) &nbsp;|&nbsp; border = this panel's device (where it just arrived)",
        2: "chip fill = this panel's device (currently computing) &nbsp;|&nbsp; no border — purely local, nothing to relate to",
        3: "chip fill = expert device that produced this result &nbsp;|&nbsp; border = this panel's device (home, already arrived)",
        4: "solid accent fill = final combined result, back in its original sequence position",
    }[step_p2]
    st.markdown(
        f'<p style="font-family:\'IBM Plex Mono\',monospace; font-size:11px; color:{INK_SOFT}; margin:0 0 8px;">{chip_key}</p>',
        unsafe_allow_html=True,
    )
    fig_p2 = render_device_grid(step_p2, n_devices, tokens_per_device, origin, routing2, device_colors, ncols2)
    st.pyplot(fig_p2, use_container_width=True)
    plt.close(fig_p2)
