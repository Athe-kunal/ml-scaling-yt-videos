"""Modeled collective-bandwidth ramp — why small messages achieve far less than
the "claimed" ICI bandwidth, and why a naive bytes/runtime metric overstates it.

Recreates the shape of the scaling book's "AllGather bandwidth on TPU v5e" plot
(jax-ml.github.io/scaling-book/sharding/) using an analytical latency+bandwidth
model, since the book only describes the curve in prose (not raw data):
  - claimed unidirectional ICI bandwidth: 4.5e10 B/s on v5e
  - buffers under ~45kB are latency-bound
  - ~95% of peak bandwidth is reached by ~10MB
Real TPU specs (HBM, FLOPs/s, ICI bandwidth) are from
jax-ml.github.io/scaling-book/tpus/.
"""

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

st.set_page_config(page_title="TPU Collective Bandwidth Ramp", page_icon="📶", layout="wide")

# ------------------------------------------------------------------ theme ---
BG = "#0A0E14"
PANEL_BG = "#111826"
CARD_BG = "#161F2E"
INK = "#EDF1F7"
INK_SOFT = "#8E9BAF"
RULE = "#26314A"
ACCENT = "#5EEAD4"
ORANGE = "#FBBF24"
INVALID_COLOR = "#FB7185"
SHADOW = "#00000055"
FIG_DPI = 200

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
            radial-gradient(900px 500px at 100% 0%, rgba(251,191,36,0.05), transparent),
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
        font-family: 'IBM Plex Mono', monospace; font-size: 13.5px; color: {INK}; line-height: 1.6;
    }}
    .case-box b {{ color: {ACCENT}; }}
    div[data-testid="stMetric"] {{ background: {CARD_BG}; border: 1px solid {RULE}; border-radius: 12px; padding: 12px 16px 8px; }}
    div[data-testid="stMetric"] label {{ font-family: 'IBM Plex Sans Condensed', sans-serif !important; font-size: 11px !important; letter-spacing: 0.08em; text-transform: uppercase; color: {INK_SOFT} !important; }}
    div[data-testid="stMetricValue"] {{ font-family: 'IBM Plex Mono', monospace !important; color: {ACCENT} !important; }}
    div[data-testid="stImage"] img {{ border-radius: 16px; }}
    table {{ font-family: 'IBM Plex Mono', monospace !important; font-size: 12.5px !important; }}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<p class="hero-title">TPU Collective Bandwidth Ramp</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="hero-sub">Why achieved AllGather bandwidth ramps up with message size instead of being flat. '
    'Based on jax-ml.github.io/scaling-book/sharding/ &amp; /tpus/</p>',
    unsafe_allow_html=True,
)

# --------------------------------------------------------------- TPU specs ---
# name: (hbm_bytes_per_s, bf16_flops_s, ici_uni_bw_bytes_s, ici_bidi_bw_bytes_s, links_per_chip)
TPU_SPECS = {
    "TPU v3":  dict(hbm=9.0e11,  flops=1.4e14,  ici_uni=1.0e11, ici_bidi=2.0e11, links=4),
    "TPU v4p": dict(hbm=1.2e12,  flops=2.75e14, ici_uni=4.5e10, ici_bidi=9.0e10, links=6),
    "TPU v5e": dict(hbm=8.2e11,  flops=1.97e14, ici_uni=4.5e10, ici_bidi=9.0e10, links=4),
    "TPU v5p": dict(hbm=2.8e12,  flops=4.59e14, ici_uni=9.0e10, ici_bidi=1.8e11, links=6),
    "TPU v6e": dict(hbm=1.6e12,  flops=9.20e14, ici_uni=9.0e10, ici_bidi=1.8e11, links=4),
    "TPU7x":   dict(hbm=7.4e12,  flops=2.30e15, ici_uni=9.0e10, ici_bidi=1.8e11, links=6),
}


def human_bytes_per_s(x):
    for unit, div in (("TB/s", 1e12), ("GB/s", 1e9), ("MB/s", 1e6), ("KB/s", 1e3)):
        if x >= div:
            return f"{x / div:.1f} {unit}"
    return f"{x:.0f} B/s"


def human_bytes(x):
    for unit, div in (("GB", 1e9), ("MB", 1e6), ("KB", 1e3)):
        if x >= div:
            return f"{x / div:.1f} {unit}"
    return f"{x:.0f} B"


# ------------------------------------------------------------------ sidebar ---
with st.sidebar:
    st.markdown("### TPU generation")
    tpu_name = st.selectbox("Chip", list(TPU_SPECS.keys()), index=list(TPU_SPECS.keys()).index("TPU v5e"))
    spec = TPU_SPECS[tpu_name]

    st.markdown("### Collective / mesh")
    n_links = spec["links"]
    parallel_links = st.slider(
        "Simultaneous ICI directions used", 1, n_links, min(2, n_links),
        help="A 2D torus chip has 4 nearest-neighbor links (6 for a 3D torus). Using more of them in "
             "parallel (e.g. sending half the data each way around a ring) multiplies effective bandwidth.",
    )
    X = st.slider(
        "Devices in the collective (X)", 2, 128, 16,
        help="Number of devices participating in the AllGather/ReduceScatter ring.",
    )
    latency_us = st.slider(
        "Per-transfer latency floor (μs)", 0.1, 5.0, 1.0, step=0.1,
        help="The book states ICI hop latency is about 1μs and that latency/bandwidth terms are additive. "
             "This is the fixed overhead every transfer pays regardless of size.",
    )

    st.markdown("### Message size sweep")
    size_min_exp = st.slider("Min size (10^x bytes)", 2, 6, 3)
    size_max_exp = st.slider("Max size (10^x bytes)", 7, 11, 10)

W_uni = spec["ici_uni"] * parallel_links
T0 = latency_us * 1e-6

sizes = np.logspace(size_min_exp, size_max_exp, 200)
# T(size) = latency floor + bandwidth term, additive (per the book's hop-latency model).
# The bandwidth term uses the classic ring-collective factor (X-1)/(2X): only a (X-1)/X share of the
# data needs to cross the network at all, and a ring can send in both directions at once (factor 2).
bw_term = (sizes * (X - 1)) / (2 * X * W_uni)
time_model = T0 + bw_term

link_bw = (sizes * (X - 1)) / (2 * X * time_model)     # implied per-link bandwidth, given the ring cost model
naive_bw = sizes / time_model                            # naive "bytes moved / wall-clock time"

# where each curve crosses 95% of its own asymptote — mirrors the book's "~10MB reaches 95% of peak" claim
asymptote = W_uni
idx_95 = np.argmax(link_bw >= 0.95 * asymptote)
size_95 = sizes[idx_95] if link_bw[idx_95] >= 0.95 * asymptote else None
latency_bound_size = W_uni * T0  # book's rule of thumb: size < claimed_BW * latency

# ------------------------------------------------------------------- metrics ---
c1, c2, c3, c4 = st.columns(4)
c1.metric("Claimed unidirectional BW", human_bytes_per_s(spec["ici_uni"]))
c2.metric("Effective BW (this config)", human_bytes_per_s(W_uni))
c3.metric("Latency-bound below", human_bytes(latency_bound_size))
c4.metric("~95% peak reached at", human_bytes(size_95) if size_95 else "—")

# ------------------------------------------------------------------- plot ---
fig, ax = plt.subplots(figsize=(11, 5.5), dpi=FIG_DPI, facecolor=BG)
ax.set_facecolor(BG)

ax.plot(sizes, link_bw, color=ACCENT, linewidth=2.2, marker="o", markersize=3.5,
        label="Link BW  (bytes·(X-1) / (2·X·time))")
ax.plot(sizes, naive_bw, color=ORANGE, linewidth=2.2, marker="o", markersize=3.5,
        label="BW  (bytes / runtime)")
ax.axhline(W_uni, color=ACCENT, linewidth=1.2, linestyle="--", alpha=0.6,
           label=f"effective bandwidth, {parallel_links} link(s) ({human_bytes_per_s(W_uni)})")
ax.axvline(latency_bound_size, color=INVALID_COLOR, linewidth=1.0, linestyle=":", alpha=0.6)

ax.set_xscale("log")
ax.set_xlabel("Message size (bytes)", color=INK_SOFT, fontsize=11)
ax.set_ylabel("Bandwidth (bytes/s)", color=INK_SOFT, fontsize=11)
ax.set_title(f"Modeled AllGather bandwidth — {tpu_name}, X={X} devices, "
             f"{parallel_links}/{n_links} ICI links used",
             color=INK, fontsize=13, fontweight="bold", pad=14)

ax.tick_params(colors=INK_SOFT, labelsize=9)
for spine in ax.spines.values():
    spine.set_color(RULE)
ax.grid(True, color=RULE, linewidth=0.6, alpha=0.6)

ymax = max(naive_bw.max(), link_bw.max()) * 1.15
ax.set_ylim(0, ymax)
ax.axvspan(sizes.min(), latency_bound_size, color=INVALID_COLOR, alpha=0.06, zorder=0)
ax.text(latency_bound_size * 0.85, ymax * 0.04, "latency-bound", color=INVALID_COLOR, fontsize=9,
        ha="right", fontfamily="monospace", alpha=0.85)

legend = ax.legend(loc="lower right", facecolor=CARD_BG, edgecolor=RULE, fontsize=9.5, labelcolor=INK)
legend.get_frame().set_alpha(0.95)

fig.tight_layout()
st.pyplot(fig, use_container_width=True)
plt.close(fig)

# --------------------------------------------------------------- explainer ---
st.markdown(
    f'''<div class="case-box">
    <b>Why two curves?</b> A ring AllGather across X devices only ever needs to move
    <b>(X-1)/X</b> of the data over the network (each device already holds its own 1/X shard locally),
    and a ring can send in both directions at once (÷2). So the <b>Link BW</b> curve
    (teal) backs out the true per-link bandwidth from the observed time using that factor — it's the
    honest number, and it's what converges to the <b>claimed</b> ICI bandwidth as messages get large.<br><br>
    The <b>naive BW</b> curve (amber) is just <code>bytes / runtime</code> — it ignores the ring's sharing
    factor entirely, so it overstates the link bandwidth by roughly <b>2·X/(X-1)</b>, which is why it
    plateaus near <b>2×</b> the claimed unidirectional number once X is reasonably large. Both curves start
    low for small messages because a fixed <b>latency floor</b> ({latency_us:.1f}&nbsp;&mu;s here) dominates the total
    time when there isn't much data to amortize it over — that's the shaded "latency-bound" region.<br><br>
    <b>Caveat:</b> the scaling book describes this curve (claimed BW, ~45kB latency-bound threshold on v5e,
    ~95% of peak by ~10MB) but doesn't publish the underlying benchmark data — this plot is an
    <b>analytical model</b> (additive latency + bandwidth term, per the book's own hop-latency description)
    calibrated to match those figures, not a replay of real measurements.
    </div>''',
    unsafe_allow_html=True,
)

with st.expander("TPU spec reference table"):
    rows = []
    for name, s in TPU_SPECS.items():
        rows.append({
            "TPU": name,
            "HBM BW": human_bytes_per_s(s["hbm"]),
            "bf16 FLOPs/s": f"{s['flops']:.2e}",
            "ICI uni BW/link": human_bytes_per_s(s["ici_uni"]),
            "ICI bidi BW/link": human_bytes_per_s(s["ici_bidi"]),
            "links/chip": s["links"],
        })
    st.table(rows)
