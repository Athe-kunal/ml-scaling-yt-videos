# ML scaling visualizers

Python apps use the repo venv (`uv sync` once). Vite apps are independent (`npm install` inside the app).

## video_1

Arithmetic-intensity / matmul roofline (Streamlit).

```bash
uv run streamlit run video_1/arithemtic_intensity.py
```

## video_2

TPU math time vs communication time (Streamlit).

```bash
uv run streamlit run video_2/tpu_comms_visualizer.py
```

## video_3

Chapter 3 sharding, collectives, and MoE (one Streamlit app per file).

```bash
uv run streamlit run video_3/sharding_visualizer.py          # I, J on mesh (x, y)
uv run streamlit run video_3/matmul_sharding_visualizer.py   # sharded matmul, 4 cases
uv run streamlit run video_3/questions_5_9_visualizer.py     # questions 5–9
uv run streamlit run video_3/question_10_alltoall_visualizer.py
uv run streamlit run video_3/moe_alltoall_visualizer.py
uv run streamlit run video_3/tpu_bandwidth_visualizer.py
```

## video_4

Transformer FLOPs, params, and activations (Vite).

```bash
make video4-install && make video4   # http://localhost:5174
```

## video_5

DP, ZeRO, FSDP, TP, vocab-parallel embedding (Vite). `NAME` opens a topic (`NONE`, `DP`, `ZERO1`, `ZERO2`, `ZERO3`, `FSDP`, `TP`, `VPE`, `FSDP_TP`, `FLOPS_COMMS`).

```bash
make video5-install && make video5
make video5 NAME=FSDP                 # http://localhost:5178
```

## video_6

Inference step time, throughput, MLP, speculative decoding (Vite).

```bash
make video6-install && make video6    # http://localhost:5179
```

## ring-attention

Ring, zig-zag, stripe, and tree attention (Vite).

```bash
make ring-install && make ring        # http://localhost:5180
```

## inference_engineering

Scheduler / paged-KV visualizer (Vite), plus `tinyserve.py` (a small inference server, not the visualizer).

```bash
cd inference_engineering/scheduler-app && npm install && npm run dev
```

## slides / Scaling_Lectures

PDFs only. No app.
