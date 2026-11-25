# Frontier four bit training resources

## What should readers use to understand the two articles?

### Takeaway
The two articles describe different recipes. Radical Numerics explains NVIDIA's NVFP4 training recipe; Thinking Machines explains its own selective NVFP4/MXFP8 production approach and quantized communication.

### Cited Findings
- [Radical Numerics Part 1](https://www.radicalnumerics.ai/blog/nvfp4-part1), January 2026, is a detailed treatment of floating-point formats, two-level scaling, random Hadamard transforms, stochastic rounding, 1D/2D scaling, selective precision, and code examples; its [Part 2](https://www.radicalnumerics.ai/blog/nvfp4-part2) traces forward, activation-gradient, and weight-gradient matrix multiplies into Transformer Engine and CUDA kernels.
- [Thinking Machines end-to-end training](https://thinkingmachines.ai/blog/end-to-end-4-bit-training/), November 2026, reports NVFP4 expert-layer forward passes, MXFP8 backward passes, higher-precision exceptions, and quantized communications. “End-to-end” refers to using the same numeric configuration across pretraining through RL, not all operations in 4 bits. It reports 8% pretraining and 20% RL wall-clock gains in its largest jobs against full MXFP8; peak hardware throughput is a different quantity.
- [NVIDIA's Transformer Engine documentation](https://docs.nvidia.com/deeplearning/transformer-engine/features/low_precision_training/nvfp4/nvfp4.html) explains NVFP4 x = E2M1 element × E4M3 block scale × FP32 global scale, 16-value 1D blocks, 16×16 weight blocks, stochastic rounding for gradients, and Hadamard transforms for weight-gradient operands.
- [Microscaling Data Formats for Deep Learning](https://arxiv.org/abs/2310.10537) introduces per-block scaling and is the bridge to NVFP4; [OCP MX specification](https://www.opencompute.org/documents/%20ocp-microscaling-formats-mx-v1-0-spec-final-pdf/) defines standardized MX formats. MXFP4 uses 32-element blocks with E8M0 scales whereas NVFP4 uses 16-element blocks with E4M3 scales and a global FP32 scale. [Radical Numerics](https://www.radicalnumerics.ai/blog/nvfp4-part1)

### Inferences
- The strongest learning order is ordinary floating-point and quantization, training matrix multiplies, microscaling, NVIDIA recipe, then Thinking Machines and performance bottlenecks.

### Gaps
- Source-reported speedups are workload-specific and should not be generalized to all models or hardware.

## Which advanced papers and video explainers complement them?

### Takeaway
Read the numerical recipe before variants that alter scaling or pursue broader quantization coverage.

### Cited Findings
- [Pretraining Large Language Models with NVFP4](https://arxiv.org/abs/2509.25149): primary NVIDIA technical report covering stability, 2D scaling, Hadamard transforms, stochastic rounding, selective precision, and evaluation.
- [Four Over Six](https://arxiv.org/abs/2512.02010): adaptive block scale selection for NVFP4 and why FP4's uneven grid can harm near-maximum values; relevant to the Thinking Machines article.
- [FP4 All the Way](https://arxiv.org/abs/2505.19115): a separate approach to training with predominantly FP4 weights, activations, and gradients; contrast the precise tensor and operation coverage against Thinking Machines' selective recipe.
- [Quartet II](https://arxiv.org/abs/2601.22813): more advanced unbiased gradient estimation in NVFP4 training.
- [Normalized Architectures are Natively 4-Bit](https://arxiv.org/abs/2605.06067): research direction co-designing the architecture to reduce the need for numerical interventions.
- [Pretraining Large Language Models with NVFP4](https://www.youtube.com/watch?v=IGn04ARFoSY): video paper explainer, description discusses the paper's recipe; video audio and quality were not independently inspected.
- [NVIDIA On-Demand: Accelerating LLM Training and Inferencing With Reduced Precision](https://www.nvidia.com/en-us/on-demand/session/gtc26-dlit81567/): technical talk covering NVFP4, Hadamard transforms, 2D blocks, stochastic rounding, higher precision exceptions; may require registration.
- [NVIDIA Developer: What Is NVFP4?](https://www.youtube.com/watch?v=UTfg-_EGurw) is an accessible introduction but is explicitly focused on inference, not training.

### Inferences
- Prerequisites include backpropagation matrix products, floating-point representation, saturation and rounding, and distribution outliers.

### Gaps
- Video content quality and full duration were not independently verified.
