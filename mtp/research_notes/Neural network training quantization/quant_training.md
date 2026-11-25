# Quantized training resources

## Which resources explain mixed precision and quantized training?

### Takeaway
Trace FP16 mixed precision through FP8 to FP4, distinguishing the precision of GEMM operands from accumulators, master weights, optimizer states, and other operators.

### Cited Findings
- [Mixed Precision Training (Micikevicius et al., ICLR 2018)](https://arxiv.org/abs/1710.03740): foundational paper on FP16 arithmetic, FP32 master weights and accumulation, and loss scaling to preserve small gradients. Read before FP8.
- [NVIDIA mixed precision training tutorial](https://docs.nvidia.com/deeplearning/performance/mixed-precision-training/index.html): practical worked explanation of half precision, underflow, loss scaling, and Tensor Cores; complement with the original paper.
- [Training Deep Neural Networks with 8-bit Floating Point Numbers (NeurIPS 2018)](https://papers.nips.cc/paper_files/paper/2018/hash/335d3d1cd7ef05ec77714a215134914c-Abstract.html): earlier FP8 study focusing on backpropagation fidelity, stochastic rounding, and chunk-based accumulation. Experiments are specific to tested CNNs and speech models, not modern LLMs.
- [FP8 Formats for Deep Learning](https://arxiv.org/abs/2209.05433): introduces E4M3 and E5M2 and the tradeoff between exponent range and mantissa precision. [NVIDIA FP8 primer](https://docs.nvidia.com/deeplearning/transformer-engine-releases/release-2.5/user-guide/examples/fp8_primer.html) explains scaling and offers runnable examples. [Transformer Engine recipes](https://docs.nvidia.com/deeplearning/transformer-engine/api/common.html) document a hybrid recipe using E4M3 forward and E5M2 backward; do not present this as universal.
- [FP4 All the Way (NeurIPS 2025)](https://papers.nips.cc/paper_files/paper/2025/hash/8340b085045cf13f1f0b6c2c4cc0a89c-Abstract-Conference.html): research on predominantly FP4 weights, activations, and gradients; stochastic rounding in backward and update, with stated results up to 200B tokens. Inspect exact precision and experiments before describing as literally all operations FP4.
- [Pretraining Large Language Models with NVFP4](https://arxiv.org/abs/2509.25149): research report covering randomized Hadamard transforms, two-dimensional weight quantization, stochastic rounding and selected higher-precision layers, with a 12B parameter run trained on 10T tokens. Distinguish its implementation and setup from other FP4 claims.
- [8-bit Optimizers via Block-wise Quantization](https://arxiv.org/abs/2110.02861): quantizes optimizer statistics to reduce optimizer-state memory; uses block-wise dynamic quantization and stable embeddings. It addresses a distinct piece of the memory budget from low-bit matrix multiplications.
- [Tim Dettmers: 8-bit Optimizers](https://www.youtube.com/watch?v=IxrlHAJtqKE): author talk about the optimizer paper. [Cohere technical talk with Tim Dettmers](https://www.youtube.com/watch?v=jyOqtw4ry2w): connects optimizer-state quantization with LLM.int8 inference; good to contrast different quantization targets.

### Inferences
- A useful teaching diagram tracks tensor dtype at each operation: weights, activations, activation gradients, weight gradients, accumulator, master weights, optimizer state, and communication. Claims of 4-bit training generally require this qualification.

### Gaps
- Actual speedup depends on hardware, kernels, quantization overhead, and benchmark; no universal speedup is implied.

## How should QAT and QLoRA be distinguished from low-precision pretraining?

### Takeaway
QAT simulates quantization during training to produce an accurate quantized inference model; QLoRA fine-tunes adapters through a frozen quantized base; neither inherently implies that all training matrix multiplications and model updates execute in 4-bit.

### Cited Findings
- [Jacob et al., Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference (CVPR 2018)](https://openaccess.thecvf.com/content_cvpr_2018/html/Jacob_Quantization_and_Training_CVPR_2018_paper.html): classic paper on 8-bit integer-only inference and training designed for it.
- [PyTorch QAT tutorial](https://pytorch.org/blog/quantization-aware-training/): practical fake-quantized training and subsequent conversion for inference. [TorchAO documentation](https://docs.pytorch.org/ao/stable/workflows/qat.html) provides current API, so prefer it over older example code when implementing.
- [QLoRA (NeurIPS 2023)](https://arxiv.org/abs/2305.14314): frozen 4-bit base, trainable low-rank adapters, NF4, double quantization, and paged optimizers. It is fine-tuning rather than full 4-bit pretraining.
- [QAT beginner tutorial video](https://www.youtube.com/watch?v=s3tqqBaRuHE): hands-on ResNet training-loop demonstration; API may be dated, so check current documentation.

### Inferences
- QAT, optimizer quantization, QLoRA, and FP4 GEMMs answer different questions and make a useful comparative slide.

### Gaps
- None material to the assigned scope.
