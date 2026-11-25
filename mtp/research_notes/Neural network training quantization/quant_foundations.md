# Numerical formats and quantization foundations

## Which resources form a progression from numbers to training?

### Takeaway
Begin with formats and error, proceed to scale and zero point, then distinguish inference quantization from low-precision training.

### Cited Findings
- [Stanford numerical formats slides](https://gfxcourses.stanford.edu/cs348k/spring24content/media/dnnhardware/07_dnnhw.pdf): sign, exponent, mantissa and range versus precision.
- [MIT HAN Lab lecture 5](https://www.youtube.com/watch?v=AlASZb93rrc): numeric types, k-means and linear quantization.
- [MIT HAN Lab lecture 6](https://www.youtube.com/watch?v=3nqUFSSJYKQ): PTQ, QAT, binary/ternary and mixed precision, principally inference.
- [PyTorch practical quantization](https://pytorch.org/blog/quantization-in-practice/): symmetric and affine schemes, calibration, practical examples.
- [MLIR quantization](https://llvm.googlesource.com/llvm-project/mlir/%2B/909d3b99f611c068636a3d328cc96b32f106999e/docs/Quantization.md): affine equation and why exact zero representation matters.
- [Jacob et al., Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference](https://openaccess.thecvf.com/content_cvpr_2018/CameraReady/0777.pdf): foundational integer inference and QAT, not low-bit training end to end.
- [A White Paper on Neural Network Quantization](https://arxiv.org/abs/2106.08295): detailed, hardware-motivated PTQ and QAT exposition.
- [Mixed Precision Training](https://arxiv.org/abs/1710.03740): FP16 computations with FP32 master weights and loss scaling.

### Inferences
- Teach QAT and low-precision training as different objectives, then bridge them through mixed precision.

### Gaps
- Video lecture duration and exact depth were not independently checked.
