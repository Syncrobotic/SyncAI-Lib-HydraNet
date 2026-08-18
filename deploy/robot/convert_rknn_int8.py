import sys

from rknn.api import RKNN

onnx_in, rknn_out, ds = sys.argv[1], sys.argv[2], sys.argv[3]
rknn = RKNN(verbose=False)
rknn.config(
    target_platform="rk3588",
    mean_values=[[0, 0, 0]],
    std_values=[[1, 1, 1]],
    quantized_dtype="asymmetric_quantized-8",
    quantized_algorithm="normal",
)
assert rknn.load_onnx(model=onnx_in) == 0
print("build INT8 with calibration...", flush=True)
assert rknn.build(do_quantization=True, dataset=ds) == 0
assert rknn.export_rknn(rknn_out) == 0
print("OK ->", rknn_out, flush=True)
rknn.release()
