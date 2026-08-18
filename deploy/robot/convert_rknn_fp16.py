import sys

from rknn.api import RKNN

onnx_in, rknn_out = sys.argv[1], sys.argv[2]
rknn = RKNN(verbose=False)
# Preprocessing (mean/std, /255) is embedded in the ONNX graph; input is raw RGB 0-255.
# So RKNN normalization is identity here.
rknn.config(target_platform="rk3588", mean_values=[[0, 0, 0]], std_values=[[1, 1, 1]])
print("load_onnx...", flush=True)
assert rknn.load_onnx(model=onnx_in) == 0, "load_onnx failed"
print("build (fp16, no quant)...", flush=True)
assert rknn.build(do_quantization=False) == 0, "build failed"
print("export_rknn...", flush=True)
assert rknn.export_rknn(rknn_out) == 0, "export failed"
print("OK ->", rknn_out, flush=True)
rknn.release()
