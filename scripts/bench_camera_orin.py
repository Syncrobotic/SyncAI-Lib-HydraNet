#!/usr/bin/env python3
"""End-to-end frame rate on a Jetson, from camera to usable output.

    python3 bench_camera_orin.py hydranet_fp16.engine --device /dev/video0

trtexec answers "how fast is the GPU". This answers "how fast is the robot", which is
the number that matters and always the lower of the two. It times each stage separately
so the bottleneck is visible rather than inferred:

    capture -> letterbox + normalise -> H2D -> inference -> D2H -> argmax + decode

Detection decode here is deliberately naive (score threshold, no NMS): the point is to
show how much host work the 80-class output costs, not to be the production decoder.
"""

from __future__ import annotations

import argparse
import ctypes
import time
from pathlib import Path

import numpy as np

# Only used for engines built before the normalisation moved into the graph. The
# engine says which convention it wants through its input binding name: `image_rgb_255`
# means the graph normalises and this script must not, `images` means the reverse.
# Two copies of a mean and standard deviation, one here and one in training, is the
# classic silent deployment failure -- so new exports carry the constants with them and
# this pair only serves the engines that predate that.
LEGACY_INPUT = "images"
RAW_INPUT = "image_rgb_255"
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class Cudart:
    """The four CUDA calls this benchmark needs, via ctypes.

    pycuda is not packaged for this board and building it on ARM is slow and fragile, so
    rather than add a dependency to a machine that will run the deployed model, bind
    libcudart directly. TensorRT's own Python module is already installed and does the
    rest.
    """

    def __init__(self):
        self.lib = ctypes.CDLL("libcudart.so.12")
        self.lib.cudaGetErrorString.restype = ctypes.c_char_p

    def _check(self, code, what):
        if code != 0:
            msg = self.lib.cudaGetErrorString(code).decode()
            raise RuntimeError(f"{what} failed: {msg} (cuda error {code})")

    def malloc(self, nbytes):
        ptr = ctypes.c_void_p()
        self._check(
            self.lib.cudaMalloc(ctypes.byref(ptr), ctypes.c_size_t(nbytes)), "cudaMalloc"
        )
        return ptr

    def stream_create(self):
        s = ctypes.c_void_p()
        self._check(self.lib.cudaStreamCreate(ctypes.byref(s)), "cudaStreamCreate")
        return s

    def memcpy_htod(self, dst, src: np.ndarray, stream):
        self._check(
            self.lib.cudaMemcpyAsync(
                dst, src.ctypes.data_as(ctypes.c_void_p), ctypes.c_size_t(src.nbytes), 1, stream
            ),
            "cudaMemcpyAsync H2D",
        )

    def memcpy_dtoh(self, dst: np.ndarray, src, stream):
        self._check(
            self.lib.cudaMemcpyAsync(
                dst.ctypes.data_as(ctypes.c_void_p), src, ctypes.c_size_t(dst.nbytes), 2, stream
            ),
            "cudaMemcpyAsync D2H",
        )

    def sync(self, stream):
        self._check(self.lib.cudaStreamSynchronize(stream), "cudaStreamSynchronize")


class Stage:
    """Wall-clock accumulator; reports the median, which ignores the first-frame spike."""

    def __init__(self, name):
        self.name, self.samples = name, []

    def __enter__(self):
        self.t = time.perf_counter()
        return self

    def __exit__(self, *_):
        self.samples.append((time.perf_counter() - self.t) * 1000)

    @property
    def ms(self):
        return float(np.median(self.samples)) if self.samples else float("nan")


def letterbox(frame, out_h, out_w):
    """Preserve aspect ratio, pad the rest. The camera is 16:9 and the model is 1.25:1,
    so a plain resize would squeeze the image horizontally and break every learned shape."""
    import cv2  # pyright: ignore[reportMissingImports]

    h, w = frame.shape[:2]
    scale = min(out_h / h, out_w / w)
    nh, nw = round(h * scale), round(w * scale)
    resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.zeros((out_h, out_w, 3), dtype=np.uint8)
    top, left = (out_h - nh) // 2, (out_w - nw) // 2
    canvas[top : top + nh, left : left + nw] = resized
    return canvas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("engine")
    ap.add_argument("--device", default="/dev/video0")
    ap.add_argument("--frames", type=int, default=200)
    ap.add_argument("--height", type=int, default=512)
    ap.add_argument("--width", type=int, default=640)
    args = ap.parse_args()

    import cv2  # pyright: ignore[reportMissingImports]
    import tensorrt as trt  # pyright: ignore[reportMissingImports]

    cuda = Cudart()

    logger = trt.Logger(trt.Logger.WARNING)
    with Path(args.engine).open("rb") as f, trt.Runtime(logger) as rt:
        engine = rt.deserialize_cuda_engine(f.read())
    ctx = engine.create_execution_context()

    names = [engine.get_tensor_name(i) for i in range(engine.num_io_tensors)]
    inputs = [n for n in names if engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT]
    outputs = [n for n in names if engine.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT]

    host, dev = {}, {}
    for n in names:
        shape = tuple(ctx.get_tensor_shape(n))
        arr = np.empty(shape, dtype=trt.nptype(engine.get_tensor_dtype(n)))
        host[n] = np.ascontiguousarray(arr)
        dev[n] = cuda.malloc(host[n].nbytes)
        ctx.set_tensor_address(n, dev[n].value)
    stream = cuda.stream_create()
    in_name = inputs[0]
    # The contract, read off the engine rather than assumed.
    graph_normalises = in_name == RAW_INPUT
    print(
        f"input '{in_name}': "
        + (
            "raw RGB 0-255, the graph normalises"
            if graph_normalises
            else "pre-normalised; this script applies ImageNet mean/std (legacy export)"
        )
    )

    cap = cv2.VideoCapture(args.device, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    if not cap.isOpened():
        raise SystemExit(f"cannot open {args.device}")
    cw = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    ch = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    print(f"camera: {cw:.0f}x{ch:.0f}")
    print(f"model:  {args.width}x{args.height}, {len(outputs)} outputs\n")

    cap_s, pre_s, h2d_s, inf_s, d2h_s, post_s, total_s = (
        Stage(n)
        for n in ("capture", "preprocess", "H2D", "inference", "D2H", "postprocess", "total")
    )

    for i in range(args.frames + 10):  # the first frames pay for allocation and clocks
        warm = i < 10
        with total_s:
            with cap_s:
                ok, frame = cap.read()
            if not ok:
                raise SystemExit("camera read failed")

            with pre_s:
                lb = letterbox(frame, args.height, args.width)
                rgb = cv2.cvtColor(lb, cv2.COLOR_BGR2RGB).astype(np.float32)
                if not graph_normalises:
                    rgb = (rgb / 255.0 - MEAN) / STD
                chw = rgb.transpose(2, 0, 1)[None]
                np.copyto(host[in_name], chw.astype(host[in_name].dtype))

            with h2d_s:
                cuda.memcpy_htod(dev[in_name], host[in_name], stream)
                cuda.sync(stream)
            with inf_s:
                ctx.execute_async_v3(stream.value)
                cuda.sync(stream)
            with d2h_s:
                for n in outputs:
                    cuda.memcpy_dtoh(host[n], dev[n], stream)
                cuda.sync(stream)

            with post_s:
                for n in outputs:
                    if n in ("traversability", "terrain"):
                        host[n][0].argmax(axis=0)
                    elif n.startswith("det_cls"):
                        # sigmoid then threshold: the cheapest honest stand-in for decode
                        _ = (1.0 / (1.0 + np.exp(-host[n]))) > 0.3
        if warm:
            for s in (cap_s, pre_s, h2d_s, inf_s, d2h_s, post_s, total_s):
                s.samples.clear()

    cap.release()
    stages = [cap_s, pre_s, h2d_s, inf_s, d2h_s, post_s]
    print(f"{'stage':<14}{'median ms':>10}{'share':>9}")
    for s in stages:
        print(f"{s.name:<14}{s.ms:>10.2f}{100 * s.ms / total_s.ms:>8.0f}%")
    print(f"{'-' * 33}\n{'total':<14}{total_s.ms:>10.2f}")
    print(f"\nend-to-end: {1000 / total_s.ms:.1f} FPS")
    print(f"GPU alone:  {1000 / inf_s.ms:.1f} FPS  <- what trtexec reports")


if __name__ == "__main__":
    main()
