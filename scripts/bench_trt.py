#!/usr/bin/env python3
"""TensorRT throughput sweep for the 96-stream x 5 fps target (= 480 frames/s).

    python3 scripts/bench_trt.py exports/pro6000/xl_b*.onnx --out runs/bench_pro6000

Exists because the pip `tensorrt` wheel ships the Python API without the `trtexec`
binary, so `scripts/bench_pro6000.sh`'s sweep needs a Python runner. Run ONLY on an
idle GPU: a throughput number taken while anything shares the card is not a
measurement.

Method: build a serialized engine per (onnx, precision), then time enqueues with CUDA
events around `execute_async_v3` on a dedicated stream, H2D **and D2H** included in a
separate "end-to-end" figure and excluded in the "compute" figure -- the gap between the
two is the PCIe story, which is the reason the in-graph argmax exists. Engines are cached
next to the ONNX so re-runs skip the build.

**D2H was missing until 2026-08-31 and the docstring claimed it was there.** Only the
inputs were copied, so the "end-to-end" figure was engine + upload, and the outputs --
which is where this model's PCIe cost actually is -- were never moved. That made the
figure blind to the one thing it was cited for: `--argmax-seg` shrinks the output payload
**9.2x** at batch 16 (296.03 MB of fp32 `terrain` logits at 16x6x640x1120 down to
32.25 MB with a uint8 `terrain_argmax`), and the harness charged it the slower engine
while measuring none of the saving. Read a pre-2026-08-31 `results.json` as upload-only.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import tensorrt as trt
from cuda.bindings import runtime as cudart  # ships with tensorrt's cuda dependency

LOGGER = trt.Logger(trt.Logger.WARNING)
# PLAN §7.4 revised the delivery target to 96 streams at 5 fps on 2026-08-26. This
# constant kept the 1,440 it replaced, so every row this script printed for six days
# was scored against a requirement the project had already dropped -- and read as a
# 3.5x shortfall on 2026-09-01 what is actually 0.77x of the real target.
TARGET_FPS = 480.0  # 96 streams x 5 fps (PLAN §7.4, revised 2026-08-26)


def check(err):
    code = err[0] if isinstance(err, tuple) else err
    if int(code) != 0:
        raise RuntimeError(f"CUDA error {code}")
    return err[1:] if isinstance(err, tuple) and len(err) > 1 else None


def to_fp16(onnx: Path) -> Path:
    """TRT 11 dropped the FP16 builder flag: precision now comes from the graph's own
    dtypes. So fp16 means converting the ONNX itself (weights + activations), IO kept
    fp32 so the harness feeds the same buffers either way."""
    out = onnx.with_suffix(".fp16.onnx")
    if out.is_file():
        return out
    import onnx as onnx_lib
    from onnxconverter_common import float16

    model = onnx_lib.load(str(onnx))
    model = float16.convert_float_to_float16(model, keep_io_types=True)

    # **The converter rewrites tensors, not a Cast's stated destination**, and a graph
    # that takes a uint8 image casts it to float32 as its first act. That Cast keeps
    # saying FLOAT while the mean/std beside it become FLOAT16, and TensorRT refuses the
    # subtraction: "SUB must have same input types. But they are of types Float and Half".
    # Only the cast fed by the graph input is rewritten -- the ones `keep_io_types` adds
    # at the outputs say FLOAT on purpose, and rewriting those would change the contract
    # the harness and every host read the buffers under.
    entry = {i.name for i in model.graph.input}
    for node in model.graph.node:
        if node.op_type != "Cast" or not set(node.input) & entry:
            continue
        for attr in node.attribute:
            if attr.name == "to" and attr.i == onnx_lib.TensorProto.FLOAT:
                attr.i = onnx_lib.TensorProto.FLOAT16

    onnx_lib.save(model, str(out))
    return out


def build_engine(onnx: Path, fp16: bool) -> Path:
    tag = "fp16" if fp16 else "fp32"
    plan = onnx.with_suffix(f".{tag}.plan")
    if plan.is_file():
        return plan
    src = to_fp16(onnx) if fp16 else onnx
    builder = trt.Builder(LOGGER)
    network = builder.create_network()
    parser = trt.OnnxParser(network, LOGGER)
    if not parser.parse(src.read_bytes()):
        raise RuntimeError(
            f"{src}: " + "; ".join(str(parser.get_error(i)) for i in range(parser.num_errors))
        )
    cfg = builder.create_builder_config()
    cfg.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 8 << 30)
    t0 = time.time()
    blob = builder.build_serialized_network(network, cfg)
    if blob is None:
        raise RuntimeError(f"engine build failed for {onnx} ({tag})")
    plan.write_bytes(blob)
    print(f"  built {plan.name} in {time.time() - t0:.0f}s")
    return plan


def bench(plan: Path, batch: int, seconds: float) -> dict:
    runtime = trt.Runtime(LOGGER)
    engine = runtime.deserialize_cuda_engine(plan.read_bytes())
    ctx = engine.create_execution_context()
    (stream,) = check(cudart.cudaStreamCreate())

    host_in, dev = {}, {}
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        shape = tuple(engine.get_tensor_shape(name))
        dtype = trt.nptype(engine.get_tensor_dtype(name))
        nbytes = int(np.prod(shape)) * np.dtype(dtype).itemsize
        (ptr,) = check(cudart.cudaMalloc(nbytes))
        dev[name] = (ptr, nbytes)
        ctx.set_tensor_address(name, ptr)
        if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
            host_in[name] = np.random.randint(0, 255, size=shape).astype(dtype)

    host_out = {
        engine.get_tensor_name(i): np.empty(
            tuple(engine.get_tensor_shape(engine.get_tensor_name(i))),
            dtype=trt.nptype(engine.get_tensor_dtype(engine.get_tensor_name(i))),
        )
        for i in range(engine.num_io_tensors)
        if engine.get_tensor_mode(engine.get_tensor_name(i)) == trt.TensorIOMode.OUTPUT
    }

    def d2h():
        """The half that was missing. This model's outputs dwarf its input -- at batch 16
        the fp32 `terrain` logits alone are 275 MB against a 137 MB image -- so an
        "end-to-end" figure without them is an upload benchmark."""
        for name, arr in host_out.items():
            ptr, nbytes = dev[name]
            check(
                cudart.cudaMemcpyAsync(
                    arr.ctypes.data,
                    ptr,
                    nbytes,
                    cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost,
                    stream,
                )
            )

    def h2d():
        for name, arr in host_in.items():
            ptr, nbytes = dev[name]
            check(
                cudart.cudaMemcpyAsync(
                    ptr,
                    arr.ctypes.data,
                    nbytes,
                    cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
                    stream,
                )
            )

    # warm-up
    for _ in range(10):
        h2d()
        ctx.execute_async_v3(stream)
    check(cudart.cudaStreamSynchronize(stream))

    def timed(with_copies: bool) -> float:
        (start,) = check(cudart.cudaEventCreate())
        (stop,) = check(cudart.cudaEventCreate())
        n = 0
        check(cudart.cudaEventRecord(start, stream))
        t0 = time.time()
        while time.time() - t0 < seconds:
            if with_copies:
                h2d()
            ctx.execute_async_v3(stream)
            if with_copies:
                d2h()
            n += 1
        check(cudart.cudaEventRecord(stop, stream))
        check(cudart.cudaEventSynchronize(stop))
        (ms,) = check(cudart.cudaEventElapsedTime(start, stop))
        return n * batch / (ms / 1000.0)

    fps_compute = timed(False)
    fps_e2e = timed(True)
    for ptr, _ in dev.values():
        check(cudart.cudaFree(ptr))
    return {
        "frames_per_s_compute": round(fps_compute, 1),
        "frames_per_s_e2e": round(fps_e2e, 1),
    }


def onnx_batch(path: Path) -> int:
    """The batch this graph was exported at, read from the graph.

    **It used to be read from the filename** -- `re.search(r"_b(\\d+)", stem)`, defaulting
    to 1 -- and the 2026-08-25 resolution sweep is what that costs. Its files were named
    `res_640x1120.onnx`, exported with `--batch 16`, so every row divided the true
    throughput by sixteen: the log says 82.8 f/s where the engine does 1,325, and
    `meets_target_compute` compared the sixteenth against 1,440. At 576x1008 that recorded
    a **False** for an engine measuring 1,622 f/s, which clears the target.

    A file knows its own batch. Falling back to the filename, or to 1, is guessing about
    the one number every other number here is divided by.
    """
    import onnx

    graph = onnx.load(str(path), load_external_data=False).graph
    dim = graph.input[0].type.tensor_type.shape.dim[0]
    if dim.HasField("dim_value") and dim.dim_value > 0:
        return int(dim.dim_value)
    raise SystemExit(
        f"{path.name}: its first input's batch dimension is "
        f"{dim.dim_param or 'unset'!r} rather than a number. A dynamic-batch engine has "
        "to be told which batch to bench; this script divides every measurement by it."
    )


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("onnx", nargs="+")
    ap.add_argument("--out", type=Path, default=Path("runs/bench_pro6000"))
    ap.add_argument("--seconds", type=float, default=15.0)
    ap.add_argument("--fp32", action="store_true", help="also sweep fp32 (slow builds)")
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)

    results = []
    for path in sorted(Path(p) for p in args.onnx):
        batch = onnx_batch(path)
        for fp16 in [True, False] if args.fp32 else [True]:
            plan = build_engine(path, fp16)
            r = bench(plan, batch, args.seconds)
            row = {
                "onnx": path.name,
                "precision": "fp16" if fp16 else "fp32",
                "batch": batch,
                **r,
                "meets_target_compute": r["frames_per_s_compute"] >= TARGET_FPS,
                "meets_target_e2e": r["frames_per_s_e2e"] >= TARGET_FPS,
            }
            results.append(row)
            print(
                f"{path.name:16s} {'fp16' if fp16 else 'fp32'} b={batch:<3d} "
                f"compute {r['frames_per_s_compute']:8.1f} f/s"
                f"   +copies {r['frames_per_s_e2e']:8.1f} f/s"
            )
    (args.out / "results.json").write_text(
        json.dumps(
            {
                "target_frames_per_s": TARGET_FPS,
                "gpu_note": "valid only if the GPU was otherwise idle",
                "results": results,
            },
            indent=2,
        )
    )
    print(f"\ntarget {TARGET_FPS:.0f} f/s (96x15). Wrote {args.out}/results.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
