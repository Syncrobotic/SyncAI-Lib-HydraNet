"""TensorRT executor with pinned staging and dual-stream, double-buffered H2D.

This is the recovery the bench asked for. runs/bench_pro6000 measured the fp16 b16
engine at 3,272 f/s compute and 1,553 f/s once synchronous fp32 H2D was included:
the copy path is the frontier. Two of the three named recoveries live here --

* **Pinned staging.** The scheduler assembles each tick directly into a
  page-locked host buffer (``acquire_input`` hands out a numpy view of it), so the
  H2D is a true async DMA rather than a hidden pageable staging copy.
* **Double-buffered overlap.** Two buffer slots, separate copy and compute
  streams: batch N+1's H2D runs while batch N computes. Events order the streams;
  the host throttles itself by waiting on the compute of two batches back, which
  bounds the pipeline at depth 2 without ever serialising copy against compute.

The third recovery (uint8 input, 4x less to copy) is the graph's job -- see
``uint8_input`` -- and composes with both of the above.

Slot discipline for callers: ``acquire_input(slot)`` -> fill the view ->
``submit(slot)`` -> consume ``outputs(slot)`` before submitting that slot again.
The pilot alternates slots, consuming tick N-1 while tick N computes, which
satisfies this by construction.

CUDA plumbing is via ``cuda.bindings.runtime`` (ships with the tensorrt wheel),
the same runtime scripts/bench_trt.py uses; the build recipe below matches that
script's so an engine built here is the engine that was benchmarked.
"""

from __future__ import annotations

import ctypes
import time
from pathlib import Path
from typing import Any

import numpy as np


def _check(err: Any) -> Any:
    """cuda.bindings returns (errcode, *values); raise on nonzero, unpack the rest."""
    code = err[0] if isinstance(err, tuple) else err
    if int(code) != 0:
        raise RuntimeError(f"CUDA error {code}")
    return err[1:] if isinstance(err, tuple) and len(err) > 1 else None


def io_specs(
    bindings: list[tuple[str, tuple[int, ...], np.dtype, bool]],
    source: str = "engine",
) -> tuple[tuple[str, tuple[int, ...], np.dtype], dict[str, tuple[tuple[int, ...], np.dtype]]]:
    """Split an engine's bindings into its one input and its outputs.

    Separated from :class:`TrtExecutor` because it is the only part of that class
    that needs no GPU, and because both of its refusals are the ones worth having:
    a two-input engine would otherwise have its second input silently classified as
    an output and read back as garbage, and an engine with no input at all would
    fail much later, at a `cudaMalloc` of zero bytes.

    `bindings` is `(name, shape, dtype, is_input)` per binding, in engine order.
    """
    input_spec = None
    outputs: dict[str, tuple[tuple[int, ...], np.dtype]] = {}
    for name, shape, dtype, is_input in bindings:
        if is_input:
            if input_spec is not None:
                raise ValueError(
                    f"{source}: executor expects a single-input engine, and this one "
                    f"binds both {input_spec[0]!r} and {name!r} as inputs"
                )
            input_spec = (name, shape, dtype)
        else:
            outputs[name] = (shape, dtype)
    if input_spec is None:
        raise ValueError(f"{source}: no input binding found")
    return input_spec, outputs


def build_plan(onnx_path: str | Path, plan_path: str | Path | None = None) -> Path:
    """Serialize a TensorRT engine for ``onnx_path``; cached next to the ONNX.

    Same recipe as scripts/bench_trt.py (TRT 11: precision comes from the graph's
    own dtypes, so an fp16 engine means an fp16-converted ONNX, not a builder
    flag). Restated rather than imported because src/ does not import scripts/.
    """
    import tensorrt as trt

    onnx_path = Path(onnx_path)
    plan = Path(plan_path) if plan_path else onnx_path.with_suffix(".plan")
    if plan.is_file():
        return plan
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network()
    parser = trt.OnnxParser(network, logger)
    if not parser.parse(onnx_path.read_bytes()):
        raise RuntimeError(
            f"{onnx_path}: "
            + "; ".join(str(parser.get_error(i)) for i in range(parser.num_errors))
        )
    cfg = builder.create_builder_config()
    cfg.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 8 << 30)
    t0 = time.time()
    blob = builder.build_serialized_network(network, cfg)
    if blob is None:
        raise RuntimeError(f"engine build failed for {onnx_path}")
    plan.write_bytes(blob)
    print(f"built {plan.name} in {time.time() - t0:.0f}s")
    return plan


def bench_sync(plan: str | Path, batch: int, seconds: float) -> dict[str, float]:
    """Synchronous-copy reference bench: the method of scripts/bench_trt.bench.

    Pageable host arrays, per-iteration memcpyAsync + execute on one stream, CUDA
    events around the loop; "compute" excludes the copies, "h2d" includes them.
    Restated here rather than imported because scripts are not libraries
    (tests/test_scripts_are_not_libraries.py ratchets on exactly that), and this
    is the baseline row every recovery in this package is measured against.
    """
    import tensorrt as trt
    from cuda.bindings import runtime as cudart

    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(Path(plan).read_bytes())
    ctx = engine.create_execution_context()
    (stream,) = _check(cudart.cudaStreamCreate())

    host_in, dev = {}, {}
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        shape = tuple(engine.get_tensor_shape(name))
        dtype = trt.nptype(engine.get_tensor_dtype(name))
        nbytes = int(np.prod(shape)) * np.dtype(dtype).itemsize
        (ptr,) = _check(cudart.cudaMalloc(nbytes))
        dev[name] = (ptr, nbytes)
        ctx.set_tensor_address(name, ptr)
        if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
            host_in[name] = np.random.randint(0, 255, size=shape).astype(dtype)

    def h2d() -> None:
        for name, arr in host_in.items():
            ptr, nbytes = dev[name]
            _check(
                cudart.cudaMemcpyAsync(
                    ptr,
                    arr.ctypes.data,
                    nbytes,
                    cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
                    stream,
                )
            )

    for _ in range(10):
        h2d()
        ctx.execute_async_v3(stream)
    _check(cudart.cudaStreamSynchronize(stream))

    def timed(with_copies: bool) -> float:
        (start,) = _check(cudart.cudaEventCreate())
        (stop,) = _check(cudart.cudaEventCreate())
        n = 0
        _check(cudart.cudaEventRecord(start, stream))
        t0 = time.time()
        while time.time() - t0 < seconds:
            if with_copies:
                h2d()
            ctx.execute_async_v3(stream)
            n += 1
        _check(cudart.cudaEventRecord(stop, stream))
        _check(cudart.cudaEventSynchronize(stop))
        (ms,) = _check(cudart.cudaEventElapsedTime(start, stop))
        return n * batch / (ms / 1000.0)

    result = {
        "frames_per_s_compute": round(timed(False), 1),
        "frames_per_s_h2d": round(timed(True), 1),
    }
    for ptr, _ in dev.values():
        _check(cudart.cudaFree(ptr))
    return result


class TrtExecutor:
    """Double-buffered executor around one fixed-batch engine."""

    SLOTS = 2

    def __init__(self, plan: str | Path, enable_d2h: bool = True):
        import tensorrt as trt
        from cuda.bindings import runtime as cudart

        self._cudart = cudart
        self.enable_d2h = enable_d2h
        logger = trt.Logger(trt.Logger.WARNING)
        self._runtime = trt.Runtime(logger)
        self.engine = self._runtime.deserialize_cuda_engine(Path(plan).read_bytes())
        self.ctx = self.engine.create_execution_context()

        bindings = []
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            bindings.append(
                (
                    name,
                    tuple(self.engine.get_tensor_shape(name)),
                    np.dtype(trt.nptype(self.engine.get_tensor_dtype(name))),
                    self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT,
                )
            )
        (self.input_name, self.input_shape, self.input_dtype), self.output_specs = io_specs(
            bindings, str(plan)
        )
        self.batch = int(self.input_shape[0])

        (self.copy_stream,) = _check(cudart.cudaStreamCreate())
        (self.compute_stream,) = _check(cudart.cudaStreamCreate())
        (self.d2h_stream,) = _check(cudart.cudaStreamCreate())

        def events() -> list[Any]:
            return [_check(cudart.cudaEventCreate())[0] for _ in range(self.SLOTS)]

        # Per-slot event pairs; the *_end events also order the streams.
        self._ev_h2d_start, self._ev_h2d_end = events(), events()
        self._ev_compute_start, self._ev_compute_end = events(), events()
        self._ev_d2h_start, self._ev_d2h_end = events(), events()

        self._dev_in: list[int] = []
        self._pinned_in: list[np.ndarray] = []
        self._host_ptrs: list[int] = []
        self._dev_out: list[dict[str, int]] = []
        self._pinned_out: list[dict[str, np.ndarray]] = []
        in_bytes = int(np.prod(self.input_shape)) * self.input_dtype.itemsize
        self.input_nbytes = in_bytes
        for _ in range(self.SLOTS):
            (dptr,) = _check(cudart.cudaMalloc(in_bytes))
            self._dev_in.append(dptr)
            self._pinned_in.append(
                self._alloc_pinned(in_bytes, self.input_shape, self.input_dtype)
            )
            douts, houts = {}, {}
            for name, (shape, dtype) in self.output_specs.items():
                nbytes = int(np.prod(shape)) * dtype.itemsize
                (optr,) = _check(cudart.cudaMalloc(nbytes))
                douts[name] = optr
                if enable_d2h:
                    houts[name] = self._alloc_pinned(nbytes, shape, dtype)
            self._dev_out.append(douts)
            self._pinned_out.append(houts)

    def _alloc_pinned(self, nbytes: int, shape: tuple[int, ...], dtype: np.dtype) -> np.ndarray:
        (ptr,) = _check(self._cudart.cudaHostAlloc(nbytes, 0))  # cudaHostAllocDefault
        self._host_ptrs.append(ptr)
        buf = (ctypes.c_ubyte * nbytes).from_address(ptr)
        return np.frombuffer(buf, dtype=dtype).reshape(shape)

    # -- the double-buffered pipeline -------------------------------------------
    def acquire_input(self, slot: int) -> np.ndarray:
        """Pinned host input view for ``slot``, safe to write once returned.

        Waits until the previous H2D from this buffer has drained, so the caller
        never overwrites bytes still on their way to the device.
        """
        _check(self._cudart.cudaEventSynchronize(self._ev_h2d_end[slot]))
        return self._pinned_in[slot]

    def submit(self, slot: int) -> None:
        """Enqueue H2D (copy stream) + inference (compute stream) for ``slot``."""
        cudart = self._cudart
        # Host throttle and device-buffer safety in one wait: the batch two back on
        # this slot must have finished computing before its input buffer is reused.
        _check(cudart.cudaEventSynchronize(self._ev_compute_end[slot]))

        _check(cudart.cudaEventRecord(self._ev_h2d_start[slot], self.copy_stream))
        _check(
            cudart.cudaMemcpyAsync(
                self._dev_in[slot],
                self._pinned_in[slot].ctypes.data,
                self.input_nbytes,
                cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
                self.copy_stream,
            )
        )
        _check(cudart.cudaEventRecord(self._ev_h2d_end[slot], self.copy_stream))

        _check(cudart.cudaStreamWaitEvent(self.compute_stream, self._ev_h2d_end[slot], 0))
        # The previous D2H from this slot's output buffers must drain before the
        # new compute overwrites them.
        _check(cudart.cudaStreamWaitEvent(self.compute_stream, self._ev_d2h_end[slot], 0))
        self.ctx.set_tensor_address(self.input_name, self._dev_in[slot])
        for name, ptr in self._dev_out[slot].items():
            self.ctx.set_tensor_address(name, ptr)
        _check(cudart.cudaEventRecord(self._ev_compute_start[slot], self.compute_stream))
        self.ctx.execute_async_v3(self.compute_stream)
        _check(cudart.cudaEventRecord(self._ev_compute_end[slot], self.compute_stream))

        if self.enable_d2h:
            _check(cudart.cudaStreamWaitEvent(self.d2h_stream, self._ev_compute_end[slot], 0))
            _check(cudart.cudaEventRecord(self._ev_d2h_start[slot], self.d2h_stream))
            for name, (shape, dtype) in self.output_specs.items():
                _check(
                    cudart.cudaMemcpyAsync(
                        self._pinned_out[slot][name].ctypes.data,
                        self._dev_out[slot][name],
                        int(np.prod(shape)) * dtype.itemsize,
                        cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost,
                        self.d2h_stream,
                    )
                )
            _check(cudart.cudaEventRecord(self._ev_d2h_end[slot], self.d2h_stream))

    def outputs(self, slot: int) -> dict[str, np.ndarray]:
        """Pinned views of ``slot``'s outputs, after the D2H has landed.

        Views are valid until the caller submits this slot again.
        """
        if not self.enable_d2h:
            raise RuntimeError("executor was built with enable_d2h=False")
        _check(self._cudart.cudaEventSynchronize(self._ev_d2h_end[slot]))
        return self._pinned_out[slot]

    def timings_ms(self, slot: int) -> dict[str, float]:
        """Measured h2d/compute/d2h durations of the last completed batch on ``slot``."""
        cudart = self._cudart
        out = {}
        for key, (start, end) in {
            "h2d": (self._ev_h2d_start, self._ev_h2d_end),
            "compute": (self._ev_compute_start, self._ev_compute_end),
            "d2h": (self._ev_d2h_start, self._ev_d2h_end),
        }.items():
            if key == "d2h" and not self.enable_d2h:
                continue
            _check(cudart.cudaEventSynchronize(end[slot]))
            (ms,) = _check(cudart.cudaEventElapsedTime(start[slot], end[slot]))
            out[key] = float(ms)
        return out

    def synchronize(self) -> None:
        for stream in (self.copy_stream, self.compute_stream, self.d2h_stream):
            _check(self._cudart.cudaStreamSynchronize(stream))

    def close(self) -> None:
        cudart = self._cudart
        self.synchronize()
        for ptr in self._dev_in:
            _check(cudart.cudaFree(ptr))
        for outs in self._dev_out:
            for ptr in outs.values():
                _check(cudart.cudaFree(ptr))
        for ptr in self._host_ptrs:
            _check(cudart.cudaFreeHost(ptr))
        self._dev_in, self._dev_out, self._host_ptrs = [], [], []
