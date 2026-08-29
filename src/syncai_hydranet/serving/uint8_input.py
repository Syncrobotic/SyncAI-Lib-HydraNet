"""Graph surgery: give an exported ONNX a uint8 [B,H,W,3] input binding.

Why this exists: the bench (runs/bench_pro6000) showed the fp16 b16 engine computes
3,272 f/s but synchronous fp32 H2D drags it to 1,553 -- the copy, not the model, is
the frontier. A fp32 NCHW frame at 512x640 is 3.93 MB on the bus; the same frame as
uint8 is 0.98 MB, and NHWC is what a decoder emits, so the host-side transpose
disappears too. The cast and the layout change move *into* the graph, where TensorRT
fuses them into the first convolution's neighbourhood for approximately nothing.

Why surgery rather than re-export: the graph being served is the graph that was
benchmarked (exports/pro6000/xl_b16.fp16.onnx). Prepending two nodes to that exact
artifact provably changes only the input path; a fresh export would re-open the
question of which checkpoint and config produced it.

Contract-by-name, the repo's export convention: the new input is
``image_rgb_u8_nhwc``. A runtime written for ``image_rgb_255`` (fp32 NCHW) fails to
find its binding in this variant, loudly, instead of feeding 0-255 floats where
bytes are expected.

TensorRT constraint honoured here: a UINT8 network input must feed a Cast before
any other op, so the node order is Cast (uint8 -> float32) then Transpose
(NHWC -> NCHW); the old input name becomes the Transpose's output so every existing
consumer is untouched.
"""

from __future__ import annotations

from pathlib import Path

INPUT_U8_NHWC = "image_rgb_u8_nhwc"


def add_uint8_nhwc_input(src: str | Path, dst: str | Path | None = None) -> Path:
    """Rewrite ``src``'s single image input to uint8 NHWC; write and return ``dst``.

    ``dst`` defaults to ``<src stem>.u8<suffix>`` next to the source. The output is
    a new file; the source is never modified.
    """
    import onnx
    from onnx import TensorProto, helper

    src = Path(src)
    if dst is None:
        dst = src.with_name(src.name.replace(".onnx", "") + ".u8.onnx")
    dst = Path(dst)

    model = onnx.load(str(src))
    graph = model.graph
    if len(graph.input) != 1:
        raise ValueError(f"{src}: expected exactly one graph input, found {len(graph.input)}")
    old = graph.input[0]
    if old.name == INPUT_U8_NHWC:
        raise ValueError(f"{src}: input is already {INPUT_U8_NHWC}")
    dims = [d.dim_value for d in old.type.tensor_type.shape.dim]
    if len(dims) != 4 or dims[1] != 3:
        raise ValueError(f"{src}: input {old.name} is {dims}, expected [B,3,H,W]")
    b, c, h, w = dims

    new_in = helper.make_tensor_value_info(INPUT_U8_NHWC, TensorProto.UINT8, [b, h, w, c])
    cast = helper.make_node(
        "Cast",
        [INPUT_U8_NHWC],
        [f"{INPUT_U8_NHWC}_as_float"],
        to=TensorProto.FLOAT,
        name="serving_input_cast",
    )
    transpose = helper.make_node(
        "Transpose",
        [f"{INPUT_U8_NHWC}_as_float"],
        [old.name],  # existing consumers keep reading the tensor they always read
        perm=[0, 3, 1, 2],
        name="serving_input_nhwc_to_nchw",
    )
    graph.input.remove(old)
    graph.input.insert(0, new_in)
    # Node lists are topologically sorted; the new producers go first.
    graph.node.insert(0, cast)
    graph.node.insert(1, transpose)

    # Keep the embedded metadata truthful for anyone reading the ONNX itself.
    props = {p.key: p.value for p in model.metadata_props}
    props["input_layout"] = "NHWC"
    props["input_range"] = "0-255 uint8"
    del model.metadata_props[:]
    for key, value in props.items():
        entry = model.metadata_props.add()
        entry.key, entry.value = key, value

    onnx.checker.check_model(model)
    onnx.save(model, str(dst))
    return dst
