"""`serving/uint8_input.py`: the graph surgery that halves the H2D copy.

It had no test at all until 2026-09-04, on a module whose whole reason for existing is
a measured throughput claim -- 1,553 f/s to 990 f/s end-to-end depending on which input
binding the engine carries (PLAN 7a.29). A rewrite that changed the *numbers* rather
than the binding would move that figure and nothing would have said so.

So the assertion that matters here is not that the surgery ran: it is that the rewritten
graph returns what the original returned, for the same image, to the bit. Everything
else -- the binding's name, its dtype, its layout, the metadata -- is contract, and
contract is what `export_onnx`'s renaming convention already argues has to fail loudly.
"""

from __future__ import annotations

import numpy as np
import pytest

from syncai_hydranet.serving.uint8_input import INPUT_U8_NHWC, add_uint8_nhwc_input

onnx = pytest.importorskip("onnx", reason="the `export` extra provides onnx")
ort = pytest.importorskip("onnxruntime", reason="the `export` extra provides onnxruntime")


def _tiny_model(path, *, b=1, c=3, h=4, w=5, inputs=1):
    """A graph with one float NCHW input and one op, which is all the surgery touches."""
    from onnx import TensorProto, helper

    ins = [
        helper.make_tensor_value_info(f"image_rgb_255{'' if i == 0 else i}", TensorProto.FLOAT,
                                      [b, c, h, w])
        for i in range(inputs)
    ]  # fmt: skip
    # Multiply by a constant: enough to prove the values reaching the body are unchanged.
    scale = helper.make_tensor("scale", TensorProto.FLOAT, [1], [2.0])
    node = helper.make_node("Mul", [ins[0].name, "scale"], ["out"], name="body")
    out = helper.make_tensor_value_info("out", TensorProto.FLOAT, [b, c, h, w])
    graph = helper.make_graph([node], "tiny", ins, [out], initializer=[scale])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 9
    onnx.checker.check_model(model)
    onnx.save(model, str(path))
    return path


def test_the_rewritten_graph_returns_what_the_original_returned(tmp_path):
    """The claim the module exists to make good on: only the input path changed.

    The frame is fed as fp32 NCHW to the original and as the identical bytes in uint8
    NHWC to the rewrite. Byte-exact, not approximate: a uint8 holds 0-255 exactly, which
    is the property that lets this be a copy-size win rather than a precision trade.
    """
    src = _tiny_model(tmp_path / "m.onnx")
    dst = add_uint8_nhwc_input(src)

    rng = np.random.default_rng(20260904)
    nchw_u8 = rng.integers(0, 256, (1, 3, 4, 5), dtype=np.uint8)

    before = ort.InferenceSession(str(src)).run(
        None, {"image_rgb_255": nchw_u8.astype(np.float32)}
    )[0]
    after = ort.InferenceSession(str(dst)).run(
        None, {INPUT_U8_NHWC: np.ascontiguousarray(nchw_u8.transpose(0, 2, 3, 1))}
    )[0]
    np.testing.assert_array_equal(before, after)


def test_the_binding_is_renamed_so_a_float_host_fails_to_find_it(tmp_path):
    """The export convention: a runtime written for the fp32 binding must not silently
    feed 0-255 floats where bytes are expected -- it must fail to find its binding."""
    dst = add_uint8_nhwc_input(_tiny_model(tmp_path / "m.onnx"))
    names = {i.name: i for i in ort.InferenceSession(str(dst)).get_inputs()}
    assert set(names) == {INPUT_U8_NHWC}
    assert names[INPUT_U8_NHWC].type == "tensor(uint8)"
    assert names[INPUT_U8_NHWC].shape == [1, 4, 5, 3]  # NHWC, not NCHW


def test_the_metadata_says_what_the_binding_now_is(tmp_path):
    dst = add_uint8_nhwc_input(_tiny_model(tmp_path / "m.onnx"))
    props = {p.key: p.value for p in onnx.load(str(dst)).metadata_props}
    assert props["input_layout"] == "NHWC"
    assert props["input_range"] == "0-255 uint8"


def test_the_source_is_never_modified(tmp_path):
    """`add_uint8_nhwc_input` writes a new file, because the graph being served is the
    graph that was benchmarked and surgery on it in place would end that."""
    src = _tiny_model(tmp_path / "m.onnx")
    before = src.read_bytes()
    dst = add_uint8_nhwc_input(src)
    assert dst != src
    assert src.read_bytes() == before


def test_a_second_pass_is_refused_rather_than_stacking_a_cast(tmp_path):
    dst = add_uint8_nhwc_input(_tiny_model(tmp_path / "m.onnx"))
    with pytest.raises(ValueError, match="already"):
        add_uint8_nhwc_input(dst)


def test_a_graph_with_two_inputs_is_refused(tmp_path):
    src = _tiny_model(tmp_path / "two.onnx", inputs=2)
    with pytest.raises(ValueError, match="exactly one graph input"):
        add_uint8_nhwc_input(src)


def test_an_input_that_is_not_three_channel_nchw_is_refused(tmp_path):
    src = _tiny_model(tmp_path / "wide.onnx", c=4)
    with pytest.raises(ValueError, match=r"expected \[B,3,H,W\]"):
        add_uint8_nhwc_input(src)
