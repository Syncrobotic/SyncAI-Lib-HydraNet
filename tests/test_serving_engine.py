"""The parts of `serving/engine.py` that do not need a GPU.

The module was at 0% coverage: 320 lines of TensorRT executor that every throughput
figure in this project rests on, exercised by nothing. Most of it genuinely needs the
hardware -- a `cudaMalloc` cannot be faked into meaning anything -- but two pieces do not,
and they are the two that decide whether the rest is even pointed at the right tensors.

What is deliberately *not* covered here, so that a later reader does not mistake this file
for a claim that the executor is tested: the stream and event choreography in `submit`,
the pinned-buffer lifetime rules in `acquire_input` and `outputs`, and `close`. Those need
a card and are the honest remaining gap.
"""

from __future__ import annotations

import numpy as np
import pytest

from syncai_hydranet.serving.engine import _check, io_specs

F32 = np.dtype(np.float32)
U8 = np.dtype(np.uint8)


def test_io_specs_splits_one_input_from_the_outputs():
    (name, shape, dtype), outputs = io_specs(
        [
            ("images", (16, 3, 384, 640), U8, True),
            ("terrain", (16, 12, 96, 160), F32, False),
            ("det_cls", (16, 80, 48, 80), F32, False),
        ],
        "plan.engine",
    )
    assert (name, shape, dtype) == ("images", (16, 3, 384, 640), U8)
    assert set(outputs) == {"terrain", "det_cls"}
    assert outputs["terrain"] == ((16, 12, 96, 160), F32)


def test_a_second_input_is_refused_rather_than_read_back_as_an_output():
    """The failure this guard exists for is silent, not loud.

    Without it the second input falls into `output_specs`, gets a device buffer and a
    pinned host buffer, and is copied *back* after every batch -- so the executor reads
    uninitialised device memory and hands it to the caller as a head's output. Nothing
    raises; the numbers are simply wrong.
    """
    with pytest.raises(ValueError, match="single-input engine"):
        io_specs(
            [
                ("images", (16, 3, 384, 640), U8, True),
                ("prev_state", (16, 64), F32, True),
                ("terrain", (16, 12, 96, 160), F32, False),
            ],
            "two_inputs.engine",
        )


def test_the_refusal_names_both_bindings_and_the_plan():
    """A plan with two inputs is a build mistake several steps upstream, and the message
    is the only thing the reader has to find it with."""
    with pytest.raises(ValueError) as exc:
        io_specs(
            [("images", (1,), U8, True), ("prev_state", (1,), F32, True)],
            "runs/x/model.engine",
        )
    said = str(exc.value)
    assert "runs/x/model.engine" in said
    assert "images" in said and "prev_state" in said


def test_an_engine_with_no_input_is_refused_at_the_point_it_is_understood():
    """Otherwise it fails later, at a `cudaMalloc` of zero bytes, which names nothing."""
    with pytest.raises(ValueError, match="no input binding found"):
        io_specs([("terrain", (16, 12, 96, 160), F32, False)], "outputs_only.engine")
    with pytest.raises(ValueError, match="no input binding found"):
        io_specs([], "empty.engine")


def test_output_order_is_the_engines_order():
    """`outputs()` hands back this dict and callers index it by name, but anything that
    zips it against another sequence depends on insertion order holding."""
    _, outputs = io_specs(
        [("images", (1,), U8, True)] + [(f"h{i}", (i,), F32, False) for i in range(5)]
    )
    assert list(outputs) == ["h0", "h1", "h2", "h3", "h4"]


def test_check_unpacks_a_success_and_raises_on_a_cuda_error():
    """`cuda.bindings` returns `(errcode, *values)` from every call, so the error code and
    the value being returned arrive in the same tuple. Reading one without the other is
    how a failed allocation becomes a pointer of 0 that is used anyway.
    """
    assert _check((0, 12345)) == (12345,)
    assert _check((0, 1, 2)) == (1, 2)
    # Success with nothing to unpack, and the bare-code form.
    assert _check((0,)) is None
    assert _check(0) is None

    with pytest.raises(RuntimeError, match="CUDA error 2"):
        _check((2, 0))
    with pytest.raises(RuntimeError, match="CUDA error 700"):
        _check(700)


def test_check_raises_before_returning_the_value_that_came_with_the_error():
    """The tuple carries a value even when the code is nonzero -- `cudaMalloc` returns a
    pointer field regardless -- and it is not a pointer. Returning it would allocate
    nothing and hand back an address that is written to."""
    with pytest.raises(RuntimeError):
        _check((1, 0xDEADBEEF))
