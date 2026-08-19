"""Folding the segmentation argmax into the graph, and the contract that has to move with it.

Measured on a GB10 at 512x640, single thread, real decode: the **host argmax was the
largest single item in the frame** -- 2.53 ms of 6.70 ms, larger than the engine's 2.09 ms.
DEPLOY.md §4 lists four ways to make the engine smaller and the engine is 31% of the
problem. Folding the argmax in measures 4.00 ms, a 40% frame reduction, from an export flag
with no retraining and no weight changed.

Which makes the risk worth naming: this is a *free* win, and free wins are the ones that
get turned on everywhere without anyone re-reading what they cost. Two things they cost:

1. **The logits are gone.** No confidence, no soft blend, no per-class probability, and
   nothing downstream says so at the point of use.
2. **The output changes rank and dtype.** A host that keeps calling `.argmax(0)` on what
   it thinks are `[C, H, W]` logits will call it on `[H, W]` uint8 ids and get a
   `[W]`-shaped array of nonsense -- which sometimes crashes and sometimes draws.

The rename is what turns (2) into a missing binding. These tests pin that, and pin that the
folded argmax is the *same* answer the host was computing rather than a cheaper one.

pytest tests/test_export_argmax.py -v
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from syncai_hydranet.cli.export_onnx import ExportWrapper, check_parity
from syncai_hydranet.models.hydranet import build_model


def _cfg(num_classes: int = 7) -> dict:
    return {
        "model": {
            "backbone": {"name": "resnet18", "pretrained": False},
            "neck": {"name": "fpn", "out_channels": 32, "num_repeats": 1, "num_levels": 5},
            "heads": {
                "terrain": {
                    "type": "semantic_fpn",
                    "num_classes": num_classes,
                    "in_levels": [0, 1, 2],
                    "channels": 32,
                },
                "traversability": {
                    "type": "semantic_fpn",
                    "num_classes": 3,
                    "in_levels": [0, 1, 2],
                    "channels": 32,
                },
            },
            "loss_balancing": "fixed",
            "fixed_weights": {"terrain": 1.0, "traversability": 1.0},
        }
    }


def _model(num_classes: int = 7):
    torch.manual_seed(0)
    return build_model(_cfg(num_classes)).eval()


IMG = torch.rand(1, 3, 128, 160) * 255.0


# --- the answer has to be the same answer ---------------------------------


def test_the_folded_argmax_equals_the_host_argmax_it_replaces():
    """Not "a class map comes out" -- the *same* class map. Taking the argmax at P3 and
    resizing the ids with nearest would also produce a plausible map, at every boundary a
    different one, and cheaper. This pins that the graph computes what the host did."""
    model = _model()
    logits_out = ExportWrapper(model, argmax_seg=False)(IMG)
    ids_out = ExportWrapper(model, argmax_seg=True)(IMG)

    for logits, ids in zip(logits_out, ids_out, strict=True):
        expected = logits.argmax(dim=1).to(torch.uint8)
        assert torch.equal(ids, expected)


def test_the_class_map_is_uint8_and_has_lost_the_channel_axis():
    out = ExportWrapper(_model(), argmax_seg=True)(IMG)
    for t in out:
        assert t.dtype == torch.uint8
        assert t.shape == (1, 128, 160)


def test_without_the_flag_nothing_moves():
    """Every engine already built has to keep working, so the default has to be exactly
    what it was."""
    out = ExportWrapper(_model(), argmax_seg=False)(IMG)
    assert out[0].dtype == torch.float32
    assert out[0].shape == (1, 7, 128, 160)


# --- the contract ----------------------------------------------------------


def test_bindings_are_renamed_only_when_the_argmax_is_folded_in():
    plain = ExportWrapper(_model(), argmax_seg=False)
    folded = ExportWrapper(_model(), argmax_seg=True)
    assert plain.seg_output_names == ["terrain", "traversability"]
    assert folded.seg_output_names == ["terrain_argmax", "traversability_argmax"]


def test_a_taxonomy_too_wide_for_uint8_is_refused_at_construction():
    """257 classes into a uint8 map wraps silently, and class 256 becomes class 0 --
    which in every scheme here is `void`, the one that means "no answer"."""
    with pytest.raises(SystemExit, match="uint8"):
        ExportWrapper(_model(300), argmax_seg=True)


def test_the_widest_shipped_taxonomy_still_fits():
    """The retail 13 and the indoor 12 are the real cases; the guard must not be in their
    way."""
    assert ExportWrapper(_model(13), argmax_seg=True).argmax_seg


# --- parity has to compare the right thing --------------------------------


class _FakeSession:
    """onnxruntime's shape, with a controllable disagreement."""

    def __init__(self, outputs):
        self._outputs = outputs

    def get_inputs(self):
        return [type("I", (), {"name": "images"})()]

    def run(self, _names, _feed):
        return self._outputs


def _parity(ref_out, got_out, monkeypatch, tol=1e-4):
    class _FakeOrt:
        @staticmethod
        def InferenceSession(*_args, **_kwargs):  # noqa: N802 - onnxruntime's own name
            return _FakeSession(got_out)

    def _wrapper(_x):
        return tuple(torch.from_numpy(a) for a in ref_out)

    monkeypatch.setitem(sys.modules, "onnxruntime", _FakeOrt)
    return check_parity(_wrapper, torch.zeros(1), "unused.onnx", ["terrain_argmax"], tol=tol)


def test_parity_on_a_class_map_counts_disagreeing_pixels(monkeypatch):
    """Relative error on label ids is meaningless: ids 2 and 3 are not "1 apart" in any
    sense the float tolerance was set for, and on a map of small ids a single wrong pixel
    can look either enormous or negligible depending on what else is in it."""
    a = np.zeros((1, 10, 10), dtype=np.uint8)
    b = a.copy()
    b[0, 0, 0] = 5  # one pixel in a hundred
    assert not _parity([a], [b], monkeypatch)  # 0.01 > 1e-4, so it fails
    assert _parity([a], [a.copy()], monkeypatch)  # identical maps pass


def test_parity_still_uses_relative_error_on_logits(monkeypatch):
    a = np.full((1, 3, 4, 4), 100.0, dtype=np.float32)
    b = a + 1e-3  # 1e-5 relative, well inside tolerance
    assert _parity([a], [b], monkeypatch)


# --- the board reads whichever form it is given ---------------------------


def test_live_view_reads_both_binding_forms():
    """The Jetson viewer cannot import this package, so this is the only place the two
    ends are checked against each other."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "robot"))
    try:
        import live_view_orin
    finally:
        sys.path.pop(0)

    logits = np.zeros((1, 3, 4, 5), dtype=np.float32)
    logits[0, 2] = 1.0  # class 2 everywhere
    assert np.array_equal(
        live_view_orin.seg_map({"terrain": logits}, "terrain"), np.full((4, 5), 2)
    )

    ids = np.full((1, 4, 5), 2, dtype=np.uint8)
    assert np.array_equal(
        live_view_orin.seg_map({"terrain_argmax": ids}, "terrain"), np.full((4, 5), 2)
    )
