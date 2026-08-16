"""`--layout` for hydranet-infer-video: which panels go in the frame, and what happens
when the head one of them needs is not in the config.

These rules used to live inside `main`'s encode loop, which meant reaching them required
a real video, a real checkpoint and a working ffmpeg on both ends. Nothing here needs any
of the three -- `render_frame` takes a decoded array and a model, and that is the whole
point of it being a function.

The no-traversability case is the one worth pinning. It is not an argument error: whether
that head exists is a property of the checkpoint's config, so `--layout trav` parses fine
and only fails once the model is loaded. A regression there would surface as a KeyError on
the first frame of somebody's clip.

pytest tests/test_infer_video_layouts.py -v
"""

from argparse import Namespace

import numpy as np
import pytest
import torch

from syncai_hydranet.cli.infer_video import render_frame

SIZE = (64, 80)  # (H, W), and deliberately not square: a layout bug that swaps the axes
FRAME = np.zeros((48, 64, 3), dtype=np.uint8)


class _FakeModel:
    """Returns whichever heads it was told to have, at the size preprocess produced.

    Not a HydraNet: `render_frame` only ever calls `.predict`, and building a real model
    would tie this test to the backbone's weights for no gain in what it checks.
    """

    def __init__(self, *, heads, detections=None):
        self.heads = heads
        self.detections = detections or {}

    def predict(self, x, **_kw):
        # Annotated `object` because the real `HydraNet.predict` return is also mixed:
        # segmentation heads map to a tensor and `detection` to a list of dicts. Inferred
        # from the comprehension alone this would be `dict[str, Tensor]` and the next line
        # would be a type error in the stand-in but not in the thing it stands in for.
        h, w = x.shape[-2:]
        out: dict[str, object] = {
            name: torch.zeros(1, h, w, dtype=torch.long) for name in self.heads
        }
        out["detection"] = [self.detections]
        return out


def _args(layout: str) -> Namespace:
    return Namespace(layout=layout, score_thr=0.3, nms_thr=0.6)


def _render(layout, *, heads=("terrain", "traversability"), detections=None):
    return render_frame(
        FRAME,
        _FakeModel(heads=heads, detections=detections),
        "cpu",
        _args(layout),
        size=SIZE,
        terrain_colors=np.zeros((256, 3), dtype=np.uint8),
        name_of=str,
    )


def test_side_is_exactly_twice_the_width_of_one_panel():
    """The side-by-side layout is two panels, so any single-panel layout is half of it.
    Comparing against `terrain` rather than a hardcoded number keeps this true when the
    letterbox geometry changes, which is the thing that would otherwise silently drift."""
    assert _render("side").width == 2 * _render("terrain").width


@pytest.mark.parametrize("layout", ["trav", "terrain"])
def test_a_single_panel_layout_keeps_the_content_size(layout):
    side = _render("side")
    one = _render(layout)
    assert (one.width, one.height) == (side.width // 2, side.height)


def test_without_a_traversability_head_the_default_layout_still_renders():
    """The object-segmentation configs drop that head. `side` has nothing to put on the
    left, and falling back to terrain is what keeps the command usable on them."""
    out = _render("side", heads=("terrain",))
    assert (out.width, out.height) == (_render("terrain").width, _render("terrain").height)


def test_asking_for_trav_without_the_head_refuses_and_names_the_config_key():
    """A KeyError here would name neither the head nor the flag that asked for it."""
    with pytest.raises(SystemExit) as e:
        _render("trav", heads=("terrain",))
    assert "model.heads.traversability" in str(e.value)


def test_boxes_outside_the_content_region_are_dropped_not_drawn():
    """A detection can land in the letterbox padding; clamping it would draw a box the
    model never predicted, at the frame edge, where it looks like a real one."""
    far_outside = {
        "boxes": torch.tensor([[9000.0, 9000.0, 9100.0, 9100.0]]),
        "scores": torch.tensor([0.9]),
        "labels": torch.tensor([0]),
    }
    drawn = np.asarray(_render("terrain", detections=far_outside))
    assert np.array_equal(drawn, np.asarray(_render("terrain")))
