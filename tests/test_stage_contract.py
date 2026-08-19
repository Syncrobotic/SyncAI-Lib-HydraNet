"""The second stage's input contract, pinned where it is actually load-bearing.

`stage.py` named the boundary and typed what crosses it, and then nothing used the type:
zero producers, zero consumers, zero coverage, eight docstring references. This is the
file that stops that happening again, and it exists because of what the measurement found
when the type was finally checked against the code it describes.

**The contract was wrong, not merely unadopted.** As one TypedDict, `StageFrame` required
six keys. `zone_stock_counts` reads five of them and never `image`; `reach_to_shelf_events`
reads two, of which one is required; and the only real producer, `scripts/pose_pilot.py`,
builds `{frame_index, terrain}` and satisfies none of it. Annotating either signature with
it would have turned a working caller into a type error -- which is why "adopt the
contract" was never as simple as adding an annotation, and why nobody had.

So the types are split by what a consumer reads, and these tests pin both halves of that:
each narrow type asks for exactly what its consumer indexes, and the full payload is
assignable to both.
"""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest

from syncai_hydranet.analytics import events as ev
from syncai_hydranet.analytics.stage import BoxFrame, FrameRef, StageFrame, TerrainFrame

# The package directory, not one file. `events` was a single 1,425-line module when this
# test was written and `Path(ev.__file__)` was the whole of it; it is now a package, so
# that path is an `__init__.py` holding re-exports and no function bodies at all. Reading
# the directory keeps the property that matters here -- the contract is read off the code
# rather than restated -- and it survives the next time a consumer moves between modules.
EVENTS = Path(ev.__file__).parent


def _keys_read(func_name: str) -> set[str]:
    """Every string key `func_name` subscripts or `.get()`s off its `frames` argument.

    Read off the source rather than asserted by hand: a test that restates the key list is
    a second copy of the contract, and the copy is what drifts.
    """
    fn = None
    for path in sorted(EVENTS.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        found = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == func_name]
        assert len(found) < 2, f"{func_name} defined twice in {path.name}"
        if found:
            assert fn is None, (
                f"{func_name} defined in more than one module under {EVENTS.name}"
            )
            fn = found[0]
    assert fn is not None, f"no top-level {func_name} anywhere under {EVENTS}"
    keys: set[str] = set()
    for n in ast.walk(fn):
        if (
            isinstance(n, ast.Subscript)
            and isinstance(n.slice, ast.Constant)
            and isinstance(n.slice.value, str)
        ):
            keys.add(n.slice.value)
        if (
            isinstance(n, ast.Call)
            and isinstance(n.func, ast.Attribute)
            and n.func.attr == "get"
            and n.args
            and isinstance(n.args[0], ast.Constant)
        ):
            keys.add(n.args[0].value)
    return keys


def test_frame_index_is_the_one_key_everything_carries():
    """It is the unit of record. Every payload states it; nothing else is universal."""
    assert FrameRef.__required_keys__ == frozenset({"frame_index"})
    for t in (BoxFrame, TerrainFrame, StageFrame):
        assert "frame_index" in (t.__required_keys__ | t.__optional_keys__)


def test_boxframe_asks_for_exactly_what_its_consumer_reads():
    read = _keys_read("zone_stock_counts")
    assert set(BoxFrame.__required_keys__) == read, (
        f"BoxFrame requires {sorted(BoxFrame.__required_keys__)} and zone_stock_counts "
        f"reads {sorted(read)}. A required key nobody reads is a producer's cost for "
        "nothing; a key that is read and not required is a KeyError waiting for a caller."
    )


def test_terrainframe_covers_what_its_consumer_reads():
    read = _keys_read("reach_to_shelf_events")
    declared = TerrainFrame.__required_keys__ | TerrainFrame.__optional_keys__
    assert read <= declared, f"reach_to_shelf_events reads {sorted(read - declared)} undeclared"
    assert TerrainFrame.__required_keys__ == frozenset({"frame_index"}), (
        "`terrain` stays optional: the consumer raises a sentence naming the missing "
        "terrain head, which is a better error than a type that cannot express the case."
    )


def test_the_full_payload_carries_both_halves():
    """`StageFrame` is what a producer with everything states, and it satisfies both."""
    full = StageFrame.__required_keys__ | StageFrame.__optional_keys__
    assert BoxFrame.__required_keys__ <= full
    assert (TerrainFrame.__required_keys__ | TerrainFrame.__optional_keys__) <= full


def test_image_is_required_of_a_full_payload_and_optional_of_a_terrain_one():
    """The asymmetry is the design. A producer that has the frame must say so -- the crop
    encoder is the reason this stage exists. A producer that only has a dense map must not
    be forced to invent one."""
    assert "image" in StageFrame.__required_keys__
    assert "image" in TerrainFrame.__optional_keys__


def test_a_terrain_payload_that_the_old_contract_rejected_still_works():
    """`scripts/pose_pilot.py`'s shape, end to end. Under the single six-key StageFrame
    this was a type error, while being the only real payload anyone had built."""
    fixture_id = 4
    terrain = np.zeros((240, 320), dtype=np.int64)
    terrain[150:200, 120:200] = fixture_id
    frames: list[TerrainFrame] = [{"frame_index": i, "terrain": terrain} for i in range(6)]
    # No track carries keypoints here, so the call refuses on that and not on the payload:
    # what is under test is that the payload shape gets far enough to be refused for the
    # right reason.
    with pytest.raises(NotImplementedError, match="keypoint"):
        ev.reach_to_shelf_events(
            [_TrackWithoutPose()], frames, 5.0, "cam01", fixture_id=fixture_id
        )


class _TrackWithoutPose:
    def __init__(self) -> None:
        self.track_id = 1
        self.frames = [0, 1, 2]
        self.boxes: list = []
        self.keypoints: list = []
