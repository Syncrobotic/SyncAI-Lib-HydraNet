"""The bench that judges a depth source, and the control that stops it judging wrongly.

`tools/commissioning/geometry_bench.py` exists because "is this depth model better" had
no answer here. Every metre rests on the 1.70 m person prior, so there is no ruler --
except the floor, which is at floor level by definition on any camera with a ground plane
and a `walkable` mask.

The bench nearly shipped measuring the wrong thing. Flatness alone is won outright by a
source that returns the ground plane and nothing else, and a reader shown only the floor
columns would have adopted it. That is why `flat(ctl)` runs on every invocation instead
of being an option, and why the two halves are printed side by side and never summed.

Constructed from synthetic arrays rather than `runs/`, which is gitignored and absent in
CI -- the same constraint `test_renderer_generations.py` records. The properties held here
are arithmetic ones, so they do not need a real store to be true.
"""

from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

REPO = Path(__file__).resolve().parent.parent


def _load():
    name = "_geometry_bench"
    spec = importlib.util.spec_from_file_location(
        name, REPO / "tools" / "commissioning" / "geometry_bench.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # `@dataclass` resolves string annotations through this
    spec.loader.exec_module(module)
    return module


gb = _load()


def _fake_camera(root: Path, cam: str, *, walkable=True, shelf=True, vfov=70.4):
    """A camera.json / calib.json / plate / masks tree the bench can read.

    Small (120x80 plate) so the 1080x1920 undistort lattice stays the only slow part.
    """
    (root / "runs/onboard01").mkdir(parents=True, exist_ok=True)
    (root / "runs/commission01").mkdir(parents=True, exist_ok=True)
    (root / "plates").mkdir(parents=True, exist_ok=True)
    plate = root / "plates" / f"{cam}.png"
    Image.fromarray(np.zeros((80, 120, 3), np.uint8)).save(plate)
    (root / f"runs/onboard01/{cam}.calib.json").write_text(
        json.dumps(
            {
                "vfov_assumed_deg": vfov,
                "k1_division_model": 0.0,
                "height_m": 2.5,
                "pitch_deg": 45.0,
                "roll_deg": 0.0,
                "scale": 1.0,
                "plate_used": str(plate),
            }
        )
    )
    masks = {}
    for name, on in (("walkable", walkable), ("display_shelf", shelf)):
        if not on:
            continue
        a = np.zeros((80, 120), np.uint8)
        a[40:, :] = 255 if name == "walkable" else 0
        if name == "display_shelf":
            a[10:30, 20:100] = 255
        Image.fromarray(a).save(root / "runs/commission01" / f"{cam}.{name}.png")
        masks[name] = f"{cam}.{name}.png"
    (root / f"runs/commission01/{cam}.camera.json").write_text(
        json.dumps({"mask_files": masks, "image_size_px": [1920, 1080]})
    )
    return gb._geometry(cam, root)


def test_the_ground_plane_reads_as_zero_height_everywhere(tmp_path):
    """The bench's whole premise, as arithmetic: the floor is at floor level.

    If this drifts, every floor number the bench prints is measuring something else and
    the comparisons it is used for are void.
    """
    geo = _fake_camera(tmp_path, "synth")
    h = gb.height_map(gb.source_flat(geo), geo)
    v = h[np.isfinite(h)]
    assert v.size > 1000
    assert abs(float(np.median(v))) < 1e-6, "the plane is not at its own height"
    assert float(np.percentile(v, 99) - np.percentile(v, 1)) < 1e-6, "the plane is not flat"


def test_the_flat_control_wins_the_floor_and_reports_no_relief(tmp_path):
    """The reason the control is unconditional, stated as the numbers it produces.

    A source that has learned only "floors are flat" scores 0.000 / 0.000 -- better than
    any real depth model can -- and finds nothing above the floor. Reading the floor
    columns as a verdict would rank it first.
    """
    geo = _fake_camera(tmp_path, "synth")
    r = gb.score("synth", "flat(ctl)", gb.source_flat(geo), geo, tmp_path)
    assert not r.abstained, r.abstained
    assert r.floor_offset_m == pytest.approx(0.0, abs=1e-6)
    assert r.floor_spread_m == pytest.approx(0.0, abs=1e-6)
    assert r.relief_m == {}, (
        f"the ground plane reported fixture relief {r.relief_m}; either the control is no "
        "longer degenerate or the relief half is reading the floor"
    )


def test_relief_separates_a_source_that_sees_a_fixture(tmp_path):
    """The other half, so the pair is shown to actually discriminate.

    A plane plus a 1.5 m step over the shelf mask must score the same floor as the plane
    and a shelf that is not None -- otherwise the two halves are not independent and the
    control proves nothing.
    """
    geo = _fake_camera(tmp_path, "synth")
    flat = gb.source_flat(geo)
    raised = flat.copy()
    # Pull the shelf band towards the camera; nearer than the floor means taller.
    raised[10:30, 20:100] *= 0.55
    r = gb.score("synth", "stepped", raised, geo, tmp_path)
    assert not r.abstained, r.abstained
    assert r.floor_offset_m == pytest.approx(0.0, abs=1e-6), "the floor was disturbed"
    assert r.relief_m.get("display_shelf", 0.0) > 0.3, (
        f"a fixture standing well clear of the floor reported {r.relief_m}"
    )


def test_a_camera_without_a_walkable_mask_abstains(tmp_path):
    """No floor, no witness. Returning zeros here would read as a perfect score."""
    geo = _fake_camera(tmp_path, "nofloor", walkable=False)
    r = gb.score("nofloor", "flat(ctl)", gb.source_flat(geo), geo, tmp_path)
    assert r.abstained
    assert "walkable" in r.abstained
    assert math.isnan(r.floor_offset_m)


def test_a_source_that_returns_nothing_abstains_rather_than_scoring(tmp_path):
    geo = _fake_camera(tmp_path, "synth")
    r = gb.score("synth", "dead", None, geo, tmp_path)
    assert r.abstained
    assert math.isnan(r.floor_offset_m)


def test_an_abstention_is_rendered_as_one():
    """It has to be visible in the table, or an absent row reads as a passing row."""
    out = gb.render([gb.Reading("cam", "src", abstained="no walkable mask")])
    assert "ABSTAINED" in out and "no walkable mask" in out


def test_the_bench_reads_heights_at_the_percentiles_the_renderer_draws():
    """A bench scored at p50 would rank sources the 3D scene does not agree with.

    `scene_mesh.HEIGHT_PCT` is keyed by class id and this is keyed by name, so the two
    cannot be one object without dragging scipy and the whole mesh builder in for four
    integers. This is the join, checked rather than commented.
    """
    from syncai_bev3d.scene_mesh import CLASS_NAMES, HEIGHT_PCT

    mesh = {CLASS_NAMES[cid]: pct for cid, pct in HEIGHT_PCT.items() if cid in CLASS_NAMES}
    for name, pct in gb.HEIGHT_PCT.items():
        assert mesh.get(name) == pct, (
            f"the bench reads {name} at p{pct}; scene_mesh draws it at p{mesh.get(name)}"
        )
