"""Two 3D panels, and the property that stops either being deleted as a duplicate.

`scene_mesh.py` arrived on 2026-08-25 and the README's figures moved to it the same week.
Ever since, `bev3d.py`'s perspective panel has looked like the previous generation of the
same thing -- and on 2026-08-30 it was read that way, by someone shown a `bev3d` panel and
told it was current. The commit that introduced `scene_mesh` (`7a83e0e`) does not mention
`bev3d` at all, so arriving at that conclusion took no carelessness.

**They take different inputs and answer different questions.**

    bev3d.render(trav_bev, terrain_bev, grid, objects, ...)   arrays, from a live forward pass
    scene_mesh.build_scene_regular(camera, root)              a camera name -> `runs/` artefacts

That difference is the whole argument, so it is what these tests hold rather than the
paragraphs that state it. A prose check would pass on documentation nobody had verified;
this fails if the property stops being true -- if `bev3d` grows a commissioning
dependency, or if `scene_mesh` learns to work without one and the pair really does become
a duplicate.

**The failure being prevented is a deletion.** `scene_mesh` can only draw the 8 of 48
cameras that have a `camera.json`. Removing `bev3d` because it looks superseded would take
the 3D panel away from the other 40, and from any footage on a camera nobody has
commissioned yet -- which is every camera on the day it is installed.
"""

from __future__ import annotations

import inspect

import numpy as np
import pytest
from PIL import Image

from syncai_bev3d import bev3d, scene_mesh
from syncai_bev3d.bev import IGNORE, BevGrid

GRID = BevGrid(z_max=8.0)
TRAV_COLORS = ((40, 40, 40), (60, 160, 90), (170, 60, 60))


def _floor(rows: int) -> np.ndarray:
    """A patch of free space, as a live traversability map would arrive."""
    trav = np.zeros(GRID.shape, np.uint8)
    trav[-rows:, :] = 1
    return trav


def test_the_perspective_panel_draws_from_arrays_alone():
    """No `camera.json`, no plate, no masks -- the reason this module cannot be retired.

    Everything it needs is in the call. A camera installed this morning has none of the
    commissioning artefacts `scene_mesh` reads, and this is the only 3D panel it can have.
    """
    panel = bev3d.render(
        _floor(20), None, GRID, [], (256, 256), trav_colors=TRAV_COLORS, bg=(0, 0, 0)
    )
    assert isinstance(panel, Image.Image)
    assert np.asarray(panel).any(), "nothing was projected; the panel is background only"


def test_the_perspective_panel_takes_no_camera_identity():
    """A commissioning dependency would arrive as a camera argument, so watch for one.

    `render` naming a camera would mean this panel had become the other one, and the
    fleet's 40 uncommissioned cameras would lose their only 3D view without anything
    failing loudly enough to notice.
    """
    params = set(inspect.signature(bev3d.render).parameters)
    assert not (params & {"camera", "camera_id", "root", "camera_file"}), (
        f"bev3d.render now takes {sorted(params)} -- if it needs commissioning artefacts "
        "it no longer answers the question it exists for"
    )


def test_the_mesh_scene_is_addressed_by_camera_and_therefore_needs_commissioning():
    """The other half: `scene_mesh` reads what was measured, so it needs a measurement.

    Asserted on the signature rather than by calling it, because calling it needs `runs/`
    -- which is gitignored and absent in CI, and a test that can only run on one box is
    the shape `test_figures_are_audited.py` opens by warning about.
    """
    params = list(inspect.signature(scene_mesh.build_scene_regular).parameters)
    assert params[0] == "camera", (
        "build_scene_regular no longer takes a camera; if it can draw a scene without "
        "commissioning artefacts then the two panels really have converged, and "
        "tools/README.md and both module docstrings need rewriting rather than trusting"
    )


def test_an_uncommissioned_camera_gets_no_mesh_scene():
    """The consequence, made concrete: a camera with no artefacts cannot be drawn.

    This is the sentence the documentation rests on -- 8 of 48 -- and it is the difference
    between "duplicate" and "two answers". A name that has never been commissioned must
    fail rather than return an empty room, because an empty room renders fine and reads
    as a store with nothing in it.
    """
    with pytest.raises(FileNotFoundError) as exc:
        scene_mesh.build_scene_regular("no-such-camera-0000")
    # The named artefact, not just any failure: `pytest.raises(Exception)` would also pass
    # on a typo in this test, which is the sort of green nobody notices.
    assert "camera.json" in str(exc.value), (
        f"expected the missing commissioning artefact to be named; got {exc.value}"
    )


def test_both_panels_are_still_reachable():
    """A guard against the tidy-up this file exists to prevent, in its simplest form."""
    assert callable(bev3d.render)
    assert callable(scene_mesh.build_scene_regular)
    assert IGNORE is not None, "the flat primitive both panels are drawn from"
