"""First-pass scene analysis: calibration, structure masks, BEV -- once per camera.

Everything here runs offline at commissioning and produces the per-camera constants
(`camera.json`, cached masks, the BEV scene) that `syncai_hydranet` consumes at runtime.
The runtime side of the geometry -- `Camera`, `GroundPlane`, the projections and
`undistort_points` -- lives in `syncai_hydranet.geometry` and is imported from there:
commissioning fits the parameters, serving applies them, and both sides sharing one
definition is what keeps the metres honest. The reverse direction is banned and tested:
no serving module imports this package (`tests/test_package_boundaries.py`).

A historical note carried over from `syncai_hydranet/geometry/__init__.py`, because it
concerns files that now live here. **A copy of these modules ran on a Lite3 quadruped**
(Python 3.8, Pillow 7.0) before that line was removed on 2026-08-19 (`cc80fc3`). Four of
the five patches that copy needed were pure re-spellings; the fifth was not: **dropping
`strict=True` from `zip` deletes a guard**, not a spelling -- 3.10 raises on ragged
inputs and 3.8 silently truncates. The three sites it holds together:

    bev.py       positions, extents, labels and scores, one per detection. A short
                 `labels` silently drops detections off the floor map and the panel
                 still looks entirely correct.
    bev3d.py     the four panel corners against the four raster corners of the
                 perspective transform. Fewer pairs is a homography fitted to fewer
                 correspondences: the floor warps wrongly rather than failing.
    shading.py   faces, depths and colours, all derived from `faces`, so a mismatch
                 is not reachable from outside.

If a divergent copy of this package ever matters again, `test_orin_standalone_copies.py`
was the pattern for keeping two copies honest.
"""

from .bev import (
    BevGrid,
    free_space_map,
    place_boxes,
    project_mask,
    ray_reach,
    scene,
)

__all__ = [
    "BevGrid",
    "free_space_map",
    "place_boxes",
    "project_mask",
    "ray_reach",
    "scene",
]
