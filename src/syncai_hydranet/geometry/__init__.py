"""Geometry between the image and the floor: ground-plane fitting and BEV projection.

**This package is copied out of the repository and run elsewhere**, which makes its
runtime assumptions someone else's problem rather than only ours. As of 2026-08-18 a copy
runs on a Lite3 quadruped, rendering `bev3d.render` panels live from the NPU's output.
That robot's Python is **3.8** and its Pillow was **7.0**; this package targets the
`requires-python = ">=3.10"` the project declares, so five things here need patching on
the way over and were patched in the robot's copy, not here:

    scene_types.py:143   `Scene = PlaneScene | DepthScene`   runtime alias, 3.10+
    meshes.py:39         `Mesh = tuple[np.ndarray, ...]`     runtime subscript, 3.9+
    bev.py, shading.py, bev3d.py   `zip(..., strict=True)`   3.10+
    bev3d.py:387,420     `Image.Resampling` / `Image.Transform`   Pillow 9.1+
    bev3d.py:639         `rounded_rectangle`                 Pillow 8.2+

Nothing above is a defect: 3.10 is the floor this project supports and these are the
idioms that floor buys. It is written down because the divergence is now real and
undocumented divergence is how two copies stop being the same code -- the failure
`test_orin_standalone_copies.py` exists to prevent for the Orin's standalone bench
script. If the robot copy is ever meant to track this one, that test is the pattern to
follow; if it is not, this note is what tells the next reader why the two differ.
"""

from .bev import (
    IGNORE,
    BevGrid,
    free_space_map,
    place_boxes,
    project_mask,
    ray_reach,
    scene,
)
from .ground import (
    Camera,
    GroundPlane,
    fit_ground_plane,
    ground_to_pixel,
    pixel_to_ground,
    unproject,
)

__all__ = [
    "IGNORE",
    "BevGrid",
    "Camera",
    "GroundPlane",
    "fit_ground_plane",
    "free_space_map",
    "ground_to_pixel",
    "pixel_to_ground",
    "place_boxes",
    "project_mask",
    "ray_reach",
    "scene",
    "unproject",
]
