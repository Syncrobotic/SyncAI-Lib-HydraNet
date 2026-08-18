"""Geometry between the image and the floor: ground-plane fitting and BEV projection.

**This package is copied out of the repository and run elsewhere**, which makes its
runtime assumptions someone else's problem rather than only ours. As of 2026-08-18 a copy
runs on a Lite3 quadruped, rendering `bev3d.render` panels live from the NPU's output.
That robot's Python is **3.8** and its Pillow was **7.0**; this package targets the
`requires-python = ">=3.10"` the project declares. Four of the five patches the copy
needs are pure re-spellings with identical behaviour:

    scene_types.py:143   `Scene = PlaneScene | DepthScene`  -> Union      runtime alias
    meshes.py:39         `Mesh = tuple[np.ndarray, ...]`    -> Tuple      runtime subscript
    bev3d.py:387,420     `Image.Resampling`/`Image.Transform` -> the flat constants
    bev3d.py:639         `rounded_rectangle`                 -> needs Pillow >= 8.2

The fifth is not, and it is the one to carry forward. **Dropping `strict=True` from
`zip` does not re-spell the call, it deletes a guard**: 3.10 raises on ragged inputs and
3.8 silently truncates to the shortest. What each of the three sites is holding together:

    bev.py:196       positions, extents, labels and scores, one per detection. A short
                     `labels` or `scores` silently drops detections off the end of the
                     floor map, and every object that survives keeps its own label -- so
                     the panel is missing boxes and looks entirely correct.
    bev3d.py:144     the four panel corners against the four raster corners of the
                     perspective transform. Fewer pairs is a homography fitted to fewer
                     correspondences: the floor warps wrongly rather than failing.
    shading.py:185   faces, depths and colours, all three derived from `faces` two lines
                     above, so a mismatch here is not reachable from outside.

None of this is a defect here: 3.10 is the floor this project supports and these are the
idioms that floor buys. It is written down because the divergence is real, and an
undocumented one is how two copies stop being the same code -- the failure
`test_orin_standalone_copies.py` exists to prevent for the Orin's bench script. The robot
copy's own report is that its zipped sequences are always equal-length and no logic was
changed beyond these five, camera parameters being passed in by its caller rather than
patched in. If that copy is ever meant to track this one, that test is the pattern; if it
is not, this note is what tells the next reader why they differ and which difference can
bite silently.

The Pillow bump is part of the cost and not a source patch: 7.0 -> 10.4 on the robot,
required by `rounded_rectangle` alone.
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
