"""The runtime geometry contract: camera, ground plane, projection, lens.

This package used to hold everything between the image and the floor. The producing
side -- plate calibration, k1 fitting, pose recovery, BEV and scene rendering -- moved
to `syncai_bev3d`, which runs once per camera at commissioning and writes `camera.json`.
What stays here is what a serving process applies every frame to read that file back
into metres. `tests/test_package_boundaries.py` holds the direction: `syncai_bev3d`
imports this module, and nothing on the serving path imports `syncai_bev3d`.
"""

from .camera_json import CameraFile, Lens, Zone
from .ground import (
    Camera,
    GroundPlane,
    fit_ground_plane,
    ground_to_pixel,
    pixel_to_ground,
    undistort_points,
    unproject,
)

__all__ = [
    "Camera",
    "CameraFile",
    "GroundPlane",
    "Lens",
    "Zone",
    "fit_ground_plane",
    "ground_to_pixel",
    "pixel_to_ground",
    "undistort_points",
    "unproject",
]
