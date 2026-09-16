"""Independent synthetic surface observations shared by geometry tests."""

import math
from types import SimpleNamespace

import numpy as np
from PIL import Image, ImageDraw

from syncai_hydranet.geometry.camera_json import CameraFile
from syncai_hydranet.geometry.ground import Camera, GroundPlane, pixel_to_ground


def evidence(bottom=0.0, top=2.1):
    w, h = 640, 480
    cf = CameraFile(
        camera_id="surface",
        image_size_px=(w, h),
        camera=Camera(fx=420, fy=420, cx=w / 2, cy=h / 2),
        plane=GroundPlane(height=2.8, pitch=math.radians(25)),
    )
    world = np.array([[-1.5, bottom, 5], [-1.5, top, 5], [1.5, top, 5], [1.5, bottom, 5]])
    level = world.copy()
    level[:, 1] = cf.plane.height - level[:, 1]
    camera = level @ cf.plane.rotation.T
    px = camera[:, :2] / camera[:, 2, None] * 420 + [w / 2, h / 2]
    im = Image.new("1", (w, h))
    ImageDraw.Draw(im).polygon(list(map(tuple, px)), fill=1)
    mask = np.asarray(im, bool)
    r, c = np.mgrid[:h, :w]
    gx, gz = pixel_to_ground(c + 0.5, r + 0.5, cf.camera, cf.plane)
    walk = np.isfinite(gz) & (gz < 4.95) & (gz > 0) & ~mask
    ev = SimpleNamespace(cf=cf, z={"gx": gx, "gz": gz}, walk=walk)
    return mask, ev
