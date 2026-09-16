"""Known camera and independently projected object silhouettes for tests."""

import math

import numpy as np
from PIL import Image, ImageDraw

from syncai_hydranet.geometry.camera_json import CameraFile
from syncai_hydranet.geometry.ground import Camera, GroundPlane, distort_points


def camera():
    return CameraFile(
        camera_id="objects",
        image_size_px=(800, 600),
        camera=Camera(fx=640, fy=640, cx=400, cy=300),
        plane=GroundPlane(height=2.8, pitch=math.radians(28), roll=math.radians(4)),
    )


def observed(mesh, cf):
    # Generate the observation from known world coordinates independently of the fitter.
    vertices, faces = mesh
    level = vertices.copy()
    level[:, 1] = cf.plane.height - level[:, 1]
    points = level @ cf.plane.rotation.T
    px = points[:, :2] / points[:, 2, None] * [cf.camera.fx, cf.camera.fy] + [
        cf.camera.cx,
        cf.camera.cy,
    ]
    if cf.lens:
        px = distort_points(px, cf.lens.k1, cf.lens.centre_px, cf.lens.radius_px)
    image = Image.new("1", cf.image_size_px)
    draw = ImageDraw.Draw(image)
    for face in faces:
        draw.polygon(list(map(tuple, px[face])), fill=1)
    return np.asarray(image, bool)
