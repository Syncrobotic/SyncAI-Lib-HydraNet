#!/usr/bin/env python3
"""Serve the 3D scene page against a synthetic frame, with no robot attached.

    python3 scripts/bev_demo.py --port 8091     # then open http://localhost:8091/3d

The renderer's input is a depth image, an intrinsic matrix and a traversability mask.
This builds one: a corridor with two objects in it, and a patch of floor that returns no
depth -- the polished-floor case measured on the real robot at 10.6% of walkable pixels.

It exists because the page is otherwise only reviewable when a robot is powered on and
on the network, which is a bad property for the piece of the system whose whole job is to
be looked at. It also makes the scene format inspectable: `curl localhost:8091/scene.json`.
"""

from __future__ import annotations

import argparse
import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
# In the repo the package is at ../src; on a robot the deploy copies it to ./src next to
# this script. Accept both rather than making the caller care which layout they are in.
for candidate in (HERE.parent / "src", HERE / "src"):
    if candidate.is_dir():
        sys.path.insert(0, str(candidate))

from bev_page import PAGE  # noqa: E402

from syncai_hydranet.geometry.depth_scene import BLOCKED, GO, build_scene  # noqa: E402

H, W = 240, 320
K = np.array([[260.0, 0.0, W / 2], [0.0, 260.0, H / 2], [0.0, 0.0, 1.0]])
CAMERA_HEIGHT = 0.6  # metres; the camera looks along the floor from a robot's height
OBJECT_BOXES: list[tuple[int, int, int, int]] = []  # filled by synthetic_frame()


def synthetic_frame() -> tuple[np.ndarray, np.ndarray]:
    """A corridor: floor below the horizon, walls either side, two objects, one blind patch."""
    us, vs = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
    fy, cy, fx, cx = K[1, 1], K[1, 2], K[0, 0], K[0, 2]

    # Floor: a horizontal plane CAMERA_HEIGHT below the optical centre. Every pixel below
    # the horizon hits it at z = fy * height / (v - cy).
    below = vs > cy + 2
    depth = np.zeros((H, W), np.float32)
    depth[below] = fy * CAMERA_HEIGHT / (vs[below] - cy)
    trav = np.full((H, W), BLOCKED, np.uint8)
    trav[below] = GO

    # Walls at +-1.6 m: anything whose floor point falls outside the corridor is wall.
    x_floor = (us - cx) * depth / fx
    wall = below & (np.abs(x_floor) > 1.6)
    trav[wall] = BLOCKED

    # Two objects standing on the floor, at 2.0 m and 3.4 m.
    _boxes = []
    for z, x_c, half_w, height in (
        (2.0, -0.6, 0.28, 0.85),
        (3.4, 0.7, 0.30, 0.9),
    ):
        u_c = cx + fx * x_c / z
        u_half = fx * half_w / z
        v_top = cy + fy * (CAMERA_HEIGHT - height) / z
        v_bot = cy + fy * CAMERA_HEIGHT / z
        sel = (np.abs(us - u_c) < u_half) & (vs > v_top) & (vs < v_bot)
        depth[sel] = z
        trav[sel] = BLOCKED
        ys, xs = np.nonzero(sel)
        _boxes.append((int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1))

    OBJECT_BOXES.clear()
    OBJECT_BOXES.extend(_boxes)

    # The polished patch: the model still calls it floor, the sensor returns nothing.
    blind = below & (np.abs(x_floor) < 0.7) & (depth > 1.2) & (depth < 1.9)
    depth[blind] = 0.0
    return depth, trav


def scene_payload() -> dict:
    depth, trav = synthetic_frame()
    dets = []
    # Boxes as a detector would give them: tight around the object, and therefore still
    # containing the floor and wall visible around its edges.
    for u0, v0, u1, v1 in OBJECT_BOXES:
        pad = 4
        dets.append(
            {
                "box": (u0 - pad, v0 - pad, u1 + pad, v1 + pad),
                "cls": "chair",
                "score": 0.8,
            }
        )
    scene = build_scene(trav, depth, K, dets)
    scene["range_m"] = 5.0
    scene["synthetic"] = True
    return scene


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_):
        pass

    def do_GET(self):
        if self.path in ("/", "/3d"):
            body, ctype = PAGE, "text/html; charset=utf-8"
        elif self.path == "/scene.json":
            body, ctype = json.dumps(scene_payload()).encode(), "application/json"
        else:
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", type=int, default=8091)
    args = ap.parse_args()
    scene = scene_payload()
    print(
        f"synthetic scene: {len(scene['objects'])} objects, "
        f"{sum(1 for c in scene['grid']['cells'] if c)} occupied cells, "
        f"{scene['grid']['blind_fraction']:.1%} of walkable pixels blind"
    )
    print(f"open http://localhost:{args.port}/3d")
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
