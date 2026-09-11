"""The floor's joints as a ruler for each camera's metres.

Every commissioned camera's metres rest on the 1.70 m person prior, read off 15-47 boxes,
and on 2026-09-10 the tile pitch measured through those metres disagreed by up to 28%
between cameras over ONE floor (Taichung: 0.47 m on cam01 against 0.61 / 0.66 / 0.57 on
cam04 / cam10 / cam11). A tile is one size everywhere it is laid, so the pitch in a
camera's metres is a scale check that needs no person in the frame -- which is what the
13 selling-floor cameras with `scale_source: unmeasured` lack.

Per camera this prints the store axis (floor lines or fixture blobs), the joint pitch in
the camera's metres, and -- given the real pitch -- the camera height the pitch implies
against the one the person prior gave. With `--write-root`, a REVIEW copy of the camera
is written under a separate checkout-shaped root, rescaled by real / measured: the
camera height and every zone vertex scaled, the masks linked, and the depth scale to
rebuild the geometry cache with printed. Nothing under runs/commission01 is touched.

Usage:
  uv run python tools/commissioning/tile_ruler.py Taichung-cam01 Taichung-cam10 --tile-m 0.60
  uv run python tools/commissioning/tile_ruler.py Taichung-cam01 --tile-m 0.60 \
      --write-root runs/commission_review/tile_cam01
"""

import argparse
import os
from pathlib import Path

import numpy as np

from syncai_bev3d import floor_axis, scene_mesh
from syncai_bev3d.rulers import write_scaled_root
from syncai_hydranet.geometry.ground import pixel_to_ground, undistort_points

ROOT = Path(os.environ.get("SYNCAI_ROOT", Path(__file__).resolve().parents[2]))
# Below this autocorrelation the pitch is not trusted to rescale anything.
STRENGTH_MIN = 0.35


def read(camera):
    ev = scene_mesh.load_evidence(camera)
    cf = ev.cf
    fh, fw = ev.z["gx"].shape
    w, h = cf.image_size_px
    sc = np.array([w / fw, h / fh])

    def ground(px):
        pts = px * sc
        if cf.lens is not None:
            pts = undistort_points(pts, cf.lens.k1, cf.lens.centre_px, cf.lens.radius_px)
        x, z = pixel_to_ground(pts[:, 0], pts[:, 1], cf.camera, cf.plane)
        return np.stack([x, z], axis=1)

    axis, _second = floor_axis.floor_line_axis(
        ev.plate, ev.walk, ev.z["gz"], ev.z["geom_ok"], ground
    )
    source = "floor lines"
    if axis is None:
        axis = scene_mesh.store_yaw(scene_mesh.cell_grids(camera, gated=False, evidence=ev)[1])
        source = "fixture blobs"
    period, strength = floor_axis.floor_period(
        ev.plate, ev.walk, ev.z["gz"], ev.z["geom_ok"], ground, axis
    )
    return ev, axis, source, period, strength


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("cameras", nargs="+")
    ap.add_argument("--tile-m", type=float, default=None, help="the real joint pitch, if known")
    ap.add_argument("--write-root", type=Path, default=None, help="review root for ONE camera")
    a = ap.parse_args()
    if a.write_root is not None and (a.tile_m is None or len(a.cameras) != 1):
        ap.error("--write-root takes one camera and needs --tile-m")
    for camera in a.cameras:
        ev, axis, source, period, strength = read(camera)
        h = ev.cf.plane.height
        line = f"{camera:16s} axis {np.degrees(axis):5.1f} deg ({source:13s})  H {h:4.2f} m"
        if period is None or strength < STRENGTH_MIN:
            print(line + f"  pitch: none (strength {strength:.2f})")
            continue
        line += f"  pitch {period:.2f} m (strength {strength:.2f})"
        if a.tile_m:
            factor = a.tile_m / period
            line += (
                f"  -> at {a.tile_m:.2f} m tiles H would be {h * factor:.2f} m (x{factor:.3f})"
            )
        print(line)
        if a.write_root is not None:
            factor = a.tile_m / period
            cache = ROOT / f"runs/site30k_qa/geometry_cache/{camera}.npz"
            with np.load(cache) as z:
                old_scale = float(z["depth_scale"]) if "depth_scale" in z.files else None
            root = write_scaled_root(
                camera,
                ev.cf,
                factor,
                ROOT,
                a.write_root,
                f"tile pitch {period:.2f} m measured vs {a.tile_m:.2f} m real",
            )
            print(f"  wrote {root}")
            if old_scale is not None:
                print(
                    "  rebuild the cache with: uv run python tools/commissioning/"
                    f"rebuild_geometry.py {root}/runs/commission01/{camera}.camera.json "
                    f"--depth-scale {old_scale * factor:.4f} "
                    f"--out {root}/runs/site30k_qa/geometry_cache/{camera}.npz"
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
