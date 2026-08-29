"""Project the commissioned 3D scene back through its own camera, onto its own plate.

The metre-grid review answers "is the floor scaled right". It cannot answer whether a
fixture is the right *size*, in the right *place*, at the right *orientation*, because a
grid has none of those. This does: every mesh the scene builds is projected with the
camera model in `camera.json` and drawn as a wireframe over the plate it was built from.
A fixture that is too deep, rotated by the store-yaw fit, or sitting a metre behind
where it really is, misses the furniture underneath it and says so in one picture.

Usage:
  uv run python tools/commissioning/scene_overlay.py <camera> [--metre-scale 1.0]
"""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from syncai_bev3d import scene_mesh
from syncai_hydranet.geometry.camera_json import CameraFile

ROOT = Path("/home/paul/SyncAI-Lib-HydraNet")


def project(verts: np.ndarray, cf: CameraFile, scale: float) -> tuple[np.ndarray, np.ndarray]:
    """Scene vertices (x lateral, y up from the floor, z forward) -> plate pixels + depth.

    The scene's y is height above the floor; the camera model's level frame has y down
    from the camera, so the floor sits at `plane.height`. That single substitution is the
    whole transform, and getting its sign wrong puts the room on the ceiling.
    """
    v = np.asarray(verts, float) / scale  # back into camera.json's own metres
    level = np.stack([v[:, 0], cf.plane.height - v[:, 1], v[:, 2]], axis=-1)
    cam = level @ cf.plane.rotation.T
    with np.errstate(divide="ignore", invalid="ignore"):
        u = cf.camera.fx * cam[:, 0] / cam[:, 2] + cf.camera.cx
        w = cf.camera.fy * cam[:, 1] / cam[:, 2] + cf.camera.cy
    return np.stack([u, w], axis=-1), cam[:, 2]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("camera")
    ap.add_argument("--metre-scale", type=float, default=1.0)
    ap.add_argument("--out", default=None)
    ap.add_argument(
        "--classes",
        default="display_table,display_shelf,column",
        help="comma-separated palette keys to draw; walls at a constant 2.4 m project "
        "as huge planes and drown everything measured",
    )
    args = ap.parse_args()

    cf = CameraFile.load(ROOT / f"runs/commission01/{args.camera}.camera.json")
    plate = Image.open(ROOT / cf.plate_file).convert("RGB")
    w_px, h_px = cf.image_size_px
    plate = plate.resize((w_px, h_px))

    scene_mesh.SS = 1
    _cf, items, _heights, shapes = scene_mesh.build_scene_regular(args.camera)
    if args.metre_scale != 1.0:
        items = [((m[0] * args.metre_scale, m[1]), *rest) for m, *rest in items]

    up = 2
    img = plate.resize((w_px * up, h_px * up), Image.Resampling.LANCZOS)
    d = ImageDraw.Draw(img, "RGBA")
    wanted = {c.strip() for c in args.classes.split(",") if c.strip()}
    drawn = 0
    for mesh, key, _alpha, _s in items:
        if key not in wanted:
            continue
        rgb = scene_mesh.PALETTE.get(key, (200, 200, 200))
        uv, depth = project(mesh[0], cf, args.metre_scale)
        for face in mesh[1]:
            if not (depth[face] > 0).all():
                continue
            pts = [(float(uv[i, 0]) * up, float(uv[i, 1]) * up) for i in face]
            d.polygon(pts, outline=(*rgb, 230))
        drawn += 1
    # This is the falsifiable view -- the wireframe over the plate it was built from --
    # so it is the one place a fixture the code does not believe in most needs saying.
    said = scene_mesh.implausible_caption(shapes)
    if said:
        # On its own the line is orange text over a lit shop floor and unreadable, which
        # is the same defect as printing it to a terminal nobody reads. The band is drawn
        # only when there is something to say, so a clean overlay is unchanged.
        d.rectangle([0, 0, img.width, 42], fill=(0, 0, 0, 175))
    d.text(
        (8, 8), f"{args.camera}  scene projected back through camera.json", fill=(255, 255, 255)
    )
    if said:
        d.text((8, 24), said, fill=(246, 178, 106))
    out = Path(args.out or ROOT / f"assets/scene_overlay_{args.camera}.png")
    img.save(out)
    print(f"wrote {out}  ({drawn} meshes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
