"""A dwell or traffic heatmap, draped on the commissioned 3D scene's floor.

    uv run python tools/commissioning/heatmap3d.py <camera> [--mode dwell|traffic]
        [--tracks PATH] [--cell 0.25] [--out PATH]

Reads the per-frame floor positions `demo_video` records in
`runs/commission01/<camera>.demo_tracks.json` (or any file of that shape via
`--tracks`), builds the grid in `syncai_bev3d.heatmap` -- where the maths lives and is
tested -- and renders amber tiles at y=0.012 into the same solid-furniture scene the
demo uses, clipped to the walkable polygon. The legend prints the scale top and the
window, because a heatmap without its scale is a picture of an opinion.

No faces are involved: the input is floor metres, the output is geometry.
"""

import argparse
import json
import sys
import time
from pathlib import Path

from PIL import Image, ImageDraw

from syncai_bev3d import scene_mesh
from syncai_bev3d.heatmap import (
    DEFAULT_CELL_M,
    LEVELS,
    MODES,
    accumulate,
    heat_colour,
    normalise,
    tile_specs,
)
from syncai_bev3d.meshes import Placement, extrude, place
from syncai_hydranet.geometry.camera_json import CameraFile

ROOT = Path("/home/paul/SyncAI-Lib-HydraNet")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("camera")
    ap.add_argument("--mode", choices=list(MODES), default="dwell")
    ap.add_argument(
        "--tracks",
        default=None,
        help="a demo_tracks.json-shaped file; default is the camera's own "
        "runs/commission01/<camera>.demo_tracks.json, i.e. the last full demo render",
    )
    ap.add_argument("--cell", type=float, default=DEFAULT_CELL_M)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    camera = args.camera

    tracks_path = Path(args.tracks or ROOT / f"runs/commission01/{camera}.demo_tracks.json")
    d = json.loads(tracks_path.read_text())
    if d.get("camera") not in (None, camera):
        print(
            f"{camera}: {tracks_path} records camera {d.get('camera')!r} -- a heatmap "
            "of one camera's tracks on another's furniture would look plausible and be "
            "false",
            file=sys.stderr,
        )
        return 1
    positions = d["positions"]
    fps = float(d.get("fps", 5.0))
    n_frames = int(d.get("frames") or (max(p["frame"] for p in positions) + 1))

    cf = CameraFile.load(ROOT / f"runs/commission01/{camera}.camera.json")
    grid, x0, z0 = accumulate(positions, fps, args.mode, args.cell)
    norm, top = normalise(grid)
    tiles = tile_specs(norm, x0, z0, args.cell, cf=cf)

    for k in range(LEVELS):
        scene_mesh.PALETTE[f"heat_{k:02d}"] = heat_colour((k + 1) / LEVELS)
    _cf2, items, heights, _shapes = scene_mesh.build_scene_regular(camera)
    half = args.cell / 2
    sq = [(-half, -half), (half, -half), (half, half), (-half, half)]
    for x_m, z_m, level, alpha in tiles:
        tile = place(extrude(sq, 0.012), Placement(x_m, z_m, None))
        items.append((tile, f"heat_{level:02d}", alpha, False))

    dev = ROOT / "assets/dev"
    dev.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    out = Path(args.out or dev / f"heatmap_{camera}_{args.mode}_{stamp}.png")
    scene_mesh.SS = 1
    scene_mesh.render(camera, items, heights, out)

    # The legend, drawn onto the render: the ramp, its top, and the window it covers.
    img = Image.open(out).convert("RGB")
    dr = ImageDraw.Draw(img)
    lx, ly, sw = 16, img.height - 40, 22
    for k in range(LEVELS):
        dr.rectangle(
            [lx + k * sw, ly, lx + (k + 1) * sw, ly + 12], fill=heat_colour((k + 1) / LEVELS)
        )
    unit = "s/cell" if args.mode == "dwell" else "tracks/cell"
    dr.text((lx, ly - 14), "0", fill=(220, 224, 232))
    dr.text((lx + LEVELS * sw - 30, ly - 14), f">={top:.1f} {unit} (p99)", fill=(220, 224, 232))
    dr.text(
        (lx, ly + 16),
        f"{args.mode} over {n_frames / fps:.0f} s  ·  cell {args.cell:g} m  ·  "
        f"{len(tiles)} cells in walkable  ·  tracks from {tracks_path.name}",
        fill=(170, 180, 195),
    )
    img.save(out)
    latest = dev / f"heatmap_{camera}_{args.mode}.png"
    latest.write_bytes(out.read_bytes())
    print(
        f"wrote {out} ({args.mode}, {len(positions)} placements, "
        f"{len(tiles)} tiles, top {top:.1f} {unit})"
    )
    print(f"  newest also at {latest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
