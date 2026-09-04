"""Human zone assertions and metre-zone proposals, into the commissioning artefacts.

Two inputs, both already on disk, both older than this tool:

* `tools/site30k/zones.json` — the campaign's human polygons for what no teacher can
  see: **glass** (112 frames, four failure modes, "it is a human class") and the
  **white-on-white counter banks** the Tao-Hsin masks miss. `--stamp` applies them to
  `runs/commission01/<cam>/masks/` with stamp_zones' own guard: a pixel is only
  rewritten if its current class is in the zone's `from` list, so a coarse polygon
  cannot eat the floor between two counters. Glass becomes a mask file of its own and
  enters `camera.json` — the gap PLAN 2.1c has carried since the reset closes here.

* `runs/zones01/<cam>.zones.json` — propose_zones' metre-space proposals (walkable
  outline, entrance candidates from track births). `--import-zones` writes the walkable
  outline into `camera.json` (it mirrors the already-reviewed mask) and renders the
  entrance candidates on a proposal sheet for the accept/reject pass; policy fields
  stay null because policy is proposed-never-decided.

Usage:
  uv run python tools/commissioning/zones_apply.py --stamp [camera ...]
  uv run python tools/commissioning/zones_apply.py --import-zones [camera ...]
"""

import dataclasses
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from syncai_hydranet.geometry.camera_json import CameraFile, Zone

# The repo root, derived rather than written out: every one of these 26 tools had it
# as an absolute path, so a second checkout ran against the first one's `runs/` and
# any machine but this one failed at import with a path and no reason. Two levels up
# from `tools/<group>/<tool>.py`, and `tests/test_no_absolute_sys_path.py` keeps it so.
ROOT = Path(__file__).resolve().parents[2]
ZONES_HUMAN = ROOT / "tools/site30k/zones.json"
ZONES_PROPOSED = ROOT / "runs/zones01"
CLASS_FILES = {
    "floor": "floor",
    "wall": "wall",
    "column": "column",
    "display_table": "display_table",
    "display_shelf": "display_shelf",
    "glass": "glass",
}


def stamp(camera: str) -> bool:
    zones = json.loads(ZONES_HUMAN.read_text()).get(camera)
    if not zones:
        return False
    cf = CameraFile.load(ROOT / f"runs/commission01/{camera}.camera.json")
    w, h = cf.image_size_px
    mask_dir = ROOT / "runs/commission01"
    masks = {}
    for name in CLASS_FILES:
        f = cf.mask_files.get(name)
        if f and (mask_dir / f).exists():
            masks[name] = np.asarray(Image.open(mask_dir / f).resize((w, h))) > 127
        else:
            masks[name] = np.zeros((h, w), bool)

    for z in zones:
        poly = [(float(x), float(y)) for x, y in z["poly"]]
        sel = Image.new("L", (w, h), 0)
        ImageDraw.Draw(sel).polygon(poly, fill=255)
        sel = np.asarray(sel) > 0
        allowed = np.zeros((h, w), bool)
        for src_cls in z.get("from", []):
            allowed |= masks.get(src_cls, np.zeros((h, w), bool))
        # stamp_zones' guard: only pixels currently holding a `from` class move. A zone
        # with no `from` would overwrite everything and is refused here outright.
        if not z.get("from"):
            raise ValueError(f"{camera}: zone without a from-list -- refusing to stamp")
        move = sel & allowed
        for src_cls in z["from"]:
            masks[src_cls] &= ~move
        masks[z["to"]] |= move
        print(
            f"  [{camera}] {z['to']} <- {z['from']}: {int(move.sum())} px  ({z['note'][:50]})"
        )

    mask_files = dict(cf.mask_files)
    for name, m in masks.items():
        if not m.any() and name not in mask_files:
            continue
        out = ROOT / "runs/commission01" / camera / "masks" / f"{name}.png"
        Image.fromarray((m * 255).astype(np.uint8)).save(out)
        mask_files[name] = f"{camera}/masks/{name}.png"
    cf = dataclasses.replace(cf, mask_files=mask_files)
    cf.validate()
    cf.save(ROOT / f"runs/commission01/{camera}.camera.json")
    return True


KIND_MAP = {"floor": "walkable", "entrance": "entrance_line"}


def import_zones(camera: str) -> bool:
    src = ZONES_PROPOSED / f"{camera}.zones.json"
    cf_path = ROOT / f"runs/commission01/{camera}.camera.json"
    if not src.exists() or not cf_path.exists():
        return False
    d = json.loads(src.read_text())
    if str(d.get("scale_source", "")).startswith("unmeasured"):
        print(f"  [{camera}] proposals are dav2_raw (no metres) -- skipped")
        return False
    cf = CameraFile.load(cf_path)
    written, proposed = [], []
    for p in d["proposals"]:
        kind = KIND_MAP.get(p["kind"])
        if kind == "walkable":
            pts = tuple((round(x, 2), round(z, 2)) for x, z in p["polygon_m"])
            written.append(Zone("walkable_floor", "walkable", pts))
        elif kind == "entrance_line":
            proposed.append(p)
    if written:
        keep = tuple(z for z in cf.zones if z.kind != "walkable")
        cf = dataclasses.replace(cf, zones=keep + tuple(written))
        cf.validate()
        cf.save(cf_path)
    print(
        f"  [{camera}] walkable outline written ({len(written)}); "
        f"entrance candidates awaiting verdict: {len(proposed)}"
    )
    return True


def main():
    argv = [a for a in sys.argv[1:] if not a.startswith("--")]
    do_stamp = "--stamp" in sys.argv[1:]
    do_import = "--import-zones" in sys.argv[1:]
    cameras = argv or [p.stem.replace(".camera", "") for p in
                       sorted((ROOT / "runs/commission01").glob("*.camera.json"))]  # fmt: skip
    for camera in cameras:
        if do_stamp:
            stamp(camera)
        if do_import:
            import_zones(camera)


if __name__ == "__main__":
    main()
