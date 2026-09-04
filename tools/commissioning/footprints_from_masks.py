"""Fixture footprints from the commissioning masks. **Two attempts, neither correct yet.**

STATUS 2026-08-26: this does not produce usable footprints and is not wired into anything.
It is committed because the input it establishes is right and the two failures are worth
not repeating. Read the FAILURES section before extending it.

    uv run python tools/commissioning/footprints_from_masks.py Taichung-cam01
    uv run python tools/commissioning/footprints_from_masks.py --all

---------------------------------------------------------------------------
WHY THIS REPLACES `runs/zones01`'s FIXTURE PROPOSALS

`scripts/propose_zones.py` derived footprints on 2026-08-19 from the terrain head's
coarse `floor` class: a fixture appeared as a *hole* in the projected floor, and the hole
was extruded 0.6 m into unobserved space. Its own confidence notes end "far edges beyond
the visible contact are assumed -- trim on confirm".

The masks pass ran on 2026-08-25 and wrote per-class furniture masks --
`display_table.png`, `display_shelf.png`, `floor.png`, `wall.png`, `column.png` -- into
each camera's commissioning directory. Read against those, the 14 proposals for
Taichung-cam01 hold up badly: reviewed by eye and by a second reader, one is a clean
footprint, two are open floor, and the rest spill onto tabletops, floor or wall. Three of
them (indices 4, 8 and 7 on the confirm sheet) sit entirely on an elevated *surface* --
their "contact edge" was the tabletop's edge, so the whole shape is at the wrong height.
The largest, which covers the staff walkway beside the main counter, is floor in the
mask and floor in the picture.

None of that is a tuning problem. The old input could not tell a counter from the floor
in front of it, and the new one can.

---------------------------------------------------------------------------
THE CONSTRUCTION, AND WHY IT NEEDS NO HOLE-FILLING

For one furniture component, per image column, take the **lowest** pixel of the component
and keep it if floor lies within a few rows below. That pixel is where the furniture meets
the floor, and it is the one place on the silhouette that cannot be a tabletop: a table's
lowest pixel in a column is its base or its leg, never its top. Columns whose lowest pixel
is not adjacent to floor are dropped -- something stands in front there and the contact is
not observed, which is a fact worth losing a column over.

Those pixels are undistorted and projected to metres, giving the *visible* base line. The
footprint is that line pushed `--depth` metres directly away from the camera and closed
into a ring. Two consequences worth stating:

* An L-shaped counter comes out as one ring, not as three pieces to merge later, because
  the line is ordered by azimuth from the camera and an L is monotonic in azimuth from a
  single viewpoint.
* The far edge is still an assumption -- the camera cannot see behind a counter -- but it
  is now an assumption applied to the right edge. The old one assumed from whatever edge
  the coarse floor hole happened to have.

---------------------------------------------------------------------------
FAILURES, IN THE ORDER THEY WERE MEASURED

**1. Ordering the base line by azimuth.** Refuted on Taichung-cam01: the main counter came
out as a star of spikes reaching across the room. `contact_pixels` yields exactly one
point per image column, so the chain is already single-valued in column order and needed
no sort at all -- and sorting by angle actively interleaved two different surfaces, since
an island counter shows its near face and, past its end, the cabinet run behind it at the
same azimuth. Fixed: column order, split where the base jumps more than `MAX_BASE_STEP_M`.

**2. "Lowest pixel per column" finds only the nearest surface in each column, and the
biggest fixture in the room has none.** On Taichung-cam01 the main counter -- the single
most important zone on this camera -- produced no footprint. Its component spans a tall
range of every column it occupies, so the per-column minimum lands on the peninsula's
near end, which is cut off by the frame edge or hidden behind the wrapped stock pile, and
those columns are dropped for having no floor below. The counter's **left face base**,
which is plainly visible, is never a per-column minimum and so is never seen. The seven
shapes this does emit are mostly strips of open floor beside the fixture that produced
them, plus one self-crossing bowtie where the normal flipped mid-chain.

What a third attempt needs: every furniture->floor transition down a column, not just the
last one, grouped into surfaces -- and a far edge that is either drawn by a human or
carried by a second view, because one camera cannot see behind a counter and no amount of
mask quality changes that.

---------------------------------------------------------------------------

`kind` comes back as `display` for both furniture classes, never `till`. Which display
table is the till is a fact about the store, and `camera_json.ZONE_KINDS` refuses an
invented kind precisely so that a rule keyed on `till` cannot fire on a guess. The
verdict file from `zones_confirm.py` is where a human upgrades one.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from syncai_bev3d.floorplan import polygon_area, simplify_chain
from syncai_hydranet.geometry.camera_json import CameraFile

# The repo root, derived rather than written out: every one of these 26 tools had it
# as an absolute path, so a second checkout ran against the first one's `runs/` and
# any machine but this one failed at import with a path and no reason. Two levels up
# from `tools/<group>/<tool>.py`, and `tests/test_no_absolute_sys_path.py` keeps it so.
ROOT = Path(__file__).resolve().parents[2]
COMMISSIONED = ROOT / "runs/commission01"
OUT = ROOT / "runs/footprints01"

# The mask classes that stand on the floor. `wall` and `column` are excluded on purpose:
# they meet the floor too, and a zone under a wall is a region no shopper can occupy.
FURNITURE = ("display_table", "display_shelf")
MIN_AREA_PX = 900  # below this a component is a reflection or a mask speck
MIN_CONTACT_COLUMNS = 12  # a base seen in fewer columns than this is not a measured base
FLOOR_LOOKAHEAD_PX = 4  # how far below the lowest pixel floor must appear
SIMPLIFY_M = 0.06  # Douglas-Peucker tolerance, a little under a floor tile's edge
MAX_BASE_STEP_M = 0.35  # a bigger jump between adjacent columns is a different surface


def components(mask: np.ndarray) -> list[np.ndarray]:
    """Connected components of a binary mask, largest first."""
    from scipy import ndimage

    lab, n = ndimage.label(mask)
    out = [(lab == i) for i in range(1, n + 1)]
    out.sort(key=lambda m: -int(m.sum()))
    return [m for m in out if int(m.sum()) >= MIN_AREA_PX]


def contact_pixels(part: np.ndarray, floor: np.ndarray) -> np.ndarray:
    """(N,2) u,v pixels where this component's silhouette bottom meets floor."""
    h, _w = part.shape
    cols = np.flatnonzero(part.any(axis=0))
    pts = []
    for c in cols:
        rows = np.flatnonzero(part[:, c])
        v = int(rows[-1])  # the lowest pixel in this column
        lo, hi = v + 1, min(v + 1 + FLOOR_LOOKAHEAD_PX, h)
        if lo < hi and bool(floor[lo:hi, c].any()):
            pts.append((float(c), float(v)))
    return np.asarray(pts, dtype=float).reshape(-1, 2)


def segments(base_m: np.ndarray, max_step_m: float) -> list[np.ndarray]:
    """Split a base line where consecutive points jump, which means it left the surface.

    `contact_pixels` yields exactly one point per image column, so the chain is already
    ordered by column and single-valued -- no sorting is needed and sorting by angle was
    wrong: an island counter shows two different base segments at the same azimuth (its
    near face and, past its end, the drawer bank behind it), and ordering by angle
    interleaved them into a star of spikes.

    What the column order does not guarantee is continuity in metres. Where one column's
    lowest pixel is the peninsula's base and the next column's is a run of cabinets two
    metres behind it, the ring would cross the whole room to connect them. A jump larger
    than `max_step_m` is that, and it ends the segment instead.
    """
    if len(base_m) < 2:
        return []
    step = np.hypot(*(base_m[1:] - base_m[:-1]).T)
    cut = np.flatnonzero(step > max_step_m) + 1
    return [seg for seg in np.split(base_m, cut) if len(seg) >= 3]


def footprint(chain: np.ndarray, depth: float) -> np.ndarray | None:
    """One base segment, offset `depth` metres to its far side, closed into a ring.

    The offset is along the segment's own **normal**, not along the ray from the camera.
    Radially is wrong wherever the base runs roughly toward the viewer -- the left face
    of a counter seen end-on -- because there the ray is parallel to the line and the
    offset slides the far edge along it instead of behind it, which is what produced the
    self-crossing shapes on the first attempt.

    Which normal is the far one is decided per point by the camera: the one whose dot
    product with the outward radial direction is positive.
    """
    chain = simplify_chain(chain, SIMPLIFY_M)
    if len(chain) < 2:
        return None
    tangent = np.gradient(chain, axis=0)
    norm = np.hypot(tangent[:, 0], tangent[:, 1])
    norm = np.where(norm < 1e-9, 1e-9, norm)
    tangent = tangent / norm[:, None]
    normal = np.stack([-tangent[:, 1], tangent[:, 0]], axis=1)
    radial = chain / np.clip(np.hypot(chain[:, 0], chain[:, 1]), 1e-6, None)[:, None]
    flip = np.sign(np.sum(normal * radial, axis=1))
    flip = np.where(flip == 0, 1.0, flip)
    far = chain + depth * normal * flip[:, None]
    return np.concatenate([chain, far[::-1]])


def derive(camera: str, depth: float) -> dict:
    cam_file = CameraFile.load(COMMISSIONED / f"{camera}.camera.json")
    masks = {}
    for name in (*FURNITURE, "floor"):
        rel = cam_file.mask_files.get(name)
        if rel is None:
            raise SystemExit(f"{camera}: camera.json has no {name!r} mask")
        masks[name] = np.asarray(Image.open(COMMISSIONED / rel)) > 0
    floor = masks["floor"]

    props = []
    for cls in FURNITURE:
        for k, part in enumerate(components(masks[cls])):
            px = contact_pixels(part, floor)
            if len(px) < MIN_CONTACT_COLUMNS:
                continue
            base = cam_file.ground_points(px)
            for part_i, seg in enumerate(segments(base, MAX_BASE_STEP_M)):
                poly = footprint(seg, depth)
                if poly is None or polygon_area(poly) < 0.10:
                    continue
                props.append(
                    {
                        "name_suggestion": f"{cls}_{k + 1:02d}_{part_i + 1:02d}",
                        "kind": "display",
                        "source_mask": cls,
                        "contact_points": len(seg),
                        "component_area_px": int(part.sum()),
                        "bev_area_m2": round(polygon_area(poly), 2),
                        "polygon_m": [
                            [round(float(a), 3), round(float(b), 3)] for a, b in poly
                        ],
                        "basis": (
                            f"lowest silhouette pixel per column of the {cls} mask where "
                            f"floor lies within {FLOOR_LOOKAHEAD_PX} px below, undistorted "
                            f"and projected, split where the base jumps more than "
                            f"{MAX_BASE_STEP_M} m, then offset {depth:.2f} m along its own "
                            "normal away from the camera. Near edge measured, far edge assumed."
                        ),
                    }
                )
    props.sort(key=lambda p: -p["bev_area_m2"])
    return {
        "schema": "hydranet-footprints/v1",
        "camera": camera,
        "masks_from": str(COMMISSIONED / camera / "masks"),
        "depth_m": depth,
        "proposals": props,
    }


def render(camera: str, book: dict) -> Path:
    cam_file = CameraFile.load(COMMISSIONED / f"{camera}.camera.json")
    from zones_confirm import PALETTE, _font, _plate, _to_px  # same sheet, same conventions

    img = _plate(cam_file)
    img = img.resize((img.width * 2, img.height * 2), Image.LANCZOS)
    scale = img.width / cam_file.image_size_px[0]
    d = ImageDraw.Draw(img, "RGBA")
    for i, p in enumerate(book["proposals"]):
        col = PALETTE[i % len(PALETTE)]
        poly = _to_px(p["polygon_m"], cam_file, scale)
        d.polygon([tuple(q) for q in poly], fill=(*col, 70), outline=col, width=4)
        c = poly.mean(axis=0)
        d.ellipse([c[0] - 20, c[1] - 20, c[0] + 20, c[1] + 20], fill=(*col, 235),
                  outline=(0, 0, 0), width=2)  # fmt: skip
        d.text((c[0], c[1]), str(i), fill=(0, 0, 0), anchor="mm", font=_font(30))
    d.rectangle([0, 0, img.width, 44], fill=(0, 0, 0))
    d.text(
        (14, 8),
        f"{camera}   {len(book['proposals'])} footprints from the 2026-08-25 masks   "
        f"near edge measured, far edge extruded {book['depth_m']:.2f} m",
        fill=(255, 255, 255),
        font=_font(24),
    )
    OUT.mkdir(parents=True, exist_ok=True)
    png = OUT / f"{camera}.png"
    img.save(png)
    return png


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cameras", nargs="*")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--depth", type=float, default=0.6, metavar="M")
    args = ap.parse_args()
    cams = args.cameras
    if args.all:
        cams = sorted(
            p.name.split(".camera.json")[0] for p in COMMISSIONED.glob("*.camera.json")
        )
    if not cams:
        raise SystemExit("name a camera, or pass --all")
    OUT.mkdir(parents=True, exist_ok=True)
    for camera in cams:
        book = derive(camera, args.depth)
        (OUT / f"{camera}.footprints.json").write_text(json.dumps(book, indent=1) + "\n")
        png = render(camera, book)
        areas = ", ".join(f"{p['bev_area_m2']}" for p in book["proposals"][:6])
        print(
            f"{camera}: {len(book['proposals'])} footprints  areas m2 [{areas}]  -> {png.name}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
