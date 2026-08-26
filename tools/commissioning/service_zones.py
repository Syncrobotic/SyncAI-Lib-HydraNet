"""Zones a shopper can stand in, from SAM 3 instances. No human draws anything.

    uv run python tools/commissioning/service_zones.py Taichung-cam01
    uv run python tools/commissioning/service_zones.py --all --apply

---------------------------------------------------------------------------
THE MISTAKE THIS REPLACES, WHICH WAS MINE AND NOT THE TEACHER'S

Two earlier attempts tried to produce **fixture footprints** -- the patch of floor a
counter stands on -- and both broke on the same wall: one camera cannot see behind a
counter, so the far edge is always assumed. That led to proposing that a human draw the
far edge, which contradicts PLAN section 4 outright: minimum human labelling, teachers do
the rest.

The wall was self-inflicted. **A fixture's footprint is a region no shopper can ever
occupy**, so as a zone it is a polygon that by construction never fires:
`analytics/journey.py` tests a person's foot point against it and the answer is always
False. What answers "did this customer spend time at the main counter" is the **floor
beside the counter** -- the approach band, on the camera's side of the fixture, entirely
in view. The occluded far edge was never needed for the thing the zone is for.

---------------------------------------------------------------------------
WHAT THE TEACHER SUPPLIES THAT THE SEMANTIC MASKS DO NOT

`masks_pass.py` already runs SAM 3, and `compose` deliberately throws away the part
needed here: it merges every prompt's instances into one id map, so a shop's five
fixtures are one `display_fixture` region. `teachers.sam3.segment` returns the instances,
one mask per detection, and the prompt that found each one carries its identity --
`data/sam3_prompts.py` measured which prompts fire on what, and "checkout counter" and
"display table" are different prompts. So the zone's *kind* comes from the teacher too,
and no human names anything.

---------------------------------------------------------------------------
THE CONSTRUCTION, ENTIRELY INSIDE WHAT THE CAMERA SEES

1. SAM 3 per prompt on the commissioning plate -> instance masks, deduped across prompts
   by mask IoU, because "retail counter" and "checkout counter" find the same desk.
2. The **contact points**: floor-mask pixels within `TOUCH_PX` of the instance. These are
   floor, they are next to the fixture, and they are visible by definition -- there is
   nothing here to assume.
3. Projected to metres through `camera.json`, exactly as a shopper's foot point will be.
4. The zone is every walkable floor cell within `REACH_M` of a contact point. That is a
   band of floor along the fixture, which is where a person stands to use it.

`REACH_M` is a stated reach, not a fitted number: 0.9 m is roughly where a person's feet
are when their hands are at a counter edge. It is an argument, like every other store
parameter in this project (`RETAIL.md` section 5: config, not weights).
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from syncai_bev3d.teachers.sam3 import MODEL_ID, load_sam3, segment, vision_features
from syncai_hydranet.data.sam3_prompts import DEFAULT_MIN_SCORE
from syncai_hydranet.geometry.camera_json import CameraFile, Zone
from syncai_hydranet.geometry.ground import (
    distort_points,
    ground_to_pixel,
    pixel_to_ground,
    undistort_points,
)
from syncai_hydranet.utils.device import pick_device

ROOT = Path("/home/paul/SyncAI-Lib-HydraNet")
COMMISSIONED = ROOT / "runs/commission01"
OUT = ROOT / "runs/service_zones01"

# Prompt groups, and the `camera_json` kind each one means. The prompts are the measured
# ones from `data/sam3_prompts.py`; what is new is keeping them apart instead of merging
# them into one `display_fixture` region, because a till and a display table are the same
# class to the terrain head and different zones to a store.
GROUPS = (
    # `display counter` started in this group and had to be moved: on Taichung-cam01 it
    # claimed seven of nine fixtures, including both left-hand demo tables and the wall
    # run. It is a shape prompt, not a function prompt -- most shop furniture is a
    # counter with a display on it -- so it says nothing about where the money changes
    # hands. `checkout counter` and `cash register` name the function, and `retail
    # counter` was measured finding the service desk specifically (4/8, 0.620, sweep A).
    ("till", ("checkout counter", "cash register", "retail counter")),
    (
        "display",
        (
            "display table",
            "display counter",
            "display podium",
            "product display stand",
            "shelving unit",
            "merchandise rack",
        ),
    ),
)
TOUCH_PX = 6  # how close a floor pixel must be to the fixture to count as its edge
REACH_M = 0.9  # a person's feet when their hands are at the fixture
CELL_M = 0.10
MIN_INSTANCE_PX = 1500  # below this SAM 3 found a poster or a reflection
MIN_ZONE_M2 = 0.6  # a band smaller than this is not somewhere a person stands
DEDUPE_IOU = 0.55


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = float(np.logical_and(a, b).sum())
    union = float(np.logical_or(a, b).sum())
    return inter / union if union else 0.0


def instances(plate: Image.Image, device: str) -> list[dict]:
    """Every fixture SAM 3 finds, once, with the group whose prompt found it."""
    proc, model = load_sam3(MODEL_ID, device)
    embeds = vision_features(proc, model, plate, device)
    found: list[dict] = []
    for kind, prompts in GROUPS:
        for prompt in prompts:
            for mask, score in segment(
                proc, model, plate, prompt, DEFAULT_MIN_SCORE, device, embeds
            ):
                if int(mask.sum()) < MIN_INSTANCE_PX:
                    continue
                dup = next((f for f in found if _iou(f["mask"], mask) >= DEDUPE_IOU), None)
                if dup is not None:
                    # Keep the higher-scoring claim, and keep `till` over `display` when
                    # both fired: a checkout counter is also a display table to SAM 3, and
                    # the more specific prompt is the one that carries information.
                    if kind == "till" and dup["kind"] != "till":
                        dup.update(kind=kind, prompt=prompt, score=score, mask=mask)
                    elif score > dup["score"] and kind == dup["kind"]:
                        dup.update(prompt=prompt, score=score, mask=mask)
                    continue
                found.append({"kind": kind, "prompt": prompt, "score": score, "mask": mask})
    return found


def contact_points_px(inst: np.ndarray, floor: np.ndarray) -> np.ndarray:
    """Floor pixels within TOUCH_PX of this instance: the fixture's visible floor edge."""
    from scipy import ndimage

    near = ndimage.binary_dilation(inst, iterations=TOUCH_PX)
    ys, xs = np.nonzero(near & floor & ~inst)
    return np.stack([xs.astype(float), ys.astype(float)], axis=1)


def to_metres(px: np.ndarray, cam_file: CameraFile) -> np.ndarray:
    lens = cam_file.lens
    if lens is not None:
        px = undistort_points(px, lens.k1, lens.centre_px, lens.radius_px)
    x, z = pixel_to_ground(px[:, 0], px[:, 1], cam_file.camera, cam_file.plane)
    out = np.stack([x, z], axis=1)
    return out[np.isfinite(out).all(axis=1)]


class FloorGrid:
    """A BEV raster of this camera's floor, shared by every zone on it.

    One grid rather than one per fixture, because the zones have to **partition** the
    floor: a shopper standing between two display tables is at one of them, and a set of
    overlapping bands makes `journeys` report a route through three zones without anyone
    moving. Overlap was the first version's real defect, visible on Taichung-cam01 as
    three zones covering the same aisle.
    """

    def __init__(self, floor_m: np.ndarray):
        self.lo = floor_m.min(axis=0) - 1.0
        hi = floor_m.max(axis=0) + 1.0
        self.nx = int(np.ceil((hi[0] - self.lo[0]) / CELL_M))
        self.nz = int(np.ceil((hi[1] - self.lo[1]) / CELL_M))
        from scipy import ndimage

        self.floor = ndimage.binary_closing(self.raster(floor_m), iterations=2)

    def raster(self, pts: np.ndarray) -> np.ndarray:
        c = ((pts[:, 0] - self.lo[0]) / CELL_M).astype(int)
        r = ((pts[:, 1] - self.lo[1]) / CELL_M).astype(int)
        ok = (c >= 0) & (c < self.nx) & (r >= 0) & (r < self.nz)
        g = np.zeros((self.nz, self.nx), dtype=bool)
        g[r[ok], c[ok]] = True
        return g

    def to_metres(self, cells: np.ndarray) -> np.ndarray:
        return np.stack(
            [
                self.lo[0] + (cells[:, 0] + 0.5) * CELL_M,
                self.lo[1] + (cells[:, 1] + 0.5) * CELL_M,
            ],
            axis=1,
        )


def assign(grid: FloorGrid, contacts: list[np.ndarray]) -> list[np.ndarray]:
    """Every floor cell within REACH_M of a fixture, given to its nearest one.

    A distance transform per fixture and an argmin across them -- a Voronoi over the
    fixtures' floor edges, clipped to the reach and to the floor. Cells nearer to nothing
    stay unassigned, which is the aisle, and cells equidistant go to the lower index,
    which is arbitrary and stated rather than hidden.
    """
    from scipy import ndimage

    dists = []
    for pts in contacts:
        seed = grid.raster(pts) if len(pts) else np.zeros((grid.nz, grid.nx), dtype=bool)
        if not seed.any():
            dists.append(np.full((grid.nz, grid.nx), np.inf))
            continue
        dists.append(ndimage.distance_transform_edt(~seed) * CELL_M)
    stack = np.stack(dists)
    nearest = stack.argmin(axis=0)
    within = stack.min(axis=0) <= REACH_M
    keep = within & grid.floor
    return [keep & (nearest == i) for i in range(len(contacts))]


def polygon(grid: FloorGrid, cells: np.ndarray) -> np.ndarray | None:
    """The outer contour of one assigned region, in metres."""
    import contourpy
    from scipy import ndimage

    if not cells.any():
        return None
    lab, n = ndimage.label(cells)
    if n > 1:  # the largest piece only: the rest are slivers the reach clipped off
        sizes = [int((lab == i).sum()) for i in range(1, n + 1)]
        cells = lab == (1 + int(np.argmax(sizes)))
    padded = np.pad(cells.astype(float), 1)
    best, best_a = None, 0.0
    for line in contourpy.contour_generator(z=padded, name="serial").lines(0.5):
        if len(line) < 4:
            continue
        x, y = line[:-1, 0], line[:-1, 1]
        a = 0.5 * abs(float(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))))
        if a > best_a:
            best, best_a = line[:-1] - 1.0, a
    if best is None:
        return None
    return _simplify(grid.to_metres(best), 0.08)


def _simplify(chain: np.ndarray, tol: float) -> np.ndarray:
    if len(chain) < 4:
        return chain
    keep = np.zeros(len(chain), dtype=bool)
    keep[0] = keep[-1] = True
    stack = [(0, len(chain) - 1)]
    while stack:
        a, b = stack.pop()
        if b <= a + 1:
            continue
        seg = chain[b] - chain[a]
        norm = float(np.hypot(*seg))
        rel = chain[a + 1 : b] - chain[a]
        d = (
            np.hypot(rel[:, 0], rel[:, 1])
            if norm < 1e-9
            else np.abs(rel[:, 0] * seg[1] - rel[:, 1] * seg[0]) / norm
        )
        i = int(d.argmax())
        if d[i] > tol:
            keep[a + 1 + i] = True
            stack += [(a, a + 1 + i), (a + 1 + i, b)]
    return chain[keep]


def _area(poly: np.ndarray) -> float:
    x, z = poly[:, 0], poly[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(z, 1)) - np.dot(z, np.roll(x, 1))))


def derive(camera: str, device: str) -> dict:
    cam_file = CameraFile.load(COMMISSIONED / f"{camera}.camera.json")
    plate = Image.open(ROOT / cam_file.plate_file).convert("RGB")
    floor = np.asarray(Image.open(COMMISSIONED / cam_file.mask_files["floor"])) > 0
    ys, xs = np.nonzero(floor)
    floor_m = to_metres(np.stack([xs.astype(float), ys.astype(float)], axis=1), cam_file)
    if len(floor_m) < 100:
        raise SystemExit(f"{camera}: the floor mask projects to {len(floor_m)} points")
    grid = FloorGrid(floor_m)

    found = instances(plate, device)
    contacts = [to_metres(contact_points_px(f["mask"], floor), cam_file) for f in found]
    regions = assign(grid, contacts)

    zones = []
    for inst, contact, cells in zip(found, contacts, regions, strict=True):
        poly = polygon(grid, cells)
        if poly is None or _area(poly) < MIN_ZONE_M2:
            continue
        zones.append(
            {
                "kind": inst["kind"],
                "prompt": inst["prompt"],
                "score": round(float(inst["score"]), 3),
                "instance_px": int(inst["mask"].sum()),
                "contact_points": len(contact),
                "area_m2": round(_area(poly), 2),
                "polygon_m": [[round(float(a), 2), round(float(b), 2)] for a, b in poly],
            }
        )
    zones.sort(key=lambda z: -z["area_m2"])
    for i, z in enumerate(zones, start=1):
        z["name"] = f"{z['kind']}_{i:02d}"
    return {
        "schema": "hydranet-service-zones/v1",
        "camera": camera,
        "teacher": MODEL_ID,
        "reach_m": REACH_M,
        "cell_m": CELL_M,
        "zones": zones,
    }


def render(camera: str, book: dict) -> Path:
    cam_file = CameraFile.load(COMMISSIONED / f"{camera}.camera.json")
    plate = Image.open(ROOT / cam_file.plate_file).convert("RGB")
    img = plate.resize((plate.width * 2, plate.height * 2), Image.LANCZOS)
    scale = img.width / cam_file.image_size_px[0]
    d = ImageDraw.Draw(img, "RGBA")
    cols = [(255, 96, 64), (90, 220, 130), (255, 205, 70), (120, 170, 255), (235, 120, 235),
            (110, 230, 230), (255, 150, 90)]  # fmt: skip
    from zones_confirm import _font

    for i, z in enumerate(book["zones"]):
        col = cols[i % len(cols)]
        a = np.asarray(z["polygon_m"], dtype=float)
        u, v, _ = ground_to_pixel(a[:, 0], a[:, 1], cam_file.camera, cam_file.plane)
        px = np.stack([u, v], axis=1)
        if cam_file.lens is not None:
            px = distort_points(
                px, cam_file.lens.k1, cam_file.lens.centre_px, cam_file.lens.radius_px
            )
        px = px * scale
        d.polygon([tuple(p) for p in px], fill=(*col, 80), outline=col, width=4)
        c = px.mean(axis=0)
        d.ellipse([c[0] - 22, c[1] - 22, c[0] + 22, c[1] + 22], fill=(*col, 235),
                  outline=(0, 0, 0), width=2)  # fmt: skip
        d.text((c[0], c[1]), str(i), fill=(0, 0, 0), anchor="mm", font=_font(30))
    d.rectangle([0, 0, img.width, 44], fill=(0, 0, 0))
    d.text(
        (14, 8),
        f"{camera}   {len(book['zones'])} service zones from SAM 3   "
        f"floor within {book['reach_m']} m of a fixture -- where a shopper stands",
        fill=(255, 255, 255),
        font=_font(24),
    )
    OUT.mkdir(parents=True, exist_ok=True)
    png = OUT / f"{camera}.png"
    img.save(png)
    return png


def apply(camera: str, book: dict) -> None:
    cf_path = COMMISSIONED / f"{camera}.camera.json"
    cam_file = CameraFile.load(cf_path)
    zones = tuple(
        Zone(z["name"], z["kind"], tuple((a, b) for a, b in z["polygon_m"]))
        for z in book["zones"]
    )
    keep = tuple(z for z in cam_file.zones if z.kind in {"walkable", "entrance_line"})
    cam_file = dataclasses.replace(cam_file, zones=keep + zones)
    cam_file.validate()
    cam_file.save(cf_path)
    print(f"  wrote {len(zones)} zones into {cf_path.name} ({len(keep)} kept)")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cameras", nargs="*")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()
    cams = args.cameras
    if args.all:
        cams = sorted(
            p.name.split(".camera.json")[0] for p in COMMISSIONED.glob("*.camera.json")
        )
    if not cams:
        raise SystemExit("name a camera, or pass --all")
    device = pick_device(None)
    OUT.mkdir(parents=True, exist_ok=True)
    for camera in cams:
        book = derive(camera, device)
        (OUT / f"{camera}.service_zones.json").write_text(json.dumps(book, indent=1) + "\n")
        png = render(camera, book)
        for z in book["zones"]:
            print(f"  {z['name']:<12} {z['kind']:<8} {z['area_m2']:>5} m2  "
                  f"<- {z['prompt']!r} {z['score']}")  # fmt: skip
        print(f"{camera}: {len(book['zones'])} zones -> {png.name}")
        if args.apply:
            apply(camera, book)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
