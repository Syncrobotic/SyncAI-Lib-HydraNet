"""The accept/reject pass that turns fixture proposals into named zones a store owns.

    uv run python tools/commissioning/zones_confirm.py --render Taichung-cam01
    # look at runs/zones_confirm01/<camera>.png, fill in <camera>.verdicts.json
    uv run python tools/commissioning/zones_confirm.py --apply Taichung-cam01

---------------------------------------------------------------------------
WHAT WAS ALREADY DONE, AND THE ONE STEP THAT WAS NOT

`scripts/propose_zones.py` produced physical zone proposals for 19 calibrated cameras on
2026-08-19 -- a walkable outline and per-fixture footprints, in metres, with every policy
field an explicit null under the ruling it recorded: **physical zones are automatable
proposals; policy is proposed-never-decided.** `zones_apply.py --import-zones` then wrote
the walkable outline into `camera.json` and stopped there, because the walkable outline
"mirrors the already-reviewed mask" and the fixtures do not.

So the fixtures have sat in `runs/zones01/` ever since, and every camera.json carries
exactly one zone. `analytics/journey.py` reads zones to answer "stood at C for how long",
and with one walkable polygon per camera the only answer available is "inside the shop".

---------------------------------------------------------------------------
WHY THIS IS NOT AN IMPORTER

`camera_json.ZONE_KINDS` is `{entrance_line, till, premium_shelf, stockroom_door,
forbidden, walkable}`. There is no `fixture`, and that absence is the design: a footprint
is a shape, and *which fixture it is* is a fact about the store that no segmentation head
can supply. An importer would have to invent a kind, and a zone whose kind was invented
fires the wrong rule for the rest of its life. `camera_json.validate` already refuses an
unknown kind for the same reason -- "an unknown kind never fires and never errors".

So this renders the proposals onto a real frame from that camera, numbered, and writes a
verdict file with `keep`, `name` and `kind` unset. A human fills it in. `--apply` refuses
any row still unset and any kind outside `ZONE_KINDS`.

`polygon_m` on a verdict row **replaces** the proposal's shape. The proposer's own
confidence notes end "far edges beyond the visible contact are assumed -- trim on
confirm", because a footprint is extruded 0.6 m into space the camera cannot see and a
closed one may swallow the fixture's occlusion shadow. Nothing here can trim that: which
part of a shape is the fixture and which is its shadow is the store's knowledge. So the
row carries an override, and a shape that is wrong is corrected by writing the metres
rather than by accepting it and explaining later.

`merge` is there because the proposer's own REPORT names the cost of its splitting rule:
one piece of furniture comes out as 2-3 adjacent polygons, and the confirm pass has to be
able to put them back together. Merged rows are unioned by convex hull, which is right for
a table cut in two and wrong for an L-shaped counter bank -- so the hull is drawn on the
render before it is written, and a shape that the hull spoils should be kept as two zones.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from syncai_hydranet.geometry.camera_json import ZONE_KINDS, CameraFile, Zone
from syncai_hydranet.geometry.ground import distort_points, ground_to_pixel

# The repo root, derived rather than written out: every one of these 26 tools had it
# as an absolute path, so a second checkout ran against the first one's `runs/` and
# any machine but this one failed at import with a path and no reason. Two levels up
# from `tools/<group>/<tool>.py`, and `tests/test_no_absolute_sys_path.py` keeps it so.
ROOT = Path(__file__).resolve().parents[2]
PROPOSED = ROOT / "runs/zones01"
COMMISSIONED = ROOT / "runs/commission01"
OUT = ROOT / "runs/zones_confirm01"
SHEET_SCALE = 2  # the plate is 960x540; a footprint label has to survive being looked at


def _font(size: int):
    """A real TrueType face if the box has one, PIL's bitmap default if it does not.

    The default font ignores `size`, so a sheet rendered on a box without DejaVu is
    legible-ish rather than illegible; it is not worth a dependency to make it exact.
    """
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ):
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


PALETTE = [
    (255, 96, 64), (90, 220, 130), (255, 205, 70), (120, 170, 255),
    (235, 120, 235), (110, 230, 230), (255, 150, 90), (170, 200, 90),
]  # fmt: skip


def _plate(cam_file: CameraFile) -> Image.Image:
    """The commissioning plate: the frame the geometry was fitted on.

    Not an arbitrary frame from a clip. The plate is what the calibration saw, so a
    polygon that looks right on it is right about the geometry rather than about a
    moment -- and it is empty of shoppers, which is what a floor outline wants.
    """
    if not cam_file.plate_file:
        raise SystemExit(f"{cam_file.camera_id}: camera.json names no plate to draw on")
    p = ROOT / cam_file.plate_file
    if not p.exists():
        raise SystemExit(f"{cam_file.camera_id}: plate {p} is missing")
    return Image.open(p).convert("RGB")


def _to_px(points_m, cam_file: CameraFile, scale: float) -> np.ndarray:
    """Floor metres -> pixels on the RAW plate this sheet is drawn on.

    `ground_to_pixel` returns ideal pixels; the plate is a decoded frame with the lens
    still in it. Skipping `distort_points` puts every outline a few pixels off its own
    floor, which reads as a loose calibration and is not one.
    """
    a = np.asarray(points_m, dtype=float).reshape(-1, 2)
    u, v, _ = ground_to_pixel(a[:, 0], a[:, 1], cam_file.camera, cam_file.plane)
    px = np.stack([u, v], axis=1)
    lens = cam_file.lens
    if lens is not None:
        px = distort_points(px, lens.k1, lens.centre_px, lens.radius_px)
    return px * scale


def _hull(points_m: np.ndarray) -> np.ndarray:
    """Monotone-chain convex hull, so merging needs no new dependency."""
    pts = sorted({(float(x), float(z)) for x, z in points_m})
    if len(pts) < 3:
        return np.asarray(pts, dtype=float)

    def half(seq):
        out: list[tuple[float, float]] = []
        for p in seq:
            while len(out) >= 2:
                (x1, y1), (x2, y2) = out[-2], out[-1]
                if (x2 - x1) * (p[1] - y1) - (y2 - y1) * (p[0] - x1) > 0:
                    break
                out.pop()
            out.append(p)
        return out

    return np.asarray(half(pts)[:-1] + half(reversed(pts))[:-1], dtype=float)


def fixtures(camera: str) -> list[dict]:
    src = PROPOSED / f"{camera}.zones.json"
    if not src.exists():
        raise SystemExit(f"no proposals at {src} -- run scripts/propose_zones.py first")
    d = json.loads(src.read_text())
    if str(d.get("scale_source", "")).startswith("unmeasured"):
        raise SystemExit(
            f"{camera}: these proposals are in dav2_raw units, not metres "
            f"(scale_source={d.get('scale_source')!r}). Confirming them would write "
            "shapes whose size is unknown into a file whose whole point is metres."
        )
    return [p for p in d["proposals"] if p["kind"] == "fixture"]


def render(camera: str) -> None:
    cam_file = CameraFile.load(COMMISSIONED / f"{camera}.camera.json")
    props = fixtures(camera)
    img = _plate(cam_file)
    # Upscaled before anything is drawn: the plate is 960x540 and a footprint is a few
    # dozen pixels across, so the labels and the outlines have to be drawn at the size
    # they will be read at rather than resized afterwards into mush.
    img = img.resize((img.width * SHEET_SCALE, img.height * SHEET_SCALE), Image.LANCZOS)
    scale = img.width / cam_file.image_size_px[0]
    d = ImageDraw.Draw(img, "RGBA")

    for z in cam_file.zones:
        if z.kind == "walkable":
            d.polygon([tuple(p) for p in _to_px(z.points_m, cam_file, scale)],
                      outline=(90, 200, 255), width=4)  # fmt: skip

    rows = []
    for i, p in enumerate(props):
        col = PALETTE[i % len(PALETTE)]
        poly = _to_px(p["polygon_m"], cam_file, scale)
        d.polygon([tuple(q) for q in poly], fill=(*col, 60), outline=col, width=4)
        c = poly.mean(axis=0)
        # A filled chip, not bare text: a numeral drawn straight onto shop shelving is
        # unreadable at the size a footprint occupies, and a sheet whose labels cannot be
        # read is a sheet nobody can return a verdict on.
        label = str(i)
        r = 20 if len(label) == 1 else 26
        d.ellipse(
            [c[0] - r, c[1] - r, c[0] + r, c[1] + r],
            fill=(*col, 235),
            outline=(0, 0, 0),
            width=2,
        )
        d.text((c[0], c[1]), label, fill=(0, 0, 0), anchor="mm", font=_font(30))
        area = np.asarray(p["polygon_m"], dtype=float)
        rows.append(
            {
                "index": i,
                "proposal": p["name_suggestion"],
                "bev_area_m2": round(float(_poly_area(area)), 2),
                "keep": None,  # true / false -- a human decides
                "name": None,  # what this fixture is called in this store
                "kind": None,  # one of ZONE_KINDS, minus walkable and entrance_line
                "merge_with": [],  # other indices to union with this one, by convex hull
                "polygon_m": None,  # replaces the proposal's shape outright, when trimming
            }
        )

    d.rectangle([0, 0, img.width, 44], fill=(0, 0, 0))
    caption = (
        f"{camera}   {len(props)} fixture footprints proposed   "
        f"blue outline = walkable floor   numbers index {camera}.verdicts.json"
    )
    d.text((14, 8), caption, fill=(255, 255, 255), font=_font(24))
    OUT.mkdir(parents=True, exist_ok=True)
    png = OUT / f"{camera}.png"
    img.save(png)
    verdicts = OUT / f"{camera}.verdicts.json"
    if verdicts.exists():
        print(f"  {verdicts} exists -- left alone, so a filled-in pass is never clobbered")
    else:
        verdicts.write_text(
            json.dumps(
                {
                    "camera": camera,
                    "source": str(PROPOSED / f"{camera}.zones.json"),
                    "kinds_allowed": sorted(ZONE_KINDS - {"walkable", "entrance_line"}),
                    "note": "keep/name/kind are null until a human fills them; --apply "
                    "refuses a null. merge_with unions footprints by convex hull; "
                    "polygon_m replaces a shape outright, for the trim the proposer's "
                    "confidence notes ask for.",
                    "fixtures": rows,
                },
                indent=1,
            )
            + "\n"
        )
    print(f"wrote {png} and {verdicts}  ({len(props)} footprints)")


def _poly_area(poly: np.ndarray) -> float:
    x, z = poly[:, 0], poly[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(z, 1)) - np.dot(z, np.roll(x, 1))))


def apply(camera: str) -> None:
    verdicts = OUT / f"{camera}.verdicts.json"
    if not verdicts.exists():
        raise SystemExit(f"no verdicts at {verdicts} -- run --render first")
    v = json.loads(verdicts.read_text())
    props = fixtures(camera)
    cf_path = COMMISSIONED / f"{camera}.camera.json"
    cam_file = CameraFile.load(cf_path)

    kept = [r for r in v["fixtures"] if r.get("keep")]
    for r in kept:
        if not r.get("name") or not r.get("kind"):
            raise SystemExit(
                f"{camera}: fixture {r['index']} is kept but has no name/kind. A zone "
                "whose identity was guessed fires the wrong rule for the rest of its life."
            )
        if r["kind"] not in ZONE_KINDS or r["kind"] in {"walkable", "entrance_line"}:
            raise SystemExit(
                f"{camera}: fixture {r['index']} has kind {r['kind']!r}; allowed here are "
                f"{sorted(ZONE_KINDS - {'walkable', 'entrance_line'})}"
            )
    merged_away = {j for r in kept for j in r.get("merge_with", [])}
    zones: list[Zone] = []
    for r in kept:
        if r["index"] in merged_away:
            continue
        override = r.get("polygon_m")
        if override is not None:
            pts = np.asarray(override, dtype=float).reshape(-1, 2)
            if len(pts) < 3:
                raise SystemExit(
                    f"{camera}: fixture {r['index']}'s polygon_m override has {len(pts)} "
                    "points; a region needs 3"
                )
        else:
            pts = np.asarray(props[r["index"]]["polygon_m"], dtype=float)
        for j in r.get("merge_with", []):
            pts = np.concatenate([pts, np.asarray(props[j]["polygon_m"], dtype=float)])
            pts = _hull(pts)
        zones.append(
            Zone(r["name"], r["kind"], tuple((round(x, 2), round(z, 2)) for x, z in pts))
        )
    # Replace only what this pass owns. The walkable outline and any entrance line were
    # written by another pass and re-deriving them here would make two writers of one field.
    keep_existing = tuple(z for z in cam_file.zones if z.kind in {"walkable", "entrance_line"})
    cam_file = dataclasses.replace(cam_file, zones=keep_existing + tuple(zones))
    cam_file.validate()
    cam_file.save(cf_path)
    print(f"{camera}: {len(zones)} zones written to {cf_path.name} "
          f"({len(keep_existing)} kept from other passes)")  # fmt: skip


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cameras", nargs="+")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--render", action="store_true")
    g.add_argument("--apply", action="store_true")
    args = ap.parse_args()
    for camera in args.cameras:
        (render if args.render else apply)(camera)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
