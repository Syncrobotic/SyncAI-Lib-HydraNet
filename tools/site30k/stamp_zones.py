#!/usr/bin/env python3
"""Human class assertions for what the teachers got wrong on immovable structure.

    python tools/site30k/stamp_zones.py --zones tools/site30k/zones.json \
        --root datasets/site30k_v1
    python tools/site30k/stamp_zones.py --zones ... --root ...
        --apply

The campaign decides structure by teacher vote, once per camera-date, which is right for
the 95% of the frame the teachers can see. Two things they cannot:

  glass    `sam3_prompts.py:203-233` records glass failing four distinct ways across two
           sessions and 112 frames, and concludes it is a human class. Four more routes
           were tried on 2026-08-20 -- cross-date plate deviation, day/night differential,
           the same restricted to `wall` pixels, and metric depth -- and none separates it.
  white-on-white   a white counter against a white wall has no photometric edge, so
           Tao-Hsin-cam03's entire counter bank reads as `wall` (69.2% wall, 0.0% shelf).
           The night plate does not rescue it: the store rolls SHUTTERS over both display
           walls, so at night the counters are not visible at all.

Both are immovable and both cameras are fixed, so both are one polygon drawn once. This
stamps those polygons over the finished masks -- no teacher re-run, no GPU.

THE GUARD THAT MAKES A COARSE POLYGON SAFE
------------------------------------------
Every zone names the classes it is allowed to overwrite (`from`). A pixel inside the
polygon whose current label is not in that list is left alone. So a display counter
standing in front of the glazing keeps `display_table`, the floor showing between two
counters keeps `floor`, and the polygon only has to be roughly right. A zone with no
`from` list overwrites everything inside it, which is almost never what you want.

Zones file::

    {"Tao-Hsin-cam04": [{"poly": [[272,0],[580,0],[575,150],[278,157]],
                         "to": "glass", "from": ["wall"],
                         "note": "street-facing shopfront glazing"}]}

Coordinates are in PLATE space (960x540) because that is the image a human looks at; they
are scaled to the mask's own size on the way in. Dry-run by default: it reports the pixel
counts per zone and writes nothing. `--apply` writes, after backing up each mask it
touches to `masks_prestamp/` so `tools/site30k/compare_masks.py` can diff the change.
"""

from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

# The campaign taxonomy, plus the id this tool introduces.
IDS = {
    "floor": 1,
    "wall": 2,
    "column": 3,
    "display_table": 4,
    "shelf": 5,
    "person": 6,
    "laptop": 7,
    "tablet": 8,
    "phone": 9,
    "boxed_stock": 10,
    "glass": 11,
    "ignore": 255,
}
PLATE_W, PLATE_H = 960, 540


def zone_mask(poly, w: int, h: int) -> np.ndarray:
    pts = [(x * w / PLATE_W, y * h / PLATE_H) for x, y in poly]
    m = Image.new("L", (w, h), 0)
    ImageDraw.Draw(m).polygon(pts, fill=255)
    return np.asarray(m) > 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--zones", type=Path, required=True)
    ap.add_argument("--root", type=Path, default=Path("datasets/site30k_v1"))
    ap.add_argument("--apply", action="store_true", help="write; default is a dry run")
    ap.add_argument("--limit", type=int, default=0, help="first N masks per camera, for a look")
    args = ap.parse_args()

    zones = json.loads(args.zones.read_text())
    unknown = {z["to"] for zs in zones.values() for z in zs} - set(IDS)
    unknown |= {c for zs in zones.values() for z in zs for c in z.get("from", [])} - set(IDS)
    if unknown:
        raise SystemExit(f"unknown class names in {args.zones}: {sorted(unknown)}")

    backup = args.root / "masks_prestamp"
    changed_px: dict[str, Counter] = defaultdict(Counter)
    touched: Counter = Counter()

    for camera, zs in sorted(zones.items()):
        masks = sorted((args.root / "masks").glob(f"{camera}__*.png"))
        if args.limit:
            masks = masks[: args.limit]
        if not masks:
            print(f"  !! {camera}: no masks under {args.root}")
            continue
        cache: dict[tuple, np.ndarray] = {}
        for mp in masks:
            a = np.asarray(Image.open(mp)).copy()
            h, w = a.shape
            before = a.copy()
            for z in zs:
                key = (id(z), h, w)
                if key not in cache:
                    cache[key] = zone_mask(z["poly"], w, h)
                sel = cache[key]
                if z.get("from"):
                    allowed = np.isin(a, [IDS[c] for c in z["from"]])
                    sel = sel & allowed
                a[sel] = IDS[z["to"]]
                changed_px[camera][z["to"]] += int(sel.sum())
            if not np.array_equal(a, before):
                touched[camera] += 1
                if args.apply:
                    dst = backup / mp.name
                    if not dst.exists():
                        dst.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(mp, dst)
                    Image.fromarray(a).save(mp)

    total_px = 0
    for camera in sorted(changed_px):
        masks_n = len(list((args.root / "masks").glob(f"{camera}__*.png")))
        px = changed_px[camera]
        total_px += sum(px.values())
        per_frame = {k: round(v / max(touched[camera], 1)) for k, v in px.items()}
        print(
            f"{camera}: {touched[camera]}/{masks_n} masks change; "
            f"px per changed mask {per_frame}"
        )
    print(
        ("APPLIED" if args.apply else "DRY RUN")
        + f": {sum(touched.values())} masks, {total_px / 1e6:.1f} Mpx reassigned"
    )
    if args.apply:
        print(f"originals backed up under {backup} -- diff with tools/site30k/compare_masks.py")
    else:
        print("nothing written. Re-run with --apply once the polygons are confirmed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
