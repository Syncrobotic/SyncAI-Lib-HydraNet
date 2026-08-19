#!/usr/bin/env python3
"""Apply the night `person` veto to a COCO file, and say what it removed.

    python3 scripts/night_person_filter.py \\
        --coco runs/gdino_night_fleet/annotations/instances_all.json \\
        --static datasets/studioa_static \\
        --out runs/night_person_filtered

Reads a COCO written by `gdino_person_boxes.py`, writes `instances_night.json` containing
only the boxes a night tranche may use unreviewed, plus `report.json` and `REPORT.md`
accounting for every box that did not make it and why.

The rule is in [night_person.py](../src/syncai_hydranet/data/night_person.py) and so is the
argument for it. In one line: Grounding DINO at 0.35 does not hold at night -- 13 of 42
cameras put a box over it on an empty shuttered store, the worst at 0.594 -- and the veto
removes a box whose pixels never moved against that camera's own midnight plate.

**Two cameras are excluded by name rather than gated.** `Kaohsiung-cam02` (parked scooters
on an outdoor street bay) and `Tao-Hsin-cam09` (a hanging accessory under a near-nadir rack)
score *below* the verified person population, so no threshold separates them. Excluding
them is honest; running them through a gate that cannot see them would not be.

Measured on the fleet re-check, this configuration removes **72 of 72** false boxes and
**0 of 12** eye-verified person boxes. Those are the acceptance criteria and
`tests/test_night_person_veto.py` holds them.

WHAT IT DOES NOT DO

It does not decide the frame is night. Pass night frames. `gdino_person_boxes.py` records
per-frame luma and chroma for exactly this reason -- an IR frame is monochrome by
construction -- and this script reports the chroma it saw so a daylight frame that wandered
in is visible rather than silently vetoed against a midnight plate.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
for candidate in (HERE.parent / "src", HERE / "src"):
    if candidate.is_dir():
        sys.path.insert(0, str(candidate))

from syncai_hydranet.data.night_person import (  # noqa: E402
    DROP_ABOVE,
    PERSON_SCORE,
    UNPROTECTED,
    NightPersonVeto,
    camera_of,
)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--coco", type=Path, required=True)
    ap.add_argument("--static", type=Path, default=Path("datasets/studioa_static"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--drop-above", type=float, default=DROP_ABOVE)
    ap.add_argument("--score-thr", type=float, default=PERSON_SCORE)
    args = ap.parse_args(argv)

    coco = json.loads(args.coco.read_text())
    images = {im["id"]: im for im in coco["images"]}
    by_image: dict[int, list] = defaultdict(list)
    for a in coco["annotations"]:
        by_image[a["image_id"]].append(a)

    veto = NightPersonVeto(args.static, drop_above=args.drop_above, score_thr=args.score_thr)
    kept_anns, reasons = [], Counter()
    per_cam: dict[str, Counter] = defaultdict(Counter)
    bright: list[str] = []

    for iid, im in images.items():
        cam = camera_of(im["file_name"])
        if im.get("chroma", 0.0) > 1.0:
            bright.append(im["file_name"])
        anns = by_image.get(iid, [])
        decisions = veto.apply(
            im["file_name"], [(a["bbox"], a["score"]) for a in anns], im["width"], im["height"]
        )
        for ann, d in zip(anns, decisions, strict=True):
            if d.score < args.score_thr:
                continue  # never counted as a removal; the detector did not want it
            key = d.reason.split(":")[0]
            per_cam[cam][key if not d.keep else "kept"] += 1
            reasons[key if not d.keep else "kept"] += 1
            if d.keep:
                kept_anns.append({**ann, "static_share": d.share})

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "instances_night.json").write_text(
        json.dumps({**coco, "annotations": kept_anns}, indent=1)
    )

    lines = [
        "# Night `person` boxes that may be used unreviewed",
        "",
        f"`{args.coco}` -> **{len(kept_anns)} boxes kept** of "
        f"{sum(reasons.values())} at score >= {args.score_thr}.",
        "",
        "| outcome | boxes |",
        "|---|---|",
        *[f"| {k} | {v} |" for k, v in reasons.most_common()],
        "",
        f"Veto threshold {args.drop_above} on static share against each camera's own "
        "midnight plate. A box at or above it never moved.",
        "",
        "| camera | status | kept | static | excluded | no plate |",
        "|---|---|---|---|---|---|",
    ]
    for cam in sorted(per_cam):
        c = per_cam[cam]
        lines.append(
            f"| {cam} | {veto.status(cam)} | {c['kept']} | {c['static']} "
            f"| {c['camera excluded']} | {c['no static plate']} |"
        )
    if bright:
        lines += [
            "",
            f"**{len(bright)} frames are not monochrome** and were vetoed against a "
            "midnight plate anyway. This script does not gate on daylight; that is a "
            "caller error and is reported rather than corrected.",
        ]
    if UNPROTECTED:
        lines += ["", "Excluded by name, with the reason:", ""]
        lines += [f"- `{c}` — {w}" for c, w in sorted(UNPROTECTED.items())]

    (args.out / "REPORT.md").write_text("\n".join(lines) + "\n")
    (args.out / "report.json").write_text(
        json.dumps(
            {
                "kept": len(kept_anns),
                "reasons": dict(reasons),
                "per_camera": {k: dict(v) for k, v in per_cam.items()},
                "drop_above": args.drop_above,
                "score_thr": args.score_thr,
                "non_monochrome_frames": bright,
            },
            indent=1,
        )
    )
    print("\n".join(lines[:10]))
    print(f"\nwrote {args.out}/instances_night.json, REPORT.md, report.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
