"""Stage 0 for one camera from one clip: the README chain as one command.

    uv run python tools/commissioning/stage0_commission.py <camera> --clip <mp4> \\
        [--note "nvr 10.12.31.9 ch12"] [--allow-bootstrap] \\
        [--freeze-out runs/stage0_<cam>] [--dry-run]

Until 2026-09-16 the chain for a second site lived in a shell script under `runs/` of a
copied checkout (FTI, `batch_one.sh`), with the bootstrap rule as inline Python nobody
could test. The steps here are the README's, in the README's order, against the tree
`SYNCAI_ROOT` names: static plate, person boxes, onboard, zones, the camera.json
contract, the review bundle, masks, extras, depth completion, the three scene renders,
and finally `stage0_baseline.py`, which freezes the result as a review candidate.

`--allow-bootstrap` is the only decision the tool takes on its own, and it says so in
the calib: a camera with no person corpus gets scale 1.0 on DA-V2's raw metres, flagged
`scale_bootstrap_dav2_raw_unverified`, so the tile ruler has a walkable mask to read
through. Nothing downstream may present those metres as measured.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import re
import subprocess
import sys
from pathlib import Path

from syncai_hydranet.paths import repo_root

ROOT = repo_root(Path(__file__).resolve().parents[2])
SLOT = re.compile(r"_(\d{8}-\d{6})_")
BOOTSTRAP_FLAG = "scale_bootstrap_dav2_raw_unverified"


def slot_from_clip(clip: Path) -> str:
    """The `_YYYYMMDD-HHMMSS_` slot in an archive name, else the file's mtime in UTC."""
    m = SLOT.search(clip.name)
    if m:
        return m.group(1)
    stamp = _dt.datetime.fromtimestamp(clip.stat().st_mtime, tz=_dt.UTC)
    return stamp.strftime("%Y%m%d-%H%M%S")


def register_camera(cameras_json: Path, camera: str, note: str) -> None:
    site = json.loads(cameras_json.read_text())
    site["cameras"].setdefault(camera, {"role": "factory", "note": note})
    cameras_json.write_text(json.dumps(site, indent=1, ensure_ascii=False) + "\n")


def bootstrap_unmeasured(calib_path: Path, allow: bool) -> str:
    """Return the calib's scale source; substitute the bootstrap when there is none.

    No floor plane at any vfov is a refusal (exit 2): the view shows too little floor to
    commission. An unmeasured scale with `allow` False is also a refusal (exit 3): the
    caller has to say it wants raw metres."""
    d = json.loads(calib_path.read_text())
    if d.get("height_dav2_raw_m") is None or d.get("pitch_deg") is None:
        raise SystemExit(2)
    source = str(d.get("scale_source", "unmeasured"))
    if not source.startswith("unmeasured"):
        return source
    if not allow:
        raise SystemExit(3)
    d.update(
        {
            "height_m": d["height_dav2_raw_m"],
            "scale": 1.0,
            "height_source": "dav2_plane_height_raw_BOOTSTRAP_unverified",
            "scale_source": "dav2_metric_indoor_raw_bootstrap_unverified",
        }
    )
    d["flags"] = [f for f in d.get("flags", []) if f != "scale_unmeasured"] + [BOOTSTRAP_FLAG]
    calib_path.write_text(json.dumps(d, indent=1) + "\n")
    return d["scale_source"]


def plan(
    root: Path,
    camera: str,
    clips_root: Path,
    plates_root: Path,
    freeze_out: Path | None,
    slot: str = "",
):
    """The chain's commands, each relative to `root`, in the README's order.

    The review bundle refuses to overwrite, so a rerun writes beside the last one, named
    by the clip's slot rather than deleting what a person may have looked at."""
    py = sys.executable
    anns = f"datasets/fti_person_gdino/{camera}/annotations/instances_all.json"
    steps = [
        (
            "plate",
            [
                py,
                "scripts/static_plates.py",
                "--root",
                str(clips_root),
                "--out",
                str(plates_root),
                "--only",
                camera,
            ],
        ),
        (
            "person boxes",
            [
                py,
                "scripts/gdino_person_boxes.py",
                "--out",
                f"datasets/fti_person_gdino/{camera}",
                "--frames",
                "24",
                "--train-thr",
                "0.35",
                "--clips",
                *sorted(str(p) for p in (root / clips_root / camera).glob("archive_*.mp4")),
            ],
        ),
        (
            "onboard",
            [
                py,
                "scripts/onboard_camera.py",
                "--camera",
                camera,
                "--out",
                "runs/onboard01",
                "--plates-root",
                str(plates_root),
                "--cameras-json",
                str(clips_root / "cameras.json"),
                "--person-anns",
                anns,
                "--skip-person-frac",
            ],
        ),
        (
            "zones",
            [
                py,
                "scripts/propose_zones.py",
                "--cameras",
                camera,
                "--calib-dir",
                "runs/onboard01",
                "--out",
                "runs/zones01",
            ],
        ),
        (
            "review bundle",
            [
                py,
                "tools/commissioning/commission_camera.py",
                f"runs/onboard01/{camera}.calib.json",
                "--out",
                f"runs/commission_review/{camera}_{slot or 'batch'}",
            ],
        ),
        (
            "masks",
            [
                py,
                "tools/commissioning/masks_pass.py",
                camera,
                "--plates-root",
                str(plates_root),
                "--out-root",
                "runs/commission01",
                "--calib-root",
                "runs/onboard01",
            ],
        ),
        ("extras", [py, "tools/commissioning/extras_pass.py", camera]),
        ("depth completion", [py, "tools/commissioning/depth_complete.py", camera]),
        ("scene3d", [py, "tools/commissioning/scene3d.py", camera]),
        ("scene overlay", [py, "tools/commissioning/scene_overlay.py", camera]),
        ("scene mesh", [py, "tools/commissioning/scene_mesh.py", camera]),
    ]
    if freeze_out is not None:
        steps.append(
            (
                "freeze",
                [
                    py,
                    "tools/commissioning/stage0_baseline.py",
                    "--root",
                    str(root),
                    "--out",
                    str(freeze_out),
                    camera,
                ],
            )
        )
    return steps


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("camera")
    ap.add_argument("--clip", type=Path, required=True, help="one recorded clip of the camera")
    ap.add_argument("--note", default="", help="where the stream came from, for cameras.json")
    ap.add_argument("--clips-root", type=Path, default=Path("datasets/fti_clips"))
    ap.add_argument("--plates-root", type=Path, default=Path("datasets/fti_static"))
    ap.add_argument(
        "--allow-bootstrap",
        action="store_true",
        help="scale 1.0 on raw DA-V2 metres when no person box; flagged in the calib",
    )
    ap.add_argument(
        "--freeze-out",
        type=Path,
        help="stage0_baseline output directory; omit to stop at the renders",
    )
    ap.add_argument("--dry-run", action="store_true", help="print the commands and stop")
    args = ap.parse_args(argv)
    root = ROOT
    clips_root, plates_root = args.clips_root, args.plates_root
    clip = args.clip.resolve()
    if not clip.is_file():
        ap.error(f"clip not found: {clip}")
    link_dir = root / clips_root / args.camera
    slot = slot_from_clip(clip)
    link = link_dir / f"archive_{slot}_fti.mp4"
    if args.dry_run:
        print(f"link {link} -> {clip}")
        for label, argv_ in plan(
            root, args.camera, clips_root, plates_root, args.freeze_out, slot
        ):
            print(f"[{label}] {' '.join(argv_)}")
        return 0
    link_dir.mkdir(parents=True, exist_ok=True)
    if link.is_symlink() or link.exists():
        link.unlink()
    link.symlink_to(clip)
    register_camera(root / clips_root / "cameras.json", args.camera, args.note)
    env = dict(os.environ, SYNCAI_ROOT=str(root), PYTHONPATH=str(root / "src"))
    for label, argv_ in plan(root, args.camera, clips_root, plates_root, args.freeze_out, slot):
        print(f"== {args.camera} {label}", flush=True)
        subprocess.run(["nice", "-n", "10", *argv_], cwd=root, env=env, check=True)
        if label == "onboard":
            calib = root / f"runs/onboard01/{args.camera}.calib.json"
            source = bootstrap_unmeasured(calib, args.allow_bootstrap)
            print(f"   scale source: {source}", flush=True)
        if label == "zones":
            from syncai_bev3d.commissioning import from_onboard_calib

            (root / "runs/commission01").mkdir(parents=True, exist_ok=True)
            from_onboard_calib(root / f"runs/onboard01/{args.camera}.calib.json").save(
                root / f"runs/commission01/{args.camera}.camera.json"
            )
    print(f"STAGE0_DONE {args.camera}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
