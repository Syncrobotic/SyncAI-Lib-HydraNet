"""Stage 0-4: convert an onboard calibration and render its raw-frame metre grid.

Optionally refine against surveyed floor controls. Outputs are a separate review bundle;
existing commissioned files are never overwritten. See README's Stage 0 runbook.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import replace
from pathlib import Path

from PIL import Image

from syncai_bev3d.commissioning import from_onboard_calib, regeometry_from_calib
from syncai_bev3d.geometry_review import render_metre_grid
from syncai_bev3d.ground_control import refine_ground_control
from syncai_bev3d.metric_scale import refine_metric_scale


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("calib", type=Path)
    parser.add_argument("--existing", type=Path, help="preserve existing masks, zones and ROIs")
    measurements = parser.add_mutually_exclusive_group()
    measurements.add_argument(
        "--controls", type=Path, help="surveyed floor coordinates and raw pixels"
    )
    measurements.add_argument(
        "--distance-controls",
        type=Path,
        help="measured floor distances; two fitting segments and one held-out segment",
    )
    parser.add_argument("--fit-focal", action="store_true", help="needs eight fitting points")
    parser.add_argument("--pixel-threshold", type=float, default=4.0)
    parser.add_argument("--validation-threshold-m", type=float, default=0.15)
    parser.add_argument("--plate", type=Path, help="override the raw static plate path")
    parser.add_argument("--out", type=Path, required=True, help="new review bundle directory")
    args = parser.parse_args(argv)
    if args.out.exists():
        raise FileExistsError(f"review bundle already exists: {args.out}")
    if args.fit_focal and not args.controls:
        parser.error("--fit-focal requires --controls")
    cf = (
        regeometry_from_calib(args.existing, args.calib)
        if args.existing
        else from_onboard_calib(args.calib)
    )
    plate_path = args.plate or (Path(cf.plate_file) if cf.plate_file else None)
    if plate_path is None:
        parser.error("calibration has no plate_file; supply --plate")
    with Image.open(plate_path) as source:
        if abs(source.width / source.height - cf.image_size_px[0] / cf.image_size_px[1]) > 0.01:
            parser.error(
                "plate aspect ratio differs from calibration; supply the original raw plate"
            )
        plate = source.convert("RGB").resize(cf.image_size_px, Image.Resampling.LANCZOS)
    plate_hash = hashlib.sha256(plate_path.read_bytes()).hexdigest()
    cf = replace(cf, plate_file=str(plate_path), plate_sha256=plate_hash)
    before = render_metre_grid(plate, cf)
    report: dict = {
        "camera_id": cf.camera_id,
        "accepted": False,
        "reasons": ["no independent metric controls; grid requires visual review"],
        "scale_source": json.loads(args.calib.read_text()).get("scale_source"),
    }
    if args.controls:
        cf, report = refine_ground_control(
            cf,
            json.loads(args.controls.read_text()),
            fit_focal=args.fit_focal,
            pixel_threshold=args.pixel_threshold,
            validation_threshold_m=args.validation_threshold_m,
        )
        report["controls_sha256"] = hashlib.sha256(args.controls.read_bytes()).hexdigest()
    if args.distance_controls:
        cf, report = refine_metric_scale(cf, json.loads(args.distance_controls.read_text()))
        report["controls_sha256"] = hashlib.sha256(
            args.distance_controls.read_bytes()
        ).hexdigest()
    report["calib_sha256"] = hashlib.sha256(args.calib.read_bytes()).hexdigest()
    report["plate_sha256"] = plate_hash
    after = render_metre_grid(plate, cf)
    # Fail before writing if the operator accidentally selects a previous bundle.
    args.out.mkdir(parents=True, exist_ok=False)
    cf.save(args.out / f"{cf.camera_id}.camera.json")
    before.save(args.out / "grid.before.png")
    after.save(args.out / "grid.after.png")
    (args.out / "geometry-review.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    print(
        f"Wrote {args.out}; metric validation: {'PASS' if report['accepted'] else 'UNVERIFIED'}"
    )
    for reason in report["reasons"]:
        print(f"  {reason}")
    # Conversion alone is useful without a survey; a requested survey has to pass.
    return 2 if (args.controls or args.distance_controls) and not report["accepted"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
