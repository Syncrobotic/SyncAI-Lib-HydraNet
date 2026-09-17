"""Extract raw-frame structural edge proposals without commissioned camera inputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from syncai_bev3d.structural_observations import observe_directory


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "camera", help="provenance label only; never selects camera-specific rules"
    )
    parser.add_argument(
        "frames", type=Path, help="one fixed camera; image filenames in time order"
    )
    parser.add_argument("--out", type=Path, required=True, help="new proposal JSON file")
    parser.add_argument(
        "--structure",
        action="store_true",
        help="add pinned surface teacher and conditional direction checks",
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args(argv)
    if args.out.exists():
        parser.error("output exists; use a new proposal path")
    report = observe_directory(args.frames, args.camera)
    if args.structure:
        from syncai_bev3d.structural_directions import add_direction_checks
        from syncai_bev3d.structural_surfaces import SurfaceTeacher, enrich_surfaces

        evidence_dir = args.out.with_suffix(".evidence")
        if evidence_dir.exists():
            parser.error("surface evidence exists; use a new proposal path")
        report = enrich_surfaces(report, SurfaceTeacher(args.device), evidence_dir)
        report = add_direction_checks(report, evidence_dir / "groups.freeze.json")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps(report["summary"]))
    print(
        "Conditional proposals only; see direction checks and reasons; calibration_ready=false."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
