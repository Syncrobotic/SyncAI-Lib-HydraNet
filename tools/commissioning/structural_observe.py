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
        "frames",
        type=Path,
        help="one fixed camera: raw frames, or a video directory with --retry",
    )
    parser.add_argument("--out", type=Path, required=True, help="new proposal JSON file")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--structure",
        action="store_true",
        help="add pinned surface teacher and conditional direction checks",
    )
    mode.add_argument(
        "--retry",
        action="store_true",
        help="bounded acquisition from a single-camera video directory",
    )
    parser.add_argument(
        "--max-attempts", type=int, default=3, help="includes the optional inherited seed"
    )
    parser.add_argument("--frames-per-window", type=int, default=3)
    parser.add_argument("--seed-surfaces", type=Path)
    parser.add_argument("--seed-groups", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args(argv)
    if args.out.exists():
        parser.error("output exists; use a new proposal path")
    if (args.seed_surfaces is not None or args.seed_groups is not None) and not args.retry:
        parser.error("seed arguments require --retry")
    if args.retry:
        from syncai_bev3d.structural_retry import RetryPolicy, run_retry

        try:
            policy = RetryPolicy(args.max_attempts, args.frames_per_window)
        except ValueError as exc:
            parser.error(str(exc))
        if (args.seed_surfaces is None) != (args.seed_groups is None):
            parser.error("seed surfaces and frozen groups must be supplied together")
        folder = args.out.with_suffix(".retry")
        if folder.exists():
            parser.error("retry evidence exists; use a new output path")
        report = run_retry(
            args.camera,
            args.frames,
            folder,
            policy=policy,
            seed_surfaces=args.seed_surfaces,
            seed_groups=args.seed_groups,
            device=args.device,
        )
        with args.out.open("x") as handle:
            json.dump(report, handle, indent=2, allow_nan=False)
            handle.write("\n")
        print(
            json.dumps(
                {
                    "status": report["status"],
                    "attempts": len(report["attempts"]),
                    "reasons": report["reasons"],
                }
            )
        )
        return int(report["status"] == "failed")
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
