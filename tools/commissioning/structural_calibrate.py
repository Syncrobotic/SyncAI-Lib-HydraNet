"""Write an isolated, source-bound structural calibration proposal for review."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from syncai_bev3d.render_provenance import sha256
from syncai_bev3d.structural_controls import load_structural_controls, refine_structural_camera
from syncai_hydranet.geometry.camera_json import CameraFile


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("camera")
    parser.add_argument("controls", type=Path)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists():
        parser.error("output exists; use a new review directory")
    root = args.root.resolve()
    controls_sha = sha256(args.controls)
    controls = load_structural_controls(args.controls, root, args.camera)
    cf = CameraFile.load(root / f"runs/commission01/{args.camera}.camera.json")
    proposal, report = refine_structural_camera(cf, controls)
    if (
        load_structural_controls(args.controls, root, args.camera) != controls
        or sha256(args.controls) != controls_sha
    ):
        raise ValueError("source observations changed while fitting")
    report["controls_sha256"] = controls_sha
    report["source_identity"] = controls["source_identity"]
    report["annotation_source"] = controls["annotation_source"]
    args.out.mkdir(parents=True)
    proposal.save(args.out / "camera.json")
    (args.out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Candidate written to {args.out}; line checks: {report['line_checks_passed']}")
    print("Diagnostic only: height, zone rederivation and deployment remain unverified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
