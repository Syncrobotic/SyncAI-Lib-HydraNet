"""Record a reviewed screen/rear-casing observation for the next Stage0 scene build.

Example: facing_review.py CAMERA --instance 6 --visible-side back
         --reviewer NAME --evidence 'Rear casing is visible in the source crop'
This records visible evidence. Geometry still has to pass the renderer's fit checks.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from syncai_bev3d.object_facing import mask_identity
from syncai_bev3d.object_instances import image_digest, load_instances
from syncai_hydranet.geometry.camera_json import CameraFile

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("camera")
    parser.add_argument("--instance", type=int, required=True)
    parser.add_argument("--visible-side", choices=["front", "back"], required=True)
    parser.add_argument("--reviewer", required=True)
    parser.add_argument("--evidence", required=True)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if not args.reviewer.strip() or not args.evidence.strip():
        parser.error("reviewer and evidence cannot be empty")
    cf = CameraFile.load(ROOT / f"runs/commission01/{args.camera}.camera.json")
    if not cf.plate_file:
        parser.error("camera has no source plate")
    source = ROOT / cf.plate_file
    archive = ROOT / f"runs/commission01/{args.camera}/masks/object_instances.npz"
    instances, _metadata = load_instances(archive, source=source)
    if not 0 <= args.instance < len(instances):
        parser.error("instance index is outside the current archive")
    out = args.out or archive.with_name("object_facing.json")
    data: dict = {
        "schema_version": 1,
        "camera": args.camera,
        "source_sha256": image_digest(source),
        "observations": [],
    }
    if out.exists():
        previous = json.loads(out.read_text())
        if any(
            previous.get(k) != data[k] for k in ("schema_version", "camera", "source_sha256")
        ):
            parser.error("existing review belongs to another source; use a new --out path")
        data = previous
    instance = instances[args.instance]
    if not instance.mask.any():
        parser.error("instance has an empty mask")
    r, c = np.nonzero(instance.mask)
    observation = {
        "instance_id": args.instance,
        "category": instance.category,
        "mask_sha256": mask_identity(instance),
        "visible_side": args.visible_side,
        "reviewer": args.reviewer,
        "evidence": args.evidence,
        "source_crop_px": [int(c.min()), int(r.min()), int(c.max() + 1), int(r.max() + 1)],
        "method": "reviewed appearance; not an automatic classifier",
    }
    data["observations"] = [
        r for r in data["observations"] if r["instance_id"] != args.instance
    ]
    data["observations"].append(observation)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    print(f"Recorded {args.camera} #{args.instance}: {args.visible_side} -> {out}")


if __name__ == "__main__":
    main()
