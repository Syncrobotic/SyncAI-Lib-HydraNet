"""Detect individual Stage0 assets: python tools/commissioning/objects_pass.py CAMERA ...

Run after commissioning/extras, before scene_mesh. No per-pixel depth is changed.
The renderer consumes masks/object_instances.npz automatically on its next build.
"""

import argparse
from pathlib import Path

from syncai_bev3d.object_instances import OBJECT_PROMPTS
from syncai_bev3d.teachers import sam3
from syncai_bev3d.teachers.scene_objects import run_camera
from syncai_hydranet.utils.device import pick_device

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cameras", nargs="+")
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"])
    parser.add_argument("--categories", nargs="+", choices=list(OBJECT_PROMPTS))
    parser.add_argument(
        "--supports-only",
        action="store_true",
        help="refresh tabletop masks and preserve device instances",
    )
    args = parser.parse_args()
    device = args.device or str(pick_device())
    proc, model = sam3.load_sam3(sam3.MODEL_ID, device)
    for camera in args.cameras:
        run_camera(
            camera,
            proc,
            model,
            device,
            root=ROOT,
            categories=args.categories,
            supports_only=args.supports_only,
        )


if __name__ == "__main__":
    main()
