"""Rebuild scene geometry after a camera correction, without rerunning mask teachers."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from syncai_bev3d.commissioning import from_onboard_calib
from syncai_bev3d.geometry_cache import (
    build_geometry_cache,
    floor_depth_scale,
    geometry_signature,
    save_geometry_cache,
)
from syncai_bev3d.plate_calibration import MODEL, MODEL_REVISION, run_depth, undistort_image
from syncai_hydranet.geometry.camera_json import CameraFile


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("camera_file", type=Path)
    scale_args = parser.add_mutually_exclusive_group(required=True)
    scale_args.add_argument(
        "--depth-scale",
        type=float,
        help="explicit scale for this depth and camera; not inherited after refitting",
    )
    scale_args.add_argument(
        "--calib",
        type=Path,
        help="read depth scale from an onboard calibration matching camera.json",
    )
    scale_args.add_argument(
        "--floor-mask",
        type=Path,
        help="fit depth scale to a raw walkable mask under the new camera geometry",
    )
    parser.add_argument(
        "--depth", type=Path, help="reuse unscaled DA-V2 depth .npy of this undistorted plate"
    )
    parser.add_argument("--out", type=Path, required=True, help="new cache path for review")
    args = parser.parse_args(argv)
    if args.out.exists():
        parser.error("output exists; write a new cache and review before replacing it")
    cf = CameraFile.load(args.camera_file)
    depth_scale = args.depth_scale
    if args.calib:
        reference = from_onboard_calib(args.calib)
        if geometry_signature(reference) != geometry_signature(cf):
            parser.error(
                "onboard calibration differs from camera.json; recompute depth scale first"
            )
        depth_scale = json.loads(args.calib.read_text())["scale"]
    if not args.floor_mask and (
        depth_scale is None or not np.isfinite(depth_scale) or depth_scale <= 0
    ):
        parser.error("depth scale must be positive and finite")
    if cf.plate_file is None:
        parser.error("camera.json has no plate_file")
    plate_path = Path(cf.plate_file)
    with Image.open(plate_path) as image:
        plate = np.asarray(image.convert("RGB"))
    if abs(plate.shape[1] / plate.shape[0] - cf.image_size_px[0] / cf.image_size_px[1]) > 0.01:
        parser.error("plate aspect ratio does not match the calibrated frame")
    k1 = cf.lens.k1 if cf.lens else 0.0
    if cf.lens and (
        cf.lens.centre_px != (cf.image_size_px[0] / 2, cf.image_size_px[1] / 2)
        or not np.isclose(cf.lens.radius_px, np.hypot(*cf.image_size_px) / 2)
    ):
        parser.error("plate undistortion requires a centred, half-diagonal division lens")
    depth = (
        np.load(args.depth, allow_pickle=False)
        if args.depth
        else run_depth(undistort_image(plate, k1))
    )
    if depth.shape != plate.shape[:2]:
        parser.error("depth must match the uncropped undistorted plate dimensions")
    floor_report = None
    if args.floor_mask:
        with Image.open(args.floor_mask) as mask_image:
            if mask_image.size != cf.image_size_px:
                parser.error("floor mask must use camera.json's raw image dimensions")
            floor_mask = np.asarray(mask_image.convert("L")) > 127
        depth_scale, floor_report = floor_depth_scale(cf, depth, floor_mask)
    arrays = build_geometry_cache(cf, depth, depth_scale=depth_scale)
    if floor_report is not None:
        arrays["floor_scale_report"] = np.array(json.dumps(floor_report))
    arrays["plate_sha256"] = np.array(hashlib.sha256(plate_path.read_bytes()).hexdigest())
    arrays["depth_model"] = np.array(MODEL)
    arrays["depth_revision"] = np.array(MODEL_REVISION)
    save_geometry_cache(args.out, arrays)
    if not args.depth:
        np.save(args.out.with_suffix(".depth.npy"), depth)
    print(f"Wrote {args.out}; calibrated geometry, object depths retain the teacher's errors")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
