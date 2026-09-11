"""Create an offline floor-distance picker for independent Stage0 scale validation.

python tools/commissioning/measure_floor.py CAMERA --out runs/scale_measurements/CAMERA
"""

import argparse
import json
from pathlib import Path

from PIL import Image

from syncai_hydranet.geometry.camera_json import CameraFile

ROOT = Path(__file__).resolve().parents[2]
PAGE = Path(__file__).with_suffix(".html").read_text()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("camera")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    cf = CameraFile.load(ROOT / f"runs/commission01/{args.camera}.camera.json")
    if not cf.plate_file:
        parser.error("camera has no source plate")
    args.out.mkdir(parents=True, exist_ok=False)
    with Image.open(ROOT / cf.plate_file) as source:
        source.convert("RGB").resize(cf.image_size_px).save(args.out / "plate.png")
    metadata = {
        "camera_id": cf.camera_id,
        "image_size_px": list(cf.image_size_px),
        "pixel_space": "raw",
        "surface": "floor",
    }
    page = PAGE.replace("__META__", json.dumps(metadata).replace("</", "<\\/"))
    (args.out / "index.html").write_text(page)
    print(args.out / "index.html")


if __name__ == "__main__":
    main()
