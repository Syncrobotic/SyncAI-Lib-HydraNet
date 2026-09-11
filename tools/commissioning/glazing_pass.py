"""Predict architectural glass into a source-bound Stage0 review bundle.

A candidate bundle never overwrites commissioned masks. --scene previews its surfaces
through the normal grounded-plane fitter. This is semantic segmentation; door handles,
hinge side, swing angle and exact metric dimensions are not detected by this model.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy import ndimage
from torch.nn import functional as F

from syncai_bev3d.render_provenance import sha256
from syncai_hydranet.data.trans10k import GLAZING_CLASSES
from syncai_hydranet.geometry.camera_json import CameraFile
from syncai_hydranet.models.glazing import GlazingEncoder, GlazingHead
from syncai_hydranet.utils.visualize import preprocess

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cameras", nargs="+")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.7)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--scene", action="store_true")
    args = parser.parse_args()
    if not 0 < args.threshold < 1 or args.threads < 1:
        parser.error("threshold must be between 0 and 1; threads must be positive")
    torch.set_num_threads(args.threads)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    spec = checkpoint["spec"]["encoder"]
    for key, digest_key in [
        ("config", "config_sha256"),
        ("encoder_checkpoint", "encoder_sha256"),
    ]:
        if sha256(Path(spec[key])) != spec[digest_key]:
            raise ValueError(f"encoder input changed: {key}")
    if spec["classes"] != list(GLAZING_CLASSES):
        raise ValueError("checkpoint glazing classes disagree")
    encoder = GlazingEncoder(spec["config"], spec["encoder_checkpoint"])
    head = GlazingHead(encoder.channels).eval()
    head.load_state_dict(checkpoint["state_dict"])
    for camera in args.cameras:
        cf = CameraFile.load(ROOT / f"runs/commission01/{camera}.camera.json")
        if not cf.plate_file:
            raise ValueError(f"{camera} has no plate")
        source = ROOT / cf.plate_file
        image = Image.open(source).convert("RGB")
        if image.size != cf.image_size_px:
            raise ValueError("plate dimensions differ from camera calibration")
        dest = args.out / camera
        dest.mkdir(parents=True, exist_ok=False)
        x, _canvas, region = preprocess(image, spec["input_size"])
        left, top, width, height = region
        with torch.inference_mode():
            logits = F.interpolate(
                head(encoder(x)), size=spec["input_size"], mode="bilinear", align_corners=False
            )
            logits = logits[:, :, top : top + height, left : left + width]
            logits = F.interpolate(
                logits, size=(image.height, image.width), mode="bilinear", align_corners=False
            )
            probability = logits.softmax(1)[0].numpy()
        predicted = probability.argmax(0)
        confident = probability.max(0) >= args.threshold
        overlay = np.array(image).copy()
        masks = {}
        (dest / "masks").mkdir()
        colours = [(0, 0, 0), (60, 235, 155), (55, 175, 255), (245, 195, 60)]
        rows = []
        detections = []
        for index, name in enumerate(GLAZING_CLASSES[1:], 1):
            mask = (predicted == index) & confident
            labels, _count = ndimage.label(mask, np.ones((3, 3)))
            for component, bounds in enumerate(ndimage.find_objects(labels), 1):
                if bounds is None:
                    continue
                part = labels[bounds] == component
                if part.sum() < max(100, mask.size * 0.0002):
                    mask[bounds][part] = False
                    continue
                rows_slice, cols_slice = bounds
                detections.append(
                    {
                        "kind": name,
                        "component": component,
                        "bbox_px": [
                            cols_slice.start,
                            rows_slice.start,
                            cols_slice.stop,
                            rows_slice.stop,
                        ],
                        "mask_pixels": int(part.sum()),
                        "score": float(probability[index][bounds][part].mean()),
                    }
                )
            masks[name] = mask
            path = dest / f"masks/{name}.png"
            Image.fromarray(mask.astype(np.uint8) * 255).save(path)
            overlay[mask] = overlay[mask] * 0.6 + np.array(colours[index]) * 0.4
            rows.append(
                {
                    "kind": name,
                    "pixels": int(mask.sum()),
                    "mean_score": float(probability[index][mask].mean())
                    if mask.any()
                    else None,
                    "mask_sha256": sha256(path),
                }
            )
        Image.fromarray(overlay).save(dest / "overlay.png")
        image.save(dest / "plate.png")
        np.savez_compressed(
            dest / "probabilities.npz", probabilities=probability.astype(np.float16)
        )
        report = {
            "schema": 1,
            "camera": camera,
            "source": str(source.relative_to(ROOT)),
            "source_sha256": sha256(source),
            "camera_sha256": sha256(ROOT / f"runs/commission01/{camera}.camera.json"),
            "checkpoint": str(args.checkpoint.resolve()),
            "checkpoint_sha256": sha256(args.checkpoint),
            "threshold": args.threshold,
            "classes": list(GLAZING_CLASSES),
            "masks": rows,
            "detections": detections,
            "status": "candidate only; commissioned masks are unchanged",
            "limitations": "Public-data model; hinge, swing and metric size are unverified.",
        }
        if args.scene:
            import trimesh

            from syncai_bev3d import scene_mesh as sm

            surfaces = []
            _cf, items, heights, shapes = sm.build_scene_regular(
                camera, ROOT, surface_masks=masks, surface_report=surfaces
            )
            report["surface_fits"] = surfaces
            accepted = [
                r
                for r in surfaces
                if r["status"] == "supported" and r.get("mask_source") == "candidate override"
            ]
            if accepted:
                scene = trimesh.Scene()
                for index, ((vertices, faces), key, alpha, _shadow) in enumerate(items):
                    if not len(faces):
                        continue
                    mesh = trimesh.Trimesh(
                        vertices=vertices,
                        faces=faces,
                        process=False,
                        vertex_colors=np.array([*sm.colour_of(key), alpha], dtype=np.uint8),
                    )
                    scene.add_geometry(mesh, node_name=f"{key}_{index}")
                scene.export(dest / "candidate.glb")
                sm.render(camera, items, heights, dest / "candidate.png", shapes=shapes)
                report["scene_status"] = "candidate preview; requires visual review"
            else:
                # Empty glazing predictions must not reveal the broad generic-door
                # fallback as a new opaque shopfront in the preview.
                report["scene_status"] = "no accepted model surfaces; current scene retained"
                current = ROOT / f"runs/commission01/{camera}/scene.glb"
                if current.exists():
                    shutil.copy2(current, dest / "retained.glb")
                    report["retained_scene_sha256"] = sha256(current)
        (dest / "prediction.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({"camera": camera, "masks": rows, "output": str(dest)}), flush=True)


if __name__ == "__main__":
    main()
