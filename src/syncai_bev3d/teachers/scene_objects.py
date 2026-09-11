"""Instance-preserving teacher pass shared by Stage0 CLI entry points."""

import numpy as np
from PIL import Image

from syncai_bev3d.object_instances import (
    OBJECT_PROMPTS,
    ObjectInstance,
    deduplicate,
    save_instances,
)
from syncai_bev3d.teachers import sam3
from syncai_hydranet.geometry.camera_json import CameraFile


def run_camera(camera, proc, model, device, *, root, categories=None, supports_only=False):
    categories = list(OBJECT_PROMPTS) if categories is None else categories
    cf = CameraFile.load(root / f"runs/commission01/{camera}.camera.json")
    if not cf.plate_file:
        raise ValueError(f"{camera}: a commissioned source plate is required")
    source = root / cf.plate_file
    with Image.open(source) as plate:
        img = plate.convert("RGB").resize((1920, 1080), Image.Resampling.LANCZOS)
    embeds = sam3.vision_features(proc, model, img, device)
    tops = []
    for prompt in ("table top", "countertop"):
        for mask, score in sam3.segment(proc, model, img, prompt, 0.65, device, embeds):
            if mask.sum() < 2000:
                continue
            small = np.asarray(
                Image.fromarray(mask).resize(cf.image_size_px, Image.Resampling.NEAREST), bool
            )
            tops.append(ObjectInstance("table_top", small, score, prompt))
    tops = deduplicate(tops)
    save_instances(
        root / f"runs/commission01/{camera}/masks/support_tops.npz",
        tops,
        shape=cf.image_size_px[::-1],
        source=source,
        categories=["table_top"],
    )
    print(f"  {camera}: saved {len(tops)} tabletop masks", flush=True)
    if supports_only:
        return tops
    instances = []
    for category in categories:
        prompts, threshold = OBJECT_PROMPTS[category]
        for prompt in prompts:
            found = sam3.segment(proc, model, img, prompt, threshold, device, embeds)
            print(f"  {camera}: {prompt}: {len(found)} candidates", flush=True)
            for mask, score in found:
                # Reject tiny noise and masks spanning furniture rather than a device.
                fraction = mask.mean()
                maximum = 0.15 if category in {"chair", "stool"} else 0.06
                if not 0.00006 <= fraction <= maximum:
                    continue
                small = np.asarray(
                    Image.fromarray(mask).resize(cf.image_size_px, Image.Resampling.NEAREST),
                    bool,
                )
                instances.append(ObjectInstance(category, small, score, prompt))
    kept = deduplicate(instances)
    dest = root / f"runs/commission01/{camera}/masks/object_instances.npz"
    save_instances(
        dest, kept, shape=cf.image_size_px[::-1], source=source, categories=categories
    )
    print(f"  {camera}: saved {len(kept)} independent objects to {dest}", flush=True)
    return kept
