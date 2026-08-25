"""Door and product masks for commissioned cameras -- the taxonomy gap pass.

The structure vote covers floor/wall/column/table/shelf; the user's spec also names
door, glass and product. Two of the three are teacher-derivable and added here from the
cleanest plate: `door` (main prompt table, 6/8 at 0.671) and `product` (objects table --
displayed merchandise is semi-static, so the median plate shows it). **Glass is not**:
the prompt table records 112 frames of it failing four distinct ways ("it stays a human
class"), so glass remains a human polygon for the zone tool, exactly as PLAN 2.1 always
had it.

Usage: uv run python tools/commissioning/extras_pass.py <camera> [...]
"""

import dataclasses
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy import ndimage

from syncai_bev3d.teachers import sam3 as SAM3
from syncai_hydranet.data import sam3_prompts as P
from syncai_hydranet.data import sam3_prompts_objects as O
from syncai_hydranet.geometry.camera_json import CameraFile

ROOT = Path("/home/paul/SyncAI-Lib-HydraNet")
W, H = 1920, 1080
MIN_PX = 2000  # doors and the generic product union
MIN_SUB_PX = 150  # a phone on a table is small and real


class _Sub:
    """A prompt bundle for a product subclass; measured usable at plate resolution
    (the old "0 instances" verdict was taken at 352x240 -- the plates are 1080p)."""

    def __init__(self, prompts, min_score=0.3):
        self.prompts = prompts
        self.min_score = min_score


def concepts():
    door = next(c for c in P.CONCEPTS if c.name == "door")
    product = next(c for c in O.CONCEPTS if c.name == "product")
    return {
        "door": door,
        "product": product,
        "product_iphone": _Sub(("iphone", "smartphone", "mobile phone")),
        "product_ipad": _Sub(("ipad", "tablet computer")),
        "product_macbook": _Sub(("macbook", "open laptop", "laptop computer")),
        "product_boxed_stock": _Sub(
            ("stack of product boxes", "product box on a shelf", "boxed product")
        ),
    }


def run_camera(camera, proc, model, device, table):
    cache = np.load(ROOT / f"runs/commission01/{camera}/structure_cache.npz")
    plate = ROOT / "datasets/studioa_static" / camera / f"plate_{cache['cleanest']}.png"
    img = Image.open(plate).convert("RGB").resize((W, H), Image.Resampling.LANCZOS)
    embeds = SAM3.vision_features(proc, model, img, device)
    cf = CameraFile.load(ROOT / f"runs/commission01/{camera}.camera.json")
    w, h = cf.image_size_px
    mask_files = dict(cf.mask_files)
    stats = []
    for name, concept in table.items():
        union = np.zeros((H, W), bool)
        n = 0
        for prompt in concept.prompts:
            for m, _s in SAM3.segment(
                proc, model, img, prompt, concept.min_score, device, embeds
            ):
                if m.sum() < (MIN_SUB_PX if name.startswith("product_") else MIN_PX):
                    continue
                union |= m
                n += 1
        if name == "door" and union.any():
            # A doorway is tall; a cabinet door is not. The height map already knows.
            z = np.load(ROOT / f"runs/site30k_qa/geometry_cache/{camera}.npz")
            height, ok = z["height"], z["geom_ok"]
            lab, ncomp = ndimage.label(union, structure=np.ones((3, 3)))
            kept = np.zeros_like(union)
            for k in range(1, ncomp + 1):
                sel = (lab == k) & ok
                hs = height[sel]
                hs = hs[np.isfinite(hs)]
                if len(hs) > 100 and np.percentile(hs, 85) >= 1.6:
                    kept |= lab == k
            union = kept
        out = ROOT / "runs/commission01" / camera / "masks" / f"{name}.png"
        small = np.asarray(
            Image.fromarray((union * 255).astype(np.uint8)).resize(
                (w, h), Image.Resampling.NEAREST
            )
        )
        Image.fromarray(small).save(out)
        mask_files[name] = f"{camera}/masks/{name}.png"
        stats.append(f"{name}: {n} instances, {100 * union.mean():.1f}% of frame")
    cf = dataclasses.replace(cf, mask_files=mask_files)
    cf.validate()
    cf.save(ROOT / f"runs/commission01/{camera}.camera.json")
    print(f"  [{camera}] " + "; ".join(stats))


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    proc, model = SAM3.load_sam3(SAM3.MODEL_ID, device)
    table = concepts()
    for camera in sys.argv[1:]:
        run_camera(camera, proc, model, device, table)


if __name__ == "__main__":
    main()
