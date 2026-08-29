"""Commissioning masks pass (PLAN 2.1c-d) for one camera, reusing the site30k recipe.

Runs the recipe's per-camera pre-pass only -- plate teacher, voted floor, SAM 3 static
concepts, cluster, decide_structure -- with no clips and no frame rendering. Output per
camera: class masks + walkable + shelf ROIs, written into runs/commission01/<camera>/
and referenced from its camera.json.
"""

import argparse
import importlib.util
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

spec = importlib.util.spec_from_file_location(
    "recipe", "/home/paul/SyncAI-Lib-HydraNet/tools/site30k/recipe.py"
)
R = importlib.util.module_from_spec(spec)
spec.loader.exec_module(R)

from syncai_hydranet.geometry.camera_json import CameraFile  # noqa: E402

ROOT = Path("/home/paul/SyncAI-Lib-HydraNet")
CLASS_NAMES = {1: "floor", 2: "wall", 3: "column", 4: "display_table", 5: "display_shelf"}
OVERLAY_RGB = {
    1: (60, 200, 90),
    2: (150, 150, 160),
    3: (240, 200, 60),
    4: (200, 90, 200),
    5: (90, 140, 240),
}


def run_camera(camera, third, proc, model, static_concepts, plates_root=None, out_root=None):
    plates_root = plates_root or ROOT / "datasets/studioa_static"
    out_root = out_root or ROOT / "runs/commission01"
    geo = R.CameraGeometry(camera, third)
    plate_dir = Path(plates_root) / camera
    slots = [p.stem.split("_", 1)[1] for p in sorted(plate_dir.glob("plate_*.png"))]
    plate_imgs, plate_cache = {}, {}
    for slot in slots:
        img = (
            Image.open(plate_dir / f"plate_{slot}.png")
            .convert("RGB")
            .resize((R.W, R.H), Image.Resampling.LANCZOS)
        )
        plate_imgs[(camera, slot)] = img
        plate_cache[(camera, slot)] = third.terrain_ids(img)

    floor, _strict = R.camera_floor(camera, geo, slots, plate_cache)

    smasks, smeta = [], []
    for slot in slots:
        pimg = plate_imgs[(camera, slot)]
        embeds = R.SAM3.vision_features(proc, model, pimg, R.device)
        for concept in static_concepts:
            for prompt in concept.prompts:
                for m_, s_ in R.SAM3.segment(
                    proc, model, pimg, prompt, concept.min_score, R.device, embeds
                ):
                    if m_.sum() < 500:
                        continue
                    smeta.append(
                        {
                            "concept": concept.name,
                            "prompt": prompt,
                            "score": float(s_),
                            "px": int(m_.sum()),
                            "slot": slot,
                        }
                    )
                    smasks.append(m_)
    cl_masks, cl_votes = R.cluster(smasks, smeta)
    static_map, decisions = R.decide_structure(
        cl_masks, cl_votes, [plate_cache[(camera, s)] for s in slots], geo, geo.lx, geo.lz
    )
    acc = sum(1 for d in decisions if d["win"] and not d["reject"])
    print(
        f"  [{camera}] {len(smasks)} instances -> {len(cl_masks)} objects, {acc} accepted; "
        f"floor {100 * floor.mean():.1f}% of frame"
    )
    (Path(out_root) / camera).mkdir(parents=True, exist_ok=True)
    cleanest = min(slots, key=lambda s: float((plate_cache[(camera, s)] == 5).mean()))
    np.savez(
        Path(out_root) / camera / "structure_cache.npz",
        static=static_map,
        floor=floor,
        cleanest=cleanest,
    )

    # The recipe's own composition (batch loop): the structure background is IGNORE, the
    # floor boundary is snapped to the plate by the guided filter, and floor claimed
    # inside a fully-enclosed object hole belongs to the object (the counter-tray case).
    from scipy import ndimage as ndi

    plate_rgb = np.asarray(plate_imgs[(camera, cleanest)])
    floor_clip = (
        R.guided(
            floor.astype(np.float32),
            plate_rgb.astype(np.float32) / 255.0,
            R.GUIDE_RADIUS,
            R.GUIDE_EPS,
        )
        >= 0.5
    )
    floor_clip = ndi.binary_fill_holes(floor_clip)
    holes = np.zeros((R.H, R.W), bool)
    lbo, no = ndi.label(static_map != R.IGNORE)
    for k in range(1, no + 1):
        comp = lbo == k
        if comp.sum() < 2000:
            continue
        holes |= ndi.binary_fill_holes(comp) & ~comp
    floor_clip &= ~holes

    combined = np.zeros((R.H, R.W), np.uint8)
    for cid in (2, 3, 4, 5):
        combined[static_map == cid] = cid
    combined[floor_clip & (combined == 0)] = 1

    out = Path(out_root) / camera
    (out / "masks").mkdir(parents=True, exist_ok=True)
    cf = CameraFile.load(Path(out_root) / f"{camera}.camera.json")
    w, h = cf.image_size_px

    mask_files = {}
    for cid, name in CLASS_NAMES.items():
        m = (combined == cid).astype(np.uint8) * 255
        small = np.asarray(Image.fromarray(m).resize((w, h), Image.Resampling.NEAREST))
        Image.fromarray(small).save(out / "masks" / f"{name}.png")
        mask_files[name] = f"{camera}/masks/{name}.png"
    walk = (combined == 1).astype(np.uint8) * 255
    Image.fromarray(
        np.asarray(Image.fromarray(walk).resize((w, h), Image.Resampling.NEAREST))
    ).save(out / "masks" / "walkable.png")
    mask_files["walkable"] = f"{camera}/masks/walkable.png"

    # Shelf ROIs: connected fixtures (table or shelf), boxes in the raw stream frame.
    fixture = (combined == 4) | (combined == 5)
    lab, _n = ndimage.label(fixture)
    rois = []
    sx, sy = w / R.W, h / R.H
    for sl in ndimage.find_objects(lab):
        if sl is None:
            continue
        area = fixture[sl].sum()
        if area < 8000:  # smaller than any real display at 1080p
            continue
        y0, y1, x0, x1 = sl[0].start, sl[0].stop, sl[1].start, sl[1].stop
        rois.append(
            (
                round(max(0.0, x0 * sx), 1),
                round(max(0.0, y0 * sy), 1),
                round(min(float(w), x1 * sx), 1),
                round(min(float(h), y1 * sy), 1),
            )
        )
    print(f"  [{camera}] shelf ROIs: {len(rois)}")

    import dataclasses

    cf = dataclasses.replace(cf, mask_files=mask_files, shelf_rois_px=tuple(rois))
    cf.validate()
    cf.save(Path(out_root) / f"{camera}.camera.json")

    # Overlay preview on the cleanest plate.
    base = np.asarray(plate_imgs[(camera, cleanest)], dtype=np.float64)
    over = base.copy()
    for cid, colour in OVERLAY_RGB.items():
        sel = combined == cid
        over[sel] = 0.55 * base[sel] + 0.45 * np.array(colour)
    img = Image.fromarray(over.astype(np.uint8)).resize((960, 540), Image.Resampling.LANCZOS)
    from PIL import ImageDraw

    d = ImageDraw.Draw(img)
    for x0, y0, x1, y1 in rois:
        d.rectangle([x0, y0, x1, y1], outline=(255, 80, 40), width=2)
    d.rectangle([0, 540 - 16, 960, 540], fill=(0, 0, 0))
    d.text(
        (6, 540 - 14),
        f"{camera}  green=floor/walkable grey=wall yellow=column purple=table blue=shelf  "
        f"orange boxes=shelf ROIs ({len(rois)})",
        fill=(255, 255, 255),
    )
    img.save(ROOT / f"assets/commission_masks_{camera}.png")
    return decisions


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("cameras", nargs="+")
    ap.add_argument("--plates-root", type=Path, default=ROOT / "datasets/studioa_static")
    ap.add_argument("--out-root", type=Path, default=ROOT / "runs/commission01")
    ap.add_argument(
        "--calib-root",
        type=Path,
        default=None,
        help="onboarding sweep whose `<camera>.calib.json` gives the geometry "
        "(default: the recipe's own runs/onboard01)",
    )
    a = ap.parse_args()
    if a.calib_root is not None:
        R.CameraGeometry.CALIB_ROOT = a.calib_root
    third = R.M.ThirdOpinion(R.M.THIRD_OPINION_RUN, R.device)
    proc, model = R.M.load_sam3(R.SAM3.MODEL_ID, R.device)
    static_concepts, _moving = R.M.build_concepts_v2()
    for camera in a.cameras:
        print(f"== {camera}")
        run_camera(camera, third, proc, model, static_concepts, a.plates_root, a.out_root)


if __name__ == "__main__":
    main()
