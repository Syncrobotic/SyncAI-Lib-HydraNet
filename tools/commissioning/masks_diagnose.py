"""Why `masks_pass` gave a camera the structure it did -- per cluster, with the picture.

`masks_pass.py` prints one line per camera ("N instances -> M objects, A accepted") and
throws the reasoning away. That total answers how many objects were kept; it cannot answer
the question the render actually raises, which is *which* fixture went missing and at which
stage. A fixture can be absent because SAM 3 never proposed it, or because the clustering
merged it into a neighbour, or because `decide_structure` named it something else. Those
have nothing in common but the symptom.

So this runs the same front half -- plates, floor, SAM 3 static concepts, cluster,
decide_structure -- and writes both halves of the record:

  runs/<out>/<camera>/decisions.json   every cluster: the per-family best score that
                                       ranked it, the three teacher readings behind the
                                       verdict, the winner and the rejection if any
  assets/masks_clusters_<camera>.png   a contact sheet, one cell per cluster, cropped to
                                       its box with the mask tinted and captioned by
                                       verdict -- so "was this proposed" is read off the
                                       image rather than inferred from a count

Nothing is written into `camera.json` and no mask file is touched: this is an instrument,
not a second pipeline.
"""

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

spec = importlib.util.spec_from_file_location(
    "recipe", "/home/paul/SyncAI-Lib-HydraNet/tools/site30k/recipe.py"
)
R = importlib.util.module_from_spec(spec)
spec.loader.exec_module(R)

ROOT = Path("/home/paul/SyncAI-Lib-HydraNet")
FAM_RGB = {
    "wall": (230, 40, 40),
    "column": (240, 200, 60),
    "table": (200, 90, 200),
    "shelf": (90, 140, 240),
    None: (120, 120, 120),
}
CELL = 320


def family(concept, prompt):
    """The recipe's own prompt -> family map (decide_structure, inlined there)."""
    if concept == "wall":
        return "wall"
    if concept == "column":
        return "column"
    if prompt in R.TABLE_FAMILY:
        return "table"
    if prompt in R.SHELF_FAMILY:
        return "shelf"
    return None


def sam3_instances(camera, out, plate_imgs, slots, proc, model, static_concepts, refresh):
    """SAM 3's raw static proposals for the camera, cached bit-packed on first run.

    The prompts are swept once per plate and never change; what does change is what a
    later stage makes of them. Caching the proposals is what lets a clustering rule be
    tried on eight cameras in seconds instead of an hour of GPU.
    """
    npz, jsn = out / "instances.npz", out / "instances.json"
    if npz.exists() and jsn.exists() and not refresh:
        packed = np.load(npz)["packed"]
        smeta = json.loads(jsn.read_text())
        masks = [np.unpackbits(p).astype(bool).reshape(R.H, R.W) for p in packed]
        print(f"  [{camera}] {len(masks)} instances from cache")
        return masks, smeta

    smasks, smeta = [], []
    for slot in slots:
        embeds = R.SAM3.vision_features(proc, model, plate_imgs[slot], R.device)
        for concept in static_concepts:
            for prompt in concept.prompts:
                for m_, s_ in R.SAM3.segment(
                    proc, model, plate_imgs[slot], prompt, concept.min_score, R.device, embeds
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
    packed = np.stack([np.packbits(m.ravel()) for m in smasks]) if smasks else np.zeros(0)
    np.savez_compressed(npz, packed=packed)
    jsn.write_text(json.dumps(smeta))
    return smasks, smeta


def diagnose(camera, third, proc, model, static_concepts, plates_root, out_root, refresh=False):
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
        plate_imgs[slot] = img
        plate_cache[(camera, slot)] = third.terrain_ids(img)

    floor, _strict = R.camera_floor(camera, geo, slots, plate_cache)

    out = Path(out_root) / camera
    out.mkdir(parents=True, exist_ok=True)
    smasks, smeta = sam3_instances(
        camera, out, plate_imgs, slots, proc, model, static_concepts, refresh
    )
    cl_masks, cl_votes = R.cluster(smasks, smeta)
    b03_maps = [plate_cache[(camera, s)] for s in slots]

    # Everything a clustering experiment needs, so trying a second merge rule costs no
    # GPU: the raw instances are already cached above, and these are the only other
    # inputs decide_structure reads.
    np.savez_compressed(
        out / "stage.npz",
        lx=geo.lx,
        lz=geo.lz,
        geom_ok=geo.geom_ok,
        horiz=geo.horiz,
        floor=floor,
        b03=np.stack(b03_maps),
        cleanest=min(slots, key=lambda s: float((plate_cache[(camera, s)] == 5).mean())),
    )
    _static_map, decisions = R.decide_structure(
        cl_masks, cl_votes, b03_maps, geo, geo.lx, geo.lz
    )

    # The ranking that produced each verdict, recomputed from the same votes so the
    # record shows what the winner beat and by how much.
    for d, votes in zip(decisions, cl_votes, strict=True):
        best = {"wall": 0.0, "column": 0.0, "table": 0.0, "shelf": 0.0}
        arg = {}
        for concept, prompt, score in votes:
            fam = family(concept, prompt)
            if fam and score > best[fam]:
                best[fam], arg[fam] = score, prompt
        d["best"] = {k: round(v, 3) for k, v in best.items()}
        d["best_prompt"] = arg
        d["n_votes"] = len(votes)

    (out / "decisions.json").write_text(json.dumps(decisions, indent=1))

    cleanest = min(slots, key=lambda s: float((plate_cache[(camera, s)] == 5).mean()))
    base = np.asarray(plate_imgs[cleanest], dtype=np.float64)
    order = sorted(range(len(cl_masks)), key=lambda k: -int(cl_masks[k].sum()))
    cols = 6
    rows = (len(order) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * CELL, rows * (CELL + 28)), (18, 18, 20))
    dr = ImageDraw.Draw(sheet)
    for i, k in enumerate(order):
        m = cl_masks[k]
        d = decisions[k]
        ys, xs = np.where(m)
        y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
        pad = 24
        y0, y1 = max(0, y0 - pad), min(R.H, y1 + pad)
        x0, x1 = max(0, x0 - pad), min(R.W, x1 + pad)
        crop = base[y0:y1, x0:x1].copy()
        sel = m[y0:y1, x0:x1]
        crop[sel] = 0.5 * crop[sel] + 0.5 * np.array(FAM_RGB[d["win"]])
        cell = Image.fromarray(crop.astype(np.uint8))
        cell.thumbnail((CELL, CELL), Image.Resampling.LANCZOS)
        cx, cy = (i % cols) * CELL, (i // cols) * (CELL + 28)
        sheet.paste(cell, (cx + (CELL - cell.width) // 2, cy + (CELL - cell.height) // 2))
        verdict = d["win"] or "contested"
        if d["reject"]:
            verdict = f"{verdict} REJECTED"
        b = d["best"]
        dr.text(
            (cx + 4, cy + CELL + 2),
            f"k{k} {int(m.sum() / 1000)}kpx {verdict}",
            fill=(255, 255, 255),
        )
        dr.text(
            (cx + 4, cy + CELL + 14),
            f"w{b['wall']:.2f} c{b['column']:.2f} t{b['table']:.2f} s{b['shelf']:.2f} "
            f"| b03w {d['b03_wall']:.2f} f {d['b03_fix']:.2f}",
            fill=(190, 190, 200),
        )
    sheet.save(ROOT / f"assets/masks_clusters_{camera}.png")

    acc = [d for d in decisions if d["win"] and not d["reject"]]
    by = {}
    for d in acc:
        by[d["win"]] = by.get(d["win"], 0) + 1
    print(f"  [{camera}] {len(cl_masks)} clusters, {len(acc)} accepted {by}")
    for d in sorted(decisions, key=lambda d: -d["px"])[:12]:
        print(
            f"    k{d['k']:>3} {d['px'] / 1000:6.1f}kpx  win={d['win']!s:<9} "
            f"{d['best']}  b03w={d['b03_wall']:.2f} fix={d['b03_fix']:.2f} "
            f"flat={d['flat']:.2f}  {d['reject'] or ''}"
        )
    return decisions


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("cameras", nargs="+")
    ap.add_argument("--plates-root", type=Path, default=ROOT / "datasets/studioa_static")
    ap.add_argument("--out-root", type=Path, default=ROOT / "runs/masks_diag01")
    ap.add_argument("--calib-root", type=Path, default=None)
    ap.add_argument(
        "--refresh", action="store_true", help="re-run SAM 3 instead of reusing instances.npz"
    )
    a = ap.parse_args()
    if a.calib_root is not None:
        R.CameraGeometry.CALIB_ROOT = a.calib_root
    third = R.M.ThirdOpinion(R.M.THIRD_OPINION_RUN, R.device)
    proc, model = R.M.load_sam3(R.SAM3.MODEL_ID, R.device)
    static_concepts, _moving = R.M.build_concepts_v2()
    for camera in a.cameras:
        print(f"== {camera}")
        diagnose(
            camera, third, proc, model, static_concepts, a.plates_root, a.out_root, a.refresh
        )


if __name__ == "__main__":
    main()
