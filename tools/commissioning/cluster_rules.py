"""Try a merge rule against the cached SAM 3 proposals, on every camera, with no GPU.

`recipe.cluster` merges a proposal into an existing object when the two overlap at
IoU >= 0.6 **or when 85% of the newcomer lies inside it**, and the object's mask is fixed
by whichever proposal arrived first -- the largest. In a white shop the `wall` prompt
returns one region covering the wall and the counters standing in front of it, so every
counter is fully contained in it and is absorbed as a *vote* rather than becoming an
object. The votes are still in the record (Tao-Hsin-cam03's 663 kpx cluster ranks
`wall 0.969 / table 0.945`), which is what says the fixtures were found and then merged
away rather than never proposed.

This runs `masks_diagnose.py`'s cached proposals through alternative merge rules and the
recipe's own `decide_structure`, so the comparison is like-for-like on everything except
the rule under test. Rules:

  current    IoU >= 0.6 or containment >= 0.85                    (recipe.cluster)
  iou        IoU >= 0.6 only                                      (containment removed)
  sized      IoU >= 0.6, or containment >= 0.85 with the newcomer at least RATIO of the
             object it would join -- containment means identity only between masks of
             comparable size, and a counter is a fifth of the wall behind it

Usage: tools/commissioning/cluster_rules.py --rule sized --ratio 0.5 [cameras...]
"""

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

# The repo root, derived rather than written out: every one of these 26 tools had it
# as an absolute path, so a second checkout ran against the first one's `runs/` and
# any machine but this one failed at import with a path and no reason. Two levels up
# from `tools/<group>/<tool>.py`, and `tests/test_no_absolute_sys_path.py` keeps it so.
ROOT = Path(__file__).resolve().parents[2]

spec = importlib.util.spec_from_file_location("recipe", str(ROOT / "tools/site30k/recipe.py"))
R = importlib.util.module_from_spec(spec)
spec.loader.exec_module(R)

CAMS = (
    "Taichung-cam01",
    "Taichung-cam04",
    "Taichung-cam07",
    "Taichung-cam10",
    "Taichung-cam11",
    "Tao-Hsin-cam03",
    "Tao-Hsin-cam04",
    "Kaohsiung-cam04",
)
RGB = {2: (230, 40, 40), 3: (240, 200, 60), 4: (200, 90, 200), 5: (90, 140, 240)}
NAMES = {2: "wall", 3: "column", 4: "table", 5: "shelf"}


class Geo:
    """The four fields `decide_structure` reads off a CameraGeometry, from stage.npz."""

    def __init__(self, z):
        self.lx, self.lz = z["lx"], z["lz"]
        self.geom_ok, self.horiz = z["geom_ok"], z["horiz"]


def bbox(m):
    ys, xs = np.where(m)
    return ys.min(), ys.max() + 1, xs.min(), xs.max() + 1


def cluster_rule(masks, meta, rule, ratio):
    """recipe.cluster, with the merge test swapped. Returns (masks, votes, merge log).

    Identical arithmetic to the recipe's loop; the boxes are only a skip. Dropping
    containment leaves hundreds of objects instead of forty, and a full-frame `&` per
    pair is then hours -- two masks whose boxes miss cannot overlap, and the ones whose
    boxes meet are compared inside the intersecting window alone.
    """
    order = np.argsort([-m["px"] for m in meta])
    boxes = {int(i): bbox(masks[i]) for i in order}
    areas = {int(i): int(masks[i].sum()) for i in order}
    cl_masks, cl_votes, cl_box, cl_area, log = [], [], [], [], []
    for i in order:
        m = masks[i]
        ay0, ay1, ax0, ax1 = boxes[int(i)]
        m_area = areas[int(i)]
        for j, cm in enumerate(cl_masks):
            by0, by1, bx0, bx1 = cl_box[j]
            y0, y1 = max(ay0, by0), min(ay1, by1)
            x0, x1 = max(ax0, bx0), min(ax1, bx1)
            if y0 >= y1 or x0 >= x1:
                continue
            inter = int((m[y0:y1, x0:x1] & cm[y0:y1, x0:x1]).sum())
            iou = inter / max(m_area + cl_area[j] - inter, 1)
            contain = inter / max(m_area, 1)
            size = m_area / max(cl_area[j], 1)
            if rule == "iou":
                hit = iou >= 0.6
            elif rule == "sized":
                hit = iou >= 0.6 or (contain >= 0.85 and size >= ratio)
            else:
                hit = iou >= 0.6 or contain >= 0.85
            if hit:
                cl_votes[j].append((meta[i]["concept"], meta[i]["prompt"], meta[i]["score"]))
                log.append(
                    {
                        "into": j,
                        "prompt": meta[i]["prompt"],
                        "concept": meta[i]["concept"],
                        "score": meta[i]["score"],
                        "iou": round(float(iou), 3),
                        "contain": round(float(contain), 3),
                        "size": round(float(size), 3),
                        "by": "iou" if iou >= 0.6 else "contain",
                    }
                )
                break
        else:
            cl_masks.append(m)
            cl_votes.append([(meta[i]["concept"], meta[i]["prompt"], meta[i]["score"])])
            cl_box.append((ay0, ay1, ax0, ax1))
            cl_area.append(m_area)
    return cl_masks, cl_votes, log


def load(camera, diag_root):
    d = Path(diag_root) / camera
    packed = np.load(d / "instances.npz")["packed"]
    masks = [np.unpackbits(p).astype(bool).reshape(R.H, R.W) for p in packed]
    meta = json.loads((d / "instances.json").read_text())
    z = np.load(d / "stage.npz")
    return masks, meta, z


def repaint(cl_masks, decisions, order):
    """decide_structure's final paint, with the priority swapped.

    The recipe paints accepted objects by descending prompt score into whatever is still
    IGNORE, so the first to claim a pixel keeps it. A wall that covers the counters
    standing in front of it scores 0.97 and paints first, which is why splitting those
    counters into their own objects moves the object count and not one pixel of the map.
    `area` paints the smallest object first: between two objects that overlap on screen,
    the smaller one is the one nearer the camera.
    """
    class_id = {"wall": 2, "column": 3, "table": 4, "shelf": 5}
    keep = [d for d in decisions if d["win"] and not d["reject"]]
    keep.sort(key=(lambda d: d["px"]) if order == "area" else (lambda d: -d["score"]))
    static = np.full((R.H, R.W), R.IGNORE, np.uint8)
    for d in keep:
        sel = cl_masks[d["k"]] & (static == R.IGNORE)
        static[sel] = class_id[d["win"]]
    return static


def run(camera, rule, ratio, diag_root, render, paint="score"):
    masks, meta, z = load(camera, diag_root)
    geo = Geo(z)
    cl_masks, cl_votes, log = cluster_rule(masks, meta, rule, ratio)
    b03 = [z["b03"][i] for i in range(z["b03"].shape[0])]
    static, decisions = R.decide_structure(cl_masks, cl_votes, b03, geo, geo.lx, geo.lz)
    if paint == "area":
        static = repaint(cl_masks, decisions, "area")

    acc = [d for d in decisions if d["win"] and not d["reject"]]
    by = {}
    for d in acc:
        by[d["win"]] = by.get(d["win"], 0) + 1
    parts = sum(1 for d in decisions if d["reject"] and "a part" in d["reject"])
    contain_merges = sum(1 for e in log if e["by"] == "contain")

    # The accepted-object count is not what the rest of the pipeline reads.
    # `scene_mesh` opens the class mask PNGs and takes connected components, so two
    # rules that accept 27 and 51 objects over the same pixels produce the same scene.
    # The comparable quantity is the painted map.
    share = {n: round(100 * float((static == c).mean()), 1) for c, n in NAMES.items()}
    fixture = (static == 4) | (static == 5)
    _lab, n_fix = ndimage.label(fixture, structure=np.ones((3, 3)))
    print(
        f"  [{camera}] {len(masks)} instances -> {len(cl_masks)} objects "
        f"({contain_merges} of {len(log)} merges by containment), "
        f"{len(acc)} accepted {by}, {parts} rejected as a part | "
        f"map {share}, {n_fix} fixture components"
    )

    if render:
        slot = str(z["cleanest"])
        plate = ROOT / f"datasets/studioa_static/{camera}/plate_{slot}.png"
        base = np.asarray(
            Image.open(plate).convert("RGB").resize((R.W, R.H), Image.Resampling.LANCZOS),
            dtype=np.float64,
        )
        over = base.copy()
        for cid, colour in RGB.items():
            sel = static == cid
            over[sel] = 0.55 * base[sel] + 0.45 * np.array(colour)
        sel = (z["floor"] > 0.5) & (static == R.IGNORE)
        over[sel] = 0.55 * base[sel] + 0.45 * np.array((60, 200, 90))
        img = Image.fromarray(np.concatenate([base, over], axis=1).astype(np.uint8)).resize(
            (1920, 540), Image.Resampling.LANCZOS
        )
        dr = ImageDraw.Draw(img)
        dr.rectangle([0, 540 - 16, 1920, 540], fill=(0, 0, 0))
        dr.text(
            (6, 540 - 13),
            f"{camera}  rule={rule} paint={paint}"
            + (f" ratio={ratio}" if rule == "sized" else "")
            + f"  RED=wall yellow=column purple=table blue=shelf green=floor  "
            f"objects {len(cl_masks)} accepted {by}",
            fill=(255, 255, 255),
        )
        img.save(ROOT / f"assets/cluster_{rule}_{camera}.png")
    return decisions, log


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("cameras", nargs="*", default=list(CAMS))
    ap.add_argument("--rule", default="current", choices=("current", "iou", "sized"))
    ap.add_argument("--ratio", type=float, default=0.5)
    ap.add_argument("--diag-root", type=Path, default=ROOT / "runs/masks_diag01")
    ap.add_argument("--render", action="store_true")
    ap.add_argument(
        "--paint",
        default="score",
        choices=("score", "area"),
        help="paint priority: the recipe's descending prompt score, or smallest-first",
    )
    a = ap.parse_args()
    for camera in a.cameras or CAMS:
        run(camera, a.rule, a.ratio, a.diag_root, a.render, a.paint)


if __name__ == "__main__":
    main()
