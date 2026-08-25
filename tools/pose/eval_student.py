"""Student vs teacher: keypoint error of the P3 head against ViTPose, on the test split.

Gate 3's instrument. The comparison is per-joint L2 in ORIGINAL image pixels (mapped
back through the sample's geom) plus PCK@0.2·box_height, both computed only where the
teacher was confident (conf >= 0.3) — a joint the teacher could not see is not a truth
to be measured against. Boxes are the teacher's own Gold boxes, so decoding is judged
with grouping held fixed; detection quality is scored elsewhere and mixing the two
would blur which head failed.

The caveat PLAN carries stays true here: this is agreement with the teacher, not
accuracy. The number that gates is comparative — is the student close enough to the
teacher for `reach_to_shelf` and `crouch` to fire the same way.

Usage:
  uv run python tools/pose/eval_student.py --config configs/hydranet_retail_pose01.yaml \\
      --checkpoint runs/hydranet_retail_pose01/best.pt [--limit N] [--render K]
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from syncai_hydranet.analytics.events.pose import KEYPOINT_NAMES
from syncai_hydranet.config import load_config
from syncai_hydranet.data.transforms import GEOM_IDENTITY, LetterboxResize, Sample, ToTensor
from syncai_hydranet.models.heads.pose import decode_boxes
from syncai_hydranet.models.hydranet import build_model
from syncai_hydranet.utils.checkpoint import load_checkpoint, select_weights
from syncai_hydranet.utils.device import pick_device

ROOT = Path("/home/paul/SyncAI-Lib-HydraNet")
CONF_MIN = 0.3


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--weights", default="ema")
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--render", type=int, default=0)
    args = ap.parse_args()

    device = pick_device(None)
    cfg = load_config(args.config)
    model = build_model(cfg).to(device).eval()
    model.load_state_dict(select_weights(load_checkpoint(args.checkpoint), args.weights))
    input_size = cfg["data"]["input_size"]

    data = json.loads(
        (ROOT / f"datasets/site30k_v1/annotations/keypoints_{args.split}.json").read_text()
    )
    by_image: dict[int, list] = {}
    for a in data["annotations"]:
        by_image.setdefault(a["image_id"], []).append(a)
    images = [im for im in data["images"] if im["id"] in by_image]
    if args.limit:
        images = images[: args.limit]

    letterbox = LetterboxResize(input_size)
    to_tensor = ToTensor()
    errs: list[np.ndarray] = []  # per-joint L2 in original px, NaN where unjudged
    pck_hits, pck_total = 0, 0
    per_joint_err: dict[int, list] = {k: [] for k in range(17)}
    rendered = 0

    for n, im in enumerate(images):
        anns = by_image[im["id"]]
        img = Image.open(ROOT / "datasets/site30k_v1/images" / im["file_name"]).convert("RGB")
        boxes = np.array(
            [
                [
                    a["bbox"][0],
                    a["bbox"][1],
                    a["bbox"][0] + a["bbox"][2],
                    a["bbox"][1] + a["bbox"][3],
                ]
                for a in anns
            ],
            dtype=np.float32,
        )
        teacher = np.stack(
            [np.asarray(a["keypoints"], np.float32).reshape(17, 3) for a in anns]
        )
        s = Sample(image=img, boxes=boxes.copy(), labels=np.zeros(len(boxes), np.int64))
        s = to_tensor(letterbox(s))
        sx, sy, px, py = s.get("geom", GEOM_IDENTITY)
        with torch.no_grad():
            out = model.forward(s["image"][None].to(device))
        student = (
            decode_boxes(out["pose"][0], torch.from_numpy(s["boxes"]).to(device)).cpu().numpy()
        )
        # student back to original pixels; teacher already lives there
        student[:, :, 0] = (student[:, :, 0] - px) / sx
        student[:, :, 1] = (student[:, :, 1] - py) / sy
        for t_kp, s_kp, box in zip(teacher, student, boxes, strict=True):
            judge = t_kp[:, 2] >= CONF_MIN
            d = np.linalg.norm(s_kp[:, :2] - t_kp[:, :2], axis=1)
            d = np.where(judge, d, np.nan)
            errs.append(d)
            box_h = max(box[3] - box[1], 1.0)
            pck_hits += int(np.nansum(d <= 0.2 * box_h))
            pck_total += int(judge.sum())
            for k in range(17):
                if judge[k]:
                    per_joint_err[k].append(float(d[k]))
        if args.render and rendered < args.render:
            d = ImageDraw.Draw(img)
            for t_kp, s_kp in zip(teacher, student, strict=True):
                for k in range(17):
                    if t_kp[k, 2] >= CONF_MIN:
                        x, y = t_kp[k, :2]
                        d.ellipse([x - 4, y - 4, x + 4, y + 4], outline=(60, 220, 90), width=2)
                    x, y = s_kp[k, :2]
                    d.ellipse([x - 2, y - 2, x + 2, y + 2], fill=(255, 60, 40))
            img.save(ROOT / f"assets/pose_eval_{args.split}_{n}.png")
            rendered += 1
        if n % 200 == 0:
            print(f"  {n}/{len(images)}", flush=True)

    all_err = np.concatenate(errs)
    valid = all_err[~np.isnan(all_err)]
    print(
        f"\n{args.split}: {len(images)} images, {len(errs)} persons, {len(valid)} judged joints"
    )
    print(
        f"L2 error px  mean {valid.mean():.1f}  p50 {np.percentile(valid, 50):.1f}  "
        f"p90 {np.percentile(valid, 90):.1f}  p99 {np.percentile(valid, 99):.1f}"
    )
    print(f"PCK@0.2h: {pck_hits / max(pck_total, 1):.3f}")
    print("\nper joint (p50 px):")
    for k in range(17):
        v = per_joint_err[k]
        print(f"  {KEYPOINT_NAMES[k]:<16} {np.median(v):6.1f}  (n={len(v)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
