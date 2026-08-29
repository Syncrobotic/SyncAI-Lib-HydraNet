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
      --checkpoint runs/hydranet_retail_security_b03_cw_xl-20260825-162131/best.pt \\
      [--limit N] [--render K]

The run directory is named for a *different* model because the pose config inherited
`output_dir` from the security run; the trainer appended a timestamp rather than
overwrite it. `config.yaml` inside the directory is what says which run it is.

`--render K` writes K per-person crops, not K full frames: at 1920x1080 a shopper is
a few percent of the pixels, and a gate that is decided by eye needs to be lookable-at.
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
CROP_PAD = 0.25  # of box size, so the limbs that leave the box stay visible
SKELETON = (
    (5, 7), (7, 9), (6, 8), (8, 10),      # arms
    (11, 13), (13, 15), (12, 14), (14, 16),  # legs
    (5, 6), (5, 11), (6, 12), (11, 12),   # torso
    (0, 1), (0, 2), (1, 3), (2, 4),       # face
)  # fmt: skip


def render_pair(img: Image.Image, t_kp, s_kp, box) -> Image.Image:
    """One person, cropped and upscaled: teacher skeleton in green, student in red.

    Both skeletons are drawn on the same crop rather than side by side -- the question
    the gate asks is where the student *disagrees*, and disagreement is a gap between
    two lines in one picture, not a difference between two pictures.
    """
    w, h = box[2] - box[0], box[3] - box[1]
    pad_x, pad_y = w * CROP_PAD, h * CROP_PAD
    x0 = max(0, int(box[0] - pad_x))
    y0 = max(0, int(box[1] - pad_y))
    x1 = min(img.width, int(box[2] + pad_x))
    y1 = min(img.height, int(box[3] + pad_y))
    crop = img.crop((x0, y0, x1, y1))
    scale = max(1.0, 420.0 / max(crop.height, 1))
    crop = crop.resize((max(1, int(crop.width * scale)), max(1, int(crop.height * scale))))
    d = ImageDraw.Draw(crop)

    def at(kp, k):
        return ((kp[k, 0] - x0) * scale, (kp[k, 1] - y0) * scale)

    for a, b in SKELETON:
        if t_kp[a, 2] >= CONF_MIN and t_kp[b, 2] >= CONF_MIN:
            d.line([at(t_kp, a), at(t_kp, b)], fill=(60, 220, 90), width=3)
            d.line([at(s_kp, a), at(s_kp, b)], fill=(255, 60, 40), width=2)
    for k in range(17):
        if t_kp[k, 2] >= CONF_MIN:
            x, y = at(t_kp, k)
            d.ellipse([x - 5, y - 5, x + 5, y + 5], outline=(60, 220, 90), width=2)
            sx_, sy_ = at(s_kp, k)
            d.ellipse([sx_ - 3, sy_ - 3, sx_ + 3, sy_ + 3], fill=(255, 60, 40))
            d.line([(x, y), (sx_, sy_)], fill=(255, 200, 60), width=1)  # the error itself
    d.text(
        (6, 6), "green = ViTPose teacher   red = student   amber = error", fill=(255, 255, 255)
    )
    return crop


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--weights", default="ema")
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--render", type=int, default=0)
    ap.add_argument(
        "--input-size",
        nargs=2,
        type=int,
        metavar=("H", "W"),
        help="evaluate at a canvas other than the config's. The throughput gate's only "
        "untested lever is a smaller input (576x1008 measures 1,622 f/s against the "
        "config's 1,325), and its accuracy cost has been carried as 'unmeasured' since "
        "the target was set. This is how it stops being unmeasured.",
    )
    args = ap.parse_args()

    device = pick_device(None)
    cfg = load_config(args.config)
    model = build_model(cfg).to(device).eval()
    model.load_state_dict(select_weights(load_checkpoint(args.checkpoint), args.weights))
    input_size = list(args.input_size) if args.input_size else cfg["data"]["input_size"]
    print(f"evaluating at {input_size[0]}x{input_size[1]}")

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
            # ToTensor has already made these tensors; from_numpy would reject them
            decode_boxes(out["pose"][0], torch.as_tensor(s["boxes"]).to(device)).cpu().numpy()
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
        for t_kp, s_kp, box in zip(teacher, student, boxes, strict=True):
            if rendered >= args.render:
                break
            crop = render_pair(img, t_kp, s_kp, box)
            crop.save(ROOT / f"assets/pose_eval_{args.split}_{rendered:02d}.png")
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
