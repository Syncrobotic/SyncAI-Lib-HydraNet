"""ViTPose keypoints over the Gold person boxes -- the pose head's training data.

The distillation chain PLAN 2.2 names: ViTPose (top-down, per-crop) labels the site
frames offline, and the bottom-up P3 head learns to reproduce it whole-frame. Boxes are
the Gold tier only -- score >= 0.50 after NMS, eyeballed 60/60 real on 2026-08-24 --
because a keypoint teacher run on a hallucinated box would distil the hallucination.
The keypoint order is asserted against `analytics/events/pose.KEYPOINT_NAMES` at load,
same as the pilot: a silent reorder would corrupt every downstream event.

Output: `datasets/site30k_v1/annotations/keypoints_{split}.json`, COCO-keypoints shaped
(image ids reused from instances_all), one annotation per Gold box with its flat
17x3 keypoints and the box that prompted them.

Usage:
  uv run python tools/pose/vitpose_teacher.py <split> [--limit N] [--render-first K]
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from syncai_bev3d.teachers.boxes import nms
from syncai_hydranet.analytics.events import pose as ev

ROOT = Path("/home/paul/SyncAI-Lib-HydraNet")
ANN_DIR = ROOT / "datasets/site30k_v1/annotations"
IMG_DIR = ROOT / "datasets/site30k_v1/images"
GOLD_MIN = 0.50
POSE_MODEL = "usyd-community/vitpose-base-simple"


def gold_boxes(split: str) -> tuple[list[dict], dict[int, list]]:
    d = json.loads((ANN_DIR / f"instances_all_{split}.json").read_text())
    per_image: dict[int, list] = {}
    for a in d["annotations"]:
        x, y, w, h = a["bbox"]
        per_image.setdefault(a["image_id"], []).append(
            (x, y, x + w, y + h, a.get("score", 1.0))
        )
    gold: dict[int, list] = {}
    for img_id, boxes in per_image.items():
        arr = nms(np.asarray(boxes, float), 0.5)
        keep = arr[arr[:, 4] >= GOLD_MIN]
        if len(keep):
            gold[img_id] = keep
    return d["images"], gold


def pose_for_frame(processor, model, device, frame: np.ndarray, xyxy: np.ndarray):
    xywh = xyxy[:, :4].copy()
    xywh[:, 2] -= xywh[:, 0]
    xywh[:, 3] -= xywh[:, 1]
    inputs = processor(frame, boxes=[xywh], return_tensors="pt").to(device)
    with torch.no_grad():
        out = model(**inputs)
    results = processor.post_process_pose_estimation(out, boxes=[xywh])[0]
    kps = np.zeros((len(xywh), 17, 3), dtype=np.float32)
    for i, person in enumerate(results):
        kps[i, :, :2] = person["keypoints"].cpu().numpy()
        kps[i, :, 2] = person["scores"].cpu().numpy()
    return kps


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("split", choices=("train", "val", "test"))
    ap.add_argument("--limit", type=int, default=0, help="images cap, 0 = all")
    ap.add_argument("--render-first", type=int, default=0)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    from transformers import AutoProcessor, VitPoseForPoseEstimation

    processor = AutoProcessor.from_pretrained(POSE_MODEL)
    model = VitPoseForPoseEstimation.from_pretrained(POSE_MODEL).to(device).eval()
    id2label = model.config.id2label
    got = tuple(
        id2label[i].lower().replace("l_", "left_").replace("r_", "right_") for i in range(17)
    )
    assert got == ev.KEYPOINT_NAMES, f"keypoint order mismatch: {got}"

    images, gold = gold_boxes(args.split)
    todo = [im for im in images if im["id"] in gold]
    if args.limit:
        todo = todo[: args.limit]
    print(f"{args.split}: {len(todo)} images with Gold boxes "
          f"({sum(len(gold[im['id']]) for im in todo)} boxes)")  # fmt: skip

    annotations, out_images = [], []
    for n, im in enumerate(todo):
        frame = np.asarray(Image.open(IMG_DIR / im["file_name"]).convert("RGB"))
        boxes = gold[im["id"]]
        kps = pose_for_frame(processor, model, device, frame, boxes)
        out_images.append(im)
        for box, kp in zip(boxes, kps, strict=True):
            annotations.append(
                {
                    "image_id": im["id"],
                    "bbox": [
                        float(box[0]),
                        float(box[1]),
                        float(box[2] - box[0]),
                        float(box[3] - box[1]),
                    ],
                    "det_score": float(box[4]),
                    "keypoints": [round(float(v), 2) for v in kp.reshape(-1)],
                    "num_keypoints": int((kp[:, 2] >= 0.3).sum()),
                }
            )
        if args.render_first and n < args.render_first:
            from PIL import ImageDraw

            img = Image.fromarray(frame)
            d = ImageDraw.Draw(img)
            for kp in kps:
                for x, y, c in kp:
                    if c >= 0.3:
                        d.ellipse([x - 3, y - 3, x + 3, y + 3], fill=(255, 60, 40))
            img.save(ROOT / f"assets/pose_teacher_sample_{args.split}_{n}.png")
        if n % 200 == 0:
            print(f"  {n}/{len(todo)}", flush=True)

    git = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=ROOT
    ).stdout.strip()
    payload = {
        "provenance": {
            "tool": "tools/pose/vitpose_teacher.py",
            "git_commit": git,
            "pose_model": POSE_MODEL,
            "boxes": f"instances_all_{args.split}.json, NMS 0.5, score >= {GOLD_MIN} (Gold)",
            "keypoint_order": list(ev.KEYPOINT_NAMES),
        },
        "images": out_images,
        "annotations": annotations,
    }
    suffix = f"_limit{args.limit}" if args.limit else ""
    out = ANN_DIR / f"keypoints_{args.split}{suffix}.json"
    out.write_text(json.dumps(payload))
    print(f"wrote {out} ({len(annotations)} annotations over {len(out_images)} images)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
