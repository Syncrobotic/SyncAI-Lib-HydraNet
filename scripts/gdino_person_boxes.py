#!/usr/bin/env python3
"""Grounding DINO `person` boxes: comparison on SAM 3's frames, or a training batch from clips.

    # frame-identical comparison against an existing SAM 3 batch
    python3 scripts/gdino_person_boxes.py --out runs/gdino_person_batch01 \\
        --images datasets/retail_person_batch01/images

    # training batch straight from clips (saves the sampled frames)
    python3 scripts/gdino_person_boxes.py --out datasets/retail_person_gdino01 \\
        --frames 8 --train-thr 0.35 --clips datasets/studioa_clips/*/archive_*.mp4

Why this exists: SAM 3's `person` prompt hallucinates hanging accessory packets as
people on IR night frames (14 on one empty Taichung-cam09 frame, scores 0.55-0.60,
`sam3_person_boxes.py` docstring), and the consensus vote keeps them because they
recur. Measured here on the exact frames SAM 3 labelled (2026-08-19): GDINO's day
person scores sit at/above 0.35 (Taichung-cam01 median 0.61; Kaohsiung-cam04 94 of
SAM 3's 103 at >=0.35) while the empty night IR clip tops out at **0.326** -- zero
boxes at 0.35 against SAM 3's 229 at its own 0.50 default. The 34 night boxes in
[0.25, 0.35) score a median 0.890 static share against the plates: the packets.
So 0.35 sits in a measured gap (ARCHITECTURE_DIRECTION.md rule 2) and no daylight
gate is needed -- the night population never reaches the threshold.

Two deliberate differences from the SAM 3 script:

1. **No daylight gate.** Night frames pass through; per-frame luma/chroma are
   recorded so day/night populations can be split afterwards by the same pixel test.
2. **Scores are kept down to a floor of 0.10** in `instances_all.json`, so the two
   populations stay visible in full. `--train-thr` additionally writes
   `instances_train.json` filtered at the working threshold with per-frame greedy
   NMS (IoU 0.55, same as `sam3_prelabel._dedupe`) for training consumption.

What survives is still a teacher's opinion and not ground truth.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image

HERE = Path(__file__).resolve().parent
for candidate in (HERE.parent / "src", HERE / "src"):
    if candidate.is_dir():
        sys.path.insert(0, str(candidate))

THRESHOLD_LADDER = (0.25, 0.35, 0.50)  # reported per camera; nothing is filtered by them


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--images", nargs="+", help="frame dirs from sam3_person_boxes")
    src.add_argument("--clips", nargs="+", help="mp4 clips; sampled frames are saved to --out")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model-id", default="IDEA-Research/grounding-dino-base")
    ap.add_argument("--prompt", default="person")
    ap.add_argument("--frames", type=int, default=8, help="frames sampled per clip")
    # Floor, not a working threshold: low enough that both score populations are
    # visible in full. Choosing between the populations happens in the analysis.
    ap.add_argument("--floor", type=float, default=0.10)
    # 0.35 is the measured gap: night IR hallucinations top out at 0.326 on the one
    # empty-store night clip measured so far, day people sit at/above it.
    ap.add_argument(
        "--train-thr",
        type=float,
        default=None,
        help="also write instances_train.json filtered here, with NMS",
    )
    ap.add_argument("--nms-iou", type=float, default=0.55)
    return ap


def load_gdino(model_id: str, device: str):
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

    proc = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device).eval()
    return proc, model


@torch.inference_mode()
def detect(proc, model, img: Image.Image, prompt: str, floor: float, device: str) -> np.ndarray:
    """Boxes as [x1, y1, x2, y2, score] above the floor, xyxy in source pixels."""
    text = prompt.strip().lower().rstrip(".") + "."
    inputs = proc(images=img, text=text, return_tensors="pt").to(device)
    outputs = model(**inputs)
    kwargs = {"threshold": floor, "text_threshold": floor, "target_sizes": [img.size[::-1]]}
    try:
        (res,) = proc.post_process_grounded_object_detection(
            outputs, inputs.input_ids, **kwargs
        )
    except TypeError:  # older signature without positional input_ids
        (res,) = proc.post_process_grounded_object_detection(outputs, **kwargs)
    boxes = res["boxes"].cpu().float().numpy().reshape(-1, 4)
    scores = res["scores"].cpu().float().numpy().reshape(-1, 1)
    return np.concatenate([boxes, scores], axis=1)


def nms(boxes: np.ndarray, iou_thr: float) -> np.ndarray:
    """Greedy per-frame NMS, same shape as sam3_prelabel._dedupe."""
    if len(boxes) < 2:
        return boxes
    from syncai_hydranet.analytics.tracker import iou

    order = np.argsort(-boxes[:, 4])
    keep: list[int] = []
    for i in order:
        if all(iou(boxes[i : i + 1, :4], boxes[j : j + 1, :4])[0, 0] < iou_thr for j in keep):
            keep.append(i)
    return boxes[sorted(keep)]


def luma_chroma(frame: np.ndarray) -> tuple[float, float]:
    """Same pixel test as sam3_person_boxes.is_daylight, reported rather than gated."""
    luma = float(frame.mean())
    chroma = float(np.abs(frame.astype(np.int16) - frame.mean(axis=2, keepdims=True)).mean())
    return luma, chroma


def iter_image_dirs(dirs: list[str]):
    for path in sorted(p for d in dirs for p in Path(d).glob("*.jpg")):
        img = Image.open(path).convert("RGB")
        yield path.name.split("__")[0], path.name, img, None


def iter_clips(clips: list[str], per_clip: int, out_images: Path):
    from syncai_hydranet.data.video import frames, probe

    out_images.mkdir(parents=True, exist_ok=True)
    for clip in clips:
        camera = Path(clip).parent.name
        w, h, _ = probe(clip)
        every = max(int(300 * 5 / max(per_clip, 1)), 1)
        picked = 0
        for i, frame in enumerate(frames(clip, w, h, 5.0)):
            if i % every:
                continue
            name = f"{camera}__{Path(clip).stem}__{i:05d}.jpg"
            img = Image.fromarray(frame)
            yield camera, name, img, out_images / name
            picked += 1
            if picked >= per_clip:
                break


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    proc, model = load_gdino(args.model_id, device)
    out_root = Path(args.out)
    (out_root / "annotations").mkdir(parents=True, exist_ok=True)

    if args.images:
        source = iter_image_dirs(args.images)
    else:
        source = iter_clips(args.clips, args.frames, out_root / "images")

    images, annotations, train_annotations = [], [], []
    per_camera: dict[str, dict] = defaultdict(
        lambda: {"frames": 0, "scores": [], "luma": [], "chroma": [], "train_boxes": 0}
    )
    img_id = ann_id = train_ann_id = 1
    for camera, name, img, save_to in source:
        arr = np.asarray(img)
        boxes = detect(proc, model, img, args.prompt, args.floor, device)
        lu, ch = luma_chroma(arr)
        if save_to is not None:
            img.save(save_to, quality=92)
        cam = per_camera[camera]
        cam["frames"] += 1
        cam["scores"].extend(float(b[4]) for b in boxes)
        cam["luma"].append(lu)
        cam["chroma"].append(ch)
        images.append(
            {
                "id": img_id,
                "file_name": name,
                "width": img.width,
                "height": img.height,
                "luma": round(lu, 2),
                "chroma": round(ch, 2),
            }
        )

        def coco_box(b, bid, img_id=img_id):
            return {
                "id": bid,
                "image_id": img_id,
                "category_id": 1,
                "bbox": [float(b[0]), float(b[1]), float(b[2] - b[0]), float(b[3] - b[1])],
                "area": float((b[2] - b[0]) * (b[3] - b[1])),
                "iscrowd": 0,
                "score": float(b[4]),
            }

        for b in boxes:
            annotations.append(coco_box(b, ann_id))
            ann_id += 1
        if args.train_thr is not None:
            kept = nms(boxes[boxes[:, 4] >= args.train_thr], args.nms_iou)
            for b in kept:
                train_annotations.append(coco_box(b, train_ann_id))
                train_ann_id += 1
            cam["train_boxes"] += len(kept)
        img_id += 1
        print(f"{name:70s} {len(boxes):3d} boxes >= {args.floor}")

    report = []
    for camera, cam in sorted(per_camera.items()):
        scores = np.asarray(cam["scores"], dtype=float)
        row = {
            "camera": camera,
            "frames": cam["frames"],
            "boxes_at_floor": len(scores),
            "luma_median": round(float(np.median(cam["luma"])), 2),
            "chroma_median": round(float(np.median(cam["chroma"])), 2),
            "score_median": round(float(np.median(scores)), 4) if len(scores) else None,
            "score_max": round(float(scores.max()), 4) if len(scores) else None,
        }
        for thr in THRESHOLD_LADDER:
            row[f"boxes_ge_{thr:.2f}"] = int((scores >= thr).sum())
        if args.train_thr is not None:
            row["train_boxes"] = cam["train_boxes"]
        report.append(row)

    info = {
        "source": f"Grounding DINO ({args.model_id}) `{args.prompt}` prompt, "
        f"floor {args.floor}",
        "provenance": "PRE-LABEL, NOT GROUND TRUTH -- no human pass; scores kept for "
        "post-hoc threshold selection; 0.35 sits in the measured day/night gap",
        "inputs": [str(d) for d in (args.images or args.clips)],
    }
    coco = {
        "info": info,
        "categories": [{"id": 1, "name": "person"}],
        "images": images,
        "annotations": annotations,
    }
    (out_root / "annotations" / "instances_all.json").write_text(json.dumps(coco))
    if args.train_thr is not None:
        coco_train = {
            "info": {**info, "train_thr": args.train_thr, "nms_iou": args.nms_iou},
            "categories": [{"id": 1, "name": "person"}],
            "images": images,
            "annotations": train_annotations,
        }
        (out_root / "annotations" / "instances_train.json").write_text(json.dumps(coco_train))
    (out_root / "report.json").write_text(json.dumps(report, indent=2))
    print(
        f"\n{len(annotations)} boxes >= {args.floor} over {len(images)} frames. "
        f"Wrote {out_root}"
    )
    for row in report:
        extra = f"  train:{row['train_boxes']:4d}" if args.train_thr is not None else ""
        print(
            f"{row['camera']:18s} {row['frames']:3d} frames  floor:{row['boxes_at_floor']:4d}  "
            + "  ".join(f">={t:.2f}:{row[f'boxes_ge_{t:.2f}']:4d}" for t in THRESHOLD_LADDER)
            + f"  median:{row['score_median']}{extra}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
