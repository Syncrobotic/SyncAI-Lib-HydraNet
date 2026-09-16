"""Decompose fixed detector gates on frozen source-train positives only.

Per-object coverage is not one-to-one recall. Counterfactual centerness=1 is
diagnostic only: it never changes the decoder, threshold, or model selection.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import torch
from torchvision.ops import box_iou

from syncai_hydranet.data.studioa_instances import CLASSES, StudioAInstanceDataset
from syncai_hydranet.data.studioa_review import digest, write_json
from syncai_hydranet.models.heads import detection
from syncai_hydranet.models.heads.detection import flatten_levels
from syncai_hydranet.models.hydranet import HydraNet
from syncai_hydranet.utils.checkpoint import load_checkpoint

SCORE = 0.20
IOU = 0.50


def decompose(overlaps, probabilities, centerness, label, retained):
    """Use joint masks: every passed gate must belong to the same grid point."""
    scores = probabilities[:, label] * centerness
    localized = overlaps >= IOU
    class_pass = localized & (probabilities[:, label] > SCORE)
    combined_pass = localized & (scores > SCORE)
    winner = probabilities.argmax(1) == label
    eligible = combined_pass & winner
    oracle_ctr = class_pass & winner
    if retained:
        assert eligible.any(), "decoded coverage requires an eligible raw candidate"
        reason = "covered_after_decode"
    elif eligible.any():
        reason = "nms_or_cap"
    elif oracle_ctr.any():
        reason = "centerness_gate"
    elif class_pass.any():
        reason = "class_competition"
    elif localized.any():
        reason = "class_probability"
    else:
        reason = "localization"
    best = None
    if localized.any():
        idx = int(scores.masked_fill(~localized, -1).argmax())
        best = {
            "point_index": idx,
            "iou": float(overlaps[idx]),
            "class_probability": float(probabilities[idx, label]),
            "centerness": float(centerness[idx]),
            "product": float(scores[idx]),
            "winning_class": int(probabilities[idx].argmax()),
        }
    return {
        "reason": reason,
        "max_raw_iou": float(overlaps.max()),
        "localized_points": int(localized.sum()),
        "class_pass_points": int(class_pass.sum()),
        "combined_pass_points": int(combined_pass.sum()),
        "eligible_points": int(eligible.sum()),
        "oracle_centerness_eligible_points": int(oracle_ctr.sum()),
        "best_localized_product": best,
        "covered_after_decode": bool(retained),
    }


def xyxy(points, distances, height, width):
    boxes = torch.stack(
        [
            points[:, 0] - distances[:, 0],
            points[:, 1] - distances[:, 1],
            points[:, 0] + distances[:, 2],
            points[:, 1] + distances[:, 3],
        ],
        1,
    )
    boxes[:, 0::2].clamp_(0, width)
    boxes[:, 1::2].clamp_(0, height)
    return boxes


def inspect_run(run: Path, device: str) -> dict:
    job = json.loads((run / "job.json").read_text())
    report = json.loads((run / "report.json").read_text())
    assert report["status"] == "completed"
    assert report["job_sha256"] == digest(run / "job.json")
    for rel, expected in job["files"].items():
        assert digest(run / rel) == expected, rel
    for rel, expected in report["outputs"].items():
        assert digest(run / rel) == expected, rel
    # Pin the assignment and decoder implementation used by this diagnostic.
    assert detection.__file__ is not None
    assert digest(Path(detection.__file__)) == digest(
        run / "snapshot/src/syncai_hydranet/models/heads/detection.py"
    )
    cfg = json.loads((run / "config.json").read_text())
    ds = StudioAInstanceDataset(
        run / "instances", job["held_out"], "train", cfg["data"]["input_size"], train=False
    )
    manifest = json.loads((run / "instances/manifest.json").read_text())
    assignments = manifest["folds"][job["held_out"]]["assignments"]
    assert all(assignments[f["id"]] == "train" for f in ds.frames)
    model = HydraNet(cfg).to(device).eval()
    checkpoints = {}
    for ck in ["best.pt", "last.pt"]:
        path = run / "model" / ck
        cp = load_checkpoint(path)
        assert cp["studioa_job_sha256"] == digest(run / "job.json")
        model.load_state_dict(cp["model"])
        rows = []
        with torch.inference_mode():
            for i, frame in enumerate(ds.frames):
                sample = ds[i]
                target = {k: v.to(device) for k, v in sample["targets"].items()}
                image = sample["image"][None].to(device)
                pred = model(image)
                cls, reg, ctr = flatten_levels(
                    pred["det_cls"], pred["det_reg"], pred["det_ctr"], len(CLASSES)
                )
                head = model.det_head
                assert head is not None
                points, labels, reg_target, ctr_target = head.get_targets(
                    [x.shape[-2:] for x in pred["det_cls"]],
                    [target["boxes"]],
                    [target["labels"]],
                    device,
                )
                height, width = image.shape[-2:]
                raw = xyxy(points, reg[0], height, width)
                assigned_boxes = xyxy(points, reg_target[0], height, width)
                decoded = head.decode(
                    pred["det_cls"],
                    pred["det_reg"],
                    pred["det_ctr"],
                    score_thr=SCORE,
                    nms_thr=0.6,
                    max_det=100,
                    img_size=(height, width),
                )[0]
                for j, (gt, label) in enumerate(
                    zip(target["boxes"], target["labels"], strict=True)
                ):
                    c = int(label)
                    retained = (
                        (decoded["labels"] == c)
                        & (box_iou(gt[None], decoded["boxes"])[0] >= IOU)
                    ).any()
                    row = decompose(
                        box_iou(gt[None], raw)[0],
                        cls[0].sigmoid(),
                        ctr[0].sigmoid(),
                        c,
                        bool(retained),
                    )
                    assigned = (labels[0] == c) & ((assigned_boxes - gt).abs().amax(1) < 1e-4)
                    row.update(
                        {
                            "frame": frame["id"],
                            "camera": frame["camera"],
                            "object_index": j,
                            "class": CLASSES[c],
                            "box": gt.tolist(),
                            "width_px": float(gt[2] - gt[0]),
                            "height_px": float(gt[3] - gt[1]),
                            "assigned_points": int(assigned.sum()),
                            "max_target_centerness": float(ctr_target[0, assigned].max())
                            if assigned.any()
                            else None,
                            "best_iou_box": raw[box_iou(gt[None], raw)[0].argmax()].tolist(),
                        }
                    )
                    rows.append(row)
        summary = {}
        for name in CLASSES:
            selected = [r for r in rows if r["class"] == name]
            summary[name] = {
                "objects": len(selected),
                "cameras": len({r["camera"] for r in selected}),
                "reasons": dict(Counter(r["reason"] for r in selected)),
                "no_assigned_points": sum(r["assigned_points"] == 0 for r in selected),
                "oracle_centerness_coverage": sum(
                    r["oracle_centerness_eligible_points"] > 0 for r in selected
                ),
                "raw_localization_coverage": sum(r["localized_points"] > 0 for r in selected),
            }
        checkpoints[ck] = {
            "sha256": digest(path),
            "epoch": cp["epoch"],
            "global_step": cp["global_step"],
            "summary": summary,
            "objects": rows,
        }
    return {
        "job_sha256": digest(run / "job.json"),
        "frames": len(ds),
        "manifest_sha256": digest(run / "instances/manifest.json"),
        "inputs_verified": len(job["files"]),
        "outputs_verified": len(report["outputs"]),
        "checkpoints": checkpoints,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, action="append", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    write_json(args.out / "status.json", {"status": "running"})
    try:
        results = {str(run): inspect_run(run, args.device) for run in args.run}
        write_json(
            args.out / "diagnosis.json",
            {
                "scope": (
                    "source-train only; full-frame float32 no augmentation; "
                    "per-object coverage, not one-to-one recall or independent accuracy"
                ),
                "protocol": {
                    "score_gt": SCORE,
                    "iou_ge": IOU,
                    "nms": 0.6,
                    "max_det": 100,
                    "test_evaluated": False,
                    "val_evaluated": False,
                    "centerness_counterfactual": (
                        "set to 1 only to classify bottlenecks; not applied to decoder"
                    ),
                },
                "script_sha256": digest(Path(__file__)),
                "runs": results,
            },
        )
        write_json(args.out / "status.json", {"status": "completed"})
    except Exception:
        import traceback

        write_json(
            args.out / "status.json", {"status": "failed", "error": traceback.format_exc()}
        )
        raise


if __name__ == "__main__":
    main()
