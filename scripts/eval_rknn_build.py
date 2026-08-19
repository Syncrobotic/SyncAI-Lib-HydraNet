#!/usr/bin/env python3
"""Score an RKNN build against the fp32 graph it came from, on the same images.

    ~/.venv-rknn/bin/python scripts/eval_rknn_build.py \\
        --onnx v1_384.onnx --dataset dataset.txt --samples valdump --out build_metrics.json

`docs/RELEASE.md` says a build's numbers are not the model's numbers, and this is what
makes that rule checkable. `releases/v1/metrics.json` records fp32 at 512x640; the Lite3
runs int8 at 384x512, two lossy steps away, and `deploy/robot/README.md` has asserted
"INT8 keeps accuracy here" with nothing behind it -- `scripts/bench_*` measure throughput
only. So the accuracy of the model actually on the robot had never been measured.

---------------------------------------------------------------------------
WHY IT REBUILDS RATHER THAN LOADING THE .rknn

`init_runtime` refuses a model that arrived via `load_rknn`: the simulator needs the graph
that `build` produced in the same session. So this converts again from ONNX with the same
recipe as `deploy/robot/convert_rknn_int8.py` rather than scoring the exported file. Same
inputs and same quantisation settings produce the same weights, but it is worth being
plain that this measures **a build made by the documented recipe**, not the exact bytes
sitting on the robot -- whose calibration frames are not archived anywhere.

Which is itself the finding: a build nobody can reproduce cannot be scored either, and
that is an argument for archiving the calibration list beside the build.

---------------------------------------------------------------------------
WHAT IT MEASURES, AND WHAT IT LEAVES OUT

Segmentation only -- traversability and terrain mIoU. Those are what the robot displays
(`HYDRA_VIEW=trav`) and they need nothing but the label maps already on disk. Detection
mAP wants pycocotools over the full COCO val set through two runtimes, which is a bigger
job than the question deserves before anyone has seen a first number.

The fp32 reference runs through onnxruntime on the *same* 384x512 graph, not through
PyTorch at 512x640. Comparing int8-384 against fp32-512 would fold the resolution change
into the quantisation number and blame one for the other.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

HEADS = {"traversability": 3, "terrain": 12}
IGNORE = 255


def confusion(pred: np.ndarray, target: np.ndarray, n: int) -> np.ndarray:
    ok = target != IGNORE
    idx = target[ok].astype(np.int64) * n + pred[ok].astype(np.int64)
    return np.bincount(idx, minlength=n * n).reshape(n, n)


def miou(cm: np.ndarray) -> tuple[float, int]:
    """mIoU over classes that appear, plus how many that was -- a mean over a shrinking
    set of classes is not comparable to one over a larger set, which is why the evaluator
    in `engine/` reports the count beside the score and this does too."""
    inter = np.diag(cm).astype(float)
    union = cm.sum(1) + cm.sum(0) - np.diag(cm)
    with np.errstate(invalid="ignore"):
        iou = np.where(union > 0, inter / np.maximum(union, 1), np.nan)
    present = np.isfinite(iou)
    return (float(np.nanmean(iou)) if present.any() else float("nan"), int(present.sum()))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--dataset", required=True, help="calibration list for quantisation")
    ap.add_argument("--samples", type=Path, required=True, help="dir of *_img.npy + labels")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    import onnxruntime as ort
    from rknn.api import RKNN

    imgs = sorted(args.samples.glob("*_img.npy"))
    if args.limit:
        imgs = imgs[: args.limit]
    if not imgs:
        raise SystemExit(f"no *_img.npy under {args.samples}")

    sess = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])
    names = [o.name for o in sess.get_outputs()]
    in_name = sess.get_inputs()[0].name

    rknn = RKNN(verbose=False)
    rknn.config(
        target_platform="rk3588",
        mean_values=[[0, 0, 0]],
        std_values=[[1, 1, 1]],
        quantized_dtype="asymmetric_quantized-8",
        quantized_algorithm="normal",
    )
    assert rknn.load_onnx(model=args.onnx) == 0
    print("building int8 with calibration...", flush=True)
    assert rknn.build(do_quantization=True, dataset=args.dataset) == 0
    assert rknn.init_runtime(target=None) == 0

    cms = {
        r: {h: np.zeros((n, n), dtype=np.int64) for h, n in HEADS.items()}
        for r in ("fp32", "int8")
    }
    agree = {h: [0, 0] for h in HEADS}
    for i, p in enumerate(imgs):
        img = np.load(p)
        fp = sess.run(None, {in_name: img.transpose(2, 0, 1)[None].astype(np.float32)})
        q = rknn.inference(inputs=[img])
        for head, n in HEADS.items():
            lab = p.with_name(p.name.replace("_img.npy", f"_{head}.npy"))
            if not lab.exists():
                continue
            target = np.load(lab)
            j = names.index(head)
            pf = np.asarray(fp[j]).squeeze(0).argmax(0)
            pq = np.asarray(q[j]).squeeze(0).argmax(0)
            cms["fp32"][head] += confusion(pf, target, n)
            cms["int8"][head] += confusion(pq, target, n)
            # Pixels where quantisation changed the answer, regardless of which is right.
            agree[head][0] += int((pf == pq).sum())
            agree[head][1] += int(pf.size)
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(imgs)}", flush=True)
    rknn.release()

    result: dict = {"onnx": args.onnx, "images": len(imgs), "heads": {}}
    for head in HEADS:
        f_iou, f_n = miou(cms["fp32"][head])
        q_iou, q_n = miou(cms["int8"][head])
        same, total = agree[head]
        result["heads"][head] = {
            "fp32_mIoU": round(f_iou, 4),
            "int8_mIoU": round(q_iou, 4),
            "delta": round(q_iou - f_iou, 4),
            "classes_present": {"fp32": f_n, "int8": q_n},
            "pixel_agreement": round(same / max(1, total), 4),
        }
        print(
            f"{head:16s} fp32 {f_iou:.4f} -> int8 {q_iou:.4f}  "
            f"({q_iou - f_iou:+.4f})  pixels unchanged {same / max(1, total):.3%}"
        )
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
