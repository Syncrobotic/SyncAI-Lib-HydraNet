#!/usr/bin/env python3
"""IDF1 for the shipped tracker and the two-stage one, against a labelled clip.

    python3 scripts/track_idf1.py --gt runs/gt_cam01

PLAN §7.18 measured what it could without labels -- of the 67 two-stage joins where both
sides are well observed, none joins two people the appearance can tell apart -- and said
plainly what that does not settle: "a colour proves a join wrong and cannot certify one
right ... IDF1/HOTA against hand-labelled track IDs is the measurement decision 1 should
be settled on". `analytics/reid_metrics.idf1` has been armed since it was written and had
never run on a site clip, because no site clip was labelled. One is now.

**Both trackers see the same detections, from one inference pass**, and that pass is the
one `track_review.py propose` runs -- same config, checkpoint, weights, fps, frame count
and geometry inversion, all read back out of the ground truth's own `provenance.json`
rather than passed again here. The boxes therefore live in the same pixel convention as
the labels, which is the whole reason this does not reuse `track_identity.py`'s pass:
that one undistorts, and IoU between an undistorted prediction and a distorted label is a
quiet loss that would land on whichever tracker happened to sit nearer the lens.

The single-stage arm reproduces `tracks.json` exactly by construction, so the run prints
whether it did. If it does not, the comparison is against a different detection set than
the labels were read from and nothing below means anything.

**What this measures and what it does not.** The labels cover the detector's boxes only,
so a shopper the detector never saw is absent from both the truth and the prediction:
this is a measurement of ASSOCIATION, not of detection. And `runs/gt_cam01`'s labels were
read by a model, not by a person -- `provenance.json` says so in its own words, and any
number out of this script inherits that sentence.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from track_review import PERSON, check_person_class_space, head_class_names

from syncai_hydranet.analytics import Tracker
from syncai_hydranet.analytics.bytetrack import OfflineForward
from syncai_hydranet.analytics.reid_metrics import id_switches, idf1
from syncai_hydranet.config import load_config
from syncai_hydranet.data.transforms import invert_geom
from syncai_hydranet.data.video import frames, probe
from syncai_hydranet.models.hydranet import build_model
from syncai_hydranet.utils.checkpoint import load_checkpoint, select_weights
from syncai_hydranet.utils.device import pick_device
from syncai_hydranet.utils.visualize import preprocess


def as_metric_input(d: dict) -> dict[int, dict[int, np.ndarray]]:
    """`{id: {frame: box}}` with the ids and frames as ints, which the metrics index by.

    Split ids from `track_review` carry a letter (`8b`), and the metric only ever compares
    ids for equality, so they are renumbered here rather than parsed. The mapping is not
    returned because nothing downstream needs it: IDF1 finds its own one-to-one mapping.
    """
    return {
        i: {int(f): np.asarray(b, dtype=float) for f, b in byframe.items()}
        for i, byframe in enumerate(d.values())
    }


def detections(prov: dict, low_thr: float) -> list[dict]:
    """One inference pass, kept down to `low_thr` so both trackers can be fed from it.

    `propose` thresholds inside `model.predict`; this cannot, because the two-stage
    tracker's whole mechanism is the band below the high threshold. The single-stage arm
    is given `scores >= provenance.score_thr` and so sees exactly what `propose` saw.
    """
    cfg = load_config(prov["config"], validate=False)
    device = pick_device(cfg.get("device"))
    model = build_model(cfg).to(device).eval()
    model.load_state_dict(select_weights(load_checkpoint(prov["checkpoint"]), prov["weights"]))
    head = model.det_head
    bias = head.cls_pred.bias if head is not None else None
    if bias is None:
        raise SystemExit(
            f"{prov['config']} builds no detection head, so there is nothing to track. "
            "This script needs the checkpoint the ground truth was proposed from."
        )
    check_person_class_space(int(bias.shape[0]), head_class_names(cfg))
    size = cfg["data"]["input_size"]
    src_w, src_h, _ = probe(prov["clip"])
    per_frame = []
    for n, frame in enumerate(frames(prov["clip"], src_w, src_h, prov["fps"])):
        if prov["max_frames"] and n >= prov["max_frames"]:
            break
        x, _, region = preprocess(Image.fromarray(frame), size)
        with torch.no_grad():
            det = model.predict(x.to(device), score_thr=low_thr)["detection"][0]
        if len(det.get("labels", [])):
            lab = det["labels"].cpu().numpy()
            keep = lab == PERSON
            box = det["boxes"].cpu().numpy()[keep]
            x0, y0, cw, ch = region
            box = invert_geom(box.astype(float), (cw / src_w, ch / src_h, x0, y0))
            sc = det["scores"].cpu().numpy()[keep]
        else:
            box, sc = np.zeros((0, 4)), np.zeros(0)
        per_frame.append({"boxes": box, "scores": sc})
        if (n + 1) % 100 == 0:
            print(f"  frame {n + 1}", flush=True)
    return per_frame


def to_json(tracks) -> dict[str, dict[str, list[float]]]:
    """`{id: {frame: box}}` from either tracker's finished list.

    The two carry their id under different names -- `Track.track_id` and
    `Fragment.frag_id` -- so the id is read by asking rather than by assuming. The names
    differ because the two were written for different jobs, and reconciling them is a
    change to shipped tracking code that a measurement script has no business making.
    """
    out = {}
    for t in tracks:
        if not getattr(t, "confirmed", True):
            continue
        tid = getattr(t, "track_id", None)
        if tid is None:
            tid = t.frag_id
        out[str(tid)] = {
            str(int(f)): [float(v) for v in b] for f, b in zip(t.frames, t.boxes, strict=True)
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gt", type=Path, required=True, help="a track_review output dir")
    ap.add_argument("--low-thr", type=float, default=0.20)
    ap.add_argument("--iou", type=float, default=0.3)
    ap.add_argument("--iou-low", type=float, default=0.5)
    ap.add_argument("--max-age", type=int, default=5)
    ap.add_argument("--min-hits", type=int, default=3)
    a = ap.parse_args()

    prov = json.loads((a.gt / "provenance.json").read_text())
    gt_raw = json.loads((a.gt / "ground_truth.json").read_text())
    proposed = json.loads((a.gt / "tracks.json").read_text())
    high = float(prov["score_thr"])

    per_frame = detections(prov, a.low_thr)

    single = Tracker(iou_threshold=a.iou, max_age=a.max_age, min_hits=a.min_hits)
    two = OfflineForward(
        high_thr=high, low_thr=a.low_thr, iou_thr=a.iou, iou_thr_low=a.iou_low,
        max_age=a.max_age, min_hits=a.min_hits, vel_scale=25.0 / prov["fps"],
    )  # fmt: skip
    for n, f in enumerate(per_frame):
        hi = f["scores"] >= high
        single.update(f["boxes"][hi], n, scores=f["scores"][hi])
        two.update(f["boxes"], f["scores"], n)

    arms = {"single-stage (shipped)": to_json(single.finished()),
            "two-stage": to_json(two.finished())}  # fmt: skip

    # The self-check named in the docstring: the single-stage arm must be the file the
    # labels were read from, or the comparison is against a different detection set.
    same = arms["single-stage (shipped)"] == proposed
    print(
        f"\nsingle-stage arm reproduces tracks.json: {same} "
        f"({len(arms['single-stage (shipped)'])} vs {len(proposed)} tracks)"
    )
    if not same:
        print("  ^ the numbers below are NOT comparable with the labels; stop and find out why")

    gt = as_metric_input(gt_raw)
    report = {"gt_identities": len(gt), "clip": prov["clip"], "arms": {}}
    print(f"\nground truth: {len(gt)} identities over {prov['max_frames']} frames")
    for name, tr in arms.items():
        pred = as_metric_input(tr)
        f1 = idf1(gt, pred)
        sw = id_switches(gt, pred)
        report["arms"][name] = {"tracks": len(pred), **f1, **sw}
        print(
            f"  {name:24s} {len(pred):3d} tracks  IDF1 {f1['idf1']:.4f}  "
            f"IDP {f1['idp']:.4f}  IDR {f1['idr']:.4f}  "
            f"switches {int(sw['switches'])}  tracks/identity {sw['tracks_per_identity']:.2f}  "
            f"mostly-tracked {int(sw['mostly_tracked'])}/{len(gt)}"
        )
    (a.gt / "idf1.json").write_text(json.dumps(report, indent=1) + "\n")
    print(f"\n-> {a.gt / 'idf1.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
