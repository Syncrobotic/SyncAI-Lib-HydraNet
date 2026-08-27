#!/usr/bin/env python3
"""Why a camera's `person` boxes score what they score, in four modes.

Opened for §7.11 -- step 2's "Kaohsiung person-score investigation" -- and kept because
each mode answers a question that a raw score histogram cannot.

A per-camera score histogram is not comparable across cameras: a busy camera emits more
low-scoring duplicates and fragments, so its median falls for a reason that has nothing
to do with how well it sees a person. Every mode here controls for that differently, and
the control in all four is the **dense head** -- the same trunk, not box-conditioned,
not the head under test. What it marks as `person` pixels is used to decide which boxes
are on people and how many people are in a frame; it never decides a score.

    solo     One person in frame, so duplicates cannot exist. For each frame whose dense
             head finds exactly one person-sized blob, the highest-scoring person box
             whose centre lands in that blob. Answers "what does this camera score a
             shopper it has entirely to itself".

    density  The same camera's quiet, mid and busy clips pooled, binned by dense person
             pixels. Answers "does the score fall as people are added". Pooling across
             clips reintroduces time of day as a confound and that is deliberate: binning
             one clip's own frames is null on a camera whose clip is crowded end to end,
             because the independent variable never moves.

    factors  `decode` forms score = sigmoid(cls) * sigmoid(centerness). Those two failing
             look identical from outside and need opposite fixes -- a low `cls` is a
             classification failure and costs a retrain, a low `centerness` on a correctly
             classified person is a geometry artefact of FCOS's smallest-box-wins
             assignment. This mode splits them at every location the dense head calls
             person, and sweeps the NMS IoU threshold in the same pass so that "NMS is
             eating the crowd" is answered by the same run rather than by argument.

    standing what the strip under a person blob stands on, from the terrain head's own
             floor / fixture / person classes: feet visible, cut off by a counter, or
             behind somebody. Answers "does a fixture crossing a shopper cost score".
             It cannot see a crowd -- connected components merge touching people into
             one blob -- so read it beside `density`, never instead of it.

Usage:
    uv run python scripts/person_score_probe.py solo --cameras Kaohsiung-cam04 ...
    uv run python scripts/person_score_probe.py density --camera Kaohsiung-cam04
    uv run python scripts/person_score_probe.py standing --cameras Tao-Hsin-cam03 ...
    uv run python scripts/person_score_probe.py factors --clip A.mp4 B.mp4 --label a b
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torchvision
from PIL import Image
from scipy import ndimage

from syncai_hydranet.config import load_config
from syncai_hydranet.data.video import frames as decode_frames
from syncai_hydranet.models.hydranet import build_model
from syncai_hydranet.utils.checkpoint import load_checkpoint, select_weights
from syncai_hydranet.utils.visualize import preprocess

ROOT = Path(__file__).resolve().parent.parent
CLIPS = ROOT / "datasets/studioa_clips"
DEFAULT_CONFIG = "configs/hydranet_retail_pose02.yaml"
DEFAULT_CHECKPOINT = "runs/hydranet_retail_pose02/last.pt"
# A dense person component smaller than this at the model canvas is noise, not a shopper.
MIN_BLOB_PX = 400


class Probe:
    """One loaded model plus the three indices every mode needs from its config."""

    def __init__(self, config: str, checkpoint: str, device: str = "cuda") -> None:
        self.cfg = load_config(config, validate=False)
        self.device = device
        self.model = build_model(self.cfg).to(device).eval()
        self.model.load_state_dict(select_weights(load_checkpoint(checkpoint), "ema"))
        self.size = self.cfg["data"]["input_size"]
        self.person = list(self.cfg["model"]["heads"]["detection"]["classes"]).index("person")
        self.seg_person = list(self.cfg["data"]["terrain_classes"]).index("person")
        det = self.model._detection()
        if det is None:
            raise ValueError(
                f"{config} builds a model with no detection head, so there is no person "
                f"score to probe."
            )
        self.head = det.head

    def forward(self, frame) -> dict:
        """One forward pass, plus the letterbox region the caller needs to undo it."""
        img = Image.fromarray(frame)
        x, _canvas, region = preprocess(img, self.size)
        with torch.no_grad():
            out = self.model(x.to(self.device))
        return {"out": out, "region": region, "image": img}


def _person_boxes(probe: Probe, out: dict, nms_thr: float = 0.6) -> tuple:
    """Decoded person boxes and scores at the model canvas.

    `img_size` is passed because `HydraNet.predict` passes it: without the clamp a box
    may extend past the canvas, which changes the centre this file tests blobs against.
    """
    h, w = probe.size
    res = probe.head.decode(
        out["det_cls"],
        out["det_reg"],
        out["det_ctr"],
        score_thr=0.03,
        nms_thr=nms_thr,
        img_size=(h, w),
    )[0]
    labels = res["labels"].cpu().numpy()
    keep = labels == probe.person
    return res["boxes"].cpu().numpy()[keep], res["scores"].cpu().numpy()[keep]


def _dense_person(probe: Probe, out: dict) -> np.ndarray:
    return out["terrain"].argmax(dim=1)[0].cpu().numpy() == probe.seg_person


def _pct(v, p):
    return float(np.percentile(v, p)) if len(v) else float("nan")


# ------------------------------------------------------------------ solo


def run_solo(a) -> dict:
    probe = Probe(a.config, a.checkpoint, a.device)
    rows = {}
    for cam in a.cameras:
        best, blob_px = [], []
        for clip in sorted((CLIPS / cam).glob("archive_*.mp4")):
            for i, frame in enumerate(decode_frames(str(clip), 1920, 1080, a.fps)):
                if i >= a.frames:
                    break
                f = probe.forward(frame)
                mask = _dense_person(probe, f["out"])
                lab, n = ndimage.label(mask)
                if n == 0:
                    continue
                sizes = ndimage.sum(mask, lab, range(1, n + 1))
                big = np.nonzero(sizes >= a.min_blob)[0]
                # exactly one shopper-sized blob, or the frame cannot answer the question
                if len(big) != 1:
                    continue
                ys, xs = np.nonzero(lab == big[0] + 1)
                boxes, scores = _person_boxes(probe, f["out"])
                if not len(boxes):
                    continue
                cx = (boxes[:, 0] + boxes[:, 2]) / 2
                cy = (boxes[:, 1] + boxes[:, 3]) / 2
                inside = (
                    (cx >= xs.min()) & (cx <= xs.max()) & (cy >= ys.min()) & (cy <= ys.max())
                )
                if not inside.any():
                    continue
                best.append(float(scores[inside].max()))
                blob_px.append(float(sizes[big[0]]))
        b = np.array(best)
        frac = float((b >= 0.35).mean()) if len(b) else float("nan")
        rows[cam] = {
            "n": len(b),
            "p10": _pct(b, 10),
            "p50": _pct(b, 50),
            "p90": _pct(b, 90),
            "frac_ge035": frac,
            "blob_px_p50": _pct(np.array(blob_px), 50),
        }
        print(
            f"{cam:18s} solo frames {len(b):4d}  best-box score p10/p50/p90 "
            f"{_pct(b, 10):.2f}/{_pct(b, 50):.2f}/{_pct(b, 90):.2f}  "
            f"frac>=0.35 {frac:.2f}"
        )
    return rows


# -------------------------------------------------------------- standing


STANDING_BINS = ("grounded", "cut", "behind", "other")


def run_standing(a) -> dict:
    """Score against what the person is standing on, or cut off by.

    The terrain head names `floor`, `fixture` and `person`, so the strip immediately
    under a person blob says whether the shopper's feet are visible (`grounded`), a
    counter crosses the body (`cut`), or somebody else is in front (`behind`). The
    dense head decides the bin and the box head is scored on it, so the head under test
    never chooses where it lands.

    **This mode cannot see a crowd.** Connected components merge touching people into
    one blob, which then counts once -- on a camera whose clip holds a dozen shoppers at
    a counter it reports about one person per frame, and every number it prints for that
    camera is a statement about its quiet frames. Use `density` for the crowd.
    """
    probe = Probe(a.config, a.checkpoint, a.device)
    terrain = list(probe.cfg["data"]["terrain_classes"])
    seg_floor, seg_fixture = terrain.index("floor"), terrain.index("fixture")
    rows: dict[str, dict[str, list]] = {}
    for cam in a.cameras:
        rows[cam] = {k: [] for k in STANDING_BINS}
        for clip in sorted((CLIPS / cam).glob("archive_*.mp4")):
            for i, frame in enumerate(decode_frames(str(clip), 1920, 1080, a.fps)):
                if i >= a.frames:
                    break
                f = probe.forward(frame)
                cmap = f["out"]["terrain"].argmax(dim=1)[0].cpu().numpy()
                mask = cmap == probe.seg_person
                lab, n = ndimage.label(mask)
                if n == 0:
                    continue
                sizes = ndimage.sum(mask, lab, range(1, n + 1))
                boxes, scores = _person_boxes(probe, f["out"])
                if not len(boxes):
                    continue
                cx = (boxes[:, 0] + boxes[:, 2]) / 2
                cy = (boxes[:, 1] + boxes[:, 3]) / 2
                for b in np.nonzero(sizes >= a.min_blob)[0]:
                    ys, xs = np.nonzero(lab == b + 1)
                    y0 = min(int(ys.max()) + 1, cmap.shape[0] - 1)
                    y1 = min(int(ys.max()) + 1 + a.band, cmap.shape[0])
                    kind = "other"
                    if y1 > y0:
                        strip = cmap[y0:y1, xs.min() : xs.max() + 1]
                        counts = {
                            "grounded": int((strip == seg_floor).sum()),
                            "cut": int((strip == seg_fixture).sum()),
                            "behind": int((strip == probe.seg_person).sum()),
                        }
                        kind = max(counts, key=lambda k: counts[k])
                        if counts[kind] == 0:
                            kind = "other"
                    inside = (
                        (cx >= xs.min())
                        & (cx <= xs.max())
                        & (cy >= ys.min())
                        & (cy <= ys.max())
                    )
                    if inside.any():
                        rows[cam][kind].append(float(scores[inside].max()))

    print(
        f"{'camera':18s} {'bin':>9s} {'people':>7s} {'p50':>6s} {'p90':>6s} {'frac>=0.35':>11s}"
    )
    report: dict[str, dict] = {}
    for cam in a.cameras:
        report[cam] = {}
        for kind in STANDING_BINS:
            v = np.array(rows[cam][kind])
            if not len(v):
                continue
            frac = float((v >= 0.35).mean())
            report[cam][kind] = {
                "n": len(v), "p50": _pct(v, 50), "p90": _pct(v, 90), "frac_ge035": frac
            }  # fmt: skip
            print(
                f"{cam:18s} {kind:>9s} {len(v):7d} {_pct(v, 50):6.2f} {_pct(v, 90):6.2f} "
                f"{frac:11.2f}"
            )
    return report


# --------------------------------------------------------------- density


def run_density(a) -> dict:
    probe = Probe(a.config, a.checkpoint, a.device)
    clips = a.clip or [str(p) for p in sorted((CLIPS / a.camera).glob("archive_*.mp4"))]
    rows = []
    for clip in clips:
        for i, frame in enumerate(decode_frames(clip, 1920, 1080, a.fps)):
            if i >= a.frames:
                break
            f = probe.forward(frame)
            px = float(_dense_person(probe, f["out"]).sum())
            _boxes, scores = _person_boxes(probe, f["out"])
            rows.append(
                {
                    "clip": Path(clip).name,
                    "i": i,
                    "person_px": px,
                    "top": float(scores.max()) if len(scores) else 0.0,
                    "n035": int((scores >= 0.35).sum()),
                    "n015": int((scores >= 0.15).sum()),
                }
            )
    px = np.array([r["person_px"] for r in rows])
    print(f"clips {len(clips)}  frames {len(rows)}")
    header = f"{'person px band':>26s} {'frames':>7s} {'top p50':>8s}"
    print(f"{header} {'n>=0.35':>8s} {'n>=0.15':>8s}")
    edges = [np.percentile(px, q) for q in [0, 15, 30, 45, 60, 75, 90, 100]]
    for k in range(len(edges) - 1):
        sel = [r for r, v in zip(rows, px, strict=True) if edges[k] <= v <= edges[k + 1]]
        if not sel:
            continue
        print(
            f"{edges[k]:11.0f}-{edges[k + 1]:<14.0f} {len(sel):7d} "
            f"{np.median([r['top'] for r in sel]):8.2f} "
            f"{np.median([r['n035'] for r in sel]):8.0f} "
            f"{np.median([r['n015'] for r in sel]):8.0f}"
        )
    return {"rows": rows}


# --------------------------------------------------------------- factors


def run_factors(a) -> dict:
    probe = Probe(a.config, a.checkpoint, a.device)
    labels = a.label or [Path(c).name for c in a.clip]
    report = {}
    for clip, name in zip(a.clip, labels, strict=True):
        cls_v, ctr_v, nms_counts = [], [], {t: [] for t in a.nms}
        for i, frame in enumerate(decode_frames(clip, 1920, 1080, a.fps)):
            if i >= a.frames:
                break
            f = probe.forward(frame)
            out = f["out"]
            cls_out, reg_out, ctr_out = out["det_cls"], out["det_reg"], out["det_ctr"]
            cmap = out["terrain"].argmax(dim=1)[0]
            shapes = [c.shape[-2:] for c in cls_out]
            points, _lv = probe.head._grid_points(shapes, cls_out[0].device)
            ncls = probe.head.num_classes
            flat_cls = torch.cat(
                [c.permute(0, 2, 3, 1).reshape(1, -1, ncls) for c in cls_out], 1
            ).sigmoid()[0][:, probe.person]
            flat_reg = torch.cat([r.permute(0, 2, 3, 1).reshape(1, -1, 4) for r in reg_out], 1)[
                0
            ]
            flat_ctr = torch.cat(
                [c.permute(0, 2, 3, 1).reshape(1, -1) for c in ctr_out], 1
            ).sigmoid()[0]

            py = points[:, 1].long().clamp(0, cmap.shape[0] - 1)
            px_ = points[:, 0].long().clamp(0, cmap.shape[1] - 1)
            # only locations standing on a person, so this is about people not background
            sel = (cmap[py, px_] == probe.seg_person) & (flat_cls > 0.05)
            if sel.any():
                cls_v.append(flat_cls[sel].cpu().numpy())
                ctr_v.append(flat_ctr[sel].cpu().numpy())

            score = flat_cls * flat_ctr
            keep = score > 0.05
            if keep.any():
                pts, ltrb = points[keep], flat_reg[keep]
                boxes = torch.stack(
                    [pts[:, 0] - ltrb[:, 0], pts[:, 1] - ltrb[:, 1],
                     pts[:, 0] + ltrb[:, 2], pts[:, 1] + ltrb[:, 3]], -1
                )  # fmt: skip
                sc = score[keep]
                for t in a.nms:
                    kept = sc[torchvision.ops.nms(boxes, sc, t)]
                    nms_counts[t].append(int((kept >= 0.35).sum()))
            else:
                for t in a.nms:
                    nms_counts[t].append(0)

        cls_all = np.concatenate(cls_v) if cls_v else np.array([0.0])
        ctr_all = np.concatenate(ctr_v) if ctr_v else np.array([0.0])
        nms_median = {str(t): float(np.median(v)) for t, v in nms_counts.items()}
        report[name] = {
            "n_locs": len(cls_all),
            "cls_p50": _pct(cls_all, 50),
            "cls_p90": _pct(cls_all, 90),
            "ctr_p50": _pct(ctr_all, 50),
            "ctr_p90": _pct(ctr_all, 90),
            "score_p90": _pct(cls_all * ctr_all, 90),
            "nms": nms_median,
        }
        swept = " ".join(f"{t}:{v:.0f}" for t, v in nms_median.items())
        print(
            f"{name:22s} locs {len(cls_all):7d}  "
            f"cls p50/p90 {_pct(cls_all, 50):.2f}/{_pct(cls_all, 90):.2f}  "
            f"ctr p50/p90 {_pct(ctr_all, 50):.2f}/{_pct(ctr_all, 90):.2f}  "
            f"n>=0.35 by NMS {swept}"
        )
    return report


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    # A parent parser rather than top-level flags: argparse accepts a top-level option
    # only *before* the subcommand, so `... solo --cameras X --out F` fails with an
    # unrecognised-argument error that names the flag it just documented. Inheriting them
    # makes both positions work.
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", default=DEFAULT_CONFIG)
    common.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    common.add_argument("--device", default="cuda")
    common.add_argument("--out")
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    ap.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out")
    sub = ap.add_subparsers(dest="mode", required=True)

    s = sub.add_parser(
        "solo", parents=[common], help="score on frames holding exactly one person"
    )
    s.add_argument("--cameras", nargs="+", required=True)
    s.add_argument("--fps", type=float, default=1.0)
    s.add_argument("--frames", type=int, default=300, help="per clip")
    s.add_argument("--min-blob", type=int, default=MIN_BLOB_PX)
    s.set_defaults(fn=run_solo)

    t = sub.add_parser(
        "standing",
        parents=[common],
        help="score against what the person stands on or is cut by",
    )
    t.add_argument("--cameras", nargs="+", required=True)
    t.add_argument("--fps", type=float, default=1.0)
    t.add_argument("--frames", type=int, default=200, help="per clip")
    t.add_argument("--min-blob", type=int, default=MIN_BLOB_PX)
    t.add_argument("--band", type=int, default=6, help="rows under the blob to classify")
    t.set_defaults(fn=run_standing)

    d = sub.add_parser(
        "density",
        parents=[common],
        help="score against crowd size, pooled over a camera's clips",
    )
    d.add_argument("--camera")
    d.add_argument("--clip", nargs="+", help="explicit clips instead of --camera's whole set")
    d.add_argument("--fps", type=float, default=2.0)
    d.add_argument("--frames", type=int, default=400, help="per clip")
    d.set_defaults(fn=run_density)

    f = sub.add_parser(
        "factors",
        parents=[common],
        help="split score into cls * centerness, and sweep NMS",
    )
    f.add_argument("--clip", nargs="+", required=True)
    f.add_argument("--label", nargs="+")
    f.add_argument("--fps", type=float, default=1.0)
    f.add_argument("--frames", type=int, default=100, help="per clip")
    f.add_argument("--nms", type=float, nargs="+", default=[0.6, 0.75, 0.9])
    f.set_defaults(fn=run_factors)

    a = ap.parse_args()
    result = a.fn(a)
    if a.out:
        Path(a.out).write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
