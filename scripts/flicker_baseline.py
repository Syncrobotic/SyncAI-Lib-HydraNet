#!/usr/bin/env python3
"""Output-stability baseline: lay the ruler down before optimising anything.

  nice -n 10 .venv/bin/python scripts/flicker_baseline.py \
    --config configs/hydranet_retail_security_b03_cw_xl.yaml \
    --checkpoint runs/hydranet_retail_security_b03_cw_xl/best.pt \
    --input datasets/studioa_clips/Kaohsiung-cam04/archive_20260816-112757_20260816-113301.mp4 \
    --static-mask datasets/studioa_static/Kaohsiung-cam04/static_20260816-112757.png \
    --out runs/flicker_baseline01

The two symptoms visible in the demo video -- fragmented segmentation and jumping labels --
each become a number here, and every later optimisation (static compositing, track
rendering, logits EMA) is compared against this same set.

Four things are measured, each split into two columns, **inside the static mask** and
outside it, because a flip in a static region is noise by construction while a flip in a
moving region may be a real change:

1. Per-frame label flip rate, and the share of flipped pixels that read back their previous
   class within ``--bounce-window`` frames -- a bounce against a real change.
2. Per-frame distribution of connected-component count and the share that are small
   (area < ``--small-area-frac`` of the frame).
3. Detection stability: per-frame variance in box count; the distribution of box survival
   length after greedy adjacent-frame matching at IoU >= ``--track-iou``; and the share of
   boxes that live for exactly one frame.
4. Per-class (6 classes) flip rate. For wall/column/fixture the modal map's class boundary
   is dilated into a +/-r band, and the flip rate inside the band is contrasted with the
   rate in the class interior.

Cross-frame agreement reuses `engine/consensus.py`'s `FrameConsensus` definition -- the
same `agree_threshold`, the same `person` exclusion, the same caveats -- rather than
inventing a parallel metric. What this script adds is the *adjacent-frame* behaviour that
one does not measure: flips, bounces, fragmentation and box survival.

**Bounce, defined.** A pixel that flips at frame t (``label[t-1] != label[t]``) counts as a
bounce if any frame within the next ``--bounce-window`` reads back ``label[t-1]``. Flips
whose window runs past the end of the clip are *undecidable* and are subtracted from the
denominator rather than counted as non-bounces -- otherwise the tail of every clip drags
the bounce rate down by construction.

When a training run shares the GPU: batch is always 1, and start the whole process under
`nice -n 10`.
"""

from __future__ import annotations

import argparse
import json
from collections import deque
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy import ndimage

from syncai_hydranet.config import load_config
from syncai_hydranet.data.video import frames, probe
from syncai_hydranet.engine.consensus import FrameConsensus
from syncai_hydranet.models.heads.detection import SCORE_THR_VIEW
from syncai_hydranet.models.hydranet import build_model
from syncai_hydranet.utils.checkpoint import load_checkpoint, select_weights
from syncai_hydranet.utils.device import pick_device
from syncai_hydranet.utils.visualize import crop_box, preprocess


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="flicker_baseline",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--input", required=True, help="a clip from a fixed camera")
    ap.add_argument(
        "--static-mask",
        required=True,
        help="the static mask static_plates.py produces (255 = a pixel that never changed "
        "across the whole clip). It must be the mask for the *same* time slot as the "
        "clip: the illumination difference between slots eats the mask's meaning",
    )
    ap.add_argument(
        "--out", type=Path, required=True, help="output directory, e.g. runs/flicker_baselineNN"
    )
    ap.add_argument("--weights", choices=["ema", "model"], default="ema")
    ap.add_argument(
        "--fps", type=float, default=5.0, help="sampling fps; the demo video's rate"
    )
    ap.add_argument("--max-frames", type=int, default=900, help="0 means the whole clip")
    ap.add_argument("--score-thr", type=float, default=SCORE_THR_VIEW)
    ap.add_argument("--nms-thr", type=float, default=0.6)
    ap.add_argument(
        "--track-iou",
        type=float,
        default=0.5,
        help="IoU threshold for the greedy cross-frame match; same class, adjacent frames only",
    )
    ap.add_argument(
        "--small-area-frac",
        type=float,
        default=0.001,
        help="a connected component below this fraction of the content area counts as small; "
        "this is the definition of a fragment for the fragmentation figure",
    )
    ap.add_argument(
        "--boundary-radius",
        type=int,
        default=5,
        help="boundary band radius in pixels; the modal map's class region is dilated and "
        "eroded by r each to give the +/-r band",
    )
    ap.add_argument(
        "--boundary-classes",
        nargs="*",
        default=["wall", "column", "fixture"],
        help="terrain classes to run the boundary-band analysis over",
    )
    ap.add_argument(
        "--bounce-window",
        type=int,
        default=25,
        help="how many frames a flip has to read back its previous class in to count as a "
        "bounce; the default of 25 is 5 seconds at 5 fps",
    )
    ap.add_argument(
        "--box-static-thr",
        type=float,
        default=0.5,
        help="a box whose static-pixel share exceeds this is counted in the static column",
    )
    ap.add_argument(
        "--agree-threshold",
        type=float,
        default=0.9,
        help="passed straight to FrameConsensus.result as agree_threshold; its definition, "
        "not a second one",
    )
    ap.add_argument("--device", default=None)
    return ap


# ---------------------------------------------------------------------------
# ------------------------------------------------- detection, matched across frames
# ---------------------------------------------------------------------------


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise IoU. a: [N, 4], b: [M, 4], both xyxy."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float64)
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.clip(rb - lt, 0, None)
    inter = wh[..., 0] * wh[..., 1]
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return inter / np.clip(union, 1e-9, None)


def greedy_tracks(per_frame: list[dict], iou_thr: float) -> list[dict]:
    """Greedy adjacent-frame, same-class matching by descending IoU, strung into tracks.

    Adjacent frames only and no max-age, deliberately. What is being measured is how
    unstable the **raw** output is before any smoothing, and a tracker allowed to bridge a
    gap would sew up exactly the breaks this exists to count.
    """
    tracks: list[dict] = []
    active: list[dict] = []  # each: {label, boxes, static_fracs, last_box}
    for det in per_frame:
        boxes, labels, sfr = det["boxes"], det["labels"], det["static_frac"]
        matched_cur = np.zeros(len(boxes), dtype=bool)
        matched_act = np.zeros(len(active), dtype=bool)
        if len(active) and len(boxes):
            last = np.stack([t["last_box"] for t in active])
            m = iou_matrix(last, boxes)
            act_labels = np.array([t["label"] for t in active])
            m[act_labels[:, None] != labels[None, :]] = 0.0
            order = np.dstack(np.unravel_index(np.argsort(-m, axis=None), m.shape))[0]
            for ai, bi in order:
                if m[ai, bi] < iou_thr:
                    break
                if matched_act[ai] or matched_cur[bi]:
                    continue
                matched_act[ai] = matched_cur[bi] = True
                active[ai]["length"] += 1
                active[ai]["last_box"] = boxes[bi]
                active[ai]["static_fracs"].append(float(sfr[bi]))
        survivors = []
        for ai, t in enumerate(active):
            if matched_act[ai]:
                survivors.append(t)
            else:
                tracks.append(t)
        active = survivors
        for bi in range(len(boxes)):
            if not matched_cur[bi]:
                active.append(
                    {
                        "label": int(labels[bi]),
                        "length": 1,
                        "last_box": boxes[bi],
                        "static_fracs": [float(sfr[bi])],
                    }
                )
    tracks.extend(active)
    return tracks


# ---------------------------------------------------------------------------
# ------------------------------------------------------------------ small statistics
# ---------------------------------------------------------------------------


def dist(values) -> dict:
    """Distribution summary of one per-frame quantity.

    An empty sequence returns None fields: a null in the JSON is better than an invented
    zero, which reads as a measurement.
    """
    a = np.asarray(values, dtype=np.float64)
    if a.size == 0:
        return {"n": 0, "mean": None, "p50": None, "p90": None, "max": None}
    return {
        "n": int(a.size),
        "mean": float(a.mean()),
        "p50": float(np.percentile(a, 50)),
        "p90": float(np.percentile(a, 90)),
        "max": float(a.max()),
    }


def rate(num: float, den: float) -> float | None:
    return (num / den) if den > 0 else None


def save_heatmap(arr: np.ndarray, path: Path, title: str, vmax: float | None = None) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(11.2, 6.7))
    im = ax.imshow(arr, cmap="inferno", vmax=vmax)
    ax.set_title(title)
    ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# ------------------------------------------------------------------------------- main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> dict:
    args = build_parser().parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)

    cfg = load_config(args.config, [])
    device = pick_device(args.device or cfg.get("device"))
    model = build_model(cfg).to(device).eval()
    ckpt = load_checkpoint(args.checkpoint)
    model.load_state_dict(select_weights(ckpt, args.weights))

    size = cfg["data"]["input_size"]  # (H, W)
    class_names: list[str] = list(cfg["data"]["terrain_classes"])
    n_cls = len(class_names)
    for c in args.boundary_classes:
        if c not in class_names:
            raise SystemExit(
                f"--boundary-classes {c!r} is not in terrain_classes {class_names}"
            )
    det_classes: list[str] = []
    det_cfg = (cfg.get("model", {}).get("heads", {}) or {}).get("detection") or {}
    det_classes = list(det_cfg.get("classes", []))

    src_w, src_h, src_fps = probe(args.input)
    sample_fps = args.fps if args.fps and args.fps < src_fps else None
    print(
        f"input {src_w}x{src_h} @ {src_fps:.2f}fps, sampling {args.fps}fps, "
        f"at most {args.max_frames or 'all'} frames, device={device}"
    )

    static_img = Image.open(args.static_mask).convert("L")

    # consensus.py's definition, reused: same person exclusion, same agree_threshold.
    consensus = FrameConsensus(class_names)

    # Per-pixel accumulators, in letterbox content-area coordinates, built on the first frame.
    flip_map = None  # flips per pixel
    bounce_map = None  # bounces per pixel
    censored_map = None  # flips whose bounce window ran past the end of the clip
    occupancy = None  # [C,H,W] frames where prev == c -- the flip-rate denominator
    flip_from = None  # [C,H,W] flips out of class c
    static = None  # the static mask, at content-area resolution
    pending: deque = deque()  # flips still inside their bounce window

    prev = None
    per_frame_flip = []  # per frame: (overall, inside static, outside static)
    comp_rows = []  # per-frame connected-component statistics
    det_frames = []  # per-frame detections
    content_area = None
    n_frames = 0

    for frame in frames(args.input, src_w, src_h, sample_fps):
        x, _lb, region = preprocess(Image.fromarray(frame), size)
        with torch.no_grad():
            res = model.predict(x.to(device), score_thr=args.score_thr, nms_thr=args.nms_thr)
        cur = crop_box(res["terrain"][0].cpu().numpy(), region).astype(np.int16)
        h, w = cur.shape
        x0, y0 = region[0], region[1]

        if flip_map is None:
            content_area = h * w
            flip_map = np.zeros((h, w), dtype=np.int32)
            bounce_map = np.zeros((h, w), dtype=np.int32)
            censored_map = np.zeros((h, w), dtype=np.int32)
            occupancy = np.zeros((n_cls, h, w), dtype=np.int32)
            flip_from = np.zeros((n_cls, h, w), dtype=np.int32)
            static = np.asarray(static_img.resize((w, h), Image.Resampling.NEAREST)) > 127
            n_static = int(static.sum())
            share = n_static / content_area
            print(f"content area {w}x{h}, static pixels {n_static} ({share:.1%})")

        # Non-None from the first frame onwards. ty cannot narrow across iterations, so the
        # assert is what tells it.
        assert flip_map is not None and bounce_map is not None and censored_map is not None
        assert occupancy is not None and flip_from is not None and static is not None
        assert content_area is not None

        consensus.update(cur.astype(np.int64))

        # -- 1. flips and bounces --------------------------------------------
        if prev is not None:
            changed = prev != cur
            flip_map += changed
            for c in range(n_cls):
                was_c = prev == c
                occupancy[c] += was_c
                flip_from[c] += was_c & changed
            per_frame_flip.append(
                (
                    float(changed.mean()),
                    float(changed[static].mean()) if static.any() else float("nan"),
                    float(changed[~static].mean()) if (~static).any() else float("nan"),
                )
            )
            # Settle the open events first: this frame is their "later"
            for ev in pending:
                back = ev["mask"] & (cur == ev["orig"])
                bounce_map += back
                ev["mask"] &= ~back
                ev["ttl"] -= 1
            while pending and pending[0]["ttl"] <= 0:
                pending.popleft()
            if changed.any():
                pending.append(
                    {"mask": changed.copy(), "orig": prev.copy(), "ttl": args.bounce_window}
                )

        # -- 2. connected components -----------------------------------------
        small_cut = args.small_area_frac * content_area
        row = {
            "total": 0,
            "small": 0,
            "static": 0,
            "dynamic": 0,
            "small_static": 0,
            "small_dynamic": 0,
        }
        for c in range(n_cls):
            mask_c = cur == c
            if not mask_c.any():
                continue
            lab, n = ndimage.label(mask_c)
            if n == 0:
                continue
            idx = np.arange(1, n + 1)
            areas = ndimage.sum_labels(np.ones_like(lab, dtype=np.int32), lab, idx)
            static_px = ndimage.sum_labels(static.astype(np.int32), lab, idx)
            is_small = areas < small_cut
            is_static = (static_px / areas) >= 0.5
            row["total"] += int(n)
            row["small"] += int(is_small.sum())
            row["static"] += int(is_static.sum())
            row["dynamic"] += int((~is_static).sum())
            row["small_static"] += int((is_small & is_static).sum())
            row["small_dynamic"] += int((is_small & ~is_static).sum())
        comp_rows.append(row)

        # -- 3. detections ---------------------------------------------------
        det = res.get("detection", [{}])[0]
        boxes = det.get("boxes")
        if boxes is not None and len(boxes):
            b = boxes.cpu().numpy().astype(np.float64)
            b[:, 0::2] -= x0
            b[:, 1::2] -= y0
            b[:, 0::2] = b[:, 0::2].clip(0, w)
            b[:, 1::2] = b[:, 1::2].clip(0, h)
            labels = det["labels"].cpu().numpy().astype(np.int64)
            keep = (b[:, 2] > b[:, 0]) & (b[:, 3] > b[:, 1])
            b, labels = b[keep], labels[keep]
            sfr = np.array(
                [
                    float(static[int(y1) : int(np.ceil(y2)), int(x1) : int(np.ceil(x2))].mean())
                    if y2 > y1 and x2 > x1
                    else 0.0
                    for x1, y1, x2, y2 in b
                ]
            )
        else:
            b = np.zeros((0, 4))
            labels = np.zeros((0,), dtype=np.int64)
            sfr = np.zeros((0,))
        det_frames.append({"boxes": b, "labels": labels, "static_frac": sfr})

        prev = cur
        n_frames += 1
        if n_frames % 100 == 0:
            print(f"  {n_frames} frames")
        if args.max_frames and n_frames >= args.max_frames:
            break

    if n_frames < 2:
        raise SystemExit(
            f"only {n_frames} frame(s) decoded; adjacent-frame behaviour needs two"
        )
    # Two frames at least by here, so the accumulators exist. Again, narrowing for ty.
    assert flip_map is not None and bounce_map is not None and censored_map is not None
    assert occupancy is not None and flip_from is not None and static is not None

    # Flips whose window ran past the end come out of the bounce denominator rather than
    # counting as non-bounces -- see the module docstring.
    for ev in pending:
        censored_map += ev["mask"]

    n_trans = n_frames - 1
    result = consensus.result(args.agree_threshold)
    modal = result.modal

    # -- summary 1: flips and bounces ---------------------------------------
    def zone_stats(zone: np.ndarray) -> dict:
        zpx = int(zone.sum())
        flips = int(flip_map[zone].sum())
        evaluable = flips - int(censored_map[zone].sum())
        bounced = int(bounce_map[zone].sum())
        return {
            "pixels": zpx,
            "flip_rate": rate(flips, zpx * n_trans),
            "flips": flips,
            "bounce_share_of_flips": rate(bounced, evaluable),
            "bounced": bounced,
            "flips_evaluable": evaluable,
        }

    flip_stats = {
        "overall": zone_stats(np.ones_like(static)),
        "in_static": zone_stats(static),
        "out_static": zone_stats(~static),
    }

    per_class_flip = {}
    for c, name in enumerate(class_names):
        entry = {}
        for zname, zone in (("in_static", static), ("out_static", ~static)):
            den = int(occupancy[c][zone].sum())
            num = int(flip_from[c][zone].sum())
            entry[zname] = {"flip_rate": rate(num, den), "pixel_frames": den, "flips": num}
        entry["share_of_all_flips"] = rate(int(flip_from[c].sum()), int(flip_map.sum()))
        per_class_flip[name] = entry

    # -- summary 4: boundary bands, +/-r around a class edge on the modal map -
    boundary = {}
    r = args.boundary_radius
    for name in args.boundary_classes:
        c = class_names.index(name)
        region_c = modal == c
        if not region_c.any():
            boundary[name] = None
            continue
        grown = ndimage.binary_dilation(region_c, iterations=r)
        shrunk = ndimage.binary_erosion(region_c, iterations=r)
        band = grown & ~shrunk
        interior = shrunk
        boundary[name] = {
            "band_pixels": int(band.sum()),
            "interior_pixels": int(interior.sum()),
            "band_flip_rate": rate(int(flip_map[band].sum()), int(band.sum()) * n_trans),
            "interior_flip_rate": rate(
                int(flip_map[interior].sum()), int(interior.sum()) * n_trans
            ),
            "band_flip_rate_in_static": rate(
                int(flip_map[band & static].sum()), int((band & static).sum()) * n_trans
            ),
            "band_flip_rate_out_static": rate(
                int(flip_map[band & ~static].sum()), int((band & ~static).sum()) * n_trans
            ),
        }

    # -- summary 2: fragmentation -------------------------------------------
    comp_stats = {
        "total_components_per_frame": dist([r_["total"] for r_ in comp_rows]),
        "small_share_per_frame": dist(
            [r_["small"] / r_["total"] for r_ in comp_rows if r_["total"]]
        ),
        "in_static_components_per_frame": dist([r_["static"] for r_ in comp_rows]),
        "out_static_components_per_frame": dist([r_["dynamic"] for r_ in comp_rows]),
        "in_static_small_share": dist(
            [r_["small_static"] / r_["static"] for r_ in comp_rows if r_["static"]]
        ),
        "out_static_small_share": dist(
            [r_["small_dynamic"] / r_["dynamic"] for r_ in comp_rows if r_["dynamic"]]
        ),
    }

    # -- summary 3: detection stability -------------------------------------
    counts = np.array([len(d["boxes"]) for d in det_frames], dtype=np.float64)
    in_counts = np.array(
        [int((d["static_frac"] >= args.box_static_thr).sum()) for d in det_frames],
        dtype=np.float64,
    )
    out_counts = counts - in_counts
    tracks = greedy_tracks(det_frames, args.track_iou)

    def track_stats(sel: list[dict]) -> dict:
        lengths = np.array([t["length"] for t in sel], dtype=np.float64)
        if lengths.size == 0:
            return {"tracks": 0, "length": dist([]), "one_frame_share": None, "ge5_share": None}
        return {
            "tracks": int(lengths.size),
            "length": dist(lengths),
            "one_frame_share": float((lengths == 1).mean()),
            "ge5_share": float((lengths >= 5).mean()),
        }

    def t_static(t: dict) -> float:
        return float(np.mean(t["static_fracs"]))

    det_stats = {
        "frames": len(det_frames),
        "boxes_per_frame": {
            "overall": {"mean": float(counts.mean()), "var": float(counts.var())},
            "in_static": {"mean": float(in_counts.mean()), "var": float(in_counts.var())},
            "out_static": {"mean": float(out_counts.mean()), "var": float(out_counts.var())},
        },
        "tracks": {
            "overall": track_stats(tracks),
            "in_static": track_stats([t for t in tracks if t_static(t) >= args.box_static_thr]),
            "out_static": track_stats([t for t in tracks if t_static(t) < args.box_static_thr]),
        },
        "per_class": {},
    }
    for ci, cname in enumerate(det_classes):
        c_counts = np.array(
            [int((d["labels"] == ci).sum()) for d in det_frames], dtype=np.float64
        )
        det_stats["per_class"][cname] = {
            "boxes_per_frame": {"mean": float(c_counts.mean()), "var": float(c_counts.var())},
            "tracks": track_stats([t for t in tracks if t["label"] == ci]),
        }

    metrics = {
        "measured": (
            "flicker baseline: adjacent-frame flips and bounces, fragmentation, "
            "box survival, boundary bands"
        ),
        "input": str(args.input),
        "static_mask": str(args.static_mask),
        "config": str(args.config),
        "checkpoint": str(args.checkpoint),
        "weights": args.weights,
        "frames": n_frames,
        "transitions": n_trans,
        "sample_fps": args.fps,
        "params": {
            "score_thr": args.score_thr,
            "nms_thr": args.nms_thr,
            "track_iou": args.track_iou,
            "small_area_frac": args.small_area_frac,
            "boundary_radius": args.boundary_radius,
            "boundary_classes": args.boundary_classes,
            "bounce_window": args.bounce_window,
            "box_static_thr": args.box_static_thr,
            "agree_threshold": args.agree_threshold,
        },
        "static_pixel_share": float(static.mean()),
        "flip": flip_stats,
        "flip_per_frame": {
            "overall": dist([v[0] for v in per_frame_flip]),
            "in_static": dist([v[1] for v in per_frame_flip]),
            "out_static": dist([v[2] for v in per_frame_flip]),
        },
        "per_class_flip": per_class_flip,
        "boundary_bands": boundary,
        "fragmentation": comp_stats,
        "detection": det_stats,
        # The existing measurement, carried in untouched, caveats and all.
        "consensus": result.to_dict(),
    }

    out_json = args.out / "metrics.json"
    out_json.write_text(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"wrote {out_json}")

    # -- heat maps -----------------------------------------------------------
    # ASCII titles: the box running the measurement usually has no CJK font for
    # matplotlib, so anything else renders as tofu.
    flip_rate_map = flip_map / n_trans
    save_heatmap(
        flip_rate_map,
        args.out / "flip_rate_heatmap.png",
        f"per-pixel flip rate over {n_trans} frame pairs",
        vmax=float(np.percentile(flip_rate_map, 99.5)) or None,
    )
    evaluable_map = np.clip(flip_map - censored_map, 1, None)
    save_heatmap(
        np.where(flip_map > 0, bounce_map / evaluable_map, 0.0),
        args.out / "bounce_share_heatmap.png",
        "bounce share of flips (returned to original class / evaluable flips)",
        vmax=1.0,
    )
    save_heatmap(
        flip_rate_map * static,
        args.out / "flip_rate_in_static_heatmap.png",
        "flip rate inside static mask (any nonzero here is noise)",
        vmax=float(np.percentile(flip_rate_map, 99.5)) or None,
    )
    print(f"heat maps written to {args.out}/")
    return metrics


if __name__ == "__main__":
    main()
