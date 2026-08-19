#!/usr/bin/env python3
"""Mine the site clips for fall-shaped postures, so the behaviour question stops being asked.

    python3 scripts/mine_fall_candidates.py --config runs/hydranet_retail_cctv/config.yaml \\
        --checkpoint runs/hydranet_retail_cctv/best.pt --out runs/fall_mining01 \\
        datasets/studioa_clips/*/*.mp4

**This is not a fall detector and its output is not a count of falls.** `events.
fall_candidates` is a box-shape proxy: wide, short, and staying that way. In a shop the
three most common things it fires on are a shopper bending to a low shelf, someone
crouching at bottom stock, and legs occluded by a fixture -- which is why the event type
is named `fall_candidate` and why nothing here reports a number as a fall.

What it is for is the precondition ARCHITECTURE.md sets before any behaviour
annotation is paid for: **confirm the behaviour occurs on these cameras at all.** MERL
Shopping is a grocery aisle of closed shelving and these are open display tables; finding
out after the annotation spend is the `column` failure at a higher price. A cheap proxy
run over footage already on disk is the only instrument available that can ask.

Two denominators are printed with every result, and they are not decoration. A clip with
zero candidates means one of two very different things -- no fall-shaped posture, or no
people detected -- and only the person-detection and track counts separate them. Person
boxes come from a COCO-trained head whose recall on these cameras **has never been
measured**, so every count here is bounded by an unknown.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
for candidate in (HERE.parent / "src", HERE / "src"):
    if candidate.is_dir():
        sys.path.insert(0, str(candidate))

# The clip loop lives in the package now, not in a sibling script. It used to be imported
# from `retail_flow.py` -- a script other scripts import is a module in the wrong place --
# and the copy here had drifted: no lens correction, while `site_events.py` had one.
from syncai_hydranet.analytics.clip_tracks import track_clip  # noqa: E402
from syncai_hydranet.analytics.delivery import report_settings  # noqa: E402
from syncai_hydranet.analytics.events import fall_candidates  # noqa: E402
from syncai_hydranet.analytics.tracker import Tracker  # noqa: E402
from syncai_hydranet.config import load_config  # noqa: E402
from syncai_hydranet.data.video import frames, probe  # noqa: E402
from syncai_hydranet.models.hydranet import build_model  # noqa: E402
from syncai_hydranet.utils.checkpoint import load_checkpoint, select_weights  # noqa: E402
from syncai_hydranet.utils.device import pick_device  # noqa: E402
from syncai_hydranet.utils.visualize import preprocess  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("clips", nargs="+")
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fps", type=float, default=5.0, help="sampling rate for tracking")
    ap.add_argument("--max-frames", type=int, default=0, help="0 means the whole clip")
    # 0.20 rather than the 0.30 viewing default, for retail_flow's measured reason: a
    # tracker cannot associate what was never detected.
    ap.add_argument("--score-thr", type=float, default=0.20)
    ap.add_argument("--min-hits", type=int, default=3)
    ap.add_argument("--max-age", type=int, default=5)
    ap.add_argument("--iou", type=float, default=0.3)
    # The lens, and it was missing until 2026-08-19. This script tracked on distorted
    # boxes while `site_events.py` ran the same loop on undistorted ones; the tracker
    # associates on IoU, so the two could link observations differently on the same clip.
    # `0.0` reproduces the old behaviour exactly, for comparing candidate sets across the
    # change. The default is Taichung-cam01's tile-grid fit and is an assumption on the
    # other 47 cameras -- but the value it replaces was an unstated assumption of zero.
    ap.add_argument("--k1", type=float, default=-0.225)
    # Deliberately more permissive than the library default of 1.5 s. This is a mining
    # pass: a candidate costs a human ten seconds to dismiss and a missed one costs the
    # whole question. Precision is not the objective here and should not be tuned for.
    ap.add_argument("--sustained", type=float, default=1.0, metavar="S")
    ap.add_argument("--aspect", type=float, default=1.0)
    ap.add_argument("--height-ratio", type=float, default=0.6)
    ap.add_argument(
        "--dump-frames",
        action="store_true",
        help="second pass over clips with candidates, writing their frames as JPEGs",
    )
    return ap


def border_contact(track, frame_start: int, frame_end: int, w: int, h: int, pad: int = 3):
    """Whether this track's boxes in the span touch the frame edge, and its median box.

    Found by looking at the first candidate this script produced rather than by reasoning
    about the rule: on Kaohsiung-cam04 it was a shopper at the counter **cut off by the
    bottom edge of frame**, so only head and shoulders were boxed -- wide, short, and
    sustained, which is the proxy's exact signature.

    That is a fourth confusion beside the three the library docstring lists (bending,
    crouching, fixture occlusion), and it is the worst-behaved of the four: on a camera
    that never moves it recurs in the same place in every clip that camera will ever
    produce, so it does not average out. Same shape as the `caution` painted on a fixed
    structural column in RETAIL.md, which was in 97.4% of 1,830 frames.

    Flagged rather than dropped. What share of candidates are border artefacts is itself
    the measurement, and a filter would delete the evidence for it.
    """
    keep = [
        b
        for f, b in zip(track.frames, track.boxes, strict=True)
        if frame_start <= f <= frame_end
    ]
    if not keep:
        return None, None
    arr = np.stack(keep)
    touches = bool(
        (arr[:, 0] <= pad).any()
        or (arr[:, 1] <= pad).any()
        or (arr[:, 2] >= w - pad).any()
        or (arr[:, 3] >= h - pad).any()
    )
    return touches, [round(float(v), 1) for v in np.median(arr, axis=0)]


def person_tracks(clip: str, model, size, device, args) -> tuple[list, int, int]:
    """Person tracks for one clip, plus the two denominators.

    **`k1` is passed now and was not before.** This function used to hold its own copy of
    the loop with no lens correction, while `site_events.py` held the same loop with one.
    The tracker associates on IoU, so that divergence could change which observations get
    linked -- and it is this script that produced the 48 candidate spans. The shared loop
    is `analytics/clip_tracks.track_clip`; it has no default for `k1`, which is what stops
    the two from drifting apart again.
    """
    tracker = Tracker(iou_threshold=args.iou, max_age=args.max_age, min_hits=args.min_hits)
    out = track_clip(
        clip,
        model,
        size,
        device,
        tracker,
        frames=frames,
        preprocess=preprocess,
        probe=probe,
        fps=args.fps,
        score_thr=args.score_thr,
        k1=args.k1,
        max_frames=args.max_frames,
    )
    return out.tracks, out.frames, out.detections


def dump(clip: str, spans: list[tuple[int, int]], out_dir: Path, args) -> None:
    """Write the frames of every candidate span, so a human can look rather than trust."""
    src_w, src_h, _ = probe(clip)
    wanted = {i for s, e in spans for i in range(max(s - 2, 0), e + 3)}
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, frame in enumerate(frames(clip, src_w, src_h, args.fps)):
        if i in wanted:
            Image.fromarray(frame).save(out_dir / f"{i:05d}.jpg", quality=88)
        if wanted and i > max(wanted):
            break


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = load_config(args.config)
    device = pick_device(cfg.get("device"))
    model = build_model(cfg).to(device).eval()
    model.load_state_dict(select_weights(load_checkpoint(args.checkpoint), "ema"))
    size = cfg["data"]["input_size"]
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    print(__doc__.strip().splitlines()[0])
    print(f"checkpoint: {args.checkpoint} (person boxes come from its COCO-trained head)")
    print(
        f"sampling at {args.fps} fps. These clips report r_frame_rate 30/1 and actually "
        "hold 7-8 fps of content, counted; `probe` reads avg_frame_rate for that "
        "reason -- see data/video._fps.\n"
    )

    report: dict = {
        "settings": report_settings(args),
        "what_this_is_not": (
            "a fall count. `fall_candidate` is a box-shape proxy that fires on bending to "
            "a low shelf and on legs occluded by a fixture. Every span below needs a human."
        ),
        "clips": [],
    }
    totals = {"clips": 0, "frames": 0, "detections": 0, "tracks": 0, "candidates": 0}
    for clip in args.clips:
        camera = Path(clip).parent.name
        session = Path(clip).stem
        src_w, src_h, src_fps = probe(clip)
        tracks, n_frames, detections = person_tracks(clip, model, size, device, args)
        got = fall_candidates(
            tracks,
            args.fps,
            camera,
            aspect_thr=args.aspect,
            height_ratio=args.height_ratio,
            sustained_seconds=args.sustained,
        )
        lengths = [len(t.frames) for t in tracks]
        by_id = {t.track_id: t for t in tracks}
        rows = []
        for e in got:
            row = e.as_row()
            track = by_id.get(e.track_ids[0])
            if track is not None:
                touches, box = border_contact(track, e.frame_start, e.frame_end, src_w, src_h)
                row["touches_frame_border"] = touches
                row["median_box"] = box
            rows.append(row)
        entry = {
            "camera": camera,
            "session": session,
            "source_fps": round(src_fps, 3),
            "frames": n_frames,
            "person_detections": detections,
            "tracks": len(tracks),
            "median_track_frames": float(np.median(lengths)) if lengths else None,
            "candidates": rows,
        }
        report["clips"].append(entry)
        totals["clips"] += 1
        totals["frames"] += n_frames
        totals["detections"] += detections
        totals["tracks"] += len(tracks)
        totals["candidates"] += len(got)
        flag = f"  <-- {len(got)} candidate(s)" if got else ""
        print(
            f"{camera:18s} {session[-13:]}  frames {n_frames:5d}  person boxes "
            f"{detections:6d}  tracks {len(tracks):4d}{flag}"
        )
        if got and args.dump_frames:
            spans = [(e.frame_start, e.frame_end) for e in got]
            dump(clip, spans, out_root / camera / session, args)
        (out_root / "candidates.json").write_text(json.dumps(report, indent=2))

    report["totals"] = totals
    (out_root / "candidates.json").write_text(json.dumps(report, indent=2))
    print(
        f"\n{totals['candidates']} candidate spans over {totals['clips']} clips, "
        f"{totals['frames']} sampled frames, {totals['detections']} person boxes, "
        f"{totals['tracks']} tracks."
    )
    print("None of those is a fall until somebody looks at it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
