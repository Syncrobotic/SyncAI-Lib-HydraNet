#!/usr/bin/env python3
"""Does the shipped detector see people in an empty shop at night, and does the veto help?

    python3 scripts/night_ghosts.py --out runs/night_ghosts01

`data/night_person.py` measured the night problem on the **teacher** -- Grounding DINO at
0.35 put a box on an empty shuttered store on 13 of 42 cameras, worst 0.594 -- and built a
veto that drops a box whose pixels never moved against that camera's own midnight plate.
It removes 72 of 72 false boxes and 0 of 12 verified people, and `test_night_person_veto.py`
holds those numbers.

**That veto has never run on the serving path.** `night_person` is imported by the
teacher, the box pass, the campaign and the filter script -- everything that makes
*training data* -- and by nothing under `serving/`. So the questions it answers about the
teacher are open about the student, and they are not the same model: the student was
trained on data the veto had already cleaned, which could mean it never learned the ghosts,
or could mean nothing at all. Neither is knowable without looking.

This looks. For each commissioned camera it runs the shipped detector over that camera's
own 23:58 store-local clip -- an empty shop, no staff, shutters down -- and counts
`person` boxes at the shipped 0.35. Every box is a false positive by construction,
because there is nobody there. Then it applies the same veto and counts what survives.

Three outcomes and each means something different:

* **no boxes at all** -- the student did not inherit the teacher's night failure, and
  `after_hours_person` needs no veto at serving time.
* **boxes that the veto removes** -- the failure is inherited and the fix already exists;
  wire `night_person` into `serving/` and the alert is usable.
* **boxes the veto keeps** -- something moves that is not a person, and neither the plate
  nor the threshold reaches it. That would be the case worth a journal entry.

The night clip (23:58 store-local -- see NIGHT_PREFIX below, and the note there on the
03:00-labelled attempt that was really 11:00) is chosen by name rather than by a luma
test on purpose: `gdino_person_boxes` records per-frame luma exactly because an IR frame
is monochrome by day too, and a threshold on brightness is a second thing to get wrong
when the filename already says when it was recorded.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from PIL import Image  # noqa: E402

from syncai_hydranet.analytics.clip_tracks import PERSON, to_source_pixels  # noqa: E402
from syncai_hydranet.analytics.delivery import report_settings  # noqa: E402
from syncai_hydranet.config import load_config  # noqa: E402
from syncai_hydranet.data.night_person import NightPersonVeto  # noqa: E402
from syncai_hydranet.data.video import frames as decode_frames  # noqa: E402
from syncai_hydranet.data.video import probe  # noqa: E402
from syncai_hydranet.models.hydranet import build_model  # noqa: E402
from syncai_hydranet.utils.checkpoint import load_checkpoint, select_weights  # noqa: E402
from syncai_hydranet.utils.device import pick_device  # noqa: E402
from syncai_hydranet.utils.visualize import preprocess  # noqa: E402

CLIPS = ROOT / "datasets/studioa_clips"
STATIC = ROOT / "datasets/studioa_static"
# **UTC, and the shops are UTC+8.** `archive_20260816-1558...` is 23:58 store-local, which
# is the slot `night_person.py` and the fleet re-check both measured. The first version of
# this script used `-0300` on the reasoning that 3 am is night, ran on a shop at 11:00
# local, and reported 45 "ghosts" in an empty store that were forty-five people. The trap
# is documented twice already -- `slot_of`'s docstring and `pull_studioa.py`, which pulled
# a closed store at "16:00" -- and it is written a third time here because reading it is
# evidently not the same as not falling into it.
NIGHT_PREFIX = "archive_20260816-1558"


def night_clip(camera: str) -> Path | None:
    hits = (
        sorted((CLIPS / camera).glob(f"{NIGHT_PREFIX}*.mp4"))
        if (CLIPS / camera).is_dir()
        else []
    )
    return hits[0] if hits else None


def run_camera(camera: str, clip: Path, model, cfg, device, veto, args) -> dict:
    src_w, src_h, _ = probe(str(clip))
    size = cfg["data"]["input_size"]
    person_label = args.person_label if args.person_label >= 0 else PERSON
    # The veto keys off the corpus filename convention: `<camera>__<session>__<frame>.jpg`.
    stem = f"{camera}__{clip.stem}__00000.jpg"
    raw = kept = 0
    per_frame = []
    for n, frame in enumerate(decode_frames(str(clip), src_w, src_h, args.fps)):
        if n >= args.frames:
            break
        x, _canvas, region = preprocess(Image.fromarray(frame), size)
        with torch.no_grad():
            res = model.predict(x.to(device), score_thr=args.score_thr)
        det = res["detection"][0]
        if not len(det.get("labels", [])):
            per_frame.append(0)
            continue
        lab = det["labels"].cpu().numpy()
        keep = lab == person_label
        boxes = to_source_pixels(det["boxes"].cpu().numpy()[keep], region, src_w, src_h)
        scores = det["scores"].cpu().numpy()[keep]
        raw += len(boxes)
        per_frame.append(len(boxes))
        xywh = [
            ((b[0], b[1], b[2] - b[0], b[3] - b[1]), float(s))
            for b, s in zip(boxes, scores, strict=True)
        ]
        kept += sum(1 for d in veto.apply(stem, xywh, src_w, src_h) if d.keep)
    return {
        "camera": camera,
        "clip": clip.name,
        "frames": len(per_frame),
        "veto_status": veto.status(camera),
        "person_boxes": raw,
        "kept_after_veto": kept,
        "removed_by_veto": raw - kept,
        "worst_frame": max(per_frame) if per_frame else 0,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("runs/night_ghosts01"))
    ap.add_argument("--cameras", nargs="*")
    ap.add_argument("--config", default="configs/hydranet_retail_pose02.yaml")
    ap.add_argument("--checkpoint", default="runs/hydranet_retail_pose02/last.pt")
    ap.add_argument("--frames", type=int, default=150, help="30 s at 5 fps of an empty shop")
    ap.add_argument("--fps", type=float, default=5.0)
    ap.add_argument("--score-thr", type=float, default=0.35, help="the shipped birth edge")
    ap.add_argument("--person-label", type=int, default=-1)
    args = ap.parse_args()

    cams = args.cameras or sorted(p.name for p in CLIPS.iterdir() if p.is_dir())
    device = pick_device(None)
    cfg = load_config(args.config)
    model = build_model(cfg).to(device).eval()
    model.load_state_dict(select_weights(load_checkpoint(args.checkpoint), "ema"))
    veto = NightPersonVeto(STATIC)

    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    for camera in cams:
        clip = night_clip(camera)
        if clip is None:
            continue
        row = run_camera(camera, clip, model, cfg, device, veto, args)
        rows.append(row)
        print(
            f"{camera:<18} {row['frames']:>4} frames  {row['person_boxes']:>4} person boxes"
            f"  worst frame {row['worst_frame']:>2}  veto removes {row['removed_by_veto']:>4}"
            f"  -> {row['kept_after_veto']:>3} kept   [{row['veto_status']}]"
        )
    total = {
        k: sum(r[k] for r in rows)
        for k in ("person_boxes", "kept_after_veto", "removed_by_veto")
    }
    print(f"\n{len(rows)} cameras at 23:58 local: {total['person_boxes']} person boxes on "
          f"an empty shop, "
          f"{total['removed_by_veto']} removed by the plate veto, "
          f"{total['kept_after_veto']} left")  # fmt: skip
    (args.out / "fleet.json").write_text(
        json.dumps(
            {"settings": report_settings(args), "cameras": rows, "total": total}, indent=1
        )
        + "\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
