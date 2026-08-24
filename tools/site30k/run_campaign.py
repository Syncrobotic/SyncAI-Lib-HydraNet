#!/usr/bin/env python3
"""Run the site30k plan: one process per camera-date unit, N units at a time.

Isolation is by PROCESS, deliberately. 4,122 clips came off a store's NVR and some of
them are truncated; `data.video.frames` raises on a short decode, which is the correct
behaviour and would otherwise take a whole worker's queue down with it. A unit that dies
here loses only its own frames, is written to failures.jsonl with ffmpeg's message, and
the next unit starts.

Every unit is resumable: the recipe skips a clip whose masks already exist, so re-running
this after a crash, a reboot, or a deliberate stop costs only the units that were
in flight.

  python tools/site30k/run_campaign.py --plan runs/site30k_qa/campaign_plan.json \\
      --out datasets/site30k_v1 --workers 6

  --smoke N   one unit per camera, N frames per clip, into a separate directory: the
              pass that has to be looked at before 30,000 frames are labelled by a
              recipe reviewed on three cameras.
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path("/home/paul/SyncAI-Lib-HydraNet")
RECIPE = ROOT / "tools/site30k/recipe.py"
PYTHON = ROOT / ".venv/bin/python"
PULL = "datasets/studioa_pull_site30k"


def unit_targets(unit: dict, frames: int) -> list[str]:
    return [f"{unit['camera']}:{stem}:{frames}" for stem in unit["clips"]]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--preview-stride", type=int, default=20)
    ap.add_argument("--smoke", type=int, default=0,
                    help="frames per clip; runs ONE unit per camera and stops")
    ap.add_argument("--limit", type=int, default=0, help="first N units only")
    args = ap.parse_args()

    plan = json.loads(args.plan.read_text())
    units = plan["plan"]
    if args.smoke:
        seen, picked = set(), []
        for u in units:
            if u["camera"] in seen:
                continue
            seen.add(u["camera"])
            picked.append(u)
        units = picked
    # Round-robin over cameras rather than the plan's alphabetical order. A run this
    # long will be looked at before it finishes, and possibly stopped: alphabetical order
    # means an interruption leaves every frame on the first camera and none on the other
    # eight, which is the one shape of partial dataset that cannot be reviewed or trained
    # on. Interleaved, whatever fraction is done covers all nine.
    by_cam: dict[str, list[dict]] = {}
    for u in units:
        by_cam.setdefault(u["camera"], []).append(u)
    interleaved = []
    while any(by_cam.values()):
        for cam in list(by_cam):
            if by_cam[cam]:
                interleaved.append(by_cam[cam].pop(0))
    units = interleaved
    if args.limit:
        units = units[:args.limit]
    frames = args.smoke or plan["frames_per_clip"]
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "logs").mkdir(exist_ok=True)
    failures = (args.out / "failures.jsonl").open("a")
    progress = (args.out / "progress.jsonl").open("a")

    print(f"{len(units)} units, {frames} frames per clip, {args.workers} workers -> "
          f"{args.out}", flush=True)
    running: list[tuple[subprocess.Popen, dict, float, object]] = []
    queue = list(units)
    done = 0
    t0 = time.time()
    while queue or running:
        while queue and len(running) < args.workers:
            u = queue.pop(0)
            tag = f"{u['camera']}_{u['date']}"
            # Append, never truncate: a resume re-runs every unit, and "w" threw away
            # the original run's per-frame log lines for all 261 units on 2026-08-20.
            # The failure tail below reads the last 800 bytes, so it still sees this run.
            log = (args.out / "logs" / f"{tag}.log").open("a")
            log.write(f"\n===== run started {time.strftime('%Y-%m-%d %H:%M:%S')}, "
                      f"{frames} frames per clip =====\n")
            log.flush()
            cmd = [str(PYTHON), str(RECIPE), "--resume",
                   f"--preview-stride={args.preview_stride}",
                   str(args.out), str(frames), *unit_targets(u, frames)]
            env = dict(os.environ, PYTHONUNBUFFERED="1",
                       SITE30K_CLIP_ROOT=PULL,
                       SITE30K_PLATE_ROOT="datasets/studioa_static_site30k")
            p = subprocess.Popen(["nice", "-n", "10", *cmd], stdout=log,
                                 stderr=subprocess.STDOUT, env=env)
            running.append((p, u, time.time(), log))
        time.sleep(2)
        for entry in list(running):
            p, u, started, log = entry
            if p.poll() is None:
                continue
            running.remove(entry)
            log.close()
            done += 1
            rec = {"camera": u["camera"], "date": u["date"], "rc": p.returncode,
                   "seconds": round(time.time() - started, 1), "frames": u["frames"]}
            progress.write(json.dumps(rec) + "\n")
            progress.flush()
            if p.returncode != 0:
                tail = (args.out / "logs" / f"{u['camera']}_{u['date']}.log").read_text()[-800:]
                failures.write(json.dumps({**rec, "tail": tail}) + "\n")
                failures.flush()
            rate = (time.time() - t0) / max(done, 1)
            left = (len(queue) + len(running)) * rate / max(args.workers, 1)
            print(f"[{done}/{len(units)}] {u['camera']} {u['date']} rc={p.returncode} "
                  f"{rec['seconds']:.0f}s   eta {left / 3600:.1f} h", flush=True)
    masks = args.out / "masks"
    n_masks = len(list(masks.glob("*.png"))) if masks.exists() else 0
    print(f"\nall units finished in {(time.time() - t0) / 3600:.2f} h; "
          f"{n_masks} masks in {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
