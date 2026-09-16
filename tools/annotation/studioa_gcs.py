#!/usr/bin/env python3
"""Bounded, generation-pinned GCS intake from existing StudioA training cameras only."""

from __future__ import annotations

import argparse
import base64
import fcntl
import hashlib
import json
import shutil
import signal
import subprocess
import sys
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image

from syncai_hydranet.data.studioa_review import digest, write_json
from syncai_hydranet.data.studioa_supervision import check_supervision

ROOT = Path(__file__).resolve().parents[2]
PROJECT = "syncrobotic-aisw"
BUCKET = "gs://studioa"


def command(args, timeout=120):
    return subprocess.run(
        args, check=True, capture_output=True, text=True, timeout=timeout
    ).stdout


def camera_prefix(camera):
    store, number = camera.split("-cam")
    if store not in ("Kaohsiung", "Taichung") or not number.isdigit():
        raise ValueError("intake is restricted to the two source stores")
    return f"{store.lower()}-{int(number)}"


def clip_time(uri):
    # This bucket uses local-looking YYYYMMDD_HHMMSS filenames, unlike the older
    # studioa-recording bucket. This is a sampling hint, never a verified event time.
    return datetime.strptime(
        Path(uri).stem.rsplit("_", 2)[1] + Path(uri).stem.rsplit("_", 2)[2], "%Y%m%d%H%M%S"
    )


def choose_clip(uris, date, target):
    requested = datetime.fromisoformat(f"{date}T{target}:00")
    candidates = [(abs((clip_time(uri) - requested).total_seconds()), uri) for uri in uris]
    gap, uri = min(candidates)
    if gap > 900:
        raise ValueError("no clip within 15 filename-minutes of requested slot")
    return uri, gap


def plan(source, out, cameras, dates, times, max_bytes):
    manifest = check_supervision(source)
    roles = {}
    for frame in manifest["frames"]:
        roles.setdefault(frame["camera"], set()).add(
            manifest["folds"]["Tao-Hsin"]["assignments"][frame["id"]]
        )
    if (
        not cameras
        or len(set(cameras)) != len(cameras)
        or len(dates) != len(times)
        or len(set(dates)) != len(dates)
    ):
        raise ValueError("unique cameras and paired date/time slots are required")
    if len(cameras) * len(dates) > 24 or any(roles.get(cam) != {"train"} for cam in cameras):
        raise ValueError("at most 24 clips, exclusively existing source-train cameras")
    selected: list[dict] = []
    for date, time in zip(dates, times, strict=True):
        for camera in cameras:
            prefix = f"{BUCKET}/{camera_prefix(camera)}/{date.replace('-', '/')}/"
            listing = command(
                ["gcloud", "storage", "ls", prefix, f"--project={PROJECT}", "--quiet"]
            )
            uri, gap = choose_clip(
                [s for s in listing.splitlines() if s.endswith(".mp4")], date, time
            )
            obj = json.loads(
                command(
                    [
                        "gcloud",
                        "storage",
                        "objects",
                        "describe",
                        uri,
                        f"--project={PROJECT}",
                        "--format=json",
                        "--quiet",
                    ]
                )
            )
            selected.append(
                {
                    "camera": camera,
                    "store": camera.split("-cam")[0],
                    "original_split": "train",
                    "target_date": date,
                    "target_filename_time": time,
                    "filename_gap_s": gap,
                    "uri": uri,
                    "object": obj,
                }
            )
    if len({s["uri"] for s in selected}) != len(selected):
        raise ValueError("duplicate requested clips")
    total = sum(int(s["object"]["size"]) for s in selected)
    if not 0 < total <= max_bytes:
        raise ValueError(f"download budget exceeded: {total} > {max_bytes}")
    out.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(__file__, out / "worker.py")
    write_json(
        out / "plan.json",
        {
            "schema": "studioa.gcs-intake.v1",
            "source_manifest_sha256": digest(source / "manifest.json"),
            "source_manifest": str((source / "manifest.json").resolve()),
            "held_out": "Tao-Hsin",
            "training_cameras": cameras,
            "clips": selected,
            "total_bytes": total,
            "max_bytes": max_bytes,
            "offsets_s": [30, 330],
            "timestamp_policy": "approximate filename time; visually verify burnt-in clock",
            "worker_sha256": digest(out / "worker.py"),
            "git_commit": command(["git", "rev-parse", "HEAD"]).strip(),
        },
    )
    write_json(
        out / "status.json", {"status": "planned", "clips": len(selected), "bytes": total}
    )


def run(out):
    lock = (out / "worker.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)

    def stopped(_signal, _frame):
        raise InterruptedError("intake stopped; pinned complete clips remain resumable")

    signal.signal(signal.SIGTERM, stopped)
    frames, clips, seen = [], [], set()
    job = json.loads((out / "plan.json").read_text())
    if digest(out / "worker.py") != job["worker_sha256"]:
        raise ValueError("frozen intake worker changed")
    source = Path(job["source_manifest"])
    if digest(source) != job["source_manifest_sha256"]:
        raise ValueError("camera split manifest changed")
    seen.update(f["pixel_sha256"] for f in json.loads(source.read_text())["frames"])
    for folder in ("clips", "images"):
        (out / folder).mkdir(exist_ok=True)
    try:
        for i, item in enumerate(job["clips"]):
            write_json(
                out / "status.json",
                {
                    "status": "downloading",
                    "completed_clips": i,
                    "total_clips": len(job["clips"]),
                    "camera": item["camera"],
                },
            )
            dest = out / "clips" / Path(item["uri"]).name
            if not dest.exists():
                part = dest.with_suffix(".partial.mp4")
                command(
                    [
                        "gcloud",
                        "storage",
                        "cp",
                        item["object"]["storage_url"],
                        str(part),
                        f"--project={PROJECT}",
                        "--quiet",
                    ],
                    timeout=600,
                )
                if part.stat().st_size != int(item["object"]["size"]):
                    raise ValueError("download size differs from pinned object")
                part.rename(dest)
            with dest.open("rb") as stream:
                md5 = base64.b64encode(hashlib.file_digest(stream, "md5").digest()).decode()
            if md5 != item["object"]["md5_hash"]:
                raise ValueError("local clip differs from pinned GCS object checksum")
            metadata = json.loads(
                command(
                    [
                        "ffprobe",
                        "-v",
                        "error",
                        "-show_entries",
                        "format=duration:stream=codec_type,width,height",
                        "-of",
                        "json",
                        str(dest),
                    ]
                )
            )
            if float(metadata["format"]["duration"]) < max(job["offsets_s"]) + 1:
                raise ValueError("clip is too short for declared extraction offsets")
            clips.append(
                {
                    **item,
                    "local_file": str(dest.relative_to(out)),
                    "sha256": digest(dest),
                    "ffprobe": metadata,
                }
            )
            for offset in job["offsets_s"]:
                fid = (
                    f"gcs-{item['target_date'].replace('-', '')}-{item['camera']}-t{offset:03d}"
                )
                name = f"images/{fid}.jpg"
                path = out / name
                if not path.exists():
                    command(
                        [
                            "ffmpeg",
                            "-nostdin",
                            "-hide_banner",
                            "-loglevel",
                            "error",
                            "-ss",
                            str(offset),
                            "-i",
                            str(dest),
                            "-frames:v",
                            "1",
                            "-q:v",
                            "2",
                            str(path),
                        ]
                    )
                with Image.open(path) as image:
                    rgb = np.asarray(image.convert("RGB"))
                pixel_hash = hashlib.sha256(str(rgb.shape).encode() + rgb.tobytes()).hexdigest()
                if pixel_hash in seen:
                    continue
                seen.add(pixel_hash)
                frames.append(
                    {
                        "id": fid,
                        "camera": item["camera"],
                        "store": item["store"],
                        "original_split": "train",
                        "image": name,
                        "image_sha256": digest(path),
                        "pixel_sha256": pixel_hash,
                        "image_size_px": list(rgb.shape[1::-1]),
                        "clip_index": i,
                        "offset_s": offset,
                        "role": "candidate_pending_AI_review",
                    }
                )
        write_json(
            out / "manifest.json",
            {
                "schema": job["schema"],
                "plan_sha256": digest(out / "plan.json"),
                "frames": frames,
                "clips": clips,
                "semantic_or_detection_labels": False,
            },
        )
        write_json(
            out / "report.json",
            {
                "status": "completed",
                "frames": len(frames),
                "clips": len(clips),
                "download_bytes": job["total_bytes"],
                "outputs": {
                    str(p.relative_to(out)): digest(p)
                    for p in sorted(out.rglob("*"))
                    if p.is_file()
                    and p.name
                    not in ("worker.lock", "worker.log", "status.json", "report.json")
                },
            },
        )
        write_json(
            out / "status.json",
            {"status": "completed", "frames": len(frames), "clips": len(clips)},
        )
    except BaseException:
        write_json(out / "status.json", {"status": "failed", "error": traceback.format_exc()})
        raise
    finally:
        lock.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("plan", "run"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--cameras", nargs="+")
    parser.add_argument("--dates", nargs="+")
    parser.add_argument("--times", nargs="+")
    parser.add_argument("--max-bytes", type=int, default=1_000_000_000)
    args = parser.parse_args()
    if args.action == "plan":
        if not all((args.source, args.cameras, args.dates, args.times)):
            parser.error("plan requires source, cameras, dates and paired times")
        plan(
            args.source,
            args.out.resolve(),
            args.cameras,
            args.dates,
            args.times,
            args.max_bytes,
        )
    else:
        target = args.out.resolve() / "worker.py"
        if Path(__file__).resolve() != target:
            subprocess.run(
                [sys.executable, str(target), "run", "--out", str(args.out.resolve())],
                check=True,
            )
        else:
            run(args.out.resolve())


if __name__ == "__main__":
    main()
