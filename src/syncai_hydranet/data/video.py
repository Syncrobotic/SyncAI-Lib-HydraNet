"""Decoding a video into frames, which is a data question and not a CLI one.

`probe` and `frames` lived in `cli/infer_video.py` and were imported from there by seven
scripts -- `sam3_prelabel`, `mine_fall_candidates`, `retail_flow`, `annotation_batch`,
`site_events`, `track_review`, `fit_camera_from_people`. Seven consumers reaching up into
a command-line entry point for a primitive is the one layering violation the package
graph had: `cli` is the top layer and nothing should depend on it.

Nothing about the behaviour changes. `cli/infer_video` re-exports both names so any
caller that still imports them from there keeps working, and the ffmpeg quirks they
encode -- rotation metadata spelled three ways, a rawvideo pipe rather than a decoder
dependency -- are the reason this is worth having in one place at all.
"""

from __future__ import annotations

import json
import subprocess

import numpy as np


def probe(path: str) -> tuple[int, int, float]:
    """Return display width, height and fps, accounting for rotation metadata."""
    # `-show_streams` rather than `-show_entries`, because the section holding rotation
    # has been spelled three ways across the ffmpeg versions this has to run on:
    # `stream_side_data_list` on 4.x, `stream_side_data` on 7.x, and naming the wrong
    # one is not a missing field but a hard `Invalid argument` exit -- ffprobe refuses
    # the whole invocation, so every video path in the project dies on the ffmpeg the
    # distro happens to ship (Ubuntu 22.04 carries 4.4). `-show_streams` needs no
    # section name, emits `side_data_list` on both, and the parsing below is unchanged.
    out = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_streams",
            "-of",
            "json",
            path,
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    st = json.loads(out)["streams"][0]
    w, h = int(st["width"]), int(st["height"])
    num, _, den = st.get("r_frame_rate", "30/1").partition("/")
    fps = float(num) / float(den or 1)
    rot = 0
    for sd in st.get("side_data_list", []):
        if "rotation" in sd:
            rot = int(sd["rotation"])
    if abs(rot) % 180 == 90:  # ffmpeg autorotates on decode, swapping the axes
        w, h = h, w
    return w, h, fps


def frames(path: str, w: int, h: int, stride_fps: float | None):
    """Yield RGB frames from a rawvideo pipe."""
    vf = f"fps={stride_fps}" if stride_fps else "null"
    proc = subprocess.Popen(
        [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            path,
            "-vf",
            vf,
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-",
        ],
        stdout=subprocess.PIPE,
    )
    # `Popen.stdout` is Optional because it is None unless `stdout=PIPE` was asked for.
    # It was, one line up. Bind it once so the fact is stated where it is true instead
    # of being re-derived at every read.
    stdout = proc.stdout
    assert stdout is not None
    n = w * h * 3
    try:
        while True:
            buf = stdout.read(n)
            if len(buf) < n:
                break
            yield np.frombuffer(buf, np.uint8).reshape(h, w, 3)
    finally:
        stdout.close()
        proc.wait()
