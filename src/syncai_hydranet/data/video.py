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
import tempfile

import numpy as np

FALLBACK_FPS = 30.0


def _rate(spec: object) -> float | None:
    """``"8/1"`` -> 8.0. None for absent, malformed, or the ``0/0`` ffprobe writes
    when the field has no answer -- a live stream or a fragmented mp4 has no average."""
    if not isinstance(spec, str):
        return None
    num, _, den = spec.partition("/")
    try:
        n, d = float(num), float(den or 1)
    except ValueError:
        return None
    if d == 0 or n <= 0:
        return None
    return n / d


def _fps(stream: dict) -> float:
    """The stream's real frame rate: ``avg_frame_rate``, then ``r_frame_rate``, then 30.

    This read ``r_frame_rate`` for a long time and that field lies on the site corpus.
    It is the container's *nominal* rate, and these cameras write ``30/1`` into it while
    holding a quarter of that:

    * Kaohsiung-cam04 -- 2,400 frames over 300.1 s, **8.0 fps**
    * Taichung-cam01 -- 2,130 frames over 304.3 s, **7.0 fps**

    Both counted with ``ffprobe -count_frames``; ``avg_frame_rate`` matched the counted
    value to three decimals on each and costs no decode. Two scripts had already found
    this independently and each grew a private `source_fps` to work around it
    (`scripts/mine_fall_candidates.py`, `scripts/offline_tracks.py`) -- the workaround
    was written twice and the primitive was never fixed.

    What the lie cost, beyond bookkeeping. `cli/infer_video` takes ``--fps`` as *both*
    the sampling rate and the output rate and defaults it to this value, so the default
    invocation on a site clip encoded 7-8 fps of content at 30 fps: a render that plays
    four times too fast, handed to a customer. It also gates resampling on
    ``args.fps < src_fps``, so ``--fps 10`` passed a comparison against 30 and then had
    ffmpeg's ``fps`` filter *duplicate* frames up from 8 -- ten frames of evidence per
    second where the camera recorded eight.

    30.0 as the last resort rather than raising: a stream with neither field is a stream
    ffprobe could not measure, and the caller's own ``--fps`` overrides this anyway.
    """
    avg = _rate(stream.get("avg_frame_rate"))
    return avg if avg is not None else (_rate(stream.get("r_frame_rate")) or FALLBACK_FPS)


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
    fps = _fps(st)
    rot = 0
    for sd in st.get("side_data_list", []):
        if "rotation" in sd:
            rot = int(sd["rotation"])
    if abs(rot) % 180 == 90:  # ffmpeg autorotates on decode, swapping the axes
        w, h = h, w
    return w, h, fps


class DecodeError(RuntimeError):
    """ffmpeg stopped before the stream did. Raised by `frames()`; never returned."""


def frames(path: str, w: int, h: int, stride_fps: float | None):
    """Yield RGB frames from a rawvideo pipe. Raises `DecodeError` on a short decode.

    **The raise is the point.** This loop reads fixed-size frames until a read comes
    back short, and a short read is how a finished video ends *and* how a truncated one
    ends. Nothing looked at ffmpeg's exit status, so the two were the same event: a clip
    that died a third of the way through produced a generator that stopped cleanly, and
    every one of the fifteen call sites recorded a successful pass over a third of the
    footage. `site_events` wrote its `events.json`, `sam3_prelabel` wrote its pre-labels,
    and neither the file nor the console said which frames never existed. Silent
    truncation of customer footage is the failure this project is least able to audit
    after the fact, because there is no artefact of it.

    An early `break` by the caller is a different thing and is not an error. `--max-frames`
    and the `if i >= n` in half the scripts are deliberate, so the exit status is only
    consulted when *this* loop reached the end of the stream. When the caller stops first,
    ffmpeg is terminated rather than waited on -- it still has frames to write and would
    otherwise sit in a blocked write until SIGPIPE reached it.

    ffmpeg's stderr goes to a temporary file rather than a pipe. Its message is what makes
    the raise actionable ("moov atom not found" and "Invalid NAL unit size" are different
    problems), and a pipe nobody drains until the process exits is a deadlock whenever the
    error is per-frame rather than per-file.
    """
    vf = f"fps={stride_fps}" if stride_fps else "null"
    with tempfile.TemporaryFile() as errors:
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
            stderr=errors,
        )
        # `Popen.stdout` is Optional because it is None unless `stdout=PIPE` was asked
        # for. It was, one line up. Bind it once so the fact is stated where it is true
        # instead of being re-derived at every read.
        stdout = proc.stdout
        assert stdout is not None
        n = w * h * 3
        n_read = 0
        tail = -1  # bytes of the final short read; -1 while the loop is still running
        try:
            while True:
                buf = stdout.read(n)
                if len(buf) < n:
                    tail = len(buf)
                    break
                n_read += 1
                yield np.frombuffer(buf, np.uint8).reshape(h, w, 3)
        finally:
            stdout.close()
            if tail < 0:
                # The caller stopped early, so ffmpeg is still producing. Terminating
                # beats waiting: it has a full pipe buffer to drain into a reader that
                # has gone.
                proc.terminate()
            code = proc.wait()
            errors.seek(0)
            message = errors.read().decode("utf-8", "replace").strip()
            if tail >= 0 and (code != 0 or tail > 0):
                partial = f", {tail} trailing bytes of an incomplete frame" if tail else ""
                raise DecodeError(
                    f"ffmpeg stopped after {n_read} whole frames of {path} "
                    f"(exit {code}{partial}). The frames it did emit are a prefix of "
                    f"the clip, not the clip.\n"
                    + (message or "ffmpeg said nothing; re-run without `-v error`.")
                )
