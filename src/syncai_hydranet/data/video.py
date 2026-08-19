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

`validate_inputs` and `session_names` are the same question one step earlier: what a
caller is allowed to conclude about a *list* of clips before any of them is opened. They
came out of `sam3_prelabel.py`, where they guarded one batch run; both refusals are about
paths rather than about pixels, and every caller that decodes a list of clips wants them.
"""

from __future__ import annotations

import contextlib
import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np

FALLBACK_FPS = 30.0
PROBE_TIMEOUT_S = 60


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

    One conclusion worth keeping in front of anyone who samples this corpus, from the
    docstring this replaced: a plan to recover temporal resolution for behaviour by
    sampling at 15 or 30 fps has **nothing to recover**, because there is no 30 fps of
    content. Track fragmentation cannot be fixed by sampling faster, which leaves
    appearance association as the only instrument for it rather than one of two.

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
        # A batch runs over 48 cameras. ffprobe with no timeout will sit forever on a
        # half-downloaded object or a path that turns out to be a fifo, and the symptom
        # is a pipeline that looks busy and is not. 60 s is far past what a local seek
        # costs and far short of a working session.
        timeout=PROBE_TIMEOUT_S,
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


def finish_encoder(proc: subprocess.Popen | None) -> int | None:
    """Close an encoder's stdin, wait for it, and report its exit status. Never raises.

    None in, None out, so a caller that never wrote a frame needs no special case.

    Both video CLIs pipe raw RGB into an ffmpeg they started themselves, and both used to
    get the shutdown subtly wrong in different ways. `cli/scene` closed the pipe in a
    `finally` and ignored the status; `cli/infer_video` checked nothing and had no
    `finally` at all, so any exception mid-loop left ffmpeg without an EOF -- an mp4 with
    no moov atom, unplayable, beside an orphaned process. Both then printed `done: N
    frames -> out.mp4` unconditionally, which is also what a full disk and an unwritable
    path produced.

    Not raising is deliberate: this runs from a `finally`, where raising would replace
    whatever sent the caller there. It returns the status and lets the caller decide,
    which it can only do sensibly once it knows the loop finished.
    """
    if proc is None:
        return None
    if proc.stdin is not None:
        with contextlib.suppress(BrokenPipeError, OSError):
            proc.stdin.close()
    return proc.wait()


def validate_inputs(clips: list[str]) -> None:
    """Refuse the run if any input does not exist, naming every one of them.

    Sibling of `session_names` and called beside it, for the same reason: the whole input
    list is checked before anything opens. Kept separate because that function has one job
    and a name that says so, and existence is a different job.

    What this replaces. A path that does not exist reached `probe()`, which runs ffprobe
    with `check=True`, and nothing here caught `CalledProcessError` -- so a single typo in a
    48-clip batch produced a raw traceback, aborted every clip after it, and did so *after*
    the session directory was created and the model load was already paid for. Worse, the
    file looks like it handles this: there is a `no frames decoded, skipped` branch nine
    lines further down, and it is unreachable for a missing file because `probe()` dies
    first. It is reachable for a *directory* input with no images, which takes the other
    branch. So the two input kinds degrade completely differently and reading top to bottom
    suggests otherwise.

    Limit, stated rather than implied: this catches a path that is not there. A file that
    exists and is not decodable still fails at `probe()`, because proving decodability means
    decoding. The common failure is a typo or a glob that matched nothing near what was
    meant, and that is what this refuses.
    """
    missing = [c for c in clips if not Path(c).exists()]
    if missing:
        listed = "\n".join(f"    {p}" for p in missing)
        raise SystemExit(
            f"{len(missing)} of {len(clips)} input(s) do not exist; refusing before writing "
            f"anything.\n{listed}\n"
            "Checked up front so a typo in a long batch fails now rather than part-way "
            "through, after the model load and with directories already created."
        )


def session_names(clips: list[str]) -> list[str]:
    """One output directory per input, and refuse the run if two inputs claim the same one.

    The session name is the input's filename stem, and **a filename is not an identity** --
    it is chosen by whoever wrote the footage. `gs://studioa-recording` names a clip by its
    recording timestamp, so two cameras that start recording in the same second produce the
    same stem. Four of 96 clips in one real pull did. Both wrote into one directory, the
    second silently replaced the first, and the run finished reporting a plausible per-class
    composition over the frames that happened to survive. It cost a day to notice, and only
    because a class share moved 33.33% -> 2.49% between a partial and a full pass, which is
    arithmetically impossible for an average over one more clip.

    Widening the key -- to include the parent directory, say -- makes that particular
    collision go away without making the name an identity, so two cameras can still collide
    under it and the failure stays silent. The assertion is the fix; the key is not. A
    caller whose names genuinely collide can pass symlinks that do not, which is how the
    affected clips were re-run.

    Scope, stated rather than implied: this catches collisions **within one invocation**.
    Running twice into the same `--out` with the same clip still overwrites, which is what a
    re-run should do.
    """
    names = [Path(c).stem for c in clips]
    seen: dict[str, list[str]] = {}
    for clip, name in zip(clips, names, strict=True):
        seen.setdefault(name, []).append(clip)
    clashes = {n: paths for n, paths in seen.items() if len(paths) > 1}
    if not clashes:
        return names

    detail = "\n".join(
        f"  {n}\n" + "\n".join(f"    {p}" for p in paths)
        for n, paths in sorted(clashes.items())
    )
    # Two different failures wear the same shape here, and the remedy for one is a dead end
    # for the other: distinct paths need names that differ, a repeated path needs removing.
    # Telling someone to symlink their way out of a doubled glob sends them in a circle.
    if all(len(set(paths)) == 1 for paths in clashes.values()):
        advice = "The same input is listed more than once. De-duplicate the list and re-run."
    else:
        advice = (
            "Each input needs its own output directory or one silently overwrites the other. "
            "Pass symlinks whose names differ -- e.g. <camera>__<stem>.mp4 -- and re-run."
        )
    raise SystemExit(
        f"{len(clashes)} session name(s) claimed by more than one input; refusing before "
        f"writing anything.\n{detail}\n{advice}"
    )
