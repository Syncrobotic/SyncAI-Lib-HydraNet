"""What `probe()` believes about a clip's frame rate, and why it is not `r_frame_rate`.

`r_frame_rate` is the container's *nominal* rate. On this project's site corpus it says
`30/1` while the stream holds a quarter of that -- Kaohsiung-cam04 is 2,400 frames over
300.1 s (8.0 fps), Taichung-cam01 is 2,130 over 304.3 s (7.0 fps), both counted with
`ffprobe -count_frames`. Two scripts found this independently and each wrote a private
`source_fps` around it rather than fixing the primitive, so the CLIs kept believing 30.

The cost was not bookkeeping: `hydranet-infer-video` defaults its output rate to this
number, so the default render of a site clip played four times too fast.

These tests drive the parsing directly with canned ffprobe payloads. The rates that
matter cannot be reproduced by a generated test clip -- a clip written at a constant 8
fps has `r_frame_rate == avg_frame_rate` and would pass under the old code too.
"""

from __future__ import annotations

import json

import pytest

from syncai_hydranet.data import video


def _probe(monkeypatch, stream: dict) -> tuple[int, int, float]:
    """Run `probe()` against one canned ffprobe stream object."""

    class _Out:
        stdout = json.dumps({"streams": [stream]})

    def _run(*_args, **_kwargs):
        return _Out()

    monkeypatch.setattr(video.subprocess, "run", _run)
    return video.probe("clip.mp4")


BASE = {"width": 1920, "height": 1080}


def test_the_nominal_rate_loses_to_the_measured_one(monkeypatch):
    """The site corpus in one assertion: 30/1 nominal, 8 fps of content."""
    w, h, fps = _probe(monkeypatch, {**BASE, "r_frame_rate": "30/1", "avg_frame_rate": "8/1"})
    assert (w, h) == (1920, 1080)
    assert fps == pytest.approx(8.0)


def test_a_fractional_average_survives_the_division(monkeypatch):
    """Taichung-cam01: 2130 frames / 304.3 s, which ffprobe writes as a ratio."""
    _, _, fps = _probe(
        monkeypatch, {**BASE, "r_frame_rate": "30/1", "avg_frame_rate": "21300/3043"}
    )
    assert fps == pytest.approx(7.0, abs=0.01)


@pytest.mark.parametrize("avg", ["0/0", "0/1", "", "N/A", None])
def test_an_unmeasurable_average_falls_back_to_the_nominal_rate(monkeypatch, avg):
    """A live stream or fragmented mp4 has no average; `0/0` is what ffprobe writes.

    Falling back rather than raising: `r_frame_rate` is a worse answer, not no answer,
    and every caller can override it with `--fps`.
    """
    stream = {**BASE, "r_frame_rate": "25/1"}
    if avg is not None:
        stream["avg_frame_rate"] = avg
    _, _, fps = _probe(monkeypatch, stream)
    assert fps == pytest.approx(25.0)


def test_neither_field_present_is_still_a_number(monkeypatch):
    """`probe()`'s contract is three numbers. A caller cannot letterbox against None."""
    _, _, fps = _probe(monkeypatch, dict(BASE))
    assert fps == pytest.approx(video.FALLBACK_FPS)


def test_rotation_still_swaps_the_axes(monkeypatch):
    """Regression guard: the fps change must not disturb the sideways-camera path.

    One camera in the corpus is mounted sideways. ffmpeg autorotates on decode, so the
    raw pipe hands back the swapped size and `frames()` reshapes against these numbers.
    """
    w, h, _ = _probe(
        monkeypatch,
        {**BASE, "avg_frame_rate": "8/1", "side_data_list": [{"rotation": -90}]},
    )
    assert (w, h) == (1080, 1920)
