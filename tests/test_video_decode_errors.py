"""A short decode has to be an error, not the end of the video.

`frames()` reads fixed-size frames until a read comes back short -- which is how a
finished clip ends and also how a truncated one ends. Nothing looked at ffmpeg's exit
status, so the two were indistinguishable: a clip that died a third of the way through
yielded a generator that stopped cleanly, and all fifteen call sites recorded a
successful pass over a third of the footage. `site_events` wrote its events.json,
`sam3_prelabel` wrote its pre-labels, and no artefact anywhere recorded which frames
never existed.

The early-`break` case is the one that makes this delicate: `--max-frames` and the
`if i >= n` in half the scripts stop the loop on purpose, and ffmpeg then dies of a
broken pipe. That must stay silent.
"""

from __future__ import annotations

import io
import shutil
import subprocess

import numpy as np
import pytest

from syncai_hydranet.data.video import DecodeError, frames, probe

W, H = 32, 24


class _FakeProc:
    """A Popen whose stdout holds `payload` and whose exit status is `code`."""

    def __init__(self, payload: bytes, code: int):
        self.stdout = io.BytesIO(payload)
        self._code = code
        self.terminated = False

    def terminate(self):
        self.terminated = True

    def wait(self):
        return self._code


@pytest.fixture
def fake_ffmpeg(monkeypatch):
    """Install a fake `Popen` and hand back the process object it produced."""
    made: list[_FakeProc] = []

    def install(payload: bytes, code: int) -> list[_FakeProc]:
        def _popen(*_args, **_kwargs):
            proc = _FakeProc(payload, code)
            made.append(proc)
            return proc

        monkeypatch.setattr(subprocess, "Popen", _popen)
        return made

    return install


def _frame_bytes(n: int) -> bytes:
    return bytes(n * W * H * 3)


def test_a_clean_decode_yields_every_frame_and_says_nothing(fake_ffmpeg):
    fake_ffmpeg(_frame_bytes(4), 0)
    got = list(frames("clip.mp4", W, H, None))
    assert len(got) == 4
    assert got[0].shape == (H, W, 3)


def test_a_non_zero_exit_after_a_prefix_raises(fake_ffmpeg):
    """Three frames then ffmpeg dies. The old code returned three frames and success."""
    fake_ffmpeg(_frame_bytes(3), 1)
    with pytest.raises(DecodeError) as excinfo:
        list(frames("clip.mp4", W, H, None))
    assert "3 whole frames" in str(excinfo.value)
    assert "exit 1" in str(excinfo.value)


def test_a_torn_final_frame_raises_even_on_a_zero_exit(fake_ffmpeg):
    """Trailing bytes that do not make a frame mean the pipe closed mid-write."""
    fake_ffmpeg(_frame_bytes(2) + b"\x00" * 17, 0)
    with pytest.raises(DecodeError) as excinfo:
        list(frames("clip.mp4", W, H, None))
    assert "17 trailing bytes" in str(excinfo.value)


def test_an_early_break_is_not_an_error_and_terminates_ffmpeg(fake_ffmpeg):
    """`--max-frames` is deliberate. ffmpeg's broken-pipe death is not a decode failure.

    It is also why `terminate()` is called rather than `wait()`: ffmpeg is mid-stream
    with a pipe buffer to drain into a reader that has gone.
    """
    made = fake_ffmpeg(_frame_bytes(50), 1)  # a non-zero code the caller must not see
    gen = frames("clip.mp4", W, H, None)
    seen = 0
    for _ in gen:
        seen += 1
        if seen == 2:
            break
    gen.close()
    assert seen == 2
    assert made[0].terminated, "ffmpeg was left running after the caller stopped reading"


def test_an_exception_in_the_caller_is_not_masked_by_a_decode_error(fake_ffmpeg):
    """The consumer's own failure closes the generator; it must survive intact."""
    fake_ffmpeg(_frame_bytes(50), 1)
    with pytest.raises(ZeroDivisionError):
        for _ in frames("clip.mp4", W, H, None):
            raise ZeroDivisionError("the consumer blew up")


# --------------------------------------------------------------- with a real ffmpeg

needs_ffmpeg = pytest.mark.skipif(
    not (shutil.which("ffmpeg") and shutil.which("ffprobe")), reason="needs ffmpeg"
)


@pytest.fixture
def clip(tmp_path):
    """Ten frames of testsrc at 5 fps, written where the test can truncate it."""
    path = tmp_path / "clip.mp4"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-f", "lavfi",
         "-i", f"testsrc=size={W}x{H}:rate=5:duration=2", str(path)],
        check=True, timeout=60,
    )  # fmt: skip
    return path


@needs_ffmpeg
def test_a_real_clip_decodes_to_the_frame_count_probe_implies(clip):
    w, h, fps = probe(str(clip))
    assert (w, h) == (W, H)
    assert fps == pytest.approx(5.0)
    got = list(frames(str(clip), w, h, None))
    assert len(got) == 10
    assert all(f.dtype == np.uint8 for f in got)


@needs_ffmpeg
def test_a_file_ffmpeg_cannot_open_raises_rather_than_yielding_nothing(tmp_path):
    """The zero-frame case. It used to be an empty loop body and a clean exit."""
    junk = tmp_path / "not_really.mp4"
    junk.write_bytes(b"\x00\x01\x02\x03" * 512)
    with pytest.raises(DecodeError) as excinfo:
        list(frames(str(junk), W, H, None))
    assert "0 whole frames" in str(excinfo.value)
