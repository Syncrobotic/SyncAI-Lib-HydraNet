"""One output directory per input clip, or the run is refused.

`sam3_prelabel.py` names an output directory after its input's filename stem. A filename is
not an identity -- it belongs to whoever wrote the footage -- and `gs://studioa-recording`
names a clip by its recording timestamp, so two cameras that start recording in the same
second produce the same stem. Four of 96 clips in one real pull did. Both wrote into one
directory, the second replaced the first, and the run finished by printing a plausible
per-class composition over whichever frames survived: 45 directories where 48 were asked
for, and a COCO file whose boxes referenced frames from the wrong camera.

Nothing errored, which is the point. It was found only because a class share moved
33.33% -> 2.49% between a partial and a full pass -- impossible for an average over one more
clip -- and it came one step from putting a contaminated camera into an approved test split.

These tests pin the refusal, its message, and its ordering. The ordering is not a detail:
detecting the collision on the way past would leave the half-written dataset the refusal
exists to prevent.

Both refusals now live in `data/video.py` -- they are about a list of paths, which every
caller that decodes a list of clips asks about. The `prelabel` fixture stays for the
tests that drive the script's `main`, which is where the ordering is decided.

pytest tests/test_sam3_sessions.py -v
"""

import json
import sys
from pathlib import Path

import pytest

from syncai_hydranet.data.video import session_names, validate_inputs

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


@pytest.fixture(scope="module")
def prelabel():
    sys.path.insert(0, str(SCRIPTS))
    try:
        import sam3_prelabel
    finally:
        sys.path.remove(str(SCRIPTS))
    return sam3_prelabel


@pytest.fixture
def two_cameras(tmp_path):
    """The real shape of the collision: one clip name, two cameras, two directories."""
    stem = "archive_20260816-063227_20260816-063727.mp4"
    clips = []
    for camera in ("Kaohsiung-cam07", "Taichung-cam08"):
        d = tmp_path / camera
        d.mkdir()
        (d / stem).write_bytes(b"")
        clips.append(str(d / stem))
    return clips


def test_distinct_names_pass_through_in_order(tmp_path):
    clips = [str(tmp_path / f"cam{i}.mp4") for i in range(3)]
    assert session_names(clips) == ["cam0", "cam1", "cam2"]


def test_two_cameras_one_clip_name_is_refused(two_cameras):
    with pytest.raises(SystemExit) as exc:
        session_names(two_cameras)
    assert "refusing before writing anything" in str(exc.value)


def test_the_message_names_both_input_paths_not_the_shared_key(two_cameras):
    """The key is what the script computed; the paths are what the operator must go look at.

    A message naming only the shared basename sends someone to the wrong place -- today's
    collision was unintelligible without the camera directory, which is the part the stem
    throws away.
    """
    with pytest.raises(SystemExit) as exc:
        session_names(two_cameras)
    message = str(exc.value)
    for path in two_cameras:
        assert path in message
    assert "Kaohsiung-cam07" in message and "Taichung-cam08" in message


def test_a_missing_input_is_refused_before_anything_runs(tmp_path):
    """The defect this replaces aborted the batch mid-way, after paying for the model load.

    A path that does not exist used to reach `probe()`, whose ffprobe call raises
    `CalledProcessError` uncaught: raw traceback, every later clip dropped, and the session
    directory already created. The `no frames decoded, skipped` branch further down looks
    like it covers this and does not -- `probe()` dies first, so that branch is reachable
    only for a directory input.
    """
    real = tmp_path / "there.mp4"
    real.write_bytes(b"")
    with pytest.raises(SystemExit) as exc:
        validate_inputs([str(real), str(tmp_path / "gone.mp4")])
    message = str(exc.value)
    assert "1 of 2 input(s) do not exist" in message
    assert "gone.mp4" in message
    assert "there.mp4" not in message, "naming a valid input as missing sends the wrong way"


def test_every_missing_input_is_named_not_just_the_first(tmp_path):
    """Refusing on the first one would make a bad glob a three-round guessing game."""
    missing = [str(tmp_path / f"gone{i}.mp4") for i in range(3)]
    with pytest.raises(SystemExit) as exc:
        validate_inputs(missing)
    for path in missing:
        assert path in str(exc.value)


def test_existing_inputs_pass_silently(tmp_path):
    clip = tmp_path / "here.mp4"
    clip.write_bytes(b"")
    stills = tmp_path / "camera_of_stills"
    stills.mkdir()
    assert validate_inputs([str(clip), str(stills)]) is None


def test_existence_is_checked_before_the_model_loads(prelabel, tmp_path, monkeypatch):
    def must_not_run(*_a, **_k):
        raise AssertionError("load_sam3 reached: the existence check ran too late")

    monkeypatch.setattr(prelabel, "load_sam3", must_not_run)
    out = tmp_path / "dataset"
    with pytest.raises(SystemExit) as exc:
        prelabel.main(
            ["--out", str(out), "--scheme", "retail_objects", str(tmp_path / "gone.mp4")]
        )
    assert "do not exist" in str(exc.value)
    assert not out.exists(), "the refused run left a dataset directory behind"


def test_the_same_path_twice_is_told_to_deduplicate_not_to_symlink(two_cameras):
    """Same refusal, different cause, and the remedies are not interchangeable.

    A doubled glob or a manifest with a repeated entry produces one path printed twice.
    Refusing is still right -- processing a clip twice is wasted model time -- but advising
    symlinks would send the operator in a circle, since renaming a path to itself is not a
    thing they can do.
    """
    with pytest.raises(SystemExit) as exc:
        session_names([two_cameras[0], two_cameras[0]])
    message = str(exc.value)
    assert "listed more than once" in message
    assert "symlink" not in message.lower()


def test_distinct_paths_are_still_told_to_symlink(two_cameras):
    with pytest.raises(SystemExit) as exc:
        session_names(two_cameras)
    assert "symlinks whose names differ" in str(exc.value)


def test_a_third_colliding_input_is_reported_with_the_other_two(two_cameras, tmp_path):
    d = tmp_path / "Tao-Hsin-cam01"
    d.mkdir()
    third = d / Path(two_cameras[0]).name
    third.write_bytes(b"")
    with pytest.raises(SystemExit) as exc:
        session_names([*two_cameras, str(third)])
    assert str(third) in str(exc.value)


def test_the_whole_input_list_is_checked_before_the_model_loads(
    prelabel, two_cameras, tmp_path, monkeypatch
):
    """Ordering, which is the half that keeps the dataset clean rather than the log tidy.

    Checking incrementally would still write N-1 directories before finding the collision at
    N. `load_sam3` is stubbed to fail loudly: reaching it at all means the check ran too
    late, and the assertion below distinguishes that from the refusal we want.

    Not redundant with the tests above, and do not delete it as such. They exercise a
    function that did not exist before the fix, so they cannot fail on the old code for an
    interesting reason. This one goes through `main()`, which did: on the old code `main()`
    walks straight into the stub, raises `AssertionError`, and `pytest.raises(SystemExit)`
    does not catch it. It is the only genuine regression test in this file.
    """

    def must_not_run(*_a, **_k):
        raise AssertionError("load_sam3 reached: the uniqueness check ran too late")

    monkeypatch.setattr(prelabel, "load_sam3", must_not_run)
    out = tmp_path / "dataset"
    with pytest.raises(SystemExit) as exc:
        prelabel.main(["--out", str(out), "--scheme", "retail_objects", *two_cameras])
    assert "refusing before writing anything" in str(exc.value)
    assert not out.exists(), "the refused run left a dataset directory behind"


# ------------------------------- a session directory means frames were written


@pytest.fixture
def empty_camera(tmp_path):
    """A directory input holding no images: decodes to nothing and is *skipped*, not an
    error. A camera that legitimately yielded nothing is a real case, not a bad path."""
    d = tmp_path / "Tao-Hsin-cam04"
    d.mkdir()
    return d


def test_a_skipped_clip_leaves_no_session_directory(
    prelabel, empty_camera, tmp_path, monkeypatch
):
    """The half `validate_inputs` cannot reach.

    Existence checking cannot tell you a real input will decode to zero frames, so the
    in-loop skips are the only place this can be handled. They used to run *after* the two
    `mkdir` calls, so a skipped clip left `images/<split>/<session>/` and
    `annotations/<split>/<session>/` behind, empty -- and an empty session directory is
    indistinguishable from a camera that produced nothing, which is exactly the ambiguity
    the refusal machinery exists to remove.

    This regresses the moment someone tidies: the natural place to write those two lines is
    beside the path construction, which is where they were and why they were wrong.
    """
    monkeypatch.setattr(prelabel, "load_sam3", lambda *_a, **_k: (None, None))
    out = tmp_path / "dataset"
    assert (
        prelabel.main(["--out", str(out), "--scheme", "retail_objects", str(empty_camera)]) == 0
    )
    for kind in ("images", "annotations"):
        assert not (out / kind / "train" / empty_camera.name).exists(), (
            f"{kind}/train/{empty_camera.name} was created for a clip that wrote no frames"
        )


def test_a_run_that_skips_everything_still_writes_its_manifest(
    prelabel, empty_camera, tmp_path, monkeypatch
):
    """`root` used to be created as a side effect of the first session's `mkdir`.

    With the session directories moved below the skips, a run where every clip skips
    creates none of them, so the manifest write has to make `root` itself. It is wanted:
    `found_nothing` is precisely what an operator needs after a run that produced no data.
    A dataset root holding a manifest that says so is an answer; an empty session directory
    is an ambiguity. Without this the same run raises FileNotFoundError at the last line.
    """
    monkeypatch.setattr(prelabel, "load_sam3", lambda *_a, **_k: (None, None))
    out = tmp_path / "dataset"
    prelabel.main(["--out", str(out), "--scheme", "retail_objects", str(empty_camera)])
    manifest = json.loads((out / "sam3_batch.json").read_text())
    assert manifest["found_nothing"], "a run that labelled nothing must say so in its manifest"
