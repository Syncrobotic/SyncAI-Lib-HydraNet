"""Camera attribution, which is the part of the split report that can be silently wrong.

The rule checks are arithmetic over pixel counts and fail loudly if they are wrong. The
resolution step is not: it turns a directory name into a camera, and a wrong answer there
produces a split report that passes R1 and R2 while the split violates both.

Two cases carry the risk. A `Camera__stem` prefix must beat the manifest, because it was
written by something that knew and it is the form that cannot collide. And a stem that two
cameras claim must resolve to nothing -- **8 of 184 real clip stems are claimed by more
than one camera**, because two cameras in different stores can start a recording in the
same second and then their files have the same name. Picking either one would be a coin
toss recorded as a fact.

pytest tests/test_site_split_report.py -v
"""

import json
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


@pytest.fixture(scope="module")
def report():
    sys.path.insert(0, str(SCRIPTS))
    try:
        import site_split_report
    finally:
        sys.path.remove(str(SCRIPTS))
    return site_split_report


def _manifest(tmp_path, pulled):
    (tmp_path / "manifest_test.json").write_text(json.dumps({"pulled": pulled}))
    return tmp_path


def test_a_stem_two_cameras_claim_resolves_to_nothing(report, tmp_path):
    """The real case: `archive_20260816-062742_20260816-063246` is claimed by both
    Kaohsiung-cam02 and Taichung-cam02, and it is in batch01's test split. Choosing one
    would put a coin toss into a split report as a fact."""
    idx = report.camera_index(
        _manifest(
            tmp_path,
            [
                {"camera": "Kaohsiung-cam02", "clips": [{"uri": "gs://b/archive_x.mp4"}]},
                {"camera": "Taichung-cam02", "clips": [{"uri": "gs://b/archive_x.mp4"}]},
            ],
        )
    )
    assert report.resolve("archive_x", idx) == report.AMBIGUOUS


def test_an_unambiguous_stem_resolves(report, tmp_path):
    idx = report.camera_index(
        _manifest(
            tmp_path, [{"camera": "Taichung-cam01", "clips": [{"uri": "gs://b/arc_y.mp4"}]}]
        )
    )
    assert report.resolve("arc_y", idx) == "Taichung-cam01"


def test_a_name_prefix_beats_the_manifest(report, tmp_path):
    """The prefix was written at export time by something that knew which camera it was,
    and it is the form that cannot collide. A manifest disagreeing with it means the
    manifest is describing a different clip that happens to share a name."""
    idx = report.camera_index(
        _manifest(
            tmp_path, [{"camera": "Kaohsiung-cam09", "clips": [{"uri": "gs://b/arc_z.mp4"}]}]
        )
    )
    assert report.resolve("Taichung-cam06__arc_z", idx) == "Taichung-cam06"


def test_a_prefix_resolves_even_with_no_manifest_at_all(report, tmp_path):
    """A camera-prefixed export has to stay checkable after the manifests are gone, which
    is the whole argument for exporting that way."""
    assert report.resolve("Tao-Hsin-cam03__whatever", report.camera_index(tmp_path)) == (
        "Tao-Hsin-cam03"
    )


def test_an_unknown_stem_is_ambiguous_rather_than_invented(report, tmp_path):
    assert report.resolve("never_seen", report.camera_index(tmp_path)) == report.AMBIGUOUS


def test_every_manifest_in_the_directory_contributes(report, tmp_path):
    """The clips were pulled in two passes -- one of them mislabelled and renamed rather
    than deleted -- so reading only the newest manifest loses half the attribution."""
    (tmp_path / "manifest_a.json").write_text(
        json.dumps({"pulled": [{"camera": "A-cam01", "clips": [{"uri": "gs://b/one.mp4"}]}]})
    )
    (tmp_path / "manifest_b-mislabelled.json").write_text(
        json.dumps({"pulled": [{"camera": "B-cam01", "clips": [{"uri": "gs://b/two.mp4"}]}]})
    )
    idx = report.camera_index(tmp_path)
    assert report.resolve("one", idx) == "A-cam01"
    assert report.resolve("two", idx) == "B-cam01"


def test_the_rare_threshold_is_the_evaluators(report):
    """1% is `evaluator.THIN_SUPPORT`, calibrated there on `column` at 0.66% -- the one
    class in this project known to have failed this way. Two thresholds for one idea would
    drift."""
    assert pytest.approx(0.01) == report.RARE_SHARE


def test_the_shipped_dataset_still_passes_r1_and_r2(report):
    """The check that would catch a future batch quietly reusing a test camera. Skipped
    where the dataset is absent, since it is gitignored."""
    root = Path(__file__).resolve().parents[1] / "datasets" / "retail_objects_batch01"
    clips = Path(__file__).resolve().parents[1] / "datasets" / "studioa_clips"
    if not (root / "annotations" / "test").is_dir() or not clips.is_dir():
        pytest.skip("site dataset or clip manifests not present in this checkout")

    idx = report.camera_index(clips)
    sides = {}
    for split in ("train", "test"):
        sides[split] = {
            report.resolve(p.name, idx)
            for p in (root / "annotations" / split).iterdir()
            if p.is_dir()
        } - {report.AMBIGUOUS}
    assert sides["train"] and sides["test"]
    assert not (sides["train"] & sides["test"]), "a camera supplies both train and test"
