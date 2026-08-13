"""Dataset fingerprints: the record of which data a run actually saw.

pytest tests/test_fingerprint.py -v
"""

from syncai_hydranet.data.fingerprint import (
    fingerprint_dataset,
    fingerprint_dir,
    fingerprint_file,
)


def _tree(root, split, n, size=10):
    for sub in ("images", "annotations"):
        d = root / sub / split
        d.mkdir(parents=True, exist_ok=True)
        for i in range(n):
            (d / f"{i}.png").write_bytes(b"x" * size)
    return root


# ------------------------------------------------------------------ directory


def test_counts_files_and_bytes(tmp_path):
    _tree(tmp_path, "train", n=3, size=100)
    fp = fingerprint_dir(tmp_path / "images" / "train")
    assert fp["present"] and fp["files"] == 3 and fp["bytes"] == 300
    assert fp["digest"].startswith("sha256:")


def test_identical_trees_agree(tmp_path):
    a = _tree(tmp_path / "a", "train", n=3)
    b = _tree(tmp_path / "b", "train", n=3)
    assert (
        fingerprint_dir(a / "images" / "train")["digest"]
        == fingerprint_dir(b / "images" / "train")["digest"]
    )


def test_an_added_file_changes_the_digest(tmp_path):
    d = _tree(tmp_path, "train", n=3) / "images" / "train"
    before = fingerprint_dir(d)["digest"]
    (d / "extra.png").write_bytes(b"x" * 10)
    assert fingerprint_dir(d)["digest"] != before


def test_a_resized_file_changes_the_digest(tmp_path):
    """Re-exported annotations are the common case: same names, different content."""
    d = _tree(tmp_path, "train", n=3) / "images" / "train"
    before = fingerprint_dir(d)["digest"]
    (d / "0.png").write_bytes(b"x" * 999)
    assert fingerprint_dir(d)["digest"] != before


def test_a_rename_changes_the_digest(tmp_path):
    d = _tree(tmp_path, "train", n=3) / "images" / "train"
    before = fingerprint_dir(d)["digest"]
    (d / "0.png").rename(d / "renamed.png")
    assert fingerprint_dir(d)["digest"] != before


def test_touching_a_file_does_not(tmp_path):
    """Copying or re-symlinking a dataset changes every mtime without changing the
    data, so mtime is deliberately not part of the digest."""
    d = _tree(tmp_path, "train", n=3) / "images" / "train"
    before = fingerprint_dir(d)["digest"]
    (d / "0.png").touch()
    assert fingerprint_dir(d)["digest"] == before


def test_missing_directory_is_reported_not_raised(tmp_path):
    assert fingerprint_dir(tmp_path / "nope") == {
        "present": False,
        "path": str(tmp_path / "nope"),
    }


def test_file_fingerprint(tmp_path):
    f = tmp_path / "instances_train2017.json"
    f.write_bytes(b"{}" * 50)
    assert fingerprint_file(f)["bytes"] == 100
    assert fingerprint_file(tmp_path / "nope.json")["present"] is False


# -------------------------------------------------------------------- config


def test_seg_folder_layout(tmp_path):
    _tree(tmp_path, "train", n=2)
    _tree(tmp_path, "val", n=1)
    fp = fingerprint_dataset(
        {"name": "ade20k", "root": str(tmp_path), "split_train": "train", "split_val": "val"}
    )
    assert fp["name"] == "ade20k"
    assert fp["splits"]["train"]["images"]["files"] == 2
    assert fp["splits"]["train"]["labels"]["files"] == 2
    assert fp["splits"]["val"]["images"]["files"] == 1


def test_coco_layout(tmp_path):
    (tmp_path / "train2017").mkdir()
    (tmp_path / "train2017" / "a.jpg").write_bytes(b"x" * 7)
    (tmp_path / "annotations").mkdir()
    (tmp_path / "annotations" / "instances_train2017.json").write_bytes(b"{}")
    fp = fingerprint_dataset(
        {
            "name": "coco",
            "root": str(tmp_path),
            "split_train": "train2017",
            "split_val": "val2017",
        }
    )
    assert fp["splits"]["train"]["images"]["files"] == 1
    assert fp["splits"]["train"]["labels"]["bytes"] == 2
    assert "val" not in fp["splits"]  # not on disk, so nothing to record


def test_test_split_is_included_when_asked(tmp_path):
    _tree(tmp_path, "holdout", n=4)
    fp = fingerprint_dataset({"root": str(tmp_path), "split_test": "holdout"}, splits=("test",))
    assert fp["splits"]["test"]["images"]["files"] == 4
