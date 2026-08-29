"""The camera overlap that a config cannot see and a finished run cannot undo."""

from __future__ import annotations

from syncai_hydranet.data.datasets import split_leaks


def _seg(name, root, **kw):
    return {"name": name, "type": "seg_folder", "root": str(root), "split_train": "train", **kw}


def _make(tmp_path, name, splits):
    for split, cams in splits.items():
        for cam in cams:
            (tmp_path / name / "images" / split / cam).mkdir(parents=True)
    return tmp_path / name


def test_a_supplement_trained_on_the_val_cameras_is_reported(tmp_path):
    """The failure this exists for: a batch selected on `SAM 3 finds a column here`
    knows nothing about a split made by camera, and both configs stay self-consistent."""
    main = _make(tmp_path, "main", {"train": ["cam01__clipA"], "val": ["cam06__clipB"]})
    supp = _make(tmp_path, "supp", {"train": ["cam06", "cam09"]})
    leaks = split_leaks([_seg("site_seg", main, split_val="val"), _seg("site_columns", supp)])
    assert leaks == [("site_columns", "site_seg", "val", ["cam06"])]


def test_the_camera_is_matched_across_naming_conventions(tmp_path):
    """One dataset names sessions `<camera>__<clip>` and the other just `<camera>`;
    a check that compared directory names would find nothing."""
    main = _make(tmp_path, "main", {"train": ["a__x"], "test": ["cam12__clip"]})
    supp = _make(tmp_path, "supp", {"train": ["cam12"]})
    assert split_leaks([_seg("m", main, split_test="test"), _seg("s", supp)])[0][3] == ["cam12"]


def test_a_dataset_without_session_directories_is_not_compared(tmp_path):
    """ADE20K puts images straight under `images/train`. It has no cameras, so it has no
    overlap to report -- and must not be made to look like one big session."""
    flat = tmp_path / "flat" / "images" / "train"
    flat.mkdir(parents=True)
    (flat / "ADE_train_00000001.jpg").write_bytes(b"")
    main = _make(tmp_path, "main", {"train": ["cam01"], "val": ["cam06"]})
    assert (
        split_leaks([_seg("ade", tmp_path / "flat"), _seg("site", main, split_val="val")]) == []
    )


def test_one_dataset_holding_its_own_split_is_not_a_leak(tmp_path):
    """train and val of the *same* dataset are different cameras by construction; a
    dataset must not be reported against itself for having a val split at all."""
    main = _make(tmp_path, "main", {"train": ["cam01"], "val": ["cam06"]})
    assert split_leaks([_seg("site", main, split_val="val")]) == []
