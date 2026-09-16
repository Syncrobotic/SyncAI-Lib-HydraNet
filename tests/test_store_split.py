import pytest

from syncai_hydranet.data.store_split import (
    STORES,
    camera_store,
    fold_split,
    validate_fold,
)


def records():
    return [
        {
            "store": store,
            "camera": f"{store}-cam{i:02}",
            "original_split": split,
            "image_sha256": f"{store}-{split}",
        }
        for store in STORES
        for i, split in enumerate(("train", "val", "test"))
    ]


def test_entire_target_store_is_unseen_and_source_test_stays_unused():
    for held_out in STORES:
        assert validate_fold(records(), held_out) == {"train": 2, "val": 2, "test": 3}
        for original in ("train", "val", "test"):
            assert fold_split(held_out, original, held_out) == "test"
        for source in set(STORES) - {held_out}:
            assert fold_split(source, "test", held_out) is None


def test_changed_camera_assignment_between_batches_is_rejected():
    rows = records()
    rows[1]["camera"] = rows[0]["camera"]
    with pytest.raises(ValueError, match="camera leakage"):
        validate_fold(rows, "Tao-Hsin")


def test_identical_images_cannot_cross_store_boundary():
    rows = records()
    rows[-1]["image_sha256"] = rows[0]["image_sha256"]
    with pytest.raises(ValueError, match="identical image leakage"):
        validate_fold(rows, "Tao-Hsin")


def test_bad_identity_and_empty_validation_fail_closed():
    assert camera_store("images/Tao-Hsin-cam03__20260816__0001.jpg") == (
        "Tao-Hsin-cam03",
        "Tao-Hsin",
    )
    with pytest.raises(ValueError, match="identify"):
        camera_store("unknown.jpg")
    with pytest.raises(ValueError, match="empty"):
        validate_fold([r for r in records() if r["original_split"] != "val"], "Taichung")
