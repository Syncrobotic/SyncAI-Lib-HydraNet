import copy
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from syncai_hydranet.data.store_split import STORES
from syncai_hydranet.data.studioa_contract import VIEWS
from syncai_hydranet.data.studioa_review import (
    check_package,
    index_source,
    prepare_package,
    review_template,
    select_frames,
    validate_review,
    write_json,
)


@pytest.fixture
def dataset(tmp_path):
    root = tmp_path / "dataset"
    for i, store in enumerate(STORES):
        for j, split in enumerate(("train", "val", "test")):
            for session in range(2):
                name = f"{store}-cam{j:02}__session{session}__0000"
                image = root / "images" / split / f"{name}.png"
                mask = root / "annotations" / split / f"{name}.png"
                image.parent.mkdir(parents=True, exist_ok=True)
                mask.parent.mkdir(parents=True, exist_ok=True)
                Image.new("RGB", (32, 24), (i * 60 + j * 10, session * 30, 0)).save(image)
                pixels = np.ones((24, 32), dtype=np.uint8)
                pixels[:10] = 2
                Image.fromarray(pixels).save(mask)
    return root


def prepared(dataset, tmp_path):
    out = tmp_path / "review"
    report = prepare_package([(dataset, "site30k_native")], out, per_camera=1)
    manifest = json.loads((out / "manifest.json").read_text())
    return out, report, manifest


def reviewed(frame):
    review = review_template(frame)
    review.update(
        status="reviewed",
        reviewer="human-test-fixture",
        reviewed_at="2026-09-16T10:00:00+08:00",
    )
    review["coverage"] = dict.fromkeys(VIEWS, "absent")
    review["coverage"]["floor"] = "present"
    review["entities"] = [
        {
            "id": "floor-1",
            "entity": "floor",
            "visibility": "occluded",
            "polygons_px": [
                [[1, 12], [12, 12], [12, 20], [1, 20]],
                [[16, 12], [22, 12], [22, 20]],
            ],
        }
    ]
    return review


def test_package_is_frozen_pending_work_with_whole_store_proposals(dataset, tmp_path):
    out, report, manifest = prepared(dataset, tmp_path)
    assert report["indexed_pairs"] == 18
    assert report["frames"] == 9 and report["reviewed_frames"] == 0
    assert report["status"] == "pending_review"
    for store, fold in manifest["prospective_folds"].items():
        assert fold["status"] == "structurally_valid"
        assert fold["counts"] == {"train": 2, "val": 2, "test": 3}
        for frame in manifest["frames"]:
            assigned = fold["assignments"][frame["id"]]
            if frame["store"] == store:
                assert assigned == "test"
            elif frame["original_split"] == "test":
                assert assigned is None
    frame = manifest["frames"][0]
    with Image.open(out / frame["proposal"]) as proposal:
        assert np.all(np.asarray(proposal)[:10] == 255)
    # Editing a live source cannot silently change the frozen review image.
    Path(frame["source_image"]).write_bytes(b"changed live source")
    assert check_package(out)["status"] == "pending_review"
    with pytest.raises(FileExistsError):
        prepare_package([(dataset, "site30k_native")], out)


def test_modified_frozen_image_and_moved_review_are_rejected(dataset, tmp_path):
    out, _, manifest = prepared(dataset, tmp_path)
    frame = manifest["frames"][0]
    wrong = review_template(frame)
    wrong["frame_id"] = "other"
    write_json(out / frame["review"], wrong)
    with pytest.raises(ValueError, match="mismatch"):
        check_package(out)
    write_json(out / frame["review"], review_template(frame))
    with (out / frame["image"]).open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="changed frozen input"):
        check_package(out)


def test_pending_empty_and_false_negative_reviews_cannot_be_accepted(dataset, tmp_path):
    out, _, manifest = prepared(dataset, tmp_path)
    frame = manifest["frames"][0]
    with pytest.raises(ValueError, match="pending"):
        validate_review(review_template(frame), frame)
    review = reviewed(frame)
    assert validate_review(review, frame)["floor"] == 1
    write_json(out / frame["review"], review)
    assert check_package(out)["reviewed_frames"] == 1
    review["coverage"]["floor"] = "absent"
    with pytest.raises(ValueError, match="disagree"):
        validate_review(review, frame)
    review["coverage"]["floor"] = "present"
    review["entities"][0]["polygons_px"][0][0] = [32, 12]
    with pytest.raises(ValueError, match="coordinates"):
        validate_review(review, frame)


@pytest.mark.parametrize(
    "entity,uncertain", [("door", "glass_door"), ("counter", "checkout_counter")]
)
def test_unknown_attributes_are_not_negative_examples(dataset, tmp_path, entity, uncertain):
    _, _, manifest = prepared(dataset, tmp_path)
    frame = manifest["frames"][0]
    review = reviewed(frame)
    item = copy.deepcopy(review["entities"][0])
    item.update(id="unknown-1", entity=entity)
    review["entities"].append(item)
    if entity == "door":
        review["coverage"]["door"] = "present"
    with pytest.raises(ValueError, match="unknown attribute"):
        validate_review(review, frame)
    review["coverage"][uncertain] = "unresolved"
    validate_review(review, frame)


def test_ambiguous_boxes_require_unresolved_coverage(dataset, tmp_path):
    _, _, manifest = prepared(dataset, tmp_path)
    frame = manifest["frames"][0]
    review = reviewed(frame)
    review["unresolved_regions"] = [
        {
            "polygon_px": [[1, 1], [5, 1], [5, 5]],
            "reason": "box role is not visible",
            "possible_entities": ["boxed_stock", "cardboard_box"],
        }
    ]
    with pytest.raises(ValueError, match="unresolved region"):
        validate_review(review, frame)
    review["coverage"].update(boxed_stock="unresolved", cardboard_box="unresolved")
    validate_review(review, frame)


def test_incomplete_pair_and_bad_limits_are_rejected(dataset):
    rows = index_source(dataset, "site30k_native")
    with pytest.raises(ValueError, match="positive"):
        select_frames(rows, 0)
    Path(rows[0]["source_mask"]).unlink()
    with pytest.raises(ValueError, match="missing paired mask"):
        index_source(dataset, "site30k_native")


def test_duplicate_pixels_do_not_pass_a_prospective_fold(dataset, tmp_path):
    images = sorted((dataset / "images").rglob("*.png"))
    content = images[0].read_bytes()
    for image in images:
        image.write_bytes(content)
    _, report, manifest = prepared(dataset, tmp_path)
    assert set(report["folds"].values()) == {"blocked"}
    assert all(
        "image leakage" in fold["reason"] for fold in manifest["prospective_folds"].values()
    )


def test_completed_reviews_still_do_not_establish_independent_acceptance(dataset, tmp_path):
    out, _, manifest = prepared(dataset, tmp_path)
    for frame in manifest["frames"]:
        write_json(out / frame["review"], reviewed(frame))
    report = check_package(out)
    assert report["status"] == "review_complete"
    assert report["reviewed_instances_by_view"]["floor"] == 9
    assert report["reviewed_instances_by_view"]["phone"] == 0
    assert report["acceptance"].startswith("not established")


def test_review_requires_identity_timestamp_complete_coverage_and_unique_instances(
    dataset, tmp_path
):
    _, _, manifest = prepared(dataset, tmp_path)
    frame = manifest["frames"][0]
    mutations = [
        ("reviewer", None, "reviewer"),
        ("reviewed_at", "2026-09-16T10:00:00", "timezone"),
        ("coverage", {}, "coverage"),
        ("image_sha256", "other", "mismatch"),
    ]
    for key, value, message in mutations:
        review = reviewed(frame)
        review[key] = value
        with pytest.raises(ValueError, match=message):
            validate_review(review, frame)
    review = reviewed(frame)
    review["entities"].append(copy.deepcopy(review["entities"][0]))
    with pytest.raises(ValueError, match="unique"):
        validate_review(review, frame)
