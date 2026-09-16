"""Reviewed additions must not turn unselected proposals into false background."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from syncai_hydranet.data.studioa_autolabel import Candidate, annotate, validate_annotation
from syncai_hydranet.data.studioa_supervision import CLASSES, semantic_target

pytestmark = pytest.mark.filterwarnings(
    "ignore:__array__ implementation doesn't accept a copy keyword:DeprecationWarning"
)


def worker():
    path = Path(__file__).resolve().parents[1] / "tools/annotation/studioa_extend.py"
    spec = importlib.util.spec_from_file_location("extension_test", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def annotation():
    mask = np.zeros((64, 80), dtype=bool)
    mask[20:40, 20:40] = True
    data = annotate(
        [
            Candidate("phone", mask, 0.9, ["phone"]),
            Candidate("floor", np.ones_like(mask), 0.9, ["floor"]),
        ],
        mask.shape,
    )
    data["image_sha256"] = "fixture"
    return data


def test_explicit_class_correction_preserves_mask_and_leaves_everything_else_unknown():
    data = annotation()
    identity = next(
        r["id"]
        for r in data["entities"] + data["unresolved_candidates"]
        if r["entity"] == "phone"
    )
    curated = worker().retain_reviewed(
        data,
        [
            {
                "id": identity,
                "entity": "tablet",
                "reason": "visual confirmation of physical tablet",
            }
        ],
    )
    validate_annotation(curated, "fixture", (64, 80))
    assert curated["entities"][0]["original_proposal_entity"] == "phone"
    target, _ = semantic_target(curated)
    assert target[25, 25] == CLASSES["tablet"]
    assert target[0, 0] == 255
    assert len(curated["unreviewed_proposals"]) >= 1
    assert data["entities"] != curated["entities"]  # source untouched


def test_empty_duplicate_or_unreasoned_acceptance_is_rejected():
    data = annotation()
    identity = next(
        r["id"]
        for r in data["entities"] + data["unresolved_candidates"]
        if r["entity"] == "phone"
    )
    with pytest.raises(ValueError, match="at least one"):
        worker().retain_reviewed(data, [])
    decision = {"id": identity, "entity": "phone", "reason": "seen"}
    with pytest.raises(ValueError, match="unique"):
        worker().retain_reviewed(data, [decision, decision])
    with pytest.raises(ValueError, match="reason"):
        worker().retain_reviewed(data, [{**decision, "reason": ""}])


def test_extension_keeps_all_prior_masks_and_splits_and_adds_only_train(tmp_path):
    from PIL import Image

    from _studioa_fixture import source_package
    from syncai_hydranet.data.studioa_review import digest, write_json
    from syncai_hydranet.data.studioa_supervision import check_supervision, export_supervision

    source, base, media = tmp_path / "source", tmp_path / "base", tmp_path / "media"
    source_package(source)
    export_supervision(source, base)
    (media / "images").mkdir(parents=True)
    (media / "frames").mkdir()
    fid = "gcs-new-Kaohsiung-cam01"
    image = media / "images" / f"{fid}.png"
    Image.new("RGB", (80, 64), (231, 19, 7)).save(image)
    frame = {
        "id": fid,
        "camera": "Kaohsiung-cam01",
        "store": "Kaohsiung",
        "original_split": "train",
        "image_size_px": [80, 64],
        "image": str(image.relative_to(media)),
        "image_sha256": digest(image),
    }
    write_json(media / "job.json", {"frames": [frame]})
    data = annotation()
    data.update(frame_id=fid, image_sha256=digest(image), job_sha256=digest(media / "job.json"))
    identity = next(
        r["id"]
        for r in data["entities"] + data["unresolved_candidates"]
        if r["entity"] == "phone"
    )
    name = f"frames/{fid}.json"
    write_json(media / name, data)
    write_json(
        media / "report.json",
        {
            "status": "completed",
            "job_sha256": digest(media / "job.json"),
            "outputs": {name: digest(media / name)},
        },
    )
    decisions = tmp_path / "decisions.json"
    write_json(
        decisions,
        {
            "reviewer_kind": "ai",
            "source_job_sha256": digest(media / "job.json"),
            "frames": [
                {
                    "frame_id": fid,
                    "annotation_sha256": digest(media / name),
                    "positives": [{"id": identity, "entity": "phone", "reason": "seen"}],
                }
            ],
        },
    )
    out = tmp_path / "extended"
    worker().extend(base, media, decisions, out)
    manifest = check_supervision(out / "semantic")
    assert manifest["folds"]["Tao-Hsin"]["counts"] == {"train": 3, "val": 2, "test": 3}
    report = json.loads((out / "extension_report.json").read_text())
    assert report["prior_masks_preserved"] == 9 and report["prior_assignments_preserved"]
