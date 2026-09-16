import copy

import numpy as np
import pytest

from syncai_hydranet.data.studioa_autolabel import (
    Candidate,
    annotate,
    coco_annotations,
    decode,
    overlap,
    validate_annotation,
)

# pycocotools' Cython bridge predates NumPy's copy keyword; exact RLE round trips
# below still verify the decoded pixels rather than hiding an annotation mismatch.
pytestmark = pytest.mark.filterwarnings(
    "ignore:__array__ implementation doesn't accept a copy keyword:DeprecationWarning"
)


def candidate(name, prompt=None, box=(5, 5, 20, 20), score=0.9):
    mask = np.zeros((50, 60), bool)
    x0, y0, x1, y1 = box
    mask[y0:y1, x0:x1] = True
    return Candidate(name, mask, score, [prompt or name])


def bound(candidates):
    data = annotate(candidates, (50, 60))
    data["image_sha256"] = "fixture"
    validate_annotation(data, "fixture", (50, 60))
    return data


def test_same_door_preserves_glazing_without_a_second_instance():
    data = bound([candidate("door"), candidate("door", "glass door", score=0.8)])
    assert len(data["entities"]) == 1
    assert data["counts_by_view"] == {"door": 1, "glass_door": 1}
    assert data["entities"][0]["glazed"] is True
    assert data["human_reviewed"] is False
    coco = coco_annotations(data, 7, 100)
    assert len(coco) == 1 and coco[0]["image_id"] == 7
    np.testing.assert_array_equal(decode(coco[0]["segmentation"]), candidate("door").mask)


def test_conflicting_products_remain_unresolved_instead_of_confidence_argmax():
    data = bound([candidate("phone"), candidate("tablet", score=0.6)])
    assert not data["entities"]
    assert len(data["unresolved_candidates"]) == 2
    assert data["coverage"]["phone"] == "uncertain"
    assert data["coverage"]["fire_equipment"] == "not_detected"
    assert not coco_annotations(data, 1, 1)


def test_printed_device_and_glass_parts_do_not_become_independent_objects():
    data = bound([candidate("poster", box=(0, 0, 30, 30)), candidate("phone")])
    assert [row["entity"] for row in data["entities"]] == ["poster"]
    assert data["unresolved_candidates"][0]["reasons"] == ["possible_poster_depiction"]
    data = bound([candidate("display_cabinet", box=(0, 0, 30, 30)), candidate("glass_panel")])
    assert [row["entity"] for row in data["entities"]] == ["display_cabinet"]
    assert data["unresolved_candidates"]


def test_unknown_role_and_material_are_not_absence():
    data = bound([candidate("door"), candidate("counter", box=(30, 30, 45, 45))])
    assert data["coverage"]["glass_door"] == "uncertain"
    assert data["coverage"]["checkout_counter"] == "uncertain"
    assert "absent" not in data["coverage"].values()


def test_mask_provenance_and_geometry_are_checked():
    data = bound([candidate("floor")])
    with pytest.raises(ValueError, match="identity"):
        validate_annotation(data, "another image", (50, 60))
    for key, value in [("area", 100), ("bbox_xywh", [0, 0, 1, 1])]:
        changed = copy.deepcopy(data)
        changed["entities"][0][key] = value
        with pytest.raises(ValueError):
            validate_annotation(changed, "fixture", (50, 60))
    data["human_reviewed"] = True
    with pytest.raises(ValueError, match="identity"):
        validate_annotation(data, "fixture", (50, 60))


def test_small_noise_is_filtered_and_disjoint_instances_stay_separate():
    data = bound(
        [
            candidate("phone", box=(1, 1, 3, 3)),
            candidate("phone"),
            candidate("phone", box=(30, 30, 45, 45)),
        ]
    )
    assert len(data["entities"]) == 2
    assert data["filtered_counts"]["small_or_low_score"] == 1
    assert overlap(candidate("phone"), candidate("phone", box=(30, 30, 45, 45))) == (0, 0)


def test_floor_tiles_are_regions_not_multiple_physical_floor_instances():
    data = bound(
        [
            candidate("floor", box=(0, 0, 30, 30)),
            candidate("floor", "floor tiles", box=(1, 1, 10, 10), score=0.7),
        ]
    )
    assert len(data["entities"]) == 1
    assert data["entities"][0]["geometry_kind"] == "semantic_region"
    assert data["entities"][0]["area"] == 900
    assert data["filtered_counts"]["semantic_regions_merged"] == 1


def test_worker_freezes_writes_resumes_and_rejects_foreign_checkpoints(tmp_path, monkeypatch):
    import importlib.util
    import json
    from pathlib import Path

    from PIL import Image

    from syncai_hydranet.data.studioa_review import prepare_package

    tool_path = Path(__file__).resolve().parents[1] / "tools/annotation/studioa_autolabel.py"
    spec = importlib.util.spec_from_file_location("ai_worker_test", tool_path)
    assert spec is not None and spec.loader is not None
    worker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(worker)
    dataset = tmp_path / "source"
    for folder in ("images", "annotations"):
        path = dataset / folder / "train/Kaohsiung-cam01__session__0000.png"
        path.parent.mkdir(parents=True)
        if folder == "images":
            Image.new("RGB", (64, 48)).save(path)
        else:
            Image.fromarray(np.ones((48, 64), np.uint8)).save(path)
    bundle, out = tmp_path / "bundle", tmp_path / "ai"
    prepare_package([(dataset, "site30k_native")], bundle, 1)
    worker.prepare(bundle, out, None)
    monkeypatch.setattr(worker.sam3, "load_sam3", lambda *_args: (None, None))
    monkeypatch.setattr(worker.sam3, "vision_features", lambda *_args: None)
    monkeypatch.setattr(
        worker.sam3,
        "segment",
        lambda _p, _m, image, prompt, *_args: (
            [(np.ones((image.height, image.width), bool), 0.9)] if prompt == "floor" else []
        ),
    )
    monkeypatch.setattr(worker.signal, "signal", lambda *_args: None)
    worker.run(out, "cpu")
    assert json.loads((out / "status.json").read_text())["status"] == "completed"
    report = json.loads((out / "report.json").read_text())
    assert report["instances_by_entity"] == {"floor": 1}
    target = next((out / "frames").glob("*.json"))
    previous = target.read_bytes()
    worker.run(out, "cpu")
    assert target.read_bytes() == previous
    refined = tmp_path / "refined"
    worker.refine(out, refined)
    assert json.loads((refined / "status.json").read_text())["status"] == "completed"
    assert json.loads((refined / "report.json").read_text())["instances_by_entity"] == {
        "floor": 1
    }
    assert target.read_bytes() == previous
    refined_target = refined / "frames" / target.name
    refined_data = json.loads(refined_target.read_text())
    decisions_path = tmp_path / "decisions.json"
    worker.write(
        decisions_path,
        {
            "reviewer_kind": "ai",
            "frames": [
                {
                    "frame_id": refined_data["frame_id"],
                    "annotation_sha256": worker.digest(refined_target),
                    "image_sha256": refined_data["image_sha256"],
                    "decisions": [
                        {
                            "id": refined_data["entities"][0]["id"],
                            "action": "defer",
                            "reason": "synthetic ambiguity",
                        }
                    ],
                }
            ],
        },
    )
    visually_reviewed = tmp_path / "visually_reviewed"
    worker.visual_review(refined, decisions_path, visually_reviewed)
    review_report = json.loads((visually_reviewed / "report.json").read_text())
    assert review_report["instances"] == 0
    assert review_report["ai_visual_review"]["decision_count"] == 1
    assert json.loads(refined_target.read_text())["entities"]

    class FakeReviewer:
        def classify(self, panels):
            return [{"entity": "floor", "reason": "synthetic floor"} for _ in panels]

    monkeypatch.setattr(worker, "LocalReviewer", lambda _device: FakeReviewer())
    relabeled = tmp_path / "relabeled"
    worker.relabel_prepare(refined, relabeled)
    worker.relabel_run(relabeled, "cpu")
    relabeled_target = relabeled / "frames" / target.name
    relabeled_bytes = relabeled_target.read_bytes()
    assert json.loads((relabeled / "report.json").read_text())["instances_by_entity"] == {
        "floor": 1
    }
    worker.relabel_run(relabeled, "cpu")
    assert relabeled_target.read_bytes() == relabeled_bytes
    raw_path = relabeled / "raw" / target.name
    raw_path.write_text("{}")
    with pytest.raises(ValueError, match="changed relabel checkpoint"):
        worker.relabel_run(relabeled, "cpu")
    data = json.loads(previous)
    data["job_sha256"] = "other-policy"
    worker.write(target, data)
    with pytest.raises(ValueError, match="another job"):
        worker.run(out, "cpu")
    assert json.loads((out / "status.json").read_text())["status"] == "failed"


def test_semantic_conflict_does_not_discard_the_entire_floor():
    data = bound(
        [candidate("floor", box=(0, 0, 30, 30)), candidate("door", box=(0, 0, 10, 10))]
    )
    floors = [row for row in data["entities"] if row["entity"] == "floor"]
    assert len(floors) == 1 and floors[0]["area"] == 800
    conflicts = [row for row in data["unresolved_candidates"] if row["entity"] == "floor"]
    assert len(conflicts) == 1 and conflicts[0]["area"] == 100
    assert not (decode(floors[0]["segmentation"]) & decode(conflicts[0]["segmentation"])).any()


def test_ai_visual_review_only_defers_and_preserves_the_mask():
    from syncai_hydranet.data.studioa_autolabel import apply_ai_review

    original = bound([candidate("counter", "checkout counter")])
    row = original["entities"][0]
    reviewed = apply_ai_review(
        original,
        [
            {
                "id": row["id"],
                "action": "defer",
                "reason": "visible printer, not a verified counter",
            }
        ],
    )
    validate_annotation(reviewed, "fixture", (50, 60))
    assert not reviewed["entities"]
    assert original["entities"] == [row]
    assert reviewed["unresolved_candidates"][0]["segmentation"] == row["segmentation"]
    assert reviewed["coverage"]["checkout_counter"] == "uncertain"
    with pytest.raises(ValueError, match="defer"):
        apply_ai_review(original, [{"id": row["id"], "action": "promote", "reason": "guess"}])
