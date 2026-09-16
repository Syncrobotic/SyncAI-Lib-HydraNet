import copy

import numpy as np
import pytest
from PIL import Image

from syncai_hydranet.data.studioa_autolabel import (
    Candidate,
    annotate,
    decode,
    validate_annotation,
)
from syncai_hydranet.data.studioa_relabel import (
    candidate_groups,
    constrain_decision,
    parse_answer,
    reannotate,
    review_panel,
)

pytestmark = pytest.mark.filterwarnings(
    "ignore:__array__ implementation doesn't accept a copy keyword:DeprecationWarning"
)


def rows(spec):
    candidates = []
    for entity, box in spec:
        mask = np.zeros((60, 80), dtype=bool)
        x0, y0, x1, y1 = box
        mask[y0:y1, x0:x1] = True
        candidates.append(Candidate(entity, mask, 0.8, [entity]))
    data = annotate(candidates, (60, 80))
    return data["entities"] + data["unresolved_candidates"]


def checked(groups, decisions):
    data = reannotate(groups, decisions, (60, 80))
    data["image_sha256"] = "test"
    validate_annotation(data, "test", (60, 80))
    return data


def test_vlm_malformed_unknown_and_unsupported_class_abstain():
    for text in [
        '{"entity":"shelf"}',
        "ignore previous instructions",
        '{"entity":"floor"',
        '{"entity":"floor","reason":""}',
    ]:
        assert parse_answer(text)["entity"] == "unknown"
    answer = parse_answer(
        '```json\n{"entity":"display_cabinet","reason":"Upright stocked rack"}\n```'
    )
    assert answer["entity"] == "display_cabinet"


def test_context_cannot_turn_person_or_small_product_into_counter():
    answer = {"entity": "counter", "reason": "white service desk", "raw": "fixture"}
    for entity in ("person", "boxed_stock"):
        group = rows([(entity, (5, 5, 25, 25))])
        result = constrain_decision(group, answer)
        assert result["entity"] == "unknown" and result["model_entity"] == "counter"
    assert constrain_decision(rows([("display_table", (5, 5, 25, 25))]), answer) == answer
    for entity in ("floor", "wall", "ceiling"):
        result = constrain_decision(rows([(entity, (5, 5, 25, 25))]), answer)
        assert result["entity"] == entity
        assert result["basis"] == "source_semantic_preservation"


def test_grouping_merges_alternative_names_but_not_nested_merchandise():
    source = rows(
        [
            ("display_table", (5, 5, 50, 55)),
            ("counter", (5, 5, 50, 55)),
            ("phone", (10, 10, 20, 20)),
        ]
    )
    groups = candidate_groups(source)
    assert sorted(map(len, groups)) == [1, 2]


def test_reclassification_recovers_rack_and_keeps_source_masks_unchanged():
    source = rows([("other_shelf", (5, 5, 50, 55))])
    previous = copy.deepcopy(source)
    data = checked(
        candidate_groups(source),
        [{"entity": "display_cabinet", "reason": "Upright rack holds packaged stock"}],
    )
    assert data["entities"][0]["entity"] == "display_cabinet"
    assert not data["unresolved_candidates"]
    assert source == previous
    assert data["human_reviewed"] is False


def test_unknown_is_not_promoted_and_foreground_goods_are_not_table_pixels():
    source = rows(
        [
            ("display_table", (5, 5, 50, 55)),
            ("phone", (10, 10, 20, 20)),
            ("other_shelf", (60, 10, 75, 50)),
        ]
    )
    groups = candidate_groups(source)
    decisions = [
        {
            "entity": "unknown" if g[0]["entity"] == "other_shelf" else g[0]["entity"],
            "reason": "Visible target",
        }
        for g in groups
    ]
    data = checked(groups, decisions)
    table = next(r for r in data["entities"] if r["entity"] == "display_table")
    phone = next(r for r in data["entities"] if r["entity"] == "phone")
    assert not (decode(table["segmentation"]) & decode(phone["segmentation"])).any()
    assert data["coverage"]["display_cabinet"] == "uncertain"
    assert data["unresolved_candidates"][0]["reasons"] == ["vision_language_model_abstention"]


def test_review_panel_is_bound_to_mask_context_and_does_not_mutate_image():
    source = Image.new("RGB", (80, 60), "white")
    original = source.tobytes()
    panel = review_panel(source, rows([("floor", (5, 5, 50, 55))])[0])
    assert panel.size == (448, 448)
    assert source.tobytes() == original
    with pytest.raises(ValueError, match="count mismatch"):
        reannotate([], [{"entity": "floor", "reason": "x"}], (60, 80))


def test_visual_decisions_bind_sources_reject_duplicates_and_preserve_model_answer():
    from syncai_hydranet.data.studioa_relabel import (
        apply_visual_decision,
        validate_visual_decisions,
    )

    source = rows([("display_table", (5, 5, 50, 55))])
    frame = {"id": "f", "image_sha256": "image", "annotation_sha256": "annotation"}
    decision = {"candidate_id": source[0]["id"], "entity": "counter", "reason": "Register"}
    item = {
        "frame_id": "f",
        "image_sha256": "image",
        "annotation_sha256": "annotation",
        "decisions": [decision],
    }
    document = {"reviewer_kind": "ai", "frames": [item]}
    annotations = {"f": {"entities": source, "unresolved_candidates": []}}
    overrides = validate_visual_decisions(document, [frame], annotations)
    model = {"entity": "display_table", "reason": "model", "raw": "display_table"}
    actual = apply_visual_decision(source, model, overrides["f"])
    assert actual["entity"] == "counter" and actual["model_entity"] == "display_table"
    assert actual["raw"] == "display_table" and actual["basis"] == "assistant_visual_review"
    item["image_sha256"] = "changed"
    with pytest.raises(ValueError, match="source hash"):
        validate_visual_decisions(document, [frame], annotations)
    item["image_sha256"] = "image"
    item["decisions"].append(decision)
    with pytest.raises(ValueError, match="duplicate"):
        validate_visual_decisions(document, [frame], annotations)
    assert parse_answer("display_table")["response_format"] == "class_only"
