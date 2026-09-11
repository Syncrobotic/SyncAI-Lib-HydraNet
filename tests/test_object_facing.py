"""Appearance reviews must never move assets or silently survive changed evidence."""

import json
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from syncai_bev3d.object_facing import (
    apply_observation,
    describe_orientation,
    load_observations,
    mask_identity,
)
from syncai_bev3d.object_instances import ObjectInstance, image_digest, save_instances
from syncai_bev3d.object_placement import Silhouette, scene_objects
from test_object_placement import camera, observed


def fixture():
    from syncai_bev3d.meshes import Placement, place
    from syncai_bev3d.object_assets import asset_mesh

    cf = camera()
    local = asset_mesh("monitor", 0.5, 0.2, 0.45)
    mesh = place((local[0] + [0, 0.8, 0], local[1]), Placement(0.1, 3.0, 0.0))
    mask = observed(mesh, cf)
    record = {
        "category": "monitor",
        "variant": "standard",
        "x_m": 0.1,
        "z_m": 3.0,
        "base_m": 0.8,
        "heading_deg": 0.0,
        "dimensions_m": [0.5, 0.2, 0.45],
        "silhouette_iou": Silhouette(mask, cf).score(mesh),
        "heading_ambiguous": True,
        "dimension_bound_hit": False,
        "status": "placed",
    }
    describe_orientation(record)
    return cf, mesh, mask, record


def test_review_flip_changes_heading_without_moving_or_resizing_asset():
    cf, mesh, _mask, record = fixture()
    original = mesh[0].copy()
    result = apply_observation(mesh, record, cf, {"visible_side": "back"}, lambda _m: 0.99)
    assert record["heading_deg"] == -180
    assert record["front_back_status"] == "reviewed_appearance"
    assert not record["heading_requires_review"]
    assert record["silhouette_heading_ambiguous"]
    assert record["axis_heading_deg"] == 0
    assert np.array_equal(mesh[0], original)
    assert np.allclose(np.ptp(result[0], axis=0), np.ptp(original, axis=0))
    assert np.array_equal(result[0][:, 1], original[:, 1])
    assert np.allclose(result[0][:, [0, 2]] + original[:, [0, 2]], [0.2, 6.0])
    assert (record["x_m"], record["z_m"], record["base_m"]) == (0.1, 3.0, 0.8)
    assert record["dimensions_m"] == [0.5, 0.2, 0.45]


def test_matching_review_keeps_mesh_and_confirms_facing():
    cf, mesh, _mask, record = fixture()
    result = apply_observation(mesh, record, cf, {"visible_side": "front"}, None)
    assert result is mesh
    assert record["facing_result"] == "confirmed existing direction"


@pytest.mark.parametrize("candidate", [0.41, 0.85, float("nan")])
def test_conflicting_silhouette_cannot_be_overridden(candidate):
    cf, mesh, _mask, record = fixture()
    record["silhouette_iou"] = 0.95
    result = apply_observation(mesh, record, cf, {"visible_side": "back"}, lambda _m: candidate)
    assert result is mesh
    assert record["heading_deg"] == 0
    assert record["heading_requires_review"]
    assert record["front_back_status"] == "unverified"


def test_grazing_view_remains_unresolved():
    cf, mesh, _mask, record = fixture()
    record["heading_deg"] = 90.0
    assert apply_observation(mesh, record, cf, {"visible_side": "front"}, None) is mesh
    assert record["heading_requires_review"]
    assert "edge-on" in record["facing_result"]


@pytest.mark.parametrize(
    "category,variant", [("phone", "flat"), ("tablet", "flat"), ("stool", "standard")]
)
def test_symmetric_templates_have_an_axis_without_claiming_front_back(category, variant):
    cf, mesh, _mask, record = fixture()
    record.update(category=category, variant=variant)
    describe_orientation(record)
    assert record["heading_ambiguous"]
    assert not record["heading_requires_review"]
    assert record["front_back_status"] == "not_represented_by_template"
    assert apply_observation(mesh, record, cf, {"visible_side": "front"}, None) is mesh
    assert record["front_back_status"] != "reviewed_appearance"


def review_file(tmp_path, instance):
    source = tmp_path / "plate.png"
    source.write_bytes(b"source image identity")
    cue = {
        "instance_id": 0,
        "visible_side": "back",
        "reviewer": "test reviewer",
        "evidence": "rear casing",
        "mask_sha256": mask_identity(instance),
    }
    data = {
        "schema_version": 1,
        "camera": "objects",
        "source_sha256": image_digest(source),
        "observations": [cue],
    }
    path = tmp_path / "object_facing.json"
    path.write_text(json.dumps(data))
    return path, source, data


def test_review_is_bound_to_plate_camera_and_mask_but_not_confidence(tmp_path):
    instance = ObjectInstance("monitor", np.eye(4, dtype=bool), 0.9, "monitor")
    path, source, _data = review_file(tmp_path, instance)
    assert load_observations(path, source, [replace(instance, score=0.8)], "objects")[0]
    assert not load_observations(path, source, [instance], "other")[0]
    assert not load_observations(
        path, source, [replace(instance, mask=~instance.mask)], "objects"
    )[0]
    assert not load_observations(
        path, source, [replace(instance, category="tablet")], "objects"
    )[0]
    source.write_bytes(b"changed plate")
    assert not load_observations(path, source, [instance], "objects")[0]


def test_duplicate_ids_are_rejected_even_when_first_observation_is_stale(tmp_path):
    instance = ObjectInstance("monitor", np.eye(4, dtype=bool), 0.9, "monitor")
    path, source, data = review_file(tmp_path, instance)
    data["observations"] *= 2
    data["observations"][0]["mask_sha256"] = "stale"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="unique"):
        load_observations(path, source, [instance], "objects")


def test_scene_build_applies_review_using_real_projected_silhouette(tmp_path, monkeypatch):
    cf, mesh, mask, record = fixture()
    instance = ObjectInstance("monitor", mask, 0.9, "monitor")
    path, source, data = review_file(tmp_path, instance)
    # The correctly facing monitor is behind the mask; the fit starts reversed.
    reversed_mesh = mesh[0].copy(), mesh[1]
    reversed_mesh[0][:, 0] = 0.2 - reversed_mesh[0][:, 0]
    reversed_mesh[0][:, 2] = 6.0 - reversed_mesh[0][:, 2]
    record["heading_deg"] = 180.0
    record["silhouette_iou"] = Silhouette(mask, cf).score(reversed_mesh)
    data["observations"][0]["visible_side"] = "front"
    archive = tmp_path / "runs/commission01/objects/masks/object_instances.npz"
    save_instances(archive, [instance], shape=mask.shape, source=source, categories=["monitor"])
    archive.with_name(path.name).write_text(json.dumps(data))
    monkeypatch.setattr(
        "syncai_bev3d.object_placement.fit_instance",
        lambda *_a, **_kw: (reversed_mesh, record.copy()),
    )
    ev = SimpleNamespace(cf=replace(cf, plate_file=str(source)), walk=np.ones_like(mask))
    reports = []
    built, _ = scene_objects("objects", ev, tmp_path, [], report=reports)
    assert len(built) == 1
    assert np.allclose(built[0][0][0], mesh[0])
    assert reports[0]["front_back_status"] == "reviewed_appearance"
    assert reports[0]["silhouette_iou"] >= record["silhouette_iou"]
