"""Stage0 assets must recover calibrated poses and obey real support boundaries."""

import math
from dataclasses import replace

import numpy as np
import pytest
from PIL import Image, ImageDraw

from syncai_bev3d.meshes import Placement, place, round_table
from syncai_bev3d.object_assets import (
    NOMINAL,
    asset_mesh,
    door_mesh,
    plausible_aspect,
    templates,
)
from syncai_bev3d.object_instances import (
    ObjectInstance,
    deduplicate,
    load_instances,
    save_instances,
)
from syncai_bev3d.object_placement import (
    Silhouette,
    Support,
    fit_instance,
    fixture_supports,
    footprint,
    overlap,
)
from syncai_hydranet.geometry.camera_json import CameraFile, Lens
from syncai_hydranet.geometry.ground import Camera, GroundPlane, distort_points


def camera():
    return CameraFile(
        camera_id="objects",
        image_size_px=(800, 600),
        camera=Camera(fx=640, fy=640, cx=400, cy=300),
        plane=GroundPlane(height=2.8, pitch=math.radians(28), roll=math.radians(4)),
    )


def observed(mesh, cf):
    # Generate the observation from known world coordinates independently of the fitter.
    vertices, faces = mesh
    level = vertices.copy()
    level[:, 1] = cf.plane.height - level[:, 1]
    points = level @ cf.plane.rotation.T
    px = points[:, :2] / points[:, 2, None] * [cf.camera.fx, cf.camera.fy] + [
        cf.camera.cx,
        cf.camera.cy,
    ]
    if cf.lens:
        px = distort_points(px, cf.lens.k1, cf.lens.centre_px, cf.lens.radius_px)
    image = Image.new("1", cf.image_size_px)
    draw = ImageDraw.Draw(image)
    for face in faces:
        draw.polygon(list(map(tuple, px[face])), fill=1)
    return np.asarray(image, bool)


@pytest.mark.parametrize("category", list(NOMINAL))
def test_templates_have_the_requested_metric_bounds_and_rest_at_zero(category):
    for variant, dimensions in templates(category):
        vertices, faces = asset_mesh(category, *dimensions, variant=variant)
        assert np.isfinite(vertices).all()
        assert np.allclose(np.ptp(vertices, axis=0), np.array(dimensions)[[0, 2, 1]])
        assert vertices[:, 1].min() == 0
        assert faces.min() >= 0 and faces.max() < len(vertices)


def test_round_and_rotated_supports_reject_empty_aabb_corners():
    supports = fixture_supports([(round_table(1, 0.8), "display_table", 255, True)])
    assert supports[0].contains([[0.30, 0.30]])
    assert not supports[0].contains([[0.45, 0.45]])
    rotated = Support("rotated", footprint([0, 0, math.pi / 4, 2, 0.4, 1]), 0.8)
    assert rotated.contains([[0.5, -0.5]])
    assert not rotated.contains([[0.5, 0.5]])


def test_overlap_uses_rotated_polygons_not_aabbs_and_allows_touching():
    a = footprint([0, 0, math.pi / 4, 2, 0.2, 1])
    b = a + np.array([0.4, 0.4])
    assert not overlap(a, b)
    assert overlap(a, a + np.array([0.01, 0.01]))
    assert not overlap(footprint([0, 0, 0, 1, 1, 1]), footprint([1, 0, 0, 1, 1, 1]))


@pytest.mark.parametrize("with_lens", [False, True])
def test_laptop_recovers_non_store_heading_scale_and_table_contact(with_lens):
    cf = camera()
    if with_lens:
        cf = replace(cf, lens=Lens(k1=-0.20, centre_px=(400, 300), radius_px=500))
    dimensions = np.array([0.37, 0.28, 0.25])
    local = asset_mesh("laptop", *dimensions)
    mesh = place((local[0] + [0, 0.83, 0], local[1]), Placement(0.2, 3.1, 0.7))
    mask = observed(mesh, cf)
    support = Support("table", np.array([[-1, 2], [1, 2], [1, 4], [-1, 4]]), 0.83)
    fitted, audit = fit_instance(
        ObjectInstance("laptop", mask, 0.95, "laptop"), cf, [support], maxiter=100
    )
    assert fitted is not None, audit
    assert audit["silhouette_iou"] > 0.85
    assert np.allclose([audit["x_m"], audit["z_m"]], [0.2, 3.1], atol=0.06)
    assert abs(audit["heading_deg"] - math.degrees(0.7)) < 12
    assert np.allclose(audit["dimensions_m"], dimensions, atol=0.055)
    assert fitted[0][:, 1].min() == pytest.approx(0.83)


def test_devices_without_support_are_not_floated_at_a_guessed_height():
    cf = camera()
    mesh = place(asset_mesh("monitor", *NOMINAL["monitor"]), Placement(0.2, 3.1, 0.7))
    mask = observed(mesh, cf)
    result, report = fit_instance(ObjectInstance("monitor", mask, 0.9, "monitor"), cf, [])
    assert result is None and report["status"] == "rejected"


def test_occluded_chair_requires_floor_evidence():
    cf = camera()
    mask = observed(place(asset_mesh("chair", *NOMINAL["chair"]), Placement(0, 3.2)), cf)
    mesh, report = fit_instance(
        ObjectInstance("chair", mask, 0.9, "chair"), cf, [], floor_mask=np.zeros_like(mask)
    )
    assert mesh is None and "floor" in report["reason"]


def test_silhouette_preserves_gaps_between_chair_legs():
    cf = camera()
    mesh = place(asset_mesh("chair", *NOMINAL["chair"]), Placement(0, 3.2, 0.5))
    mask = observed(mesh, cf)
    assert Silhouette(mask, cf).score(mesh) > 0.90
    from syncai_bev3d.meshes import box

    hull = place(box(0.48, 0.88, 0.52), Placement(0, 3.2, 0.5))
    assert Silhouette(mask, cf).score(hull) < 0.8


def test_prompt_dedup_prefers_whole_laptop_but_does_not_relabel_an_imac():
    full = np.zeros((30, 30), bool)
    full[5:25, 5:25] = True
    lid = full.copy()
    lid[18:] = False
    laptop = ObjectInstance("laptop", full, 0.7, "laptop")
    monitor = ObjectInstance("monitor", lid, 0.9, "monitor")
    assert deduplicate([monitor, laptop])[0].category == "laptop"
    monitor.mask = full
    result = deduplicate([monitor, laptop])
    assert len(result) == 1 and result[0].category == "monitor"


def test_archive_preserves_identity_and_rejects_a_different_plate(tmp_path):
    source = tmp_path / "plate.png"
    source.write_bytes(b"plate one")
    path = tmp_path / "objects.npz"
    mask = np.zeros((17, 21), bool)
    mask[3:8, 7:10] = True
    save_instances(
        path,
        [ObjectInstance("laptop", mask, 0.9, "open laptop")],
        shape=mask.shape,
        source=source,
        categories=["laptop"],
    )
    loaded, metadata = load_instances(path, source=source)
    assert len(loaded) == 1 and np.array_equal(loaded[0].mask, mask)
    assert metadata["categories"] == ["laptop"]
    source.write_bytes(b"plate two")
    with pytest.raises(ValueError, match="different plate"):
        load_instances(path, source=source)


def test_empty_detection_is_a_valid_completed_pass(tmp_path):
    source = tmp_path / "plate"
    source.write_bytes(b"empty")
    path = tmp_path / "objects.npz"
    save_instances(path, [], shape=(17, 21), source=source, categories=["laptop"])
    instances, metadata = load_instances(path)
    assert instances == [] and metadata["categories"] == ["laptop"]


def test_door_jambs_keep_the_fitted_plane_and_width():
    vertices, _faces = door_mesh([[1, 3], [1, 4]], 2.1)
    assert vertices[:, 1].min() == 0
    assert np.allclose(np.ptp(vertices, axis=0), [0.08, 2.1, 1])
    assert np.allclose(vertices[:, [0, 2]].mean(0), [1, 3.5])


def test_dimension_bounds_alone_do_not_allow_a_portrait_shaped_laptop():
    assert not plausible_aspect("laptop", [0.24, 0.30, 0.23], NOMINAL["laptop"], "standard")
    assert plausible_aspect("laptop", [0.34, 0.26, 0.23], NOMINAL["laptop"], "standard")
    assert plausible_aspect("tablet", [0.19, 0.25, 0.012], [0.19, 0.25, 0.012], "flat")
    assert not plausible_aspect("tablet", [0.19, 0.40, 0.012], [0.19, 0.25, 0.012], "flat")
