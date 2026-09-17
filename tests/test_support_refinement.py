"""A tabletop corrects the support geometry without silently losing supported devices."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image, ImageDraw
from scipy.ndimage import distance_transform_edt

from syncai_bev3d import support_refinement as sr
from syncai_bev3d.meshes import Placement, place, round_table
from syncai_bev3d.object_instances import ObjectInstance, save_instances
from syncai_bev3d.scene_mesh import counter
from syncai_hydranet.geometry.camera_json import CameraFile, Lens
from syncai_hydranet.geometry.ground import (
    Camera,
    GroundPlane,
    pixel_to_ground,
    undistort_points,
)


def example():
    cf = CameraFile(
        camera_id="support",
        image_size_px=(800, 600),
        camera=Camera(640, 640, 400, 300),
        plane=GroundPlane(2.8, np.radians(35), 0),
    )
    truth = place(counter(1, 2.2, 0.85), Placement(0.2, 3.5, 0.3))
    initial = place(counter(1.18, 2, 0.78), Placement(0.3, 3.4, 0.5))
    return cf, truth, initial


def observed(mesh, cf, *, top=False):
    vertices, faces = mesh
    if top:
        vertices = vertices[np.isclose(vertices[:, 1], vertices[:, 1].max())]
    level = vertices.copy()
    level[:, 1] = cf.plane.height - level[:, 1]
    p = level @ cf.plane.rotation.T
    px = p[:, :2] / p[:, 2, None] * [cf.camera.fx, cf.camera.fy] + [cf.camera.cx, cf.camera.cy]
    image = Image.new("1", cf.image_size_px)
    draw = ImageDraw.Draw(image)
    if top:
        centre = px.mean(0)
        order = np.argsort(np.arctan2(px[:, 1] - centre[1], px[:, 0] - centre[0]))
        draw.polygon(list(map(tuple, px[order])), fill=1)
    else:
        for f in faces:
            draw.polygon(list(map(tuple, px[f])), fill=1)
    return np.asarray(image, bool)


def test_joint_top_body_fit_recovers_known_support_pose_and_height():
    cf, truth, initial = example()
    candidate, audit = sr.refine_support(
        initial, 0.5, observed(truth, cf, top=True), observed(truth, cf), cf, counter
    )
    assert audit["accepted"], audit
    assert audit["after_top_iou"] > 0.9
    assert audit["after_body_iou"] > 0.9
    assert np.allclose(audit["after_parameters"], [0.2, 3.5, 0.3, 1, 2.2, 0.85], atol=0.10)
    assert candidate[0][:, 1].min() == 0


def test_already_aligned_surface_is_preserved_exactly():
    cf, truth, _ = example()
    candidate, audit = sr.refine_support(
        truth, 0.3, observed(truth, cf, top=True), observed(truth, cf), cf, counter
    )
    assert not audit["accepted"]
    assert candidate is truth


def test_cropped_body_cannot_change_support_height():
    cf, truth, initial = example()
    body = observed(truth, cf).copy()
    body[-1, 300:340] = True
    _, audit = sr.refine_support(initial, 0.5, observed(truth, cf, top=True), body, cf, counter)
    assert audit["frame_cropped"]
    assert audit["after_parameters"][5] == audit["before_parameters"][5]


def fixture_case(tmp_path, *, circular=False):
    cf, truth, initial = example()
    source = tmp_path / "plate.png"
    Image.new("RGB", cf.image_size_px).save(source)
    cf = replace(cf, plate_file="plate.png")
    mask = observed(truth, cf, top=True)
    dest = tmp_path / "runs/commission01/support/masks"
    save_instances(
        dest / "support_tops.npz",
        [ObjectInstance("table_top", mask, 0.9, "table top")],
        shape=mask.shape,
        source=source,
        categories=["table_top"],
    )
    ev = SimpleNamespace(cf=cf, objects=np.ones(mask.shape, int), static=np.full(mask.shape, 4))
    if circular:
        initial = place(round_table(1, 0.85), Placement(0.2, 3.5))
    return ev, initial, truth, mask, dest, source


def test_round_table_is_not_deformed_by_rectangular_refinement(tmp_path):
    ev, initial, _, _, _, _ = fixture_case(tmp_path, circular=True)
    items = [(initial, "display_table", 255, True)]
    report = []
    result = sr.refine_scene_supports(
        "support", ev, tmp_path, items, 0.5, counter, report=report
    )
    assert result[0][0] is initial and report == []


def test_candidate_that_loses_existing_device_is_rejected(tmp_path, monkeypatch):
    ev, initial, truth, mask, dest, source = fixture_case(tmp_path)
    save_instances(
        dest / "object_instances.npz",
        [ObjectInstance("laptop", mask, 0.9, "laptop")],
        shape=mask.shape,
        source=source,
        categories=["laptop"],
    )

    def refine(*_args, **_kwargs):
        return truth, {
            "accepted": True,
            "reason": "candidate",
            "before_top_iou": 0.5,
            "after_top_iou": 0.9,
            "before_parameters": sr.mesh_parameters(initial, 0.5).tolist(),
            "after_parameters": sr.mesh_parameters(truth, 0.3).tolist(),
        }

    monkeypatch.setattr(sr, "refine_support", refine)

    def device(_device, _cf, supports):
        return (
            (initial, {"silhouette_iou": 0.8}) if supports[0].name == "before" else (None, {})
        )

    monkeypatch.setattr(sr, "fit_instance", device)
    report = []
    result = sr.refine_scene_supports(
        "support",
        ev,
        tmp_path,
        [(initial, "display_table", 255, True)],
        0.5,
        counter,
        report=report,
    )
    assert result[0][0] is initial
    assert not report[0]["accepted"] and "device" in report[0]["reason"]


def test_geometry_body_is_matched_by_pixels_after_student_ids_change(tmp_path):
    ev, initial, truth, _top, dest, source = fixture_case(tmp_path)
    full_body = observed(truth, ev.cf)
    student_body = full_body.copy()
    student_body[:, 390:415] = False  # confidence hole, not a hole in the cabinet
    ev.objects = student_body.astype(int) * 47  # a newly assigned student ID
    save_instances(
        dest / "support_bodies.npz",
        [ObjectInstance("fixture_body", full_body, 1.0, "source geometry")],
        shape=full_body.shape,
        source=source,
        categories=["fixture_body"],
    )
    report = []
    result = sr.refine_scene_supports(
        "support",
        ev,
        tmp_path,
        [(initial, "display_table", 255, True)],
        0.5,
        counter,
        report=report,
    )
    assert report[0]["object_id"] == 47
    assert report[0]["body_reference"]["kind"] == "source-bound commissioning geometry proposal"
    assert report[0]["accepted"], report
    assert report[0]["after_top_iou"] > 0.9 and report[0]["after_body_iou"] > 0.9
    assert result[0][0] is not initial


@pytest.mark.parametrize("shape", [(540, 960), (180, 320)])
def test_distorted_support_projection_matches_inverse_camera_rays(shape):
    """Independent pixel-to-plane reference catches straight raw-edge shortcuts."""
    from syncai_bev3d.scene_audit import raw_silhouette

    cf = CameraFile(
        camera_id="wide",
        image_size_px=(960, 540),
        camera=Camera(380, 380, 480, 270),
        plane=GroundPlane(2.8, 0.6, -0.2),
        lens=Lens(-0.45, (480, 270), np.hypot(960, 540) / 2),
    )
    mesh = place(counter(3.5, 2.5, 0.8), Placement(-0.25, 2.75, 0))
    rows, cols = np.indices(shape)
    raw = np.c_[cols.ravel(), rows.ravel()] * (np.array(cf.image_size_px) / shape[::-1])
    ideal = undistort_points(raw, cf.lens.k1, cf.lens.centre_px, cf.lens.radius_px)
    x, z = pixel_to_ground(ideal[:, 0], ideal[:, 1], cf.camera, replace(cf.plane, height=2.0))
    reference = ((x >= -2) & (x <= 1.5) & (z >= 1.5) & (z <= 4)).reshape(shape)
    projected = sr.raster(mesh, cf, shape, top=True)
    # Polygon rasterisation covers boundary pixels; inverse rays sample their centres.
    # Integer filling plus sampled curved edges may differ by two boundary pixels.
    # Never permit errors deeper in the interior, even at low raster resolution.
    assert distance_transform_edt(~reference)[projected & ~reference].max(initial=0) <= 2
    assert distance_transform_edt(reference)[reference & ~projected].max(initial=0) <= 2
    vertices, faces = mesh
    top_faces = faces[np.isclose(vertices[faces, 1], 0.8).all(1)]
    assert np.array_equal(projected, raw_silhouette(vertices, top_faces, cf, shape=shape))


def test_support_crossing_near_plane_keeps_its_visible_triangles():
    cf = CameraFile(
        camera_id="near",
        image_size_px=(100, 100),
        camera=Camera(50, 50, 50, 50),
        plane=GroundPlane(2, 0, 0),
    )
    vertices = np.array([[-0.02, 1.98, -0.2], [0.5, 1.5, 2], [-0.5, 1.5, 2]])
    mask = sr.raster((vertices, np.array([[0, 1, 2]])), cf, (100, 100))
    assert mask.any()
    assert not sr.raster((vertices, np.empty((0, 3), int)), cf, (100, 100)).any()
