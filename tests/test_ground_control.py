"""Survey recovery with distorted cameras, withheld points and bad clicks."""

from dataclasses import replace

import numpy as np
import pytest
from PIL import Image

from syncai_bev3d.geometry_review import cache_ground_error, render_metre_grid
from syncai_bev3d.ground_control import floor_pixels, refine_ground_control, transfer_zones
from syncai_hydranet.geometry.camera_json import CameraFile, Lens, Zone
from syncai_hydranet.geometry.ground import Camera, GroundPlane


def survey_case():
    truth = CameraFile(
        camera_id="test",
        image_size_px=(960, 540),
        camera=Camera.from_vfov(540, 960, 70.4),
        plane=GroundPlane(2.7, np.radians(52), np.radians(3)),
        lens=Lens(-0.225, (480, 270), np.hypot(960, 540) / 2),
    )
    x, z = np.meshgrid(np.linspace(-1.8, 1.8, 5), np.linspace(1.4, 5.2, 4))
    local = np.column_stack((x.ravel(), z.ravel()))
    yaw = 0.38
    rotation = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
    points = (local - [0.6, 1.1]) @ rotation
    px = floor_pixels(truth, local)
    controls = {
        "camera_id": "test",
        "image_size_px": [960, 540],
        "pixel_space": "raw",
        "measurement_source": "synthetic surveyed grid; exact ground truth",
        "points_px": px.tolist(),
        "points_m": points.tolist(),
        "validation": [i in (2, 8, 11, 17) for i in range(len(px))],
    }
    initial = replace(truth, plane=GroundPlane(3.0, np.radians(48), np.radians(1)))
    return truth, initial, controls


@pytest.mark.parametrize("fit_focal", [False, True])
def test_recovers_pose_and_focal_from_independent_metric_grid(fit_focal):
    truth, initial, controls = survey_case()
    if fit_focal:
        initial = replace(
            initial,
            camera=replace(
                initial.camera, fx=initial.camera.fx * 1.13, fy=initial.camera.fy * 1.13
            ),
        )
    candidate, report = refine_ground_control(initial, controls, fit_focal=fit_focal)
    assert report["accepted"], report["reasons"]
    assert candidate.plane.height == pytest.approx(truth.plane.height, abs=1e-5)
    assert candidate.plane.pitch == pytest.approx(truth.plane.pitch, abs=1e-5)
    assert candidate.plane.roll == pytest.approx(truth.plane.roll, abs=1e-5)
    assert candidate.camera.fx == pytest.approx(truth.camera.fx, abs=1e-4)
    assert max(report["after"]["error_m"]) < 1e-5
    assert np.median(report["before"]["error_m"]) > 0.1


def test_bad_click_is_reported_and_does_not_set_the_scale():
    truth, initial, controls = survey_case()
    controls["points_px"][0][0] += 75
    candidate, report = refine_ground_control(initial, controls)
    assert report["accepted"], report["reasons"]
    assert not report["inliers"][0]
    assert candidate.plane.height == pytest.approx(truth.plane.height, abs=0.01)


def test_validation_error_cannot_be_fitted_away():
    _, initial, controls = survey_case()
    _, clean = refine_ground_control(initial, controls)
    controls["points_m"][2][0] += 1.0  # a held-out point
    _, bad = refine_ground_control(initial, controls)
    assert not bad["accepted"]
    assert bad["survey_to_camera_floor"] == clean["survey_to_camera_floor"]
    assert any("held-out" in reason for reason in bad["reasons"])


def test_no_holdout_is_a_candidate_not_a_metric_pass():
    _, initial, controls = survey_case()
    controls.pop("validation")
    _, report = refine_ground_control(initial, controls)
    assert not report["accepted"]
    assert any("validation" in reason for reason in report["reasons"])


@pytest.mark.parametrize(
    "field,value,match",
    [
        ("camera_id", "another", "camera_id"),
        ("image_size_px", [1920, 1080], "image_size"),
        ("pixel_space", "undistorted", "pixel_space"),
        ("measurement_source", "", "measurement_source"),
        ("validation", [1] * 20, "boolean"),
    ],
)
def test_mismatched_control_contract_is_refused(field, value, match):
    _, initial, controls = survey_case()
    controls[field] = value
    with pytest.raises(ValueError, match=match):
        refine_ground_control(initial, controls)


def test_collinear_survey_is_refused():
    _, initial, controls = survey_case()
    controls["points_m"] = [[float(i), 0] for i in range(20)]
    with pytest.raises(ValueError, match="collinear"):
        refine_ground_control(initial, controls)


def test_zone_transfer_keeps_its_raw_image_boundary_after_pose_change():
    truth, initial, _ = survey_case()
    initial = replace(initial, zones=(Zone("floor", "walkable", ((-1, 2), (1, 2), (0, 4))),))
    fresh = transfer_zones(initial, truth)
    np.testing.assert_allclose(
        floor_pixels(initial, np.asarray(initial.zones[0].points_m)),
        floor_pixels(fresh, np.asarray(fresh.zones[0].points_m)),
        atol=1e-9,
    )


def test_cache_drift_is_visible_at_a_different_resolution():
    truth, initial, _ = survey_case()
    yy, xx = np.mgrid[0:108, 0:192]
    ground = truth.ground_points(
        np.column_stack((xx.ravel() * 5, yy.ravel() * 5)), above_horizon="raise"
    )
    arrays = {"gx": ground[:, 0].reshape(108, 192), "gz": ground[:, 1].reshape(108, 192)}
    assert cache_ground_error(truth, arrays) < 1e-9
    assert cache_ground_error(initial, arrays) > 0.1


def test_grid_uses_raw_distortion_and_does_not_mutate_plate():
    truth, _, _ = survey_case()
    plate = Image.new("RGB", truth.image_size_px)
    grid = render_metre_grid(plate, truth)
    assert not np.asarray(plate).any()
    raw = floor_pixels(truth, np.array([[1.0, 2.0]]))[0]
    u, v = np.round(raw).astype(int)
    patch = np.asarray(grid)[v - 2 : v + 3, u - 2 : u + 3]
    assert patch.any(), "a metre intersection must be drawn in the distorted frame"
    with pytest.raises(ValueError, match="plate size"):
        render_metre_grid(plate.resize((480, 270)), truth)


def test_rebuilt_cache_tracks_camera_pose_depth_scale_and_unknown_pixels(tmp_path):
    from syncai_bev3d.geometry_cache import build_geometry_cache, save_geometry_cache
    from syncai_bev3d.geometry_review import load_geometry_cache

    truth, initial, _ = survey_case()
    truth = replace(truth, lens=None)
    yy, xx = np.mgrid[0:108, 0:192]
    cam = truth.camera.scaled_to(108, 192, (540, 960))
    rays = np.stack(((xx - cam.cx) / cam.fx, (yy - cam.cy) / cam.fy, np.ones_like(xx)), -1)
    down = (rays @ truth.plane.rotation)[..., 1]
    depth = truth.plane.height / down
    depth[20, 20] = 0
    arrays = build_geometry_cache(truth, depth / 2, depth_scale=2, frame_hw=(108, 192))
    assert np.nanmax(np.abs(arrays["height"])) < 1e-5
    assert not arrays["geom_ok"][20, 20]
    assert np.nanmedian(arrays["horiz"]) == pytest.approx(1, abs=1e-5)
    cache = tmp_path / "cache.npz"
    save_geometry_cache(cache, arrays)
    assert load_geometry_cache(cache, truth)["gx"].shape == (108, 192)
    with pytest.raises(ValueError, match="stale geometry cache"):
        load_geometry_cache(cache, initial)


def test_scene_mesh_refuses_stale_geometry_before_building_fixtures(tmp_path):
    from syncai_bev3d import scene_mesh
    from syncai_bev3d.geometry_cache import build_geometry_cache, save_geometry_cache

    truth, initial, _ = survey_case()
    commission = tmp_path / "runs/commission01"
    commission.mkdir(parents=True)
    initial.save(commission / "test.camera.json")
    cache = tmp_path / "runs/site30k_qa/geometry_cache/test.npz"
    arrays = build_geometry_cache(truth, np.ones((54, 96)), depth_scale=1, frame_hw=(54, 96))
    save_geometry_cache(cache, arrays)
    with pytest.raises(ValueError, match="stale geometry cache"):
        scene_mesh.cell_grids("test", tmp_path)


def _tool(name):
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "tools/commissioning" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_scene_overlay_projects_back_through_the_raw_lens():
    truth, _, _ = survey_case()
    points = np.array([[-1.7, 0, 2], [1.3, 0, 3.4]])
    pixels, depth = _tool("scene_overlay").project(points, truth, 1.0)
    assert (depth > 0).all()
    np.testing.assert_allclose(truth.ground_points(pixels), points[:, [0, 2]], atol=1e-9)


def test_commissioning_cli_writes_reviewable_bundle_and_refuses_overwrite(tmp_path):
    import json

    truth, _, controls = survey_case()
    plate = tmp_path / "plate.png"
    Image.new("RGB", truth.image_size_px).save(plate)
    calib = tmp_path / "calib.json"
    calib.write_text(
        json.dumps(
            {
                "camera": "test",
                "frame_hw_px": [540, 960],
                "height_m": 3.0,
                "pitch_deg": 48,
                "roll_deg": 1,
                "vfov_assumed_deg": 70.4,
                "k1_division_model": -0.225,
                "scale_source": "synthetic",
                "plate_used": str(plate),
            }
        )
    )
    control = tmp_path / "controls.json"
    control.write_text(json.dumps(controls))
    out = tmp_path / "review"
    args = [str(calib), "--controls", str(control), "--out", str(out)]
    cli = _tool("commission_camera")
    assert cli.main(args) == 0
    assert (out / "grid.before.png").exists() and (out / "grid.after.png").exists()
    report = json.loads((out / "geometry-review.json").read_text())
    assert report["accepted"]
    assert len(report["controls_sha256"]) == 64
    saved = CameraFile.load(out / "test.camera.json")
    assert saved.plane.height == pytest.approx(2.7)
    assert saved.plate_sha256 == report["plate_sha256"]
    assert saved.plate_file == str(plate)
    with pytest.raises(FileExistsError):
        cli.main(args)
    controls.pop("validation")
    control.write_text(json.dumps(controls))
    assert (
        cli.main(
            [str(calib), "--controls", str(control), "--out", str(tmp_path / "unverified")]
        )
        == 2
    )


def test_rebuild_cli_can_reuse_depth_without_running_teachers(tmp_path):
    truth, _, _ = survey_case()
    plate = tmp_path / "plate.png"
    Image.new("RGB", truth.image_size_px).save(plate)
    cf = replace(truth, plate_file=str(plate))
    camera_file = tmp_path / "camera.json"
    cf.save(camera_file)
    depth = tmp_path / "depth.npy"
    np.save(depth, np.full((540, 960), 3.0))
    cache = tmp_path / "review.npz"
    cli = _tool("rebuild_geometry")
    assert (
        cli.main(
            [str(camera_file), "--depth", str(depth), "--depth-scale", "1", "--out", str(cache)]
        )
        == 0
    )
    from syncai_bev3d.geometry_review import load_geometry_cache

    assert load_geometry_cache(cache, cf)["gx"].shape == (1080, 1920)


def test_floor_scale_uses_new_camera_and_checks_disjoint_image_blocks():
    from syncai_bev3d.geometry_cache import floor_depth_scale

    truth, _, _ = survey_case()
    truth = replace(truth, lens=None)
    yy, xx = np.mgrid[0:108, 0:192]
    cam = truth.camera.scaled_to(108, 192, (540, 960))
    rays = np.stack(((xx - cam.cx) / cam.fx, (yy - cam.cy) / cam.fy, np.ones_like(xx)), -1)
    depth = truth.plane.height / (rays @ truth.plane.rotation)[..., 1]
    scale, report = floor_depth_scale(truth, depth * 1.4, np.ones_like(depth, bool))
    assert scale == pytest.approx(1 / 1.4, abs=1e-6)
    assert report["check_p95_abs_height_m"] < 1e-5
    assert report["fit_pixels"] > 100 and report["check_pixels"] > 100
    with pytest.raises(ValueError, match="100 valid floor pixels"):
        floor_depth_scale(truth, depth, np.zeros_like(depth, bool))
