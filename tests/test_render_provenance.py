"""A later render or edited input must not silently change an earlier figure's audit."""

import argparse
import ast
import json
from pathlib import Path

from syncai_bev3d.render_provenance import changed_inputs, scene_code, sha256

ROOT = Path(__file__).resolve().parents[1]


def _tool_function(name):
    tree = ast.parse((ROOT / "tools/commissioning/demo_video.py").read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    namespace = {"ROOT": ROOT, "json": json, "sha256": sha256}
    exec(compile(ast.Module(body=[node], type_ignores=[]), "demo_video.py", "exec"), namespace)
    return namespace


def test_sidecars_stay_bound_to_each_render_when_camera_log_is_replaced(tmp_path):
    ns = _tool_function("_write_demo_tracks")
    ns["ROOT"] = tmp_path
    (tmp_path / "runs/commission01").mkdir(parents=True)
    args = argparse.Namespace(checkpoint="last.pt", score_thr=0.35, no_blur=False, fps=5)
    ns["BLUR_THR"] = 0.07
    first, second = tmp_path / "first.mp4", tmp_path / "second.mp4"
    first.write_bytes(b"first render")
    second.write_bytes(b"second render")
    write = ns["_write_demo_tracks"]
    write(args, "cam", tmp_path / "clip.mp4", None, 1, [{"frame": 0}], first, {"version": 1})
    saved = first.with_suffix(".render.json").read_bytes()
    write(args, "cam", tmp_path / "clip.mp4", None, 2, [], second, {"version": 2})
    assert first.with_suffix(".render.json").read_bytes() == saved
    meta = json.loads(saved)
    assert meta["render_sha256"] == sha256(first)
    assert meta["provenance"] == {"version": 1}
    assert meta["frames"] == 1
    assert json.loads(second.with_suffix(".render.json").read_text())["frames"] == 2


def test_changed_or_missing_inputs_are_detected(tmp_path):
    source = tmp_path / "camera.json"
    source.write_text("first")
    provenance = {"inputs": {"camera.json": sha256(source)}}
    assert changed_inputs(tmp_path, provenance) == []
    source.write_text("second")
    assert changed_inputs(tmp_path, provenance) == ["camera.json"]
    source.unlink()
    assert changed_inputs(tmp_path, provenance) == ["camera.json"]


def test_code_identity_includes_untracked_modules_and_content_changes(tmp_path):
    module = tmp_path / "src/syncai_bev3d/new_geometry.py"
    module.parent.mkdir(parents=True)
    module.write_text("first")
    before = scene_code(tmp_path)
    assert str(module.relative_to(tmp_path)) in before
    module.write_text("second")
    assert scene_code(tmp_path) != before


def test_chunk_workers_preserve_heatmap_choice():
    import sys

    ns = _tool_function("_worker_cmd")
    ns.update(sys=sys, Path=Path, __file__=str(ROOT / "tools/commissioning/demo_video.py"))
    args = argparse.Namespace(
        frames=900,
        fps=5,
        checkpoint="last.pt",
        score_thr=0.35,
        metre_scale=1,
        clip=None,
        no_blur=False,
        posed_figures=False,
        staff_colours=None,
        no_heatmap=True,
    )
    assert "--no-heatmap" in ns["_worker_cmd"](args, "cam")
    args.no_heatmap = False
    assert "--no-heatmap" not in ns["_worker_cmd"](args, "cam")
