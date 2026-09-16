"""Finite Stage0 baseline worker: freeze inputs, build review scenes, retain progress.

Run under a persistent user service for SSH-independent execution. This tool never
installs candidates into the source checkout. Each subprocess imports the frozen src/.
"""

# ruff: noqa: RUF001 -- Traditional Chinese progress text uses Chinese punctuation.

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import time
import traceback
from datetime import UTC, datetime
from pathlib import Path

from syncai_bev3d.render_provenance import sha256
from syncai_bev3d.scene_audit import capture_inputs


def save(out, state):
    state["updated_at"] = datetime.now(UTC).isoformat()
    temporary = out / "status.tmp"
    temporary.write_text(json.dumps(state, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(out / "status.json")
    lines = [
        "# Stage0 九鏡頭背景驗證",
        "",
        f"狀態：{state['status']}；目前：{state.get('current_camera', '準備快照')}",
        f"更新：{state['updated_at']}；PID：{state['pid']}",
        "",
        "使用凍結程式與輸入，CPU 四執行緒。輸出為 review 候選，不安裝到正式場景。",
        "",
        "| 相機 | 狀態 | 個別物件步驟 | 已放置 | 最終開口／牆片 |",
        "|---|---|---|---:|---:|",
    ]
    for row in state["cameras"]:
        lines.append(
            f"| {row['camera']} | {row['status']} | {row.get('device_stage', '—')} | "
            f"{row.get('placed', '—')} | {row.get('surfaces', '—')} |"
        )
    lines += [
        "",
        "每台 candidates/<相機>/ 含原圖回投、GLB、逐項報告與來源 manifest。",
        "分數是既有遮罩的一致性，不是現場公尺精度；未執行的步驟不算零偵測。",
    ]
    if state.get("error"):
        lines += ["", "錯誤：", "```", state["error"], "```"]
    report = out / "REPORT.tmp"
    report.write_text("\n".join(lines) + "\n")
    report.replace(out / "REPORT.zh-TW.md")


def freeze(root, out, cameras):
    snapshot = out / "snapshot"
    snapshot.mkdir()
    # A physical copy, not a symlink into a checkout that another session can edit.
    shutil.copytree(
        root / "src", snapshot / "src", ignore=shutil.ignore_patterns("__pycache__", "*.pyc")
    )
    tool = Path("tools/commissioning/scene_mesh.py")
    (snapshot / tool).parent.mkdir(parents=True)
    shutil.copy2(root / tool, snapshot / tool)
    shutil.copy2(Path(__file__), snapshot / "tools/commissioning/stage0_baseline.py")
    for name in ("pyproject.toml", "uv.lock"):
        shutil.copy2(root / name, snapshot / name)
    inputs = {camera: capture_inputs(root, camera) for camera in cameras}
    for record in inputs.values():
        for name, digest in record["inputs"].items():
            if digest is None:
                continue
            relative = Path(name)
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError(f"snapshot requires checkout-relative input: {name}")
            target = snapshot / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(root / relative, target)
            if sha256(target) != digest:
                raise RuntimeError(f"input changed while copying: {name}")
    for camera in cameras:
        if capture_inputs(snapshot, camera) != inputs[camera]:
            raise RuntimeError(f"snapshot is incomplete or source changed: {camera}")
        if capture_inputs(root, camera) != inputs[camera]:
            raise RuntimeError(f"source changed while freezing: {camera}")
    (out / "frozen-inputs.json").write_text(json.dumps(inputs, indent=2) + "\n")
    return snapshot


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--camera-timeout", type=int, default=1200)
    ap.add_argument("cameras", nargs="*")
    args = ap.parse_args()
    if args.camera_timeout < 1:
        ap.error("camera-timeout must be positive")
    root, out = args.root.resolve(), args.out.resolve()
    cameras = args.cameras or sorted(
        p.name.removesuffix(".camera.json")
        for p in (root / "runs/commission01").glob("*.camera.json")
        if not p.name.startswith("test")
    )
    if not cameras or any(Path(c).name != c or c in {".", ".."} for c in cameras):
        ap.error("supply valid camera identifiers")
    out.mkdir(parents=True, exist_ok=False)
    state = {
        "schema": 1,
        "pid": os.getpid(),
        "status": "freezing",
        "started_at": datetime.now(UTC).isoformat(),
        "cameras": [{"camera": c, "status": "pending"} for c in cameras],
    }
    save(out, state)
    started = time.monotonic()

    def interrupted(_signal, _frame):
        raise RuntimeError("background worker received a termination signal")

    signal.signal(signal.SIGTERM, interrupted)
    try:
        snapshot = freeze(root, out, cameras)
        (out / "candidates").mkdir()
        (out / "logs").mkdir()
        env = dict(
            os.environ,
            PYTHONPATH=str(snapshot / "src"),
            PYTHONUNBUFFERED="1",
            OMP_NUM_THREADS="4",
            OPENBLAS_NUM_THREADS="4",
            MKL_NUM_THREADS="4",
        )
        protected_paths = [
            p
            for c in cameras
            for p in (root / f"runs/commission01/{c}").glob("scene.*")
            if p.is_file()
        ]
        protected = {str(p.relative_to(root)): sha256(p) for p in protected_paths}
        (out / "commissioned-before.json").write_text(json.dumps(protected, indent=2) + "\n")
        for row in state["cameras"]:
            camera = row["camera"]
            state.update(status="running", current_camera=camera)
            row["status"] = "running"
            save(out, state)
            command = [
                sys.executable,
                str(snapshot / "tools/commissioning/scene_mesh.py"),
                camera,
                "--root",
                str(snapshot),
                "--out",
                str(out / "builds" / camera),
            ]
            with (out / "logs" / f"{camera}.log").open("w") as log:
                process = subprocess.Popen(
                    command, cwd=snapshot, env=env, stdout=log, stderr=subprocess.STDOUT
                )
                deadline = time.monotonic() + args.camera_timeout
                while process.poll() is None:
                    if time.monotonic() > deadline:
                        process.kill()
                        process.wait()
                        row["error"] = "camera timeout"
                        break
                    state["elapsed_seconds"] = round(time.monotonic() - started, 1)
                    save(out, state)
                    time.sleep(10)
            if process.returncode:
                row["status"] = "failed"
                row["returncode"] = process.returncode
            else:
                candidate = out / "builds" / camera / camera
                manifest = json.loads((candidate / "scene.manifest.json").read_text())
                for name, digest in manifest["outputs"].items():
                    if sha256(candidate / name) != digest:
                        raise RuntimeError(f"output identity mismatch: {camera}/{name}")
                audit = json.loads((candidate / "scene.surfaces.json").read_text())
                candidate.rename(out / "candidates" / camera)
                row.update(
                    status="complete",
                    device_stage=audit["devices"]["stage"],
                    placed=audit["devices"]["placed"],
                    surfaces=len(audit["final_surfaces"]),
                )
                row["glb_identical_to_previous"] = audit["glb_sha256"] == protected.get(
                    f"runs/commission01/{camera}/scene.glb"
                )
            save(out, state)
        state["commissioned_changes"] = [
            name
            for name, digest in protected.items()
            if not (root / name).is_file() or sha256(root / name) != digest
        ]
        state["status"] = (
            "completed; review required"
            if all(r["status"] == "complete" for r in state["cameras"])
            else "completed with failures"
        )
    except BaseException:
        state.update(status="failed", error=traceback.format_exc())
    state["elapsed_seconds"] = round(time.monotonic() - started, 1)
    save(out, state)
    return 0 if state["status"] == "completed; review required" else 1


if __name__ == "__main__":
    raise SystemExit(main())
