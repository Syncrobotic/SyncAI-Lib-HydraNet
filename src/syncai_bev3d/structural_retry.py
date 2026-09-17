"""Bounded raw-video acquisition with an append-only structural holdout ledger.

Retries address missing evidence only. A completed geometric test is terminal,
whether it passes or fails: repeated holdout scores cannot select a lucky run.
Old partitions, representative pixels and semantic hints never change.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from syncai_bev3d.render_provenance import sha256
from syncai_bev3d.structural_directions import (
    _coherent,
    _samples,
    freeze_line_groups,
    infer_directions,
)
from syncai_bev3d.structural_observations import ObservationConfig, observe_directory
from syncai_bev3d.structural_surfaces import MIN_SCORE, PROMPTS, SurfaceTeacher, enrich_surfaces

RETRYABLE = frozenset(
    {
        "fewer_than_four_fitting_floor_groups",
        "no_supported_floor_pair_and_orthogonal_vertical",
        "each direction needs fitting lines and an independent held-out line",
    }
)
ARCHIVE = re.compile(r"archive_\d{8}-\d{6}_\d{8}-\d{6}")


@dataclass(frozen=True)
class RetryPolicy:
    max_attempts: int = 3  # Includes an optional inherited seed assessment.
    frames_per_window: int = 3

    def __post_init__(self):
        if type(self.max_attempts) is not int or not 1 <= self.max_attempts <= 8:
            raise ValueError("max_attempts must be an integer in [1, 8]")
        if type(self.frames_per_window) is not int or not 2 <= self.frames_per_window <= 6:
            raise ValueError("frames_per_window must be an integer in [2, 6]")


def _digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()


def _seal(value):
    value = copy.deepcopy(value)
    value.pop("freeze_sha256", None)
    value["freeze_sha256"] = _digest(value)
    return value


def _verify_seal(value):
    expected = dict(value)
    digest = expected.pop("freeze_sha256", None)
    if digest != _digest(expected):
        raise ValueError("frozen groups changed")


def _write(path: Path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def load_seed(surfaces_path: Path, groups_path: Path, camera: str):
    """Accept only source-bound automatic observations, never commissioned geometry."""
    from syncai_bev3d.teachers.sam3 import MODEL_ID, MODEL_REVISION

    surfaces = json.loads(surfaces_path.read_text())
    groups = json.loads(groups_path.read_text())
    _verify_seal(groups)
    if (
        groups.get("camera") != camera
        or surfaces.get("camera") != camera
        or groups.get("schema") != "structural-line-groups-v1"
        or groups.get("pixel_space") != "raw"
        or groups.get("image_size_px") != surfaces.get("image_size_px")
        or surfaces.get("schema") != "structural-edge-proposals-v1"
        or groups.get("proposals_sha256") != _digest(surfaces)
    ):
        raise ValueError("seed groups do not bind this camera's automatic surface report")
    evidence = surfaces.get("surface_evidence", {})
    identity = evidence.get("teacher", {})
    expected = {
        "model": MODEL_ID,
        "revision": MODEL_REVISION,
        "prompts": list(PROMPTS),
        "min_score": MIN_SCORE,
        "mask_threshold": 0.5,
    }
    if evidence.get("status") != "predictions_only" or any(
        identity.get(k) != v for k, v in expected.items()
    ):
        raise ValueError("seed must use the same fixed surface teacher policy")
    selected = {f["id"]: f for f in surfaces["frames"] if f["selected"]}
    records = evidence.get("frames", [])
    if (
        not selected
        or len(records) != len(selected)
        or {r["frame_id"] for r in records} != set(selected)
    ):
        raise ValueError("seed surface frames do not cover selected raw frames")
    for frame in selected.values():
        if sha256(Path(frame["path"])) != frame["sha256"]:
            raise ValueError("seed raw image changed")
    for record in records:
        if (
            record["source_sha256"] != selected[record["frame_id"]]["sha256"]
            or sha256(Path(record["mask_file"])) != record["mask_sha256"]
        ):
            raise ValueError("seed surface mask changed or has another source")
    return surfaces, groups


def plan_windows(root: Path, policy: RetryPolicy, seed_surfaces=None):
    """Freeze chronological filename order before observing any new scores.

    StudioA archive names encode time. Other filenames retain deterministic order
    but have unknown wall-clock times. A known seed archive is excluded; legacy
    image-only seeds may not identify their capture interval, which is reported.
    """
    paths = sorted(
        p
        for p in root.iterdir()
        if p.is_file() and p.suffix.lower() in {".mp4", ".mkv", ".mov"}
    )
    known_seed = set()
    if seed_surfaces is not None:
        for frame in seed_surfaces["frames"]:
            match = ARCHIVE.search(frame["path"])
            if match:
                known_seed.add(match.group())
    budget = policy.max_attempts - int(seed_surfaces is not None)
    windows, omitted, digests = [], [], set()
    for path in paths:
        if path.stem in known_seed:
            omitted.append({"path": str(path.resolve()), "reason": "seed_archive"})
            continue
        if len(windows) >= budget:
            omitted.append({"path": str(path.resolve()), "reason": "attempt_budget"})
            continue
        digest = sha256(path)
        if digest in digests:
            omitted.append({"path": str(path.resolve()), "reason": "duplicate_video_content"})
            continue
        digests.add(digest)
        match = ARCHIVE.fullmatch(path.stem)
        windows.append(
            {
                "id": "window-" + digest[:16],
                "path": str(path.resolve()),
                "sha256": digest,
                "archive_interval": match.group() if match else None,
            }
        )
    return {
        "windows": windows,
        "omitted": omitted,
        "order": "filename_ascending",
        "seed_archive_interval_known": bool(known_seed) if seed_surfaces is not None else None,
    }


def decode_window(window: dict, output: Path, count: int, seen_pixels: set[str]):
    """Native PNG frames at fixed 10--90% offsets; bounded external decoder calls."""
    source = Path(window["path"])
    if sha256(source) != window["sha256"]:
        raise ValueError("planned source video changed")
    probe = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,sample_aspect_ratio:stream_tags=rotate:stream_side_data=rotation:format=duration",
            "-of",
            "json",
            str(source),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    metadata = json.loads(probe.stdout)
    stream = metadata["streams"][0]
    duration = float(metadata["format"]["duration"])
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("video duration must be positive and finite")
    if stream.get("sample_aspect_ratio") not in {None, "N/A", "1:1"}:
        raise ValueError("non-square video pixels require an explicit coordinate contract")
    rotations = [float(stream.get("tags", {}).get("rotate", 0))]
    rotations += [float(s.get("rotation", 0)) for s in stream.get("side_data_list", [])]
    if any(r != 0 for r in rotations):
        raise ValueError("rotated video requires an explicit raw coordinate contract")
    output.mkdir(parents=True)
    frames = []
    from syncai_bev3d.structural_observations import _read

    for i, seconds in enumerate(np.linspace(0.1 * duration, 0.9 * duration, count)):
        path = output / f"{i:04d}.png"
        subprocess.run(
            [
                "ffmpeg",
                "-nostdin",
                "-v",
                "error",
                "-noautorotate",
                "-ss",
                str(seconds),
                "-i",
                str(source),
                "-frames:v",
                "1",
                "-threads",
                "1",
                "-n",
                str(path),
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=60,
        )
        rgb, digest = _read(path)
        if [rgb.shape[1], rgb.shape[0]] != [stream["width"], stream["height"]]:
            raise ValueError("decoder changed native image dimensions")
        pixels = hashlib.sha256(rgb.tobytes()).hexdigest()
        duplicate = pixels in seen_pixels
        frames.append(
            {
                "path": str(path.resolve()),
                "sha256": digest,
                "pixels_sha256": pixels,
                "requested_offset_seconds": float(seconds),
                "duplicate": duplicate,
            }
        )
        if duplicate:
            path.unlink()  # Only this attempt's newly decoded duplicate, never a source file.
        else:
            seen_pixels.add(pixels)
    if sha256(source) != window["sha256"]:
        raise ValueError("source video changed during decoding")
    return {
        "metadata": metadata,
        "frames": frames,
        "unique_frames": sum(not f["duplicate"] for f in frames),
    }


def extend_ledger(previous: dict | None, incoming: dict, window_id: str):
    """Inherit known line assignments; ambiguous bridges fail instead of repartitioning.

    Representative pixels and floor hints of old lines are immutable. Repeated
    observations add provenance only. Novel lines receive a partition exactly once.
    """
    _verify_seal(incoming)
    if previous is None:
        ledger = copy.deepcopy(incoming)
        for row in ledger["groups"]:
            row["seen_windows"] = [window_id]
        ledger["windows"] = [{"id": window_id, "groups_sha256": incoming["freeze_sha256"]}]
        ledger["ledger_policy"] = "inherit_or_assign_once_v1"
        return _seal(ledger), {
            "new_groups": len(ledger["groups"]),
            "repeat_groups": 0,
            "conflicts": [],
        }
    _verify_seal(previous)
    if any(
        previous.get(k) != incoming.get(k)
        for k in ("camera", "image_size_px", "pixel_space", "lens_grid")
    ):
        raise ValueError("new window does not share the ledger's camera coordinate contract")
    ledger = copy.deepcopy(previous)
    old = ledger["groups"]
    size = ledger["image_size_px"]
    tolerance = float(np.linalg.norm(np.asarray(size) / 2)) * 0.005
    repeats = 0
    added = []
    conflicts = []
    for group in incoming["groups"]:
        if group["state"] != "geometric_candidate":
            continue
        raw = _samples(group["points_px"])
        matches = [
            g
            for g in old + added
            if _coherent(np.concatenate([_samples(g["points_px"]), raw]), size, tolerance)
        ]
        if len(matches) > 1:
            conflicts.append(
                {
                    "incoming": group["id"],
                    "existing": [g["id"] for g in matches],
                    "partitions": [g["validation"] for g in matches],
                    "reason": "ambiguous_line_bridge",
                }
            )
        elif matches:
            row = matches[0]
            if window_id not in row["seen_windows"]:
                row["seen_windows"].append(window_id)
            repeats += 1
        else:
            row = copy.deepcopy(group)
            digest = hashlib.sha256(f"{window_id}:{group['id']}".encode()).hexdigest()
            row.update(
                id="line-" + digest[:12],
                validation=int(digest, 16) % 4 == 0,
                seen_windows=[window_id],
                origin_group_id=group["id"],
            )
            added.append(row)
    # Fail atomically: no partial update can silently discard a holdout conflict.
    change = {"new_groups": len(added), "repeat_groups": repeats, "conflicts": conflicts}
    if conflicts:
        return copy.deepcopy(previous), change
    old.extend(added)
    ledger["windows"].append({"id": window_id, "groups_sha256": incoming["freeze_sha256"]})
    return _seal(ledger), change


def may_retry(directions):
    return (
        directions.get("status") == "insufficient_evidence"
        and "line_checks" not in directions
        and bool(directions.get("reasons"))
        and set(directions["reasons"]) <= RETRYABLE
    )


def run_retry(
    camera: str,
    source_root: Path,
    output: Path,
    *,
    policy=RetryPolicy(),
    seed_surfaces: Path | None = None,
    seed_groups: Path | None = None,
    teacher=None,
    device="cpu",
    processor=None,
    solver=None,
):
    """Run a predeclared acquisition schedule; persist every attempt and terminal reason."""
    if (seed_surfaces is None) != (seed_groups is None):
        raise ValueError("seed surfaces and frozen groups must be supplied together")
    output.mkdir(parents=True, exist_ok=False)
    solver = infer_directions if solver is None else solver
    state = {
        "schema": "structural-retry-run-v1",
        "camera": camera,
        "status": "running",
        "attempts": [],
        "policy": asdict(policy),
        "calibration_ready": False,
        "deployment_ready": False,
        "scale_status": "unknown",
        "reasons": [],
    }
    _write(output / "status.json", state)
    try:
        seed, initial = None, None
        if seed_surfaces is not None and seed_groups is not None:
            seed, initial = load_seed(seed_surfaces, seed_groups, camera)
        plan = plan_windows(source_root, policy, seed)
        plan.update(
            camera=camera, policy=asdict(policy), implementation_sha256=sha256(Path(__file__))
        )
        if seed is not None:
            assert seed_surfaces is not None and seed_groups is not None
            plan["seed"] = {
                "surfaces": str(seed_surfaces),
                "groups": str(seed_groups),
                "surfaces_sha256": sha256(seed_surfaces),
                "groups_sha256": sha256(seed_groups),
            }
        _write(output / "plan.freeze.json", plan)
        state["plan_sha256"] = sha256(output / "plan.freeze.json")
        seen_pixels = (
            {f["pixels_sha256"] for f in seed["frames"] if f["selected"]} if seed else set()
        )
        ledger = None
        schedule = ([{"id": "inherited-seed"}] if seed is not None else []) + plan["windows"]
        for index, window in enumerate(schedule):
            folder = output / f"attempt-{index + 1:02d}"
            folder.mkdir()
            state["active_window"] = window["id"]
            _write(output / "status.json", state)
            if window["id"] == "inherited-seed":
                incoming = initial
            else:
                decoded = decode_window(
                    window, folder / "frames", policy.frames_per_window, seen_pixels
                )
                _write(folder / "decode.json", decoded)
                if decoded["unique_frames"] < 2:
                    state["attempts"].append(
                        {
                            "window": window["id"],
                            "status": "insufficient_evidence",
                            "reasons": ["fewer_than_two_unique_frames"],
                        }
                    )
                    _write(output / "status.json", state)
                    continue
                if processor is None:
                    raw = observe_directory(
                        folder / "frames",
                        camera,
                        ObservationConfig(max_frames=policy.frames_per_window),
                    )
                    if teacher is None:
                        teacher = SurfaceTeacher(device)
                    surfaces = enrich_surfaces(raw, teacher, folder / "surfaces")
                else:
                    surfaces = processor(folder / "frames", camera, folder)
                _write(folder / "surfaces.json", surfaces)
                incoming = freeze_line_groups(surfaces)
            assert incoming is not None
            _write(folder / "window.groups.freeze.json", incoming)
            ledger, change = extend_ledger(ledger, incoming, window["id"])
            _write(folder / "merge.json", change)
            if change["conflicts"]:
                state["status"] = "partition_conflict"
                state["stop_reason"] = "ambiguous_line_identity"
                state["reasons"] = ["ambiguous_line_bridge_requires_independent_evidence"]
                state["attempts"].append(
                    {"window": window["id"], "status": "partition_conflict", "merge": change}
                )
                break
            _write(
                folder / "ledger.freeze.json", ledger
            )  # Before any new fit/held-out scoring.
            directions = solver(ledger)
            _write(folder / "directions.json", directions)
            state["attempts"].append(
                {
                    "window": window["id"],
                    "status": directions["status"],
                    "reasons": directions["reasons"],
                    "merge": change,
                    "ledger_sha256": ledger["freeze_sha256"],
                }
            )
            state["latest_directions"] = str((folder / "directions.json").resolve())
            state["reasons"] = directions["reasons"]
            _write(output / "status.json", state)
            if not may_retry(directions):
                state["status"] = directions["status"]
                state["stop_reason"] = "terminal_direction_decision"
                break
        else:
            state["status"] = "insufficient_evidence"
            state["stop_reason"] = (
                "attempt_budget_exhausted"
                if len(schedule) == policy.max_attempts
                else "source_windows_exhausted"
            )
            state["reasons"] = [state["stop_reason"], *state["reasons"]]
        if ledger is not None:
            _write(output / "ledger.final.json", ledger)
    except Exception as exc:
        state.update(
            status="failed",
            stop_reason="execution_error",
            failed_window=state.get("active_window"),
            reasons=[f"{type(exc).__name__}: {exc}"],
        )
    state.pop("active_window", None)
    _write(output / "status.json", state)
    return state
