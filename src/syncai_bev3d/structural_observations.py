"""Image-only structural edge proposals; never calibrated controls or world axes.

The Hough transform only seeds short corridors. Returned points are native raw
image gradient maxima, not synthetic points sampled from the fitted line. A
quadratic allows local lens curvature without assuming a camera or lens model.
Temporal matches are image tracks, not proof of physical identity or structure:
stationary shadows, text and reflections can persist too. No fit/holdout split
is assigned until a downstream physical-line and axis hypothesis is established.
"""

from __future__ import annotations

import hashlib
import io
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import PIL
import scipy
from PIL import Image
from scipy import ndimage


@dataclass(frozen=True)
class ObservationConfig:
    """Global development policy, in native pixels relative to image diagonal."""

    max_scan_frames: int = 24
    max_frames: int = 6
    max_candidates: int = 48
    max_hough_seeds: int = 180
    min_span_fraction: float = 0.07
    min_persistence: float = 0.5

    def __post_init__(self):
        for value in (
            self.max_scan_frames,
            self.max_frames,
            self.max_candidates,
            self.max_hough_seeds,
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError("frame/candidate budgets must be positive integers")
        if self.max_frames > self.max_scan_frames:
            raise ValueError("max_frames exceeds scan budget")
        if not 0 < self.min_span_fraction < 1 or not 0 < self.min_persistence <= 1:
            raise ValueError("span and persistence fractions must be in (0, 1]")


def _read(path: Path):
    data = path.read_bytes()
    with Image.open(io.BytesIO(data)) as image:
        if image.getexif().get(274, 1) != 1:
            raise ValueError(f"nontrivial EXIF orientation: {path}")
        rgb = np.asarray(image.convert("RGB"))
    return rgb, hashlib.sha256(data).hexdigest()


def _gray(rgb):
    return np.asarray(rgb, dtype=float) @ np.array([0.299, 0.587, 0.114]) / 255


def _edges(rgb):
    gray = _gray(rgb)
    smooth = ndimage.gaussian_filter(gray, 1.0)
    gy, gx = np.gradient(smooth)
    magnitude = np.hypot(gx, gy)
    y, x = np.indices(gray.shape)
    dx = gx / np.maximum(magnitude, 1e-12)
    dy = gy / np.maximum(magnitude, 1e-12)
    ahead = ndimage.map_coordinates(magnitude, [y + dy, x + dx], order=1)
    behind = ndimage.map_coordinates(magnitude, [y - dy, x - dx], order=1)
    threshold = max(0.015, float(np.quantile(magnitude, 0.90)))
    mask = (magnitude > threshold) & (magnitude >= ahead) & (magnitude > behind)
    mask[:4] = mask[-4:] = False
    mask[:, :4] = mask[:, -4:] = False
    yy, xx = np.nonzero(mask)
    return np.column_stack([xx, yy]), magnitude[mask], np.arctan2(gy[mask], gx[mask])


def _similar(a, b, tolerance):
    """Overlapping, similarly directed traces; deliberately not physical identity."""
    pa, pb = np.asarray(a["points_px"]), np.asarray(b["points_px"])
    direction = pa[-1] - pa[0]
    direction = direction / np.linalg.norm(direction)
    other = pb[-1] - pb[0]
    other = other / np.linalg.norm(other)
    if abs(float(direction @ other)) < np.cos(np.deg2rad(8)):
        return False
    sa, sb = pa @ direction, pb @ direction
    low, high = max(sa.min(), sb.min()), min(sa.max(), sb.max())
    if high - low < 0.5 * min(np.ptp(sa), np.ptp(sb)):
        return False
    normal = np.array([-direction[1], direction[0]])
    samples = np.linspace(low, high, 12)
    ia, ib = np.argsort(sa), np.argsort(sb)
    da = np.interp(samples, sa[ia], (pa @ normal)[ia])
    db = np.interp(samples, sb[ib], (pb @ normal)[ib])
    return bool(np.max(np.abs(da - db)) <= tolerance)


def extract_edges(rgb: np.ndarray, config: ObservationConfig = ObservationConfig()):
    """Return bounded raw-pixel proposals with unresolved world-axis labels."""
    if rgb.ndim != 3 or rgb.shape[2] != 3 or min(rgb.shape[:2]) < 32:
        raise ValueError("expected an RGB image of at least 32 x 32")
    if rgb.dtype != np.uint8:
        raise ValueError("expected uint8 RGB pixels")
    height, width = rgb.shape[:2]
    diagonal = float(np.hypot(width, height))
    points, strength, angles = _edges(rgb)
    if len(points) < 40:
        return [], {"edge_pixels": len(points), "reason": "insufficient_contrast"}
    # Orientation-constrained Hough voting. Subsampling is spatially deterministic.
    stride = max(1, int(np.ceil(len(points) / 60000)))
    seed_points = points[::stride]
    orientation = np.rint(np.rad2deg(angles[::stride])).astype(int) % 180
    rho_step = max(1.0, diagonal / 1100)
    rho_count = int(np.ceil(2 * diagonal / rho_step)) + 1
    accumulator = np.zeros((180, rho_count), dtype=np.int32)
    for offset in range(-6, 7):
        theta = (orientation + offset) % 180
        rad = np.deg2rad(theta)
        rho = seed_points[:, 0] * np.cos(rad) + seed_points[:, 1] * np.sin(rad)
        bins = np.rint((rho + diagonal) / rho_step).astype(int)
        np.add.at(accumulator, (theta, bins), 1)
    peaks = (accumulator == ndimage.maximum_filter(accumulator, size=(7, 9))) & (
        accumulator >= 12
    )
    ti, ri = np.nonzero(peaks)
    order = np.argsort(-accumulator[ti, ri], kind="stable")[: config.max_hough_seeds]
    min_span = diagonal * config.min_span_fraction
    corridor = max(4.0, diagonal * 0.004)
    step = max(4.0, diagonal * 0.006)
    proposals = []
    for index in order:
        theta = np.deg2rad(ti[index])
        normal = np.array([np.cos(theta), np.sin(theta)])
        tangent = np.array([-normal[1], normal[0]])
        rho = ri[index] * rho_step - diagonal
        distance = points @ normal - rho
        eligible = (np.abs(distance) <= corridor) & (
            np.abs(np.cos(angles - theta)) >= np.cos(np.deg2rad(20))
        )
        subset, weights = points[eligible], strength[eligible]
        if len(subset) < 12:
            continue
        along = subset @ tangent
        bins = np.floor(along / step).astype(int)
        # Follow one side of an edge rather than alternating between two nearby
        # contrast boundaries. All selected positions remain measured maxima.
        chosen = []
        previous_bin = None
        previous_across = rho
        across_subset = subset @ normal
        for bin_id in np.unique(bins):
            members = np.flatnonzero(bins == bin_id)
            if previous_bin is None or bin_id - previous_bin > 2:
                previous_across = rho
            continuity = np.exp(
                -np.abs(across_subset[members] - previous_across) / max(1.5, diagonal / 1100)
            )
            picked = members[np.argmax(weights[members] * continuity)]
            chosen.append(picked)
            previous_bin = bin_id
            previous_across = across_subset[picked]
        chosen = np.asarray(chosen)
        gaps = np.flatnonzero(np.diff(bins[chosen]) > 2) + 1
        for segment in np.split(chosen, gaps):
            if len(segment) < 8:
                continue
            trace = subset[segment]
            s = trace @ tangent
            if np.ptp(s) < min_span:
                continue
            normalized = (s - s.mean()) / np.ptp(s)
            across = trace @ normal
            coefficients = np.polyfit(normalized, across, 2)
            error = np.abs(across - np.polyval(coefficients, normalized))
            tolerance = max(1.5, diagonal / 1100)
            if np.quantile(error, 0.9) > tolerance or error.max() > 3 * tolerance:
                continue
            row = {
                "points_px": trace.tolist(),
                "span_px": float(np.linalg.norm(trace[-1] - trace[0])),
                "edge_strength_median": float(np.median(weights[segment])),
                "curve_residual_p90_px": float(np.quantile(error, 0.9)),
                "axis": None,
                "calibration_eligible": False,
            }
            if any(_similar(row, old, tolerance * 2) for old in proposals):
                continue
            proposals.append(row)
            if len(proposals) >= config.max_candidates:
                return proposals, {"edge_pixels": len(points), "budget_reached": True}
    return proposals, {"edge_pixels": len(points), "budget_reached": False}


def group_tracks(
    observations: list[dict], frame_count: int, diagonal: float, persistence: float
):
    """Conservative image-track components, with ambiguous matches excluded."""
    parents = list(range(len(observations)))

    def root(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i

    for i, first in enumerate(observations):
        for j in range(i):
            if _similar(first, observations[j], max(2.0, diagonal * 0.002)):
                parents[root(i)] = root(j)
    components: dict[int, list[int]] = {}
    for i in range(len(observations)):
        components.setdefault(root(i), []).append(i)
    tracks = []
    for indices in components.values():
        frames = [observations[i]["frame_id"] for i in indices]
        ambiguous = len(frames) != len(set(frames)) or any(
            not _similar(observations[i], observations[j], max(2.0, diagonal * 0.002))
            for n, i in enumerate(indices)
            for j in indices[:n]
        )
        enough = len(set(frames)) >= max(2, int(np.ceil(frame_count * persistence)))
        state = "ambiguous_match" if ambiguous else "persistent" if enough else "transient"
        track_id = f"track-{len(tracks):04d}"
        for i in indices:
            observations[i]["track_id"] = track_id
        tracks.append(
            {
                "id": track_id,
                "observation_ids": [observations[i]["id"] for i in indices],
                "distinct_frames": len(set(frames)),
                "state": state,
                "physical_line_identity": "unresolved",
                "split": "unassigned",
            }
        )
    return tracks


def observe_directory(
    directory: Path, camera_id: str, config: ObservationConfig = ObservationConfig()
):
    """Deterministically sample a single camera's raw frame directory, without caches.

    Input must be frames from one fixed camera, ordered by filename. File order
    supplies temporal bins; it does not establish timestamps or elapsed seconds.
    """
    paths = sorted(
        p
        for p in directory.iterdir()
        if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    if not paths:
        raise ValueError("no raw image frames found")
    scan_indices = np.unique(
        np.linspace(0, len(paths) - 1, min(len(paths), config.max_scan_frames), dtype=int)
    )
    scan_set = set(scan_indices.tolist())
    frames = []
    seen = set()
    size = None
    for index, path in enumerate(paths):
        record = {"id": f"frame-{index:05d}", "path": str(path.resolve()), "selected": False}
        if index not in scan_set:
            record["reason"] = "scan_budget"
            frames.append(record)
            continue
        rgb, digest = _read(path)
        if min(rgb.shape[:2]) < 32:
            raise ValueError("raw frames must be at least 32 x 32")
        current_size = [rgb.shape[1], rgb.shape[0]]
        if size is not None and current_size != size:
            raise ValueError("mixed native image sizes; separate camera streams first")
        size = current_size
        pixels_digest = hashlib.sha256(rgb.tobytes()).hexdigest()
        record.update(sha256=digest, pixels_sha256=pixels_digest, image_size_px=size)
        if pixels_digest in seen:
            record["reason"] = "duplicate_content"
        else:
            seen.add(pixels_digest)
            thumbnail = Image.fromarray(rgb)
            thumbnail.thumbnail((640, 640))
            gy, gx = np.gradient(_gray(np.asarray(thumbnail)))
            quality = float(np.mean(gx**2 + gy**2))
            record.update(quality=quality, reason="not_selected")
        frames.append(record)
    available = [i for i, row in enumerate(frames) if "quality" in row]
    selected = []
    for group in np.array_split(available, min(len(available), config.max_frames)):
        winner = max(group.tolist(), key=lambda i: frames[i]["quality"])
        frames[winner].update(selected=True, reason="selected")
        selected.append(winner)
    observations = []
    for index in selected:
        frame = frames[index]
        rgb, digest = _read(Path(frame["path"]))
        if digest != frame["sha256"]:
            raise ValueError("source frame changed during observation extraction")
        candidates, diagnostic = extract_edges(rgb, config)
        frame["extraction"] = diagnostic
        for candidate in candidates:
            candidate.update(id=f"edge-{len(observations):05d}", frame_id=frame["id"])
            observations.append(candidate)
    assert size is not None
    tracks = group_tracks(
        observations, len(selected), float(np.hypot(*size)), config.min_persistence
    )
    persistent = sum(t["state"] == "persistent" for t in tracks)
    return {
        "schema": "structural-edge-proposals-v1",
        "algorithm": "native-gradient-hough-corridor-v1",
        "implementation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "dependencies": {
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "pillow": PIL.__version__,
        },
        "camera": camera_id,
        "image_size_px": size,
        "pixel_space": "raw",
        "config": asdict(config),
        "frames": frames,
        "observations": observations,
        "tracks": tracks,
        "summary": {
            "input_frames": len(paths),
            "scanned_frames": len(scan_indices),
            "selected_frames": len(selected),
            "observations": len(observations),
            "persistent_tracks": persistent,
            "ambiguous_tracks": sum(t["state"] == "ambiguous_match" for t in tracks),
            "transient_tracks": sum(t["state"] == "transient" for t in tracks),
        },
        "status": "partial" if persistent else "insufficient_evidence",
        "calibration_ready": False,
        "deployment_ready": False,
        "scale_status": "unknown",
        "reasons": [
            "world_axes_unresolved",
            "physical_line_identity_unresolved",
            "static_shadows_reflections_and_text_not_semantically_rejected",
            "no_calibration_or_metric_scale_estimated",
        ],
    }
