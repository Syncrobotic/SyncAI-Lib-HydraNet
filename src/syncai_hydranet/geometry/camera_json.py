"""`camera.json` -- the one file that crosses the package boundary.

`syncai_bev3d` writes it once per camera at commissioning; every runtime consumer reads
it back into metres through this module. It lives on the hydranet side because the
serving path is forbidden from importing `syncai_bev3d`
(`tests/test_package_boundaries.py`), and a contract a reader cannot load is not a
contract. The writer imports this module too, so there is exactly one definition of what
the file means.

Units are in the field names (`_m`, `_rad`, `_px`) because this file is edited and
diffed by humans, and a bare `height: 2.38` invites the reader to guess. Two coordinate
systems appear and are never mixed: **floor metres** (x lateral, z forward, origin under
the camera -- what `pixel_to_ground` returns) for zones, and **pixels on the raw stream
frame at `image_size_px`** for shelf ROIs, false-positive polygons and mask files --
raw, not undistorted, because these exist to be compared against detections and to crop
decoded frames, and both of those live in the raw frame. The lens applies to *points* on
their way to the floor (`undistort_points`), never to the pixel-space artefacts.

What is deliberately *not* here:

* **No raw 3x3 homography.** The runtime projection is `Camera` + `GroundPlane`
  (`ground.py`), which a homography can be derived from but not the reverse -- the
  decomposition is where the metres come from. Storing both would be two answers.
* **No thresholds, schedules or class lists.** Those are serving config a manager
  changes; this file is what the *camera* is, valid until the camera or the store moves.
* **No mask pixels.** Masks are PNG artefacts next to the file; this holds their paths
  and the ignore value they were written with, so a reader that disagrees about the
  sentinel fails loudly at load rather than quietly at mask time.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from ..labels import IGNORE
from .ground import Camera, GroundPlane, pixel_to_ground, undistort_points

SCHEMA_VERSION = 3

# Versions this reader accepts, and why each older one is safe to read rather than
# re-commission. A refusal is the default -- a file from a *newer* writer carries fields
# whose meaning this code does not know -- so an entry here is a claim that nothing a file
# of that version can contain has changed meaning. Deliberately no literal version number
# in this sentence: one written here goes stale the next time the schema moves, and then
# names a perfectly readable file as refusable.
READABLE_VERSIONS = {
    # v1 -> v2 added the `display` zone kind and changed nothing else. Every v1 file is a
    # valid v2 file: no field moved, no unit changed, and the widened set can only make a
    # kind legal that was previously refused. Re-commissioning 8 cameras to gain a word
    # would be a cost with no measurement behind it.
    1: "v2 only widened ZONE_KINDS; no field changed meaning",
    # v2 -> v3 added `teachers`, which is optional and defaults to None. A v2 file did
    # not record its teachers and says so by carrying None; no field moved and no unit
    # changed, so the 16 cameras commissioned before this stay readable rather than being
    # re-run to gain a provenance entry they cannot retroactively know.
    2: "v3 only added the optional `teachers` map; no field changed meaning",
    3: "current",
}

# The zone kinds the event layer knows how to consume (`analytics/events/zones.py`).
# A kind outside this set is refused at load: a typo'd kind would not raise downstream,
# the zone would simply never fire, which is the silent-failure mode this file exists to
# prevent.
#
# `display` was added 2026-08-26, when the fixture footprints from `propose_zones.py`
# reached their accept/reject pass and the shop turned out to be mostly neither tills nor
# premium shelves: a STUDIO A floor is display tables. Forcing them into `premium_shelf`
# would have made the kind mean "a fixture", after which no rule could key on it -- the
# same failure this set exists to prevent, arrived at from the other direction.
ZONE_KINDS = frozenset(
    {
        "entrance_line",
        "till",
        "display",
        "premium_shelf",
        "stockroom_door",
        "forbidden",
        "walkable",
    }
)


@dataclass(frozen=True)
class Zone:
    """A named region on the floor, in metres. `entrance_line` reads as a polyline."""

    name: str
    kind: str
    points_m: tuple[tuple[float, float], ...]  # (x lateral, z forward)


@dataclass(frozen=True)
class Lens:
    """Fitzgibbon division model, as `ground.undistort_points` applies it."""

    k1: float
    centre_px: tuple[float, float]
    radius_px: float


@dataclass(frozen=True)
class CameraFile:
    """Everything constant about one mounted camera. The commissioning output."""

    camera_id: str
    image_size_px: tuple[int, int]  # (width, height) the pixel-space fields refer to
    camera: Camera
    plane: GroundPlane
    lens: Lens | None = None  # None: the lens was measured as good enough to skip
    zones: tuple[Zone, ...] = ()
    shelf_rois_px: tuple[tuple[float, float, float, float], ...] = ()
    false_positive_polygons_px: tuple[tuple[tuple[float, float], ...], ...] = ()
    mask_files: dict[str, str] = field(default_factory=dict)  # class name -> relative path
    mask_ignore: int = IGNORE
    plate_file: str | None = None
    plate_sha256: str | None = None
    commissioned_at: str | None = None  # ISO 8601, stamped by the writer
    # Which teacher models produced this file, as `{model_id: revision}`. The plate is
    # hashed (`plate_sha256`) and the models that read it were not, so two camera.jsons
    # from either side of an upstream push were indistinguishable while describing
    # different geometry. `None` means "not recorded" -- a v2 file, or a writer that did
    # not know -- and is deliberately not `{}`, which would claim no teacher was used.
    teachers: dict[str, str] | None = None

    def save(self, path: str | Path) -> None:
        payload = {
            "schema_version": SCHEMA_VERSION,
            **asdict(self),
        }
        Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    @classmethod
    def load(cls, path: str | Path) -> CameraFile:
        raw = json.loads(Path(path).read_text())
        version = raw.pop("schema_version", None)
        if version not in READABLE_VERSIONS:
            raise ValueError(
                f"{path}: schema_version {version!r}, this reader speaks "
                f"{sorted(READABLE_VERSIONS)}. Re-commission the camera rather than "
                "guessing at field meanings."
            )
        out = cls(
            camera_id=raw["camera_id"],
            image_size_px=tuple(raw["image_size_px"]),
            camera=Camera(**raw["camera"]),
            plane=GroundPlane(**raw["plane"]),
            lens=Lens(
                k1=raw["lens"]["k1"],
                centre_px=tuple(raw["lens"]["centre_px"]),
                radius_px=raw["lens"]["radius_px"],
            )
            if raw.get("lens")
            else None,
            zones=tuple(
                Zone(z["name"], z["kind"], tuple(map(tuple, z["points_m"])))
                for z in raw.get("zones", ())
            ),
            shelf_rois_px=tuple(map(tuple, raw.get("shelf_rois_px", ()))),
            false_positive_polygons_px=tuple(
                tuple(map(tuple, poly)) for poly in raw.get("false_positive_polygons_px", ())
            ),
            mask_files=dict(raw.get("mask_files", {})),
            mask_ignore=raw.get("mask_ignore", IGNORE),
            plate_file=raw.get("plate_file"),
            plate_sha256=raw.get("plate_sha256"),
            commissioned_at=raw.get("commissioned_at"),
            teachers=raw.get("teachers"),
        )
        out.validate()
        return out

    def ground_points(
        self,
        points_px: np.ndarray,
        *,
        above_horizon: str = "drop",
        what: str = "points",
    ) -> np.ndarray:
        """Pixels on this camera's frame -> `(N, 2)` floor metres, lens included.

        Three tools under `tools/commissioning/` each wrote this same undistort-then-
        project pair, and the copies disagreed about the one thing that matters: what a
        point at or above the horizon means. `pixel_to_ground` returns NaN there rather
        than inventing a distance, and two of the three silently dropped those rows while
        the third refused -- and the third's argument was the correct one *for its own
        input*. So the disagreement is not resolved by picking a winner; it is made an
        argument, because the two inputs are genuinely different:

        `above_horizon="drop"` is for projected **masks**, thousands of pixels of which
        some legitimately reach past the horizon; the floor is what is below it and the
        rest is not a failure. `above_horizon="raise"` is for hand-drawn **vertices**,
        where a single NaN is an operator who clicked above the floor line: a zone with a
        NaN corner tests False for every point inside it, forever and silently, and the
        commissioning session that would notice has already ended.

        There is no third policy and no default of `None`: a caller that has not decided
        which of these its input is has not understood its input.
        """
        pts = np.asarray(points_px, dtype=float)
        if self.lens is not None:
            pts = undistort_points(pts, self.lens.k1, self.lens.centre_px, self.lens.radius_px)
        x, z = pixel_to_ground(pts[:, 0], pts[:, 1], self.camera, self.plane)
        out = np.stack([x, z], axis=1)
        finite = np.isfinite(out).all(axis=1)
        if above_horizon == "drop":
            return out[finite]
        if above_horizon != "raise":
            raise ValueError(f"above_horizon must be 'drop' or 'raise', not {above_horizon!r}")
        if not finite.all():
            raise ValueError(
                f"{what}: {int((~finite).sum())} of {len(out)} points are at or above the "
                "horizon, where `pixel_to_ground` declines to invent a distance. A zone "
                "with a NaN corner tests False for every point inside it, forever and "
                "silently. Redraw those vertices on visible floor."
            )
        return out

    def validate(self) -> None:
        """Refuse the states that would fail silently downstream.

        Each check names the failure it forecloses, because "invalid" is not a reason.
        """
        problems: list[str] = []
        w, h = self.image_size_px
        if w <= 0 or h <= 0:
            problems.append(f"image_size_px {self.image_size_px} is not a size")
        for name, value in (
            ("fx", self.camera.fx),
            ("fy", self.camera.fy),
            ("height_m", self.plane.height),
        ):
            if not (math.isfinite(value) and value > 0):
                problems.append(
                    f"{name}={value!r}: a non-positive value makes every metre downstream NaN"
                )
        if not -math.pi / 2 < self.plane.pitch < math.pi / 2:
            problems.append(
                f"pitch {self.plane.pitch} rad is outside (-pi/2, pi/2) -- degrees in a "
                "radian field is the usual way this happens"
            )
        for z in self.zones:
            if z.kind not in ZONE_KINDS:
                problems.append(
                    f"zone {z.name!r} has kind {z.kind!r}, not one of {sorted(ZONE_KINDS)} "
                    "-- an unknown kind never fires and never errors"
                )
            minimum = 2 if z.kind == "entrance_line" else 3
            if len(z.points_m) < minimum:
                problems.append(
                    f"zone {z.name!r} has {len(z.points_m)} points, needs {minimum}"
                )
        for i, (x0, y0, x1, y1) in enumerate(self.shelf_rois_px):
            if not (0 <= x0 < x1 <= w and 0 <= y0 < y1 <= h):
                problems.append(
                    f"shelf_rois_px[{i}] {(x0, y0, x1, y1)} is not inside {w}x{h} -- an ROI "
                    "outside the frame crops nothing and the product classes go quietly blind"
                )
        # The principal point of a real mount sits near the frame centre. One in the
        # outer quarter means the intrinsics were fitted at a different resolution than
        # `image_size_px` declares -- the half-res trap, which returns metres rather
        # than an error when a consumer divides by the wrong factor.
        if (w > 0 and h > 0) and (
            not (0.25 * w <= self.camera.cx <= 0.75 * w)
            or not (0.25 * h <= self.camera.cy <= 0.75 * h)
        ):
            problems.append(
                f"principal point ({self.camera.cx}, {self.camera.cy}) is in the outer "
                f"quarter of a {w}x{h} frame -- intrinsics fitted at one resolution "
                "paired with another's image_size_px is the usual way this happens"
            )
        if self.lens is not None:
            # Past k1 = -1 the division model's singularity moves inside the frame:
            # `undistort_image` still LOOKS right while point mapping silently breaks
            # (`load_person_boxes` returned 0 boxes under k1 = -1.05, no error anywhere).
            # `calibrate.fit_k1` refuses to search there; this refuses a file that
            # carries such a value regardless of who wrote it.
            if not (-1.0 < self.lens.k1 < 1.0):
                problems.append(
                    f"lens.k1={self.lens.k1} is outside (-1, 1): the division model's "
                    "singularity is inside the frame and point mapping breaks silently "
                    "while the undistorted image still looks right"
                )
            lcx, lcy = self.lens.centre_px
            if not (0 <= lcx <= w and 0 <= lcy <= h):
                problems.append(
                    f"lens.centre_px {self.lens.centre_px} is outside the {w}x{h} frame"
                )
            half_diag = math.hypot(w, h) / 2.0
            if half_diag > 0 and not (
                0.5 * half_diag <= self.lens.radius_px <= 1.5 * half_diag
            ):
                problems.append(
                    f"lens.radius_px={self.lens.radius_px} is far from the half-diagonal "
                    f"({half_diag:.1f}) the k1 convention is normalised to -- a radius from "
                    "another resolution rescales k1's meaning silently"
                )
        if self.mask_ignore != IGNORE:
            problems.append(
                f"mask_ignore={self.mask_ignore}, this build masks with {IGNORE} -- the file "
                "was written under a different sentinel and its masks cannot be read as-is"
            )
        if problems:
            raise ValueError("camera.json refused:\n  " + "\n  ".join(problems))
