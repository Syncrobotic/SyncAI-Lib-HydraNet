"""The rulers a camera's metres can be checked against, and how they are combined.

The user's requirement (2026-09-10): the project finds and estimates its own reference
frame -- no operator, no model looking at the picture each time. So every ruler here is
read off what commissioning already produces (the plate, the masks, the calibration),
and the one prior that cannot be read off a picture -- which standard tile a store is
laid with -- is chosen by consensus across the store's cameras from a catalogue of
standard sizes, not typed in.

A `Ruler` says by what factor the camera's metres should be multiplied, with a
fractional uncertainty. Three are implemented:

* **person**: the 1.70 m prior the calibration already used. Factor 1 by definition,
  uncertainty from the calibration's own record (statistical MAD and the prior's
  systematic term). It is the anchor the others are compared to.
* **tile**: the floor's joint pitch in the camera's metres (`floor_axis.floor_period`)
  against the store's tile size. The size is the catalogue entry that the store's
  cameras agree on best -- Taichung reads 0.47 / 0.62 / 0.66 / 0.57 and only 0.60
  makes three of four agree within 10%, which is also what the fourth is wrong by.
* **table**: one store buys one counter, so every camera's measured table height should
  be the store's. Relative, like the tile without its catalogue: it moves an outlier
  toward its siblings and moves the siblings nowhere.

`combine` takes the weighted median of the log factors and decides: APPLY when the
combined factor is more than APPLY_MIN off unity AND at least two rulers put it on the
same side by more than AGREE_MIN; otherwise KEEP, with the reason. Two witnesses, not
one, because every ruler here has failed alone: the person prior by 28% on cam01, the
tile where a plank floor has no joints, the table where a camera sees one bar-height
counter.
"""

from __future__ import annotations

import dataclasses
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from syncai_hydranet.geometry.camera_json import CameraFile, Zone

# Commercial floor tiles, in metres. The consensus picks one of these per store.
STANDARD_TILES_M = (0.30, 0.40, 0.45, 0.50, 0.60, 0.80, 0.90, 1.00)
# A store's cameras must put the chosen tile within this of the catalogue size, as the
# median over cameras, or the store is not on a standard tile and the ruler abstains.
TILE_CONSENSUS_MAX = 0.12
# Fractional uncertainty of one tile reading: the pitch is read to ~2 cm on a 0.6 m
# tile, and the catalogue choice adds nothing once the consensus holds.
TILE_SIGMA = 0.05
# Fractional uncertainty of a table-height reading: p85 of a depth model's heights on a
# white counter, measured 0.78-0.97 across the fleet for what is one product.
TABLE_SIGMA = 0.08
# The factor beyond which a camera's metres are changed, and how far two rulers must
# both sit off unity, on the same side, before that change is made.
APPLY_MIN = 0.10
AGREE_MIN = 0.05


@dataclass(frozen=True)
class Ruler:
    name: str
    factor: float  # multiply the camera's current metres by this
    sigma: float  # fractional uncertainty of `factor`
    note: str = ""
    #: Whether the reference length came from outside the camera's own metres. Every
    #: ruler *reads* through those metres; what separates a witness from an echo is where
    #: its reference came from. The 1.70 m prior and a known object are anchored. A
    #: catalogue tile chosen because the store's cameras agree on it is not: two cameras
    #: through one over-scaled depth model agree with each other, not with the floor --
    #: FTI's 600 mm raised floor read 0.85 / 0.74 m and the catalogue picked 0.80
    #: (2026-09-10). A store-median counter is relative by construction.
    anchored: bool = True


def person_ruler(calib: dict) -> Ruler | None:
    """The calibration's own person-prior scale: factor 1, uncertainty from its record.

    None when the calibration says its scale was never measured -- `unmeasured`, or a
    bootstrap on the depth model's raw metres: factor 1 there would anchor the other
    rulers to the very reading they are meant to check. A record with no `scale_source`
    at all is read as measured; absence is not a declaration.
    """
    source = str(calib.get("scale_source") or "")
    if source.startswith("unmeasured") or "bootstrap" in source:
        return None
    u = calib.get("uncertainty") or {}
    stat = float(u.get("scale_frac_stat_person_mad") or 0.09)
    sys_ = float(u.get("scale_frac_sys_person_prior") or 0.11)
    n = calib.get("person_boxes_after_gates")
    return Ruler("person", 1.0, math.hypot(stat, sys_), f"{n} boxes, 1.70 m prior")


def standard_tile(periods_m: dict[str, float]) -> tuple[float | None, float]:
    """(catalogue tile size the store's cameras agree on, median disagreement), or (None, ..).

    Needs at least two cameras with a pitch; a single reading agrees with any size.
    """
    vals = np.array([p for p in periods_m.values() if p and np.isfinite(p)], float)
    if len(vals) < 2:
        return None, float("inf")
    best, best_err = None, float("inf")
    for size in STANDARD_TILES_M:
        err = float(np.median(np.abs(np.log(size / vals))))
        if err < best_err:
            best, best_err = size, err
    if best_err > TILE_CONSENSUS_MAX:
        return None, best_err
    return best, best_err


def tile_ruler(
    period_m: float | None, tile_m: float | None, *, anchored: bool = True
) -> Ruler | None:
    """`anchored=False` when `tile_m` is `standard_tile`'s consensus pick rather than a
    size known independently of the cameras' readings."""
    if not period_m or not tile_m:
        return None
    return Ruler(
        "tile",
        tile_m / period_m,
        TILE_SIGMA,
        f"pitch {period_m:.2f} m vs {tile_m:.2f} m",
        anchored=anchored,
    )


def table_ruler(table_h_m: float | None, store_table_h_m: list[float]) -> Ruler | None:
    """The store's median counter height over this camera's; None with fewer than three
    cameras to take a median of (this one included)."""
    if not table_h_m or len(store_table_h_m) < 3:
        return None
    ref = float(np.median(store_table_h_m))
    return Ruler(
        "table",
        ref / table_h_m,
        TABLE_SIGMA,
        f"{table_h_m:.2f} m vs store {ref:.2f} m",
        anchored=False,
    )


@dataclass(frozen=True)
class Verdict:
    factor: float
    apply: bool
    reason: str
    rulers: tuple[Ruler, ...]


def combine(rulers: list[Ruler]) -> Verdict:
    rs = tuple(r for r in rulers if r is not None)
    if not rs:
        return Verdict(1.0, False, "no ruler", ())
    if not any(r.anchored for r in rs):
        # A verdict here would be an echo: the rulers can agree with each other and all be
        # wrong by the same factor. Say so rather than "within 10%".
        names = ", ".join(r.name for r in rs)
        return Verdict(
            1.0,
            False,
            f"unanchored: {names} take their reference from the same metres they read; "
            "needs a person prior or a size known independently",
            rs,
        )
    logs = np.array([math.log(r.factor) for r in rs])
    w = np.array([1.0 / (r.sigma**2) for r in rs])
    order = np.argsort(logs)
    cw = np.cumsum(w[order]) / w.sum()
    f = float(math.exp(logs[order][int(np.searchsorted(cw, 0.5))]))
    off = abs(math.log(f))
    if off <= math.log(1 + APPLY_MIN):
        return Verdict(f, False, f"within {APPLY_MIN:.0%} of the current metres", rs)
    side = np.sign(math.log(f))
    agreeing = [
        r
        for r in rs
        if np.sign(math.log(r.factor)) == side
        and abs(math.log(r.factor)) > math.log(1 + AGREE_MIN)
    ]
    if len(agreeing) < 2:
        names = ", ".join(r.name for r in agreeing) or "none"
        return Verdict(f, False, f"x{f:.3f} but only one ruler ({names}) says so", rs)
    return Verdict(f, True, f"x{f:.3f}: {', '.join(r.name for r in agreeing)} agree", rs)


def scaled_camera(cf: CameraFile, factor: float) -> CameraFile:
    """The camera file with its metres multiplied: the plane height and every zone vertex.
    Pixels, lens and masks are untouched -- they never were in metres."""
    return dataclasses.replace(
        cf,
        plane=dataclasses.replace(cf.plane, height=cf.plane.height * factor),
        zones=tuple(
            Zone(z.name, z.kind, tuple((x * factor, y * factor) for x, y in z.points_m))
            for z in cf.zones
        ),
    )


def write_scaled_root(
    camera: str, cf: CameraFile, factor: float, repo: Path, out_root: Path, note: str
) -> Path:
    """A checkout-shaped review root: the rescaled camera file, links to its masks and
    plate, and a record of the factor. `repo` is the checkout the originals live in."""
    commission = out_root / "runs/commission01"
    commission.mkdir(parents=True, exist_ok=True)
    (out_root / "runs/site30k_qa/geometry_cache").mkdir(parents=True, exist_ok=True)
    scaled_camera(cf, factor).save(commission / f"{camera}.camera.json")
    for rel in set(cf.mask_files.values()):
        dst = commission / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists():
            dst.symlink_to(repo / "runs/commission01" / rel)
    extras = repo / "runs/commission01" / camera / "masks"
    if extras.exists() and not (commission / camera / "masks").exists():
        (commission / camera).mkdir(parents=True, exist_ok=True)
        (commission / camera / "masks").symlink_to(extras)
    if cf.plate_file:
        dst = out_root / cf.plate_file
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists():
            dst.symlink_to(repo / cf.plate_file)
    (out_root / f"{camera}.scale.json").write_text(
        json.dumps(
            {
                "camera": camera,
                "factor": factor,
                "height_m_before": cf.plane.height,
                "height_m_after": cf.plane.height * factor,
                "note": note,
            },
            indent=1,
        )
    )
    return out_root
