#!/usr/bin/env python3
"""Fleet onboarding calibration sweep: the per-camera spatial calibration that calib01 and
calib02 validated, strung into a repeatable pipeline.

  nice -n 10 .venv/bin/python scripts/onboard_camera.py --out runs/onboard01
  nice -n 10 .venv/bin/python scripts/onboard_camera.py --camera Taichung-cam01
  .venv/bin/python scripts/onboard_camera.py --report-only --out runs/onboard01

For every selling_floor camera (`datasets/studioa_clips/cameras.json`):

1. **Orientation**: pick a daytime plate and reuse the calib01 pipeline from
   `syncai_bev3d.plate_calibration` (the package module extracted from the since-deleted
   `calibrate_from_plate.py`; imported, not copied -- a second copy of the geometry
   arithmetic is a second chance to get it wrong): undistort (division model
   k1 = -0.225, a fleet-hardware assumption, tile-grid measured only on Taichung-cam01)
   -> DA-V2 once -> RANSAC "lowest plausible horizontal plane" -> pitch/roll/plane
   height. calib01 showed this path holds pitch to ±0.7° of the anchor (provided the
   vfov is pinned).
2. **Scale (the automatable subset)**: DA-V2's metres are not to be trusted (this fleet
   overestimates 1.45-1.6x, calib01); what is automated is the person-height statistic --
   boxes from `datasets/retail_person_gdino01` pass the same gates (score ≥ 0.5, edge,
   aspect ratio), are pushed through the fitted plane, and the median against the 1.70 m
   prior gives the scale correction back. **≥ 10 usable heights or no number** -- below
   that, flag `unmeasured`. The door-height / floor-tile visual method (calib02's ±5-8%
   method) cannot be automated; the fields stay null, flagged `needs_visual_reference`.
3. **Measurement discipline**: every camera runs and stores the full three-point vfov
   sensitivity at 55/70.4/85; the primary value is 70.4° (the cam01 tile-grid pinned
   value, a fleet assumption for the rest). No single number enters a table without a
   band -- the band is the spread of the vfov scan.
4. **Plate dirty region**: the share of the plate HydraNet (stable01's exact
   config/checkpoint pair) reads as person -- stable_infer's empirical figure:
   Kaohsiung-cam04's 112757 plate is 8.6%. What a dirty plate means: someone is standing
   inside the median, static compositing never takes over there, and geometry measured on
   the plate should stay out of that region too.

Writes `runs/onboard01/<camera>.calib.json` (field names carry their units, consumed by
hydranet-scene/events) and the fleet-wide `runs/onboard01/REPORT.md`.

When a shared training run is on the GPU: DA-V2 and HydraNet are both single-image and
batch is always 1; start the whole process under `nice -n 10`.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys
import traceback
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))

# The geometry and its checks, all reused from the package -- not from
# `calibrate_from_plate.py`, which was the same code's CLI: the pipeline moved into the
# package on 2026-08-19 so a script would stop being another script's library, and
# `500cdd2` deleted the emptied CLI on 2026-08-25.
from syncai_bev3d import plate_calibration as pc  # noqa: E402
from syncai_bev3d.calibrate import horizon_row  # noqa: E402
from syncai_hydranet.geometry.ground import Camera  # noqa: E402
from syncai_hydranet.shipped import SHIPPED_CONFIG, for_terrain  # noqa: E402

SCHEMA = "hydranet-onboard-calib/v1"
# Defaults, not constants. A second fleet arrived 2026-08-27 -- ten RTSP channels with
# their own corpus and their own person boxes -- and these two paths were the only reason
# this sweep was single-corpus; every other input was already a flag. They are threaded
# as arguments rather than rebound as globals, because `provenance` records the path the
# run actually used and a global would record whichever one was set last.
CAMERAS_JSON = Path("datasets/studioa_clips/cameras.json")
PERSON_ANNS = Path("datasets/retail_person_gdino01/annotations/instances_all.json")
# The plate meter reads the terrain head's `person` channel, so it takes the
# terrain-selected checkpoint -- see `syncai_hydranet.shipped` for why that is a question
# with two answers and why a caller has to name which one it is asking.
PLATE_MODEL_CONFIG = SHIPPED_CONFIG
PLATE_MODEL_CKPT = for_terrain()
K1_FLEET = pc.K1_FLEET  # the fleet lens; defined once in syncai_bev3d.plate_calibration
VFOV_PRIMARY = 70.4  # likewise: pinned on cam01, a fleet assumption for the rest
MIN_HEIGHTS = 10  # min samples for the person-height statistic to emit a number (task spec)
DIRTY_PLATE_FRAC = 0.05  # person share above this marks a dirty plate (cam04's 8.6% is above)
# calib01: the gap between the person-height scale and the tile-grid anchor (pose bias,
# a systematic term)
PERSON_PRIOR_SYS_FRAC = 0.11
VFOV_UNPINNED_SYS_FRAC = 0.05  # calib02: the ±5% systematic term for vfov-unpinned cameras


# ---------------------------------------------------------------------------
# DA-V2 pipeline cache: pc.run_depth rebuilds the pipeline on every call, so sweeping
# 22 cameras would reload the Large weights 22 times. Swap transformers.pipeline for a
# keyed-cache version -- not a line of run_depth's pre/post-processing moves; the same
# (task, model, device) triple just gets the same instance back.


def _install_pipeline_cache() -> None:
    import transformers

    real = transformers.pipeline
    cache: dict = {}

    def cached(task, model=None, device=None, **kw):
        key = (task, model, device)
        if key not in cache:
            cache[key] = real(task, model=model, device=device, **kw)
        return cache[key]

    transformers.pipeline = cached


# ---------------------------------------------------------------------------
# Plate dirty region: measured exactly the way stable_infer measures it (same config /
# checkpoint / preprocess / argmax), except no background board is built and only the
# person share is reported.


class PlatePersonMeter:
    def __init__(self, config: Path, checkpoint: Path):
        import torch

        from syncai_hydranet.config import load_config
        from syncai_hydranet.models.hydranet import build_model
        from syncai_hydranet.utils.checkpoint import load_checkpoint, select_weights
        from syncai_hydranet.utils.device import pick_device

        self._torch = torch
        cfg = load_config(str(config), [])
        self.device = pick_device(cfg.get("device"))
        self.model = build_model(cfg).to(self.device).eval()
        self.model.load_state_dict(select_weights(load_checkpoint(checkpoint), "ema"))
        self.size = tuple(cfg["data"]["input_size"])  # (H, W)
        self.classes = list(cfg["data"]["terrain_classes"])
        self.person_idx = self.classes.index("person")

    def person_frac(self, plate: Image.Image) -> dict:
        from syncai_hydranet.utils.visualize import crop_box, preprocess

        px, _canvas, region = preprocess(plate, self.size)
        with self._torch.no_grad():
            lab = self.model.forward(px.to(self.device))["terrain"][0].argmax(0).cpu().numpy()
        content = crop_box(lab, region)
        return {
            "person_frac_plate": round(float((content == self.person_idx).mean()), 4),
            # stable_infer prints the whole-canvas share (letterbox borders included);
            # kept so the two tables can be compared
            "person_frac_canvas": round(float((lab == self.person_idx).mean()), 4),
        }


# ---------------------------------------------------------------------------
# One camera


def selling_floor_cameras(cameras_json: Path = CAMERAS_JSON) -> list[str]:
    d = json.loads(Path(cameras_json).read_text())
    return sorted(c for c, v in d["cameras"].items() if v.get("role") == "selling_floor")


def _band(vals: list[float | None]) -> list[float] | None:
    xs = [v for v in vals if v is not None]
    if not xs:
        return None
    return [round(min(xs), 2), round(max(xs), 2)]


def onboard_one(
    camera: str,
    vfovs: list[float],
    k1: float,
    meter: PlatePersonMeter | None,
    plates_root: Path,
    person_anns: Path = PERSON_ANNS,
) -> dict:
    now = _dt.date.today().isoformat()
    is_pinned = camera == "Taichung-cam01"  # tile grid: k1 and vfov both measured
    flags: list[str] = []
    if not is_pinned:
        flags += ["vfov_fleet_assumed", "k1_fleet_assumed"]

    result: dict = {
        "schema": SCHEMA,
        "camera": camera,
        "generated": now,
        "vfov_assumed_deg": VFOV_PRIMARY,
        "vfov_source": "tile_grid_pinned" if is_pinned else "fleet_hardware_assumed",
        "k1_division_model": k1,
        "k1_source": "tile_grid_measured" if is_pinned else "fleet_hardware_assumed",
        # calib02's visual priors (door height / floor tiles) cannot be automated: the
        # fields are kept, the values left empty, and the SOP's manual step back-fills
        # them. This is "not measured", not "measured out as null".
        "visual_reference": {
            "door_height_implied_height_m": None,
            "tile_pitch_implied_height_m": None,
            "status": "needs_visual_reference",
        },
        "provenance": {
            "orientation_pipeline": "syncai_bev3d.plate_calibration (imported)",
            "depth_model": pc.MODEL,
            "depth_model_revision": pc.MODEL_REVISION,
            "person_boxes": str(person_anns),
            "person_height_prior_m": pc.ADULT_M,
            "plate_person_model": f"{PLATE_MODEL_CONFIG} + {PLATE_MODEL_CKPT} (ema)",
        },
    }

    cam_dir = plates_root / camera
    if not cam_dir.is_dir():
        result.update(
            {
                "plate_used": None,
                "pitch_deg": None,
                "roll_deg": None,
                "height_m": None,
                "height_source": "unmeasured",
                "scale": None,
                "scale_source": "unmeasured",
                "uncertainty": None,
                "flags": [*flags, "no_plate", "scale_unmeasured", "needs_visual_reference"],
            }
        )
        return result

    slot = pc.pick_daytime_slot(cam_dir)
    plate_path = cam_dir / f"plate_{slot}.png"
    rgb = np.asarray(Image.open(plate_path).convert("RGB"))
    h, w = rgb.shape[:2]
    result.update(
        {"plate_used": str(plate_path), "plate_slot_utc": slot, "frame_hw_px": [h, w]}
    )

    # Plate dirty region (the raw plate, not the undistorted one -- stable_infer
    # measures the raw board)
    if meter is not None:
        result.update(meter.person_frac(Image.fromarray(rgb)))
        if result["person_frac_plate"] > DIRTY_PLATE_FRAC:
            flags.append("dirty_plate_person_frac_gt_0p05")

    # Orientation: undistort -> DA-V2 once -> a RANSAC floor pick per vfov
    rgb_u = pc.undistort_image(rgb, k1)
    depth = pc.run_depth(rgb_u)

    boxes = pc.load_person_boxes(person_anns, camera, w, h, k1)
    result["person_boxes_after_gates"] = len(boxes)

    by_vfov: list[dict] = []
    for vfov in vfovs:
        cam = Camera.from_vfov(h, w, vfov)
        cands = pc.floor_candidates(depth, cam, inlier_m=0.05)
        plane, residual, cand_rows = pc.choose_floor(cands, 0.05)
        row: dict = {"vfov_deg": vfov, "candidates": cand_rows}
        if plane is None:
            row["failed"] = "no plausible floor plane among RANSAC candidates"
            flags.append(f"no_floor_at_vfov_{vfov:g}")
            by_vfov.append(row)
            continue
        import math

        pitch_deg = math.degrees(plane.pitch)
        hrow = horizon_row(pitch_deg, cam.fy, h)
        finite = np.isfinite(residual)
        row.update(
            {
                "height_dav2_raw_m": round(plane.height, 3),
                "pitch_deg": round(pitch_deg, 2),
                "roll_deg": round(math.degrees(plane.roll), 2),
                "inlier_frac_frame": round(float((np.abs(residual[finite]) < 0.05).mean()), 4),
                "horizon_row": round(hrow, 1),
                "horizon_inside_frame": bool(0 <= hrow < h),
                "column": pc.column_health(cam, plane, h, w),
                "floor_scale_dav2_raw": pc.floor_scale(cam, plane, h, w),
            }
        )
        if len(boxes):
            row["person"] = pc.person_checks(boxes, cam, plane, (h, w), vfov)
        by_vfov.append(row)
    result["by_vfov"] = by_vfov

    ok_rows = [r for r in by_vfov if "pitch_deg" in r]
    primary = next((r for r in by_vfov if r["vfov_deg"] == VFOV_PRIMARY), None)
    if primary is None or "pitch_deg" not in primary:
        primary = ok_rows[0] if ok_rows else None
        if primary is not None:
            flags.append("primary_vfov_failed_using_fallback")

    if primary is None:
        result.update(
            {
                "pitch_deg": None,
                "roll_deg": None,
                "height_m": None,
                "height_source": "unmeasured",
                "scale": None,
                "scale_source": "unmeasured",
                "uncertainty": None,
                "flags": [
                    *flags,
                    "no_floor_any_vfov",
                    "scale_unmeasured",
                    "needs_visual_reference",
                ],
            }
        )
        return result

    if primary.get("horizon_inside_frame"):
        flags.append("horizon_inside_frame_pose_rejected")
    if abs(primary["roll_deg"]) > 5.0:
        flags.append("roll_abs_gt_5deg")

    # Scale: the person-height statistic (the automatable subset). `scale` is "the metre
    # correction multiplied onto DA-V2 plane quantities".
    person = primary.get("person", {})
    n_heights = person.get("n_heights", 0)
    scale = person.get("scale_person")
    scale_rows = [(r["vfov_deg"], r.get("person", {}).get("scale_person")) for r in ok_rows]
    if scale is not None and n_heights >= MIN_HEIGHTS:
        med = person["implied_person_height_med_m"]
        mad = person["implied_person_height_mad_m"]
        height_m = round(primary["height_dav2_raw_m"] * scale, 2)
        height_src = "dav2_plane_height_x_person_height_scale"
        scale_src = f"person_height_median_vs_{pc.ADULT_M}m_prior_n{n_heights}"
        stat_frac = round(mad / med, 3)
        # The calibrated-height vfov band: each vfov uses its own plane height times its
        # own person-height scale -- the two slide with vfov in the same direction, so
        # the product is steadier than the raw height, and the band measures that fact
        # rather than assuming it
        heights_scaled = []
        for r in ok_rows:
            s = r.get("person", {}).get("scale_person")
            if s:
                heights_scaled.append(round(r["height_dav2_raw_m"] * s, 2))
        scale_band = _band([s for _, s in scale_rows])
    else:
        height_m, height_src = None, "unmeasured"
        scale, scale_src = None, "unmeasured"
        stat_frac, heights_scaled, scale_band = None, [], None
        flags += ["scale_unmeasured", "needs_visual_reference"]
        if n_heights:
            flags.append(f"person_heights_n{n_heights}_below_min{MIN_HEIGHTS}")

    unc: dict = {
        "vfov_scan_deg": [r["vfov_deg"] for r in ok_rows],
        "pitch_deg_band_vfov_scan": _band([r["pitch_deg"] for r in ok_rows]),
        "pitch_deg_vs_anchor_at_pinned_vfov": 0.7,  # calib01 tile-grid anchor cross-check
        "roll_deg_band_vfov_scan": _band([r["roll_deg"] for r in ok_rows]),
        "height_dav2_raw_m_band_vfov_scan": _band([r["height_dav2_raw_m"] for r in ok_rows]),
        "height_m_band_vfov_scan": _band(heights_scaled) if heights_scaled else None,
        "scale_band_vfov_scan": scale_band,
        "scale_frac_stat_person_mad": stat_frac,
        "scale_frac_sys_person_prior": PERSON_PRIOR_SYS_FRAC if scale else None,
        "scale_frac_sys_vfov_unpinned": (None if is_pinned else VFOV_UNPINNED_SYS_FRAC),
        "dominant_unknown": "vfov",  # the consistent calib01/calib02 conclusion
    }

    result.update(
        {
            "pitch_deg": primary["pitch_deg"],
            "roll_deg": primary["roll_deg"],
            "height_m": height_m,
            "height_source": height_src,
            "height_dav2_raw_m": primary["height_dav2_raw_m"],
            "scale": scale,
            "scale_source": scale_src,
            "floor_m_per_px_u_at_ref": (
                round(primary["floor_scale_dav2_raw"]["m_per_px_u"] * scale, 4)
                if scale and primary["floor_scale_dav2_raw"].get("on_floor")
                else None
            ),
            "floor_m_per_px_v_at_ref": (
                round(primary["floor_scale_dav2_raw"]["m_per_px_v"] * scale, 4)
                if scale and primary["floor_scale_dav2_raw"].get("on_floor")
                else None
            ),
            "floor_ref_px": primary["floor_scale_dav2_raw"].get("ref_px"),
            "uncertainty": unc,
            "flags": flags,
        }
    )
    return result


# ---------------------------------------------------------------------------
# The fleet report


def _fmt_band(band: list[float] | None, unit: str = "") -> str:
    if not band:
        return "—"
    return f"[{band[0]:g}, {band[1]:g}]{unit}"


def write_report(out_dir: Path, cameras: list[str]) -> None:
    rows = []
    for cam in cameras:
        p = out_dir / f"{cam}.calib.json"
        if p.exists():
            rows.append(json.loads(p.read_text()))
    done = [r for r in rows if r.get("pitch_deg") is not None]
    scaled = [r for r in rows if r.get("scale") is not None]
    unscaled = [r for r in rows if r.get("scale") is None]
    rolled = [r for r in done if abs(r["roll_deg"]) > 5.0]
    dirty = sorted(
        (r for r in rows if r.get("person_frac_plate") is not None),
        key=lambda r: -r["person_frac_plate"],
    )

    lines: list[str] = []
    lines.append("# onboard01 — fleet onboarding calibration sweep (23 selling_floor)\n")
    lines.append(
        f"Generated: {_dt.date.today().isoformat()} - `scripts/onboard_camera.py` - "
        "orientation pipeline imported from `syncai_bev3d/plate_calibration.py` "
        "(calib01-validated: "
        "pitch ±0.7° of the anchor, provided the vfov is pinned) - automated scale subset "
        "= the person-height statistic (calib02's validated visual-prior method, ±5-8%, "
        "needs a human; fields stay null, flagged `needs_visual_reference`).\n"
    )
    lines.append(
        f"**Completion: {len(done)}/{len(cameras)} cameras with an orientation; "
        f"{len(scaled)} scale automatically, {len(unscaled)} need a visual reference.**\n"
    )
    lines.append(
        "Band discipline: primary values are at vfov 70.4° (tile-grid pinned only on "
        "Taichung-cam01, a fleet-hardware assumption for the rest); the bracketed band "
        "is the spread of the vfov 55-85° scan. **The band is not error-bar decoration "
        "-- vfov is the dominant unknown, and calib01 measured height sliding ~0.9 m and "
        "scale sliding 0.96 -> 0.56 across vfov.**\n"
    )
    lines.append("## Fleet table\n")
    lines.append(
        "| Camera | pitch°@70.4 (55-85 band) | roll° (band) | H_DA-V2 raw m (band) | "
        "scale (source) | H m calibrated (band) | plate person share | flags |"
    )
    lines.append("|---|---|---|---|---|---|---|---|")
    for r in rows:
        u = r.get("uncertainty") or {}
        cam = r["camera"]
        if r.get("pitch_deg") is None:
            lines.append(f"| {cam} | — | — | — | — | — | — | {', '.join(r['flags'])} |")
            continue
        n_boxes = ""
        if r.get("scale"):
            n_boxes = r["scale_source"].rsplit("_n", 1)[-1]
        scale_cell = (
            f"{r['scale']:.3f} (person n={n_boxes})" if r.get("scale") else "unmeasured"
        )
        height_cell = (
            f"**{r['height_m']:.2f}** {_fmt_band(u.get('height_m_band_vfov_scan'))}"
            if r.get("height_m")
            else "— (needs visual reference)"
        )
        pf = r.get("person_frac_plate")
        pf_cell = f"{pf:.1%}" if pf is not None else "—"
        interesting = [
            f for f in r["flags"]
            if f not in ("vfov_fleet_assumed", "k1_fleet_assumed")
        ]  # fmt: skip
        lines.append(
            f"| {cam} | {r['pitch_deg']:.1f} "
            f"{_fmt_band(u.get('pitch_deg_band_vfov_scan'))} | "
            f"{r['roll_deg']:+.1f} {_fmt_band(u.get('roll_deg_band_vfov_scan'))} | "
            f"{r['height_dav2_raw_m']:.2f} "
            f"{_fmt_band(u.get('height_dav2_raw_m_band_vfov_scan'))} | "
            f"{scale_cell} | {height_cell} | {pf_cell} | "
            f"{', '.join(interesting) or '—'} |"
        )
    lines.append("")

    lines.append("## Sideways-mounted cameras (|roll| > 5°)\n")
    if rolled:
        lines.append(
            "The VLM pilot said at least three cameras are mounted sideways; below is "
            "the fleet's first per-camera measured roll -- a quantity the plane fit "
            "gives away for free, and nothing else in the pipeline models roll:\n"
        )
        for r in sorted(rolled, key=lambda r: -abs(r["roll_deg"])):
            u = r.get("uncertainty") or {}
            lines.append(
                f"- **{r['camera']}: roll {r['roll_deg']:+.1f}°**"
                f" (vfov band {_fmt_band(u.get('roll_deg_band_vfov_scan'))})"
            )
    else:
        lines.append("(none)")
    lines.append("")

    lines.append("## Plate dirty regions (model person share, measured stable_infer's way)\n")
    lines.append(
        "Reference point: stable01 measured Kaohsiung-cam04's 112757 plate at 8.6%. "
        f"Above {DIRTY_PLATE_FRAC:.0%} flags `dirty_plate_person_frac_gt_0p05` -- static "
        "compositing never takes over in a dirty region, and geometry measured on the "
        "plate (tile joints, door edges) should keep out of it too.\n"
    )
    for r in dirty[:8]:
        mark = "  <- dirty plate" if r["person_frac_plate"] > DIRTY_PLATE_FRAC else ""
        lines.append(
            f"- {r['camera']} ({r.get('plate_slot_utc', '?')}): "
            f"{r['person_frac_plate']:.1%}{mark}"
        )
    lines.append("")

    if scaled:
        svals = [r["scale"] for r in scaled]
        lines.append("## Scale: automatic (person height) vs needs a visual reference\n")
        lines.append(
            f"- Person-height scale produced a number (≥{MIN_HEIGHTS} height samples): "
            f"**{len(scaled)} cameras**, median scale {float(np.median(svals)):.3f}, "
            f"range [{min(svals):.3f}, {max(svals):.3f}]"
            " (calib01's three cameras were 0.69-0.76; if the spread is of the same "
            'order, a fleet-constant correction is still "plausible but unproven").'
        )
        lines.append(
            f"- Needs a visual reference (door height / floor tiles, the calib02 "
            f"method): **{len(unscaled)} cameras**. Note also that the person-height "
            "scale carries its own ±11% systematic term (the 1.70 m prior includes pose "
            "bias, calib01), and **every camera's visual fields are left null** -- "
            "converging to ±5-8% still takes one pass of calib02's door/tile step; "
            'person height only turns "no metres at all" into "metres to within ±11%".'
        )
        lines.append("")

    lines.append("## Measurement-discipline notes\n")
    lines.append(
        "- Every camera's three-point vfov sensitivity is stored in full under "
        "`by_vfov` in `<camera>.calib.json`; the bracket next to each primary value in "
        "the table is its band.\n"
        "- k1 and vfov are measured (tile grid) only on Taichung-cam01; the other 22 "
        "cameras carry `vfov_fleet_assumed`/`k1_fleet_assumed` and each shoulders a "
        "±5% systematic term (calib02).\n"
        "- DA-V2 never sets metres: `height_dav2_raw_m` is a shape quantity, and only "
        "multiplied by `scale` is it metres; a camera with `scale_source: unmeasured` "
        "has no metres.\n"
    )
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n")
    print(f"REPORT.md written: {len(done)}/{len(cameras)} cameras calibrated")


# ---------------------------------------------------------------------------


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--camera", action="append", help="run only these cameras (repeatable)")
    ap.add_argument("--out", type=Path, default=Path("runs/onboard01"))
    ap.add_argument("--plates-root", type=Path, default=pc.PLATES)
    ap.add_argument("--k1", type=float, default=K1_FLEET)
    ap.add_argument("--vfovs", default="55,70.4,85")
    ap.add_argument("--skip-person-frac", action="store_true")
    ap.add_argument("--cameras-json", type=Path, default=CAMERAS_JSON)
    ap.add_argument(
        "--person-anns",
        type=Path,
        default=PERSON_ANNS,
        help="COCO boxes whose image file_names start with `<camera>__`",
    )
    ap.add_argument("--report-only", action="store_true")
    args = ap.parse_args(argv)

    fleet = selling_floor_cameras(args.cameras_json)
    cameras = args.camera or fleet
    args.out.mkdir(parents=True, exist_ok=True)

    if args.report_only:
        write_report(args.out, fleet)
        return 0

    _install_pipeline_cache()
    meter = None
    if not args.skip_person_frac:
        meter = PlatePersonMeter(PLATE_MODEL_CONFIG, PLATE_MODEL_CKPT)
    vfovs = [float(x) for x in args.vfovs.split(",")]

    for i, cam in enumerate(cameras, 1):
        print(f"[{i}/{len(cameras)}] {cam}")
        try:
            result = onboard_one(cam, vfovs, args.k1, meter, args.plates_root, args.person_anns)
        except Exception:
            traceback.print_exc()
            result = {
                "schema": SCHEMA,
                "camera": cam,
                "generated": _dt.date.today().isoformat(),
                "pitch_deg": None,
                "roll_deg": None,
                "height_m": None,
                "height_source": "unmeasured",
                "scale": None,
                "scale_source": "unmeasured",
                "uncertainty": None,
                "flags": ["error_see_log"],
            }
        (args.out / f"{cam}.calib.json").write_text(
            json.dumps(result, indent=2, ensure_ascii=False) + "\n"
        )
        p, r0 = result.get("pitch_deg"), result.get("roll_deg")
        s = result.get("scale")
        print(
            f"  pitch {p if p is not None else '—'}  roll {r0 if r0 is not None else '—'}  "
            f"scale {s if s is not None else 'unmeasured'}  "
            f"flags {result['flags']}"
        )

    write_report(args.out, fleet)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
