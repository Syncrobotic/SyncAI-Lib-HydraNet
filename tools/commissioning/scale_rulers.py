"""Every commissioned camera's metres, checked against every ruler the scene carries.

The project finds its own reference frame here. Per store (the camera name's prefix):
the tile pitch of each camera through its own metres, the catalogue tile the store's
cameras agree on, each camera's measured counter height against the store's median, and
the person prior the calibration used. `rulers.combine` turns them into one factor and a
decision -- APPLY only when two rulers agree it is more than 10% off. Nothing is applied
without `--apply`; the verdicts are written either way.

  uv run python tools/commissioning/scale_rulers.py                # every commissioned camera
  uv run python tools/commissioning/scale_rulers.py Taichung-cam01 --apply

`--apply` rescales the camera in place: the previous camera.json and geometry cache are
moved to runs/commission01/stale_<date>/ and runs/site30k_qa/geometry_cache/stale_<date>/,
the cache is rebuilt at the new depth scale (`rebuild_geometry.py`), and the scene is
re-rendered. The verdict beside it records every ruler that voted.
"""

import argparse
import datetime as dt
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

from syncai_bev3d import floor_axis, rulers, scene_mesh
from syncai_hydranet.geometry.ground import pixel_to_ground, undistort_points

ROOT = Path(os.environ.get("SYNCAI_ROOT", Path(__file__).resolve().parents[2]))
VERDICTS = ROOT / "runs/commission_review/scale"


def store_of(camera: str) -> str:
    return camera.rsplit("-cam", 1)[0]


def read_camera(camera: str) -> dict:
    ev = scene_mesh.load_evidence(camera)
    cf = ev.cf
    fh, fw = ev.z["gx"].shape
    w, h = cf.image_size_px
    sc = np.array([w / fw, h / fh])

    def ground(px):
        pts = px * sc
        if cf.lens is not None:
            pts = undistort_points(pts, cf.lens.k1, cf.lens.centre_px, cf.lens.radius_px)
        x, z = pixel_to_ground(pts[:, 0], pts[:, 1], cf.camera, cf.plane)
        return np.stack([x, z], axis=1)

    axis = scene_mesh.store_axis(ev, camera)
    period, strength = floor_axis.floor_period(
        ev.plate, ev.walk, ev.z["gz"], ev.z["geom_ok"], ground, axis
    )
    if strength < 0.35:
        period = None
    _cf, _grids, heights, _gh = scene_mesh.cell_grids(camera, evidence=ev)
    calib_path = ROOT / f"runs/onboard01/{camera}.calib.json"
    calib = json.loads(calib_path.read_text()) if calib_path.exists() else {}
    return {
        "cf": cf,
        "height_m": cf.plane.height,
        "period": period,
        "period_strength": strength,
        "table_h": heights.get(4),
        "calib": calib,
    }


def verdicts(cameras: list[str]) -> dict[str, rulers.Verdict]:
    read = {c: read_camera(c) for c in cameras}
    out = {}
    for store in sorted({store_of(c) for c in cameras}):
        members = [c for c in cameras if store_of(c) == store]
        tile, err = rulers.standard_tile({c: read[c]["period"] for c in members})
        tables = [read[c]["table_h"] for c in members if read[c]["table_h"]]
        print(
            f"== {store}: tile {tile} (median disagreement {err:.3f}), "
            f"counters {[round(t, 2) for t in tables]}"
        )
        for c in members:
            r = read[c]
            rs = [
                rulers.person_ruler(r["calib"]),
                # the catalogue size is the cameras' consensus, not an independent fact
                rulers.tile_ruler(r["period"], tile, anchored=False),
                rulers.table_ruler(r["table_h"], tables),
            ]
            v = rulers.combine([x for x in rs if x is not None])
            out[c] = v
            votes = "  ".join(f"{x.name} x{x.factor:.3f} ({x.note})" for x in v.rulers)
            verdict = "APPLY" if v.apply else "keep "
            print(f"  {c:16s} H {r['height_m']:.2f}  {verdict} {v.reason}\n      {votes}")
            VERDICTS.mkdir(parents=True, exist_ok=True)
            (VERDICTS / f"{c}.json").write_text(
                json.dumps(
                    {
                        "camera": c,
                        "store": store,
                        "store_tile_m": tile,
                        "height_m": r["height_m"],
                        "tile_pitch_m": r["period"],
                        "table_h_m": r["table_h"],
                        "factor": v.factor,
                        "apply": v.apply,
                        "reason": v.reason,
                        "rulers": [x.__dict__ for x in v.rulers],
                        "read_at": dt.datetime.now().isoformat(timespec="seconds"),
                    },
                    indent=1,
                )
            )
    return out


def update_calib(camera: str, factor: float, note: str) -> Path:
    """The onboarding calibration carries the applied scale too, or the next pass undoes it.

    `masks_pass` (through `recipe.CameraGeometry`) reads its geometry from
    `runs/onboard01/<camera>.calib.json`, not from camera.json, and regenerates the cache
    from it when the two disagree: the first time cam01 was rescaled in camera.json alone,
    the re-commission that followed rebuilt its cache at the old scale and the scene
    refused the pair (floor drift 9.3 m). The previous calibration is kept beside it.
    """
    stamp = dt.datetime.now().strftime("%Y%m%d")
    path = ROOT / f"runs/onboard01/{camera}.calib.json"
    calib = json.loads(path.read_text())
    stale = ROOT / f"runs/onboard01/stale_{stamp}"
    stale.mkdir(exist_ok=True)
    (stale / path.name).write_text(json.dumps(calib, indent=1))
    calib["scale"] = float(calib["scale"]) * factor
    calib["height_m"] = float(calib["height_m"]) * factor
    calib["scale_source"] = f"{calib.get('scale_source')} x{factor:.3f} by rulers ({note})"
    calib.setdefault("rulers", []).append(
        {"factor": factor, "note": note, "at": dt.datetime.now().isoformat(timespec="seconds")}
    )
    path.write_text(json.dumps(calib, indent=1))
    return path


def apply(camera: str, cf, factor: float, note: str) -> None:
    stamp = dt.datetime.now().strftime("%Y%m%d")
    cam_json = ROOT / f"runs/commission01/{camera}.camera.json"
    cache = ROOT / f"runs/site30k_qa/geometry_cache/{camera}.npz"
    with np.load(cache) as z:
        if "depth_scale" not in z.files:
            raise SystemExit(f"{cache} carries no depth_scale; rebuild it first")
        depth_scale = float(z["depth_scale"])
    stale_c = ROOT / f"runs/commission01/stale_{stamp}"
    stale_g = ROOT / f"runs/site30k_qa/geometry_cache/stale_{stamp}"
    stale_c.mkdir(exist_ok=True)
    stale_g.mkdir(exist_ok=True)
    shutil.copy2(cam_json, stale_c / cam_json.name)
    shutil.move(cache, stale_g / cache.name)
    rulers.scaled_camera(cf, factor).save(cam_json)
    update_calib(camera, factor, note)
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools/commissioning/rebuild_geometry.py"),
            str(cam_json),
            "--depth-scale",
            f"{depth_scale * factor:.5f}",
            "--out",
            str(cache),
        ],
        check=True,
    )
    subprocess.run(
        [sys.executable, str(ROOT / "tools/commissioning/scene_mesh.py"), camera], check=True
    )
    (VERDICTS / f"{camera}.applied.json").write_text(
        json.dumps(
            {
                "camera": camera,
                "factor": factor,
                "note": note,
                "height_m": cf.plane.height * factor,
                "depth_scale": depth_scale * factor,
                "previous": {
                    "camera_json": str(stale_c / cam_json.name),
                    "cache": str(stale_g / cache.name),
                },
                "applied_at": dt.datetime.now().isoformat(timespec="seconds"),
            },
            indent=1,
        )
    )
    print(f"  applied x{factor:.3f} to {camera}; previous files in {stale_c} and {stale_g}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("cameras", nargs="*")
    ap.add_argument(
        "--apply", action="store_true", help="rescale the cameras the verdict says to"
    )
    a = ap.parse_args()
    cameras = a.cameras or sorted(
        p.name.replace(".camera.json", "")
        for p in (ROOT / "runs/commission01").glob("*.camera.json")
        if not p.name.startswith("test")
    )
    # every store-mate is read, so a single named camera still gets its store's consensus
    stores = {store_of(c) for c in cameras}
    everyone = sorted(
        p.name.replace(".camera.json", "")
        for p in (ROOT / "runs/commission01").glob("*.camera.json")
        if store_of(p.name.replace(".camera.json", "")) in stores
    )
    vs = verdicts(everyone)
    if a.apply:
        for c in cameras:
            v = vs[c]
            if v.apply:
                apply(c, read_camera(c)["cf"], v.factor, v.reason)
            else:
                print(f"  {c}: not applied -- {v.reason}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
