#!/usr/bin/env python3
"""v5.6: the batch recipe. v5.5 plus chroma-gated change withdrawal.

Nothing else is retuned, so the 30 frames stay comparable with runs/site30k_qa/batch30.
The three changes, each traced to a frame the user rejected (see
`git show b7457c2:docs/journal/2026-08-19-small-batch-breaks-three-gates.md` section 5):

  (1) THE FLOOR IS DECIDED ONCE PER CAMERA, over every day plate that camera has, not
      once per clip. K04's 06:30 plate carries a person smeared across the left floor;
      `dirty` deleted ground that is clean in the same camera's other two plates, and a
      per-clip decision can never look again.
  (2) CANDIDACY IS THE UNION OF THE TWO FLOOR TEACHERS, not the geometry alone. On
      Taichung-cam01 b03's floor channel (15.63% of frame) and the metric on-plane test
      (14.01%) overlap on only 8.55%, and v5.2 required the geometry for candidacy, so
      b03's own aisle was evidence that could never be admitted -- the holes the user saw.
  (3) A COMPUTER IS A COMPUTER. `computer monitor` and `imac` join the laptop family
      (both are in the repo's swept prompt library), and `keyboard` / `computer mouse`
      stop counting as exclusions for that family: a keyboard in front of a monitor is
      part of the workstation, not evidence against it.
  (4) A THIRD FLOOR SOURCE (v5.4, second eye pass): ground that is on-plane at a tight
      tolerance and horizontal is floor whatever b03 says. Taichung-cam01's aisle hole
      survived (1)-(3) because b03 reads the polished tile's reflection of the counter as
      `fixture` in all three plates, while the geometry measures |height| p90 0.015 m.

Otherwise the rules the user signed off on K04 frame 0002:

  floor      b03 plate floor OR on-plane within tol(r)=0.06+0.035r (cap 0.30), voted in
             BEV metre cells, boundary rebuilt by a guided filter on the plate, shadow
             enclosed by floor+person closed back into floor
  structure  SAM 3 static concepts on the PLATE -> instances -> clusters -> ONE class per
             object from three teachers (prompt family / b03 channel / metric geometry),
             painted whole, strongest first
  products   SAM 3 on the FRAME; laptop/tablet/phone at 0.45, boxed_stock at 0.40; each
             claim refused if an exclusion prompt beats it, if an exclusion lands within
             0.15 of it, or if the floor is what is underneath it

Per camera the geometry is built once; per clip the plate teachers run once; per frame
only the frame-dependent teachers run.

Usage: batch30.py <out_dir> <default_frames_per_clip> <camera:clip_stem[:frames]> ...
"""

import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

# Derived from this file rather than written out: an absolute path here breaks on any
# machine but the one it was typed on, and `sys.path` is the one place where that
# fails before anything else can report it. `parents[2]` is the repo root --
# tools/site30k/<file>.py.
_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_REPO / "scripts"))
import importlib.util  # noqa: E402

spec = importlib.util.spec_from_file_location(
    "campaign", str(_REPO / "scripts/campaign_site30k.py")
)
M = importlib.util.module_from_spec(spec)
spec.loader.exec_module(M)
from syncai_bev3d.teachers import sam3 as SAM3  # noqa: E402
from syncai_hydranet.data.frame_selection import describe, farthest_first  # noqa: E402
from syncai_hydranet.data.video import frames as decode_frames  # noqa: E402
from syncai_hydranet.data.video import probe  # noqa: E402
from syncai_hydranet.geometry.ground import Camera as GCamera  # noqa: E402
from syncai_hydranet.geometry.ground import (  # noqa: E402
    GroundPlane,
    pixel_to_ground,
    undistort_points,
)
from syncai_hydranet.utils.device import pick_device  # noqa: E402

# The repo root, derived rather than written out: every one of these 26 tools had it
# as an absolute path, so a second checkout ran against the first one's `runs/` and
# any machine but this one failed at import with a path and no reason. Two levels up
# from `tools/<group>/<tool>.py`, and `tests/test_no_absolute_sys_path.py` keeps it so.
ROOT = Path(__file__).resolve().parents[2]
# The pilot ran off datasets/studioa_clips; the campaign runs off the phase-2 pull. Both
# are the same layout (<root>/<camera>/<stem>.mp4), so the root is a setting rather than
# a fork in the code, and the plates follow it so a campaign plate never lands in the
# pilot's directory.
CLIP_ROOT = os.environ.get("SITE30K_CLIP_ROOT", "datasets/studioa_clips")
PLATE_ROOT = os.environ.get("SITE30K_PLATE_ROOT", "datasets/studioa_static")
H, W = 1080, 1920
IGNORE = 255
IDS = {
    "floor": 1,
    "wall": 2,
    "column": 3,
    "display_table": 4,
    "shelf": 5,
    "person": 6,
    "laptop": 7,
    "tablet": 8,
    "phone": 9,
    "boxed_stock": 10,
}
COLORS = {
    1: (60, 200, 60),
    2: (70, 130, 240),
    3: (200, 80, 220),
    4: (255, 190, 40),
    5: (0, 200, 200),
    6: (230, 25, 75),
    7: (90, 0, 200),
    8: (255, 0, 150),
    9: (255, 120, 0),
    10: (140, 60, 20),
}
TABLE_FAMILY = (
    "display table",
    "display podium",
    "retail counter",
    "display counter",
    "checkout counter",
    "table",
)
SHELF_FAMILY = (
    "shelving unit",
    "shelf",
    "merchandise rack",
    "product display rack",
    "product display stand",
    "showcase cabinet",
    "display case",
)
PRODUCT_PROMPTS = {
    # v5.3: the class covers every computer on a counter, portable or not. `computer
    # monitor` holds 27 of 48 cameras at peak 0.930 in the swept library and `imac` fires
    # on the reseller's desktop machines; without them the service-counter monitor on
    # Taichung-cam04 drew no claim at all at 06:31 and only 0.46-0.50 at 11:30.
    # `desktop computer` stays out: the same sweep has it masking upright price cards.
    "laptop": ("laptop", "open laptop", "notebook computer", "computer monitor", "imac"),
    "tablet": ("tablet", "tablet computer", "ipad on a stand"),
    "phone": ("phone", "smartphone", "mobile phone"),
    "boxed_stock": (
        "product box on a shelf",
        "boxed product",
        "packaged goods",
        "merchandise on a shelf",
        "headphones box",
    ),
}
NEGATIVE = (
    "desk telephone",
    "landline telephone",
    "office telephone",
    "printer",
    "receipt printer",
    "cash register",
    "point of sale terminal",
    "trolley",
    "cart",
    "trash can",
    "keyboard",
    "computer mouse",
    "card reader",
    "payment terminal",
    "credit card machine",
    "barcode scanner",
    "handheld scanner",
)
PRODUCT_MIN = {"laptop": 0.45, "tablet": 0.45, "phone": 0.45, "boxed_stock": 0.40}
# v5.3: exclusions that are PARTS of what they would veto. A keyboard covering 60% of a
# `laptop` claim is the laptop's own keyboard, or the one in front of the iMac; it is
# still a valid exclusion for every other family, where it means "this is a desk, not
# merchandise". Measured: 6 of 6 laptop refusals across Taichung-cam04's 11:30 frames
# were `contested with keyboard`, on the machine the user pointed at.
NEG_EXEMPT = {"laptop": ("keyboard", "computer mouse")}
# v5.4: a THIRD floor source, for ground that both teachers refuse. Taichung-cam01's
# aisle hole (36,042 px, refused in all three slots) measures |height| p90 = 0.015 m with
# 94.9% of its pixels horizontal -- it is the ground, and only b03 objects, because it
# reads the polished tile's reflection of the counter as `fixture` (100% / 96.8% / 99.4%
# across the three plates). So: on-plane at a TIGHT tolerance and horizontal is floor
# whatever b03 says. 0.05 m separates it from everything that matters -- the aisle patch
# is p90 0.015 and the accepted floor p50 0.022, while a fixture's top is a metre up and
# its vertical faces fail the horizontal test. Measured cost across the three pilot
# cameras: 96.6-100% of what it adds is currently UNLABELLED, and it claims no pixel of
# any accepted wall, table or shelf.
STRICT_TOL = 0.05
B03_WALL_MIN, B03_FIXTURE_MIN = 0.60, 0.50
# v6.0: the top-surface test is WITHDRAWN, on its own measured distribution over three
# cameras (every `column` candidate in the batch, telemetry in batch30_v59/report.json):
#
#   real pillars   K04 k5  63,078 px  score 0.887  flat 0.04  b03 fixture 0.02
#                  T01 k8  57,632 px  score 0.922  flat 0.18  b03 fixture 0.17
#   junk           T01 k16 12,902 px  score 0.000  flat 0.12  b03 fixture 0.99
#                  T01 k18 11,156 px  score 0.000  flat 0.34  b03 fixture 1.00
#                  T01 k20 10,277 px  score 0.000  flat 0.31  b03 fixture 0.99
#                  T01 k22  9,631 px  score 0.000  flat 0.07  b03 fixture 1.00
#                  T04 k23   2,724 px score 0.000  flat 0.43  b03 fixture 0.82
#                  T04 k25   2,124 px score 0.000  flat 0.49  b03 fixture 0.74
#                  T04 k26   1,835 px score 0.000  flat 0.48  b03 fixture 0.70
#
# `flat` does not separate them -- junk sits at 0.07 and 0.12, BELOW the real T01 pillar
# at 0.18, which is exactly what the first batch measured and this batch confirms on more
# cameras. Two other readings separate them completely, with no overlap at all:
# the prompt score (0.77-0.92 against 0.000 -- those candidates carry no `column` claim,
# they inherited the class from a tiebreak) and b03's FIXTURE channel (0.02-0.17 against
# 0.70-1.00). b03 still gets no say in what IS a column -- it has no column class and
# reads a real pillar as wall -- only in what is plainly a fixture.
COLUMN_SCORE_MIN, COLUMN_B03_FIX_MAX = 0.50, 0.50
COLUMN_GEOM_MIN = 0.50
TIE_MARGIN, TABLE_FLAT_MIN = 0.15, 0.30
FIXTURE_SCORE_MIN = 0.65
PART_CONTAINMENT, FOOTPRINT_TOL = 0.80, 0.15
CHANGE_WITHDRAW = 0.50
CHROMA_CHANGE = 4  # YCbCr levels; measured knee, see the withdrawal comment
ISLAND_PX = 800
PRODUCT_IOU, NEG_TIE, NEG_COVER, FLOOR_SUPPORT_MAX = 0.55, 0.15, 0.60, 0.30
GUIDE_RADIUS, GUIDE_EPS = 8, 1e-4
device = str(pick_device())


def box_sum(a, r):
    c = np.cumsum(np.pad(a, ((1, 0), (0, 0))), axis=0)
    a = c[np.minimum(np.arange(H) + r + 1, H)] - c[np.maximum(np.arange(H) - r, 0)]
    c = np.cumsum(np.pad(a, ((0, 0), (1, 0))), axis=1)
    return c[:, np.minimum(np.arange(W) + r + 1, W)] - c[:, np.maximum(np.arange(W) - r, 0)]


def guided(p, guide, r, eps):
    n_pts = box_sum(np.ones((H, W), np.float32), r)
    gm = np.stack([box_sum(guide[..., c], r) / n_pts for c in range(3)], -1)
    pm = box_sum(p, r) / n_pts
    cov = np.stack(
        [box_sum(guide[..., c] * p, r) / n_pts - gm[..., c] * pm for c in range(3)], -1
    )
    var = np.empty((H, W, 3, 3), np.float32)
    for i in range(3):
        for j in range(3):
            var[..., i, j] = (
                box_sum(guide[..., i] * guide[..., j], r) / n_pts - gm[..., i] * gm[..., j]
            )
    var += eps * np.eye(3, dtype=np.float32)
    a = np.linalg.solve(var, cov[..., None]).squeeze(-1)
    b = pm - (a * gm).sum(-1)
    return (np.stack([box_sum(a[..., c], r) / n_pts for c in range(3)], -1) * guide).sum(
        -1
    ) + box_sum(b, r) / n_pts


def slic_numpy(rgb, spacing, iters=8, compact=5.0):
    hh, ww = rgb.shape[:2]
    lab_ = rgb.astype(np.float32) / 255.0
    ys = np.arange(spacing // 2, hh, spacing)
    xs = np.arange(spacing // 2, ww, spacing)
    cy, cx = np.meshgrid(ys, xs, indexing="ij")
    centers = np.stack([cy.ravel().astype(np.float32), cx.ravel().astype(np.float32)], 1)
    cfeat = lab_[centers[:, 0].astype(int), centers[:, 1].astype(int)]
    label = -np.ones((hh, ww), np.int32)
    dist = np.full((hh, ww), np.inf, np.float32)
    yy, xx = np.mgrid[0:hh, 0:ww].astype(np.float32)
    for _ in range(iters):
        dist[:] = np.inf
        label[:] = -1
        for k in range(len(centers)):
            y0, x0 = centers[k]
            ylo, yhi = max(int(y0) - spacing, 0), min(int(y0) + spacing + 1, hh)
            xlo, xhi = max(int(x0) - spacing, 0), min(int(x0) + spacing + 1, ww)
            dc = ((lab_[ylo:yhi, xlo:xhi] - cfeat[k]) ** 2).sum(-1)
            ds = ((yy[ylo:yhi, xlo:xhi] - y0) ** 2 + (xx[ylo:yhi, xlo:xhi] - x0) ** 2) / (
                spacing**2
            )
            d = dc + compact * ds
            win = d < dist[ylo:yhi, xlo:xhi]
            dist[ylo:yhi, xlo:xhi][win] = d[win]
            label[ylo:yhi, xlo:xhi][win] = k
        for k in range(len(centers)):
            m = label == k
            if m.any():
                centers[k] = (yy[m].mean(), xx[m].mean())
                cfeat[k] = lab_[m].mean(0)
    return label


class CameraGeometry:
    """Per-camera geometry, built once: heights, ground points, and the invalid ring."""

    CACHE = ROOT / "runs/site30k_qa/geometry_cache"
    CALIB_ROOT = ROOT / "runs/onboard01"

    def __init__(self, camera, third):
        self.camera = camera
        # Every array here is a function of the calibration and the depth of ONE plate,
        # so it is the same for every clip of that camera and for every worker that ever
        # touches it. Building it costs ~40 s (Depth-Anything on the calib plate); over a
        # campaign that starts one process per camera-date that is hours of repeated
        # work, so it is cached on disk and the cache is keyed by the camera name.
        cache = self.CACHE / f"{camera}.npz"
        if cache.exists():
            z = np.load(cache)
            for k in ("gx", "gz", "oob", "height", "horiz", "geom_ok", "lx", "lz"):
                setattr(self, k, z[k])
            self.lx = None if not np.isfinite(self.lx).any() else self.lx
            self.lz = None if self.lx is None else self.lz
            self.geom = None
            print(
                f"  [{camera}] geometry from cache ({100 * self.oob.mean():.1f}% unmeasurable)",
                flush=True,
            )
            return
        self.geom = M.GeomTeacher(camera, third, calib_root=self.CALIB_ROOT)
        # A default, not a constant: a second fleet was onboarded to `runs/onboard02`
        # on 2026-08-27 and its cameras carry the same schema. Rebindable by a caller
        # (`CameraGeometry.CALIB_ROOT = ...`) the same way `CACHE` above already is,
        # so a tool can point at another sweep without a second copy of this loader.
        calib = json.loads((self.CALIB_ROOT / f"{camera}.calib.json").read_text())
        k1 = float(calib.get("k1_division_model") or 0.0)
        vfov = float(calib["vfov_assumed_deg"])
        plane = GroundPlane(
            height=float(calib["height_m"]),
            pitch=math.radians(calib["pitch_deg"]),
            roll=math.radians(calib["roll_deg"]),
        )
        vv, uu = np.mgrid[0:H, 0:W].astype(np.float64)
        und = undistort_points(
            np.stack([uu.ravel(), vv.ravel()], 1),
            k1,
            (W / 2.0, H / 2.0),
            math.hypot(H, W) / 2.0,
        )
        uu_u, vv_u = und[:, 0].reshape(H, W), und[:, 1].reshape(H, W)
        gx, gz = pixel_to_ground(uu_u, vv_u, GCamera.from_vfov(H, W, vfov), plane)
        self.gx, self.gz = gx.astype(np.float32), gz.astype(np.float32)
        ph, pw = np.asarray(Image.open(calib["plate_used"]).convert("RGB")).shape[:2]
        su, sv = uu_u * (pw / W), vv_u * (ph / H)
        # Where undistortion samples outside the plate the height map is the border's,
        # not the pixel's -- 21.6% of the frame on K04. Rules that need geometry abstain.
        self.oob = (su < 0) | (su > pw - 1) | (sv < 0) | (sv > ph - 1)
        self.height, self.horiz = self.geom.height, self.geom.horiz
        self.geom_ok = ~self.oob & np.isfinite(self.height)
        # level-frame horizontal coordinates, for footprints of off-plane objects
        self.lx, self.lz = None, None
        print(
            f"  [{camera}] geometry: {100 * self.oob.mean():.1f}% of the frame has no "
            f"measurable geometry (undistort samples outside the plate)"
        )

    def ensure_level_frame(self):
        """Fill `lx`/`lz` -- the level-frame X/Z used for footprints -- and cache them.

        **This lived in the campaign loop until 2026-09-01, which made `masks_pass.py`
        work only on cameras whose cache someone else had already built.** The
        constructor sets `lx`/`lz` to None on the fresh path and only the campaign filled
        them, so a genuinely new camera reached `decide_structure` with None and died in
        `np.isfinite(None)` several GPU-minutes in. Studio A's twenty-three never showed
        it because their caches predate the tool. It belongs here: this class already owns
        the cache and its `CALIB_ROOT`.
        """
        if self.lx is not None:
            return
        import math as _math

        from syncai_bev3d.plate_calibration import run_depth, undistort_image
        from syncai_hydranet.geometry.ground import unproject

        calib = json.loads((self.CALIB_ROOT / f"{self.camera}.calib.json").read_text())
        k1 = float(calib.get("k1_division_model") or 0.0)
        vfov = float(calib["vfov_assumed_deg"])
        plane = GroundPlane(
            height=float(calib["height_m"]),
            pitch=_math.radians(calib["pitch_deg"]),
            roll=_math.radians(calib["roll_deg"]),
        )
        native = np.asarray(Image.open(calib["plate_used"]).convert("RGB"))
        ph, pw = native.shape[:2]
        depth = run_depth(undistort_image(native, k1)) * float(calib["scale"])
        level = unproject(depth, GCamera.from_vfov(ph, pw, vfov)) @ plane.rotation
        vv, uu = np.mgrid[0:H, 0:W].astype(np.float64)
        und = undistort_points(
            np.stack([uu.ravel(), vv.ravel()], 1),
            k1,
            (W / 2.0, H / 2.0),
            _math.hypot(H, W) / 2.0,
        )
        su = np.clip(np.round(und[:, 0].reshape(H, W) * (pw / W)).astype(int), 0, pw - 1)
        sv = np.clip(np.round(und[:, 1].reshape(H, W) * (ph / H)).astype(int), 0, ph - 1)
        self.lx = level[..., 0][sv, su].astype(np.float32)
        self.lz = level[..., 2][sv, su].astype(np.float32)
        self.save_cache()

    def save_cache(self):
        """Written only once lx/lz exist, so the cache is never a partial one."""
        if self.lx is None or (self.CACHE / f"{self.camera}.npz").exists():
            return
        self.CACHE.mkdir(parents=True, exist_ok=True)
        # np.savez appends .npz unless the name already ends in it, so the temporary
        # name has to carry the suffix or the rename below chases a file that was never
        # written under that name.
        tmp = self.CACHE / f"{self.camera}.tmp{os.getpid()}.npz"
        np.savez(
            tmp,
            gx=self.gx,
            gz=self.gz,
            oob=self.oob,
            height=self.height,
            horiz=self.horiz,
            geom_ok=self.geom_ok,
            lx=self.lx,
            lz=self.lz,
        )
        Path(tmp).replace(self.CACHE / f"{self.camera}.npz")


def decide_structure(cl_masks, cl_votes, b03_maps, geo, lx, lz):
    """One class per object, then paint whole. Returns (mask, decisions).

    v5.8: `b03_maps` is every plate of the camera, and a cluster's b03 share is the BEST
    any plate gives it. The gates were written for a clean plate, and a busy slot has no
    such thing: Kaohsiung-cam04's 19:27 plate is 10.1% dirty and its 14:30 one 13.6%,
    which is why the wall behind the crowd lost patches in every frame of those clips
    while the same wall is intact on the 02:58 plate (0.0% dirty). One clean view is
    enough to prove what an immovable object is.
    """
    decisions = []
    for k, m in enumerate(cl_masks):
        best = {"wall": 0.0, "column": 0.0, "table": 0.0, "shelf": 0.0}
        for concept, prompt, score in cl_votes[k]:
            fam = (
                "wall"
                if concept == "wall"
                else "column"
                if concept == "column"
                else "table"
                if prompt in TABLE_FAMILY
                else "shelf"
                if prompt in SHELF_FAMILY
                else None
            )
            if fam:
                best[fam] = max(best[fam], score)
        ranked = sorted(best.items(), key=lambda kv: -kv[1])
        win, top = ranked[0]
        margin = top - ranked[1][1]
        good = m & geo.geom_ok
        flat = float((good & (geo.horiz >= 0.8)).sum() / max(good.sum(), 1))
        b03_wall = max(float((b[m] == 2).mean()) for b in b03_maps)
        b03_fix = max(float((b[m] == 4).mean()) for b in b03_maps)
        gshare = float(good.sum() / max(m.sum(), 1))
        note = ""
        if margin < TIE_MARGIN:
            pair = {win, ranked[1][0]}
            if pair <= {"table", "shelf"}:
                if gshare >= COLUMN_GEOM_MIN and (good & (geo.horiz >= 0.8)).sum() > 200:
                    win = "table" if flat >= TABLE_FLAT_MIN else "shelf"
                    note = f"tiebreak by top-surface share {flat:.2f}"
                else:
                    note = (
                        f"low margin {margin:.3f}, geometry abstains ({gshare:.2f}) "
                        "-- prompt winner kept, FLAGGED"
                    )
            elif "wall" in pair and (b03_fix >= B03_FIXTURE_MIN or b03_wall >= B03_WALL_MIN):
                win = (
                    "wall" if b03_wall >= B03_WALL_MIN else next(f for f in pair if f != "wall")
                )
                note = f"tiebreak by b03 (wall {b03_wall:.2f} / fixture {b03_fix:.2f})"
            else:
                win, note = None, f"contested, margin {margin:.3f}, b03 supports neither"
        reject = None
        if win in ("table", "shelf") and top < FIXTURE_SCORE_MIN:
            reject = f"fixture score {top:.2f} < {FIXTURE_SCORE_MIN}"
        elif win == "wall" and b03_wall < B03_WALL_MIN:
            reject = f"b03 wall share {b03_wall:.2f} < {B03_WALL_MIN}"
        elif win in ("table", "shelf") and b03_fix < B03_FIXTURE_MIN:
            reject = f"b03 fixture share {b03_fix:.2f} < {B03_FIXTURE_MIN}"
        elif win == "column":
            if top < COLUMN_SCORE_MIN:
                reject = f"no column claim ({top:.2f} < {COLUMN_SCORE_MIN}), class inherited"
            elif b03_fix > COLUMN_B03_FIX_MAX:
                reject = f"b03 fixture share {b03_fix:.2f} > {COLUMN_B03_FIX_MAX}: a fixture"
            elif gshare < COLUMN_GEOM_MIN:
                reject = f"column unadjudicable, {gshare:.2f} measurable"
        if win is None and reject is None:
            reject = note
        decisions.append(
            {
                "k": k,
                "px": int(m.sum()),
                "win": win,
                "score": round(top, 3),
                "margin": round(margin, 3),
                "note": note,
                "reject": reject,
                "flagged": "FLAGGED" in note,
                # the three teacher readings behind the verdict, recorded for
                # every cluster and not just the ones a tiebreak explained --
                # a gate can only be judged against the distribution it sees
                "b03_wall": round(b03_wall, 3),
                "b03_fix": round(b03_fix, 3),
                "flat": round(flat, 3),
                "gshare": round(gshare, 3),
            }
        )

    gfin = geo.geom_ok & np.isfinite(lx) & np.isfinite(lz)
    accept = sorted(
        [d for d in decisions if d["win"] and not d["reject"]], key=lambda d: -d["score"]
    )
    foot = {}
    for d in accept:
        m = cl_masks[d["k"]] & gfin
        foot[d["k"]] = (
            None
            if m.sum() < 500
            else (
                np.percentile(lx[m], 5),
                np.percentile(lx[m], 95),
                np.percentile(lz[m], 5),
                np.percentile(lz[m], 95),
            )
        )
    for i, d in enumerate(accept):
        m = cl_masks[d["k"]] & gfin
        gshare = float((cl_masks[d["k"]] & geo.geom_ok).sum() / max(cl_masks[d["k"]].sum(), 1))
        if foot[d["k"]] is None or gshare < COLUMN_GEOM_MIN:
            continue
        dy, dx = np.where(cl_masks[d["k"]])
        for e in accept[:i]:
            if foot[e["k"]] is None or e["win"] == d["win"]:
                continue
            ey, ex = np.where(cl_masks[e["k"]])
            in_box = float(
                (
                    (dy >= ey.min()) & (dy <= ey.max()) & (dx >= ex.min()) & (dx <= ex.max())
                ).mean()
            )
            if in_box < 0.5:
                continue
            x0, x1, z0, z1 = foot[e["k"]]
            share = float(
                (
                    (lx[m] >= x0 - FOOTPRINT_TOL)
                    & (lx[m] <= x1 + FOOTPRINT_TOL)
                    & (lz[m] >= z0 - FOOTPRINT_TOL)
                    & (lz[m] <= z1 + FOOTPRINT_TOL)
                ).mean()
            )
            if share >= PART_CONTAINMENT:
                d["reject"] = f"{share:.0%} inside C{e['k']} ({e['win']}): a part"
                break

    class_id = {"wall": 2, "column": 3, "table": 4, "shelf": 5}
    static = np.full((H, W), IGNORE, np.uint8)
    for d in sorted(
        [d for d in decisions if d["win"] and not d["reject"]], key=lambda d: -d["score"]
    ):
        sel = cl_masks[d["k"]] & (static == IGNORE)
        static[sel] = class_id[d["win"]]
    return static, decisions


def cluster(masks, meta):
    order = np.argsort([-m["px"] for m in meta])
    cl_masks, cl_votes = [], []
    for i in order:
        m = masks[i]
        for j, cm in enumerate(cl_masks):
            inter = (m & cm).sum()
            if inter / max((m | cm).sum(), 1) >= 0.6 or inter / max(m.sum(), 1) >= 0.85:
                cl_votes[j].append((meta[i]["concept"], meta[i]["prompt"], meta[i]["score"]))
                break
        else:
            cl_masks.append(m)
            cl_votes.append([(meta[i]["concept"], meta[i]["prompt"], meta[i]["score"])])
    return cl_masks, cl_votes


def ensure_plate(camera: str, slot: str, clip: Path, plate_path: Path) -> bool:
    """The clip's own person-free median, made here if the static pass never ran.

    Same definition as scripts/static_plates.py -- 0.5 Hz, half resolution, temporal
    median -- because the recipe reads a plate as "what this view looks like with nobody
    in it", and two definitions of that would be two datasets. Returns False if the clip
    cannot be decoded, which is a skip and not a crash: 4,122 clips came off a store's
    NVR and some of them are truncated.
    """
    if plate_path.exists():
        return True
    try:
        w, h, _ = probe(str(clip))
        frames_ = list(decode_frames(str(clip), w, h, 0.5))
    except Exception as exc:
        print(f"  !! {camera} {slot}: plate decode failed: {exc}", flush=True)
        return False
    if len(frames_) < 20:
        print(f"  !! {camera} {slot}: only {len(frames_)} frames, no plate", flush=True)
        return False
    small = np.stack(
        [
            np.asarray(Image.fromarray(f).resize((960, 540), Image.Resampling.BILINEAR))
            for f in frames_
        ]
    )
    plate = np.median(small, axis=0).astype(np.uint8)
    plate_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(plate).save(plate_path)
    print(f"  made plate {plate_path.name} from {len(frames_)} frames", flush=True)
    return True


def camera_floor(camera, geo, slots, plate_cache):
    """v5.3 (1)+(2): ONE floor per camera, from every day plate it has.

    Pooling is what makes a dirty plate survivable: a pixel counts as dirty only where
    EVERY plate of that camera is dirty, and counts as evidence wherever ANY plate saw
    floor. Candidacy is the union of the two teachers -- b03's floor channel is admitted
    on its own, where v5.2 required the metric geometry to agree and so threw away half
    of Taichung-cam01's evidence. The BEV vote, the island and hole sizes, the tolerance
    and the guided boundary are all unchanged from the recipe the user approved.
    """
    rng = np.hypot(geo.gx, geo.gz)
    tol = np.minimum(0.06 + 0.035 * np.where(np.isfinite(rng), rng, 1e9), 0.30)
    finh = np.isfinite(geo.height) & np.isfinite(rng)
    onplane = finh & (np.abs(geo.height) <= tol)
    onplane_flat = onplane & (geo.horiz >= 0.8)

    dirty_all = np.ones((H, W), bool)
    ev = np.zeros((H, W), bool)
    b03_floor_any = np.zeros((H, W), bool)
    ctx_any = np.zeros((H, W), bool)
    for slot in slots:
        b03p = plate_cache[(camera, slot)]
        dirty = ndimage.binary_dilation(b03p == 5, iterations=3)
        pf = (~dirty) & ((b03p == 1) | (onplane_flat & np.isin(b03p, (0, 2))))
        dirty_all &= dirty
        ev |= pf
        b03_floor_any |= b03p == 1
        ctx_any |= np.isin(b03p, (0, 1, 2))
        print(
            f"  [{camera} {slot}] plate dirty {100 * dirty.mean():.1f}%, "
            f"plate floor {100 * pf.mean():.1f}% of frame"
        )
    # v5.4 third source: unambiguous ground geometry, admitted whatever b03 says.
    strict = (
        np.isfinite(geo.height)
        & ~geo.oob
        & (~dirty_all)
        & (np.abs(geo.height) <= np.minimum(tol, STRICT_TOL))
        & (geo.horiz >= 0.8)
    )
    new_ev = strict & ~ev
    ev_teach = ev.copy()  # what the two teachers claim on their own, before the strict source
    ev = (ev | strict) & ~dirty_all
    cand = (~dirty_all) & ((onplane & ctx_any) | b03_floor_any | strict)
    print(
        f"  [{camera}] strict-geometry source: {100 * strict.mean():.1f}% of frame, "
        f"new to the evidence {100 * new_ev.mean():.1f}%"
    )
    print(
        f"  [{camera}] pooled: dirty in every plate {100 * dirty_all.mean():.1f}%, "
        f"candidates {100 * cand.mean():.1f}%, evidence {100 * ev.mean():.1f}% of frame"
    )

    cell = 0.05
    gw, gh = int(30 / cell), int(12 / cell)
    ok = cand & np.isfinite(geo.gx) & np.isfinite(geo.gz) & (rng < 12)
    cxi = np.floor((geo.gx[ok] + 15) / cell).astype(int)
    czi = np.floor(geo.gz[ok] / cell).astype(int)
    inb = (cxi >= 0) & (cxi < gw) & (czi >= 0) & (czi < gh)
    cc = np.zeros((gh, gw), np.int32)
    pc = np.zeros((gh, gw), np.int32)
    np.add.at(cc, (czi[inb], cxi[inb]), 1)
    np.add.at(pc, (czi[inb], cxi[inb]), ev[ok][inb].astype(np.int32))
    bev = (cc > 0) & (pc / np.maximum(cc, 1) >= 0.45)
    bev = ndimage.binary_closing(bev, structure=np.ones((3, 3)))
    lb, nn = ndimage.label(bev)
    ar = ndimage.sum_labels(np.ones_like(lb), lb, np.arange(1, nn + 1)) * cell * cell
    bev = np.isin(lb, np.flatnonzero(ar >= 0.15) + 1)
    hh_ = ndimage.binary_fill_holes(bev) & ~bev
    lh, nh = ndimage.label(hh_)
    hs = ndimage.sum_labels(np.ones_like(lh), lh, np.arange(1, nh + 1)) * cell * cell
    bev |= np.isin(lh, np.flatnonzero(hs <= 0.25) + 1)
    cxa = np.floor((geo.gx + 15) / cell).astype(np.int64)
    cza = np.floor(geo.gz / cell).astype(np.int64)
    valid = (
        cand
        & np.isfinite(geo.gx)
        & np.isfinite(geo.gz)
        & (cxa >= 0)
        & (cxa < gw)
        & (cza >= 0)
        & (cza < gh)
    )
    floor_raw = np.zeros(cand.shape, bool)
    vi = np.where(valid)
    floor_raw[vi] = bev[cza[vi], cxa[vi]]
    floor_raw[cand & ~valid] = ev[cand & ~valid]
    print(
        f"  [{camera}] floor voted (before the per-plate boundary): "
        f"{100 * floor_raw.mean():.1f}% of frame"
    )
    # Pixels only the strict source claims are returned separately: they are the ones the
    # per-clip object layer is allowed to veto (see the enclosed-hole rule in main).
    return floor_raw, floor_raw & ~ev_teach


def main():
    argv = sys.argv[1:]
    # --only=stem[,stem]: pool every plate that is passed, but RENDER only these clips.
    # This is what lets one camera be split across workers without changing the recipe:
    # the pooled floor still sees all of that camera's plates in every worker.
    only = set()
    preview_stride = 1  # 1 = an overlay for every frame; N = one in N (QA sample)
    resume = False
    rest = []
    for a in argv:
        if a.startswith("--only="):
            only |= {s for s in a[len("--only=") :].split(",") if s}
        elif a.startswith("--preview-stride="):
            preview_stride = max(1, int(a.split("=", 1)[1]))
        elif a == "--resume":
            resume = True
        else:
            rest.append(a)
    out_dir = Path(rest[0])
    per_clip = int(rest[1])
    targets = rest[2:]
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "masks").mkdir(exist_ok=True)
    (out_dir / "preview").mkdir(exist_ok=True)
    (out_dir / "images").mkdir(exist_ok=True)
    status = (out_dir / "clips.jsonl").open("a")

    third = M.ThirdOpinion(M.THIRD_OPINION_RUN, device)
    proc, model = M.load_sam3(SAM3.MODEL_ID, device)
    # frame_masks lost its args-namespace in the 00:05 refactor that moved the SAM 3
    # teacher into the wheel: device and upscale are keyword-only now, and max_box_frac
    # is the module's own constant.
    static_concepts, moving = M.build_concepts_v2()
    geo_cache, report = {}, []

    parsed = []
    for target in targets:
        parts = target.split(":")
        camera, stem = parts[0], parts[1]
        n_here = int(parts[2]) if len(parts) > 2 else per_clip
        clip = ROOT / CLIP_ROOT / camera / f"{stem}.mp4"
        slot = stem.split("_")[1]
        plate_path = ROOT / PLATE_ROOT / camera / f"plate_{slot}.png"
        if not clip.exists():
            print(f"!! {target}: missing clip, skipped")
            continue
        if not ensure_plate(camera, slot, clip, plate_path):
            continue
        parsed.append((camera, stem, slot, n_here, clip, plate_path))

    # v5.3 pre-pass: the plate teacher runs once per plate and the floor once per CAMERA,
    # over every plate that camera contributes to this batch. v5.2 ran both per clip.
    plate_imgs, plate_cache, floor_cam, strict_only = {}, {}, {}, {}
    slic_cam = {}
    static_cam, decisions_cam = {}, {}
    for camera in dict.fromkeys(p[0] for p in parsed):
        print(f"== {camera} plates")
        if camera not in geo_cache:
            geo_cache[camera] = CameraGeometry(camera, third)
        for cam_i, _stem, slot_i, _n, _clip, plate_path in parsed:
            if cam_i != camera or (camera, slot_i) in plate_cache:
                continue
            img = Image.open(plate_path).convert("RGB").resize((W, H), Image.Resampling.LANCZOS)
            plate_imgs[(camera, slot_i)] = img
            plate_cache[(camera, slot_i)] = third.terrain_ids(img)
        slots = list(dict.fromkeys(p[2] for p in parsed if p[0] == camera))
        geo = geo_cache[camera]
        # v6.1: the superpixels are a spatial partition of a camera's view, not a
        # property of one clip, so they are computed ONCE on that camera's cleanest
        # plate (least person-smear) instead of once per clip. 5.1 s per clip in the
        # measured budget, 7.2% of the whole run.
        cleanest = min(slots, key=lambda s: float((plate_cache[(camera, s)] == 5).mean()))
        slic_cam[camera] = np.asarray(
            Image.fromarray(
                slic_numpy(
                    np.asarray(plate_imgs[(camera, cleanest)].resize((960, 540))), 30
                ).astype(np.int32),
                "I",
            ).resize((W, H), Image.Resampling.NEAREST)
        )
        print(
            f"  [{camera}] superpixels from the cleanest plate ({cleanest}, "
            f"{100 * float((plate_cache[(camera, cleanest)] == 5).mean()):.1f}% dirty)"
        )
        geo.ensure_level_frame()

        floor_cam[camera], strict_only[camera] = camera_floor(camera, geo, slots, plate_cache)

        # v5.8: the immovable objects are decided ONCE PER CAMERA too, over the SAM 3
        # instances of every plate. Same reason as the floor, and the same evidence
        # rule -- a busy slot's plate is smeared where the crowd stands, and a class
        # that changes between tranches is a class the student cannot learn.
        smasks, smeta = [], []
        for slot_i in slots:
            pimg = plate_imgs[(camera, slot_i)]
            embeds = SAM3.vision_features(proc, model, pimg, device)
            for concept in static_concepts:
                for prompt in concept.prompts:
                    for m_, s_ in SAM3.segment(
                        proc, model, pimg, prompt, concept.min_score, device, embeds
                    ):
                        if m_.sum() < 500:
                            continue
                        smeta.append(
                            {
                                "concept": concept.name,
                                "prompt": prompt,
                                "score": float(s_),
                                "px": int(m_.sum()),
                                "slot": slot_i,
                            }
                        )
                        smasks.append(m_)
        cl_masks, cl_votes = cluster(smasks, smeta)
        static_cam[camera], decisions_cam[camera] = decide_structure(
            cl_masks, cl_votes, [plate_cache[(camera, s)] for s in slots], geo, geo.lx, geo.lz
        )
        acc = sum(1 for d in decisions_cam[camera] if d["win"] and not d["reject"])
        print(
            f"  [{camera}] structure over {len(slots)} plates: {len(smasks)} instances "
            f"-> {len(cl_masks)} objects, {acc} accepted"
        )

    render = [p for p in parsed if not only or p[1] in only]
    if only:
        print(
            f"rendering {len(render)} of {len(parsed)} clips passed "
            f"(the rest are here for the pooled floor only)"
        )
    for camera, stem, slot, n_here, clip, plate_path in render:
        if resume and all(
            (out_dir / "masks" / f"{camera}__{slot}__{i:04d}.png").exists()
            for i in range(n_here)
        ):
            print(f"== {camera} {slot}: {n_here} frames already written, skipped")
            continue
        print(f"== {camera} {slot}")
        t_clip = time.time()
        geo = geo_cache[camera]
        plate_img = plate_imgs[(camera, slot)]
        plate_rgb = np.asarray(plate_img, dtype=np.int16)
        static, decisions = static_cam[camera], decisions_cam[camera]
        slic = slic_cam[camera]

        # ---- the floor: the camera's, with this clip's plate for the boundary -----
        # The vote happened once per camera (camera_floor). What stays per clip is only
        # the BOUNDARY: the guided filter snaps the camera's floor to the edges of THIS
        # plate, so a fixture that moved between slots still cuts the floor where it
        # actually stands. Each frame then subtracts what stands ON the floor -- people,
        # products, fixtures -- exactly as in v5.2.
        floor_clip = (
            guided(
                floor_cam[camera].astype(np.float32),
                plate_rgb.astype(np.float32) / 255.0,
                GUIDE_RADIUS,
                GUIDE_EPS,
            )
            >= 0.5
        )
        floor_clip = ndimage.binary_fill_holes(floor_clip)
        # The object layer vetoes the strict source INSIDE ITSELF. A pixel that only the
        # strict-geometry source claims and that lies in a hole fully enclosed by one
        # accepted object is part of that object, not ground the object surrounds: the
        # measured case is the tray of dark accessories on Taichung-cam01's counter, which
        # the depth reads at floor height (3,112 px, 100% inside the counter's enclosed
        # hole, against 0.0% for the aisle patch the source exists for).
        holes = np.zeros((H, W), bool)
        lbo, no = ndimage.label(static != IGNORE)
        for k in range(1, no + 1):
            comp = lbo == k
            if comp.sum() < 2000:
                continue
            holes |= ndimage.binary_fill_holes(comp) & ~comp
        # Measured over the whole batch: floor inside an enclosed object hole is
        # 9,336 px in 30 frames, ALL of it the Taichung-cam01 counter tray. So the
        # veto applies to floor from any source, not just the strict one -- the
        # narrower version missed the tray, which reaches the floor through the
        # teachers' own evidence.
        veto = floor_clip & holes
        floor_clip &= ~veto
        print(
            f"  floor (per camera, boundary on this plate): "
            f"{100 * floor_clip.mean():.1f}% of frame"
            f"   (object-hole veto removed {int(veto.sum())} px)"
        )

        kept, day = M.load_clip(str(clip), 1.0)
        pool = [i for i, d in enumerate(day) if d] or list(range(len(kept)))
        # v6.1: farthest-first was hard-capped at 10, so `--frames 25` silently gave 10.
        # The cap is now the request itself, which is what makes the per-clip fixed cost
        # (32 s: decode, superpixels, noise floor) amortise over more frames.
        picks = [
            pool[i] for i in farthest_first([describe(kept[i]) for i in pool], max(n_here, 10))
        ]
        sub = np.linspace(0, len(kept) - 1, min(48, len(kept))).astype(int)
        devs = np.stack(
            [
                np.abs(kept[i].astype(np.int16) - plate_rgb).mean(axis=2).astype(np.uint8)
                for i in sub
            ]
        )
        thr_map = np.maximum(np.percentile(devs, 10, axis=0), 1.0) * 8.0
        del devs

        for n, idx in enumerate(picks[:n_here]):
            frame = kept[idx]
            img = Image.fromarray(frame)
            name = f"{camera}__{slot}__{n:04d}"
            b03f = third.terrain_ids(img)
            static_px = np.abs(frame.astype(np.int16) - plate_rgb).mean(axis=2) <= thr_map
            _, mv = SAM3.frame_masks(proc, model, frame, moving, device=device)
            person = (mv != M.IGNORE) & (b03f == 5)

            floor = floor_clip & ~person

            fembeds = SAM3.vision_features(proc, model, img, device)
            pmeta, pmasks = [], []
            for cls, prompts in PRODUCT_PROMPTS.items():
                for prompt in prompts:
                    for m, s in SAM3.segment(proc, model, img, prompt, 0.25, device, fembeds):
                        if m.sum() < 80 or s < PRODUCT_MIN[cls]:
                            continue
                        pmeta.append(
                            {
                                "cls": cls,
                                "prompt": prompt,
                                "score": float(s),
                                "px": int(m.sum()),
                            }
                        )
                        pmasks.append(m)
            nmeta, nmasks = [], []
            for prompt in NEGATIVE:
                for m, s in SAM3.segment(proc, model, img, prompt, 0.25, device, fembeds):
                    if m.sum() >= 80:
                        nmeta.append({"prompt": prompt, "score": float(s)})
                        nmasks.append(m)

            keep = []
            for i in sorted(range(len(pmeta)), key=lambda i: -pmeta[i]["score"]):
                m = pmasks[i]
                for kk in keep:
                    inter = (m & pmasks[kk]).sum()
                    if (
                        inter / max((m | pmasks[kk]).sum(), 1) >= PRODUCT_IOU
                        or inter / m.sum() >= 0.8
                    ):
                        break
                else:
                    keep.append(i)
            structure_floor = np.where(floor, IDS["floor"], IGNORE).astype(np.uint8)
            structure_floor[static != IGNORE] = static[static != IGNORE]
            products = np.full((H, W), IGNORE, np.uint8)
            refused = []
            for i in keep:
                m, r = pmasks[i], pmeta[i]
                neg, neg_name = 0.0, None
                exempt = NEG_EXEMPT.get(r["cls"], ())
                for j, nr in enumerate(nmeta):
                    if nr["prompt"] in exempt:
                        continue
                    if (m & nmasks[j]).sum() / m.sum() >= NEG_COVER and nr["score"] > neg:
                        neg, neg_name = nr["score"], nr["prompt"]
                below = (
                    ndimage.binary_dilation(
                        m, structure=np.array([[0, 0, 0], [0, 1, 0], [0, 1, 0]]), iterations=12
                    )
                    & ~m
                )
                fs = (
                    float((structure_floor[below] == IDS["floor"]).mean())
                    if below.sum() > 20
                    else 0.0
                )
                if neg > r["score"]:
                    refused.append((r, f"`{neg_name}` {neg:.2f} beats {r['score']:.2f}"))
                elif r["score"] - neg < NEG_TIE:
                    refused.append((r, f"contested with `{neg_name}` {neg:.2f}"))
                elif fs > FLOOR_SUPPORT_MAX:
                    refused.append((r, f"stands on the floor ({fs:.0%})"))
                else:
                    sel = m & (products == IGNORE) & ~person
                    products[sel] = IDS[r["cls"]]

            mask = np.full((H, W), IGNORE, np.uint8)
            mask[floor] = IDS["floor"]
            mask[static != IGNORE] = static[static != IGNORE]
            # v5.6: a change with no COLOUR in it is a shadow or a reflection, not a
            # change. Measured on Taichung-cam04's middle cabinet, whose front face lost
            # SLIC-shaped blocks: those pixels move 15 levels of luma at the median and
            # 2 of chroma (94% under dC 4), while the person's own pixels move 53 and 7
            # (only 39% under dC 4). Gating on dC > 4 keeps 100% of the cabinet's blocks
            # and still withdraws the person's cells, and it is the same principle the
            # floor already uses -- a shadow on the floor is floor.
            fy = np.asarray(img.convert("YCbCr"), dtype=np.int16)
            py = np.asarray(plate_img.convert("YCbCr"), dtype=np.int16)
            dchroma = np.maximum(
                np.abs(fy[..., 1] - py[..., 1]), np.abs(fy[..., 2] - py[..., 2])
            )
            changed = ~static_px & ~person & (dchroma > CHROMA_CHANGE)
            seg_ch = np.bincount(slic[changed], minlength=int(slic.max()) + 1) / np.maximum(
                np.bincount(slic.ravel(), minlength=int(slic.max()) + 1), 1
            )
            withdraw = np.isin(mask, [2, 3, 4, 5]) & (seg_ch[slic] >= CHANGE_WITHDRAW)
            mask[withdraw] = IGNORE
            mask[products != IGNORE] = products[products != IGNORE]
            mask[person] = IDS["person"]
            shade = ndimage.binary_fill_holes(np.isin(mask, [IDS["floor"], IDS["person"]])) & (
                mask == IGNORE
            )
            mask[shade] = IDS["floor"]
            for cid in (1, 2, 3, 4, 5):
                cls = mask == cid
                if not cls.any():
                    continue
                mask[ndimage.binary_fill_holes(cls) & (mask == IGNORE)] = cid
                lb2, n2 = ndimage.label(mask == cid)
                if n2:
                    sz = ndimage.sum_labels(np.ones_like(lb2), lb2, np.arange(1, n2 + 1))
                    mask[np.isin(lb2, np.flatnonzero(sz < ISLAND_PX) + 1)] = IGNORE

            # v5.8: close the hairline seams between two finished classes. Measured on
            # the v5.7 batch, 15.6% of all unlabelled pixels (1.67% of every frame) sit
            # within 2 px of a label -- the line between the counter and the floor that
            # neither claim reaches, which is what the user sees as a chipped edge. Each
            # such pixel takes the class of its nearest labelled neighbour. Anything
            # wider stays IGNORE: a real gap is not a seam.
            gap = mask == IGNORE
            if gap.any():
                dist, idx = ndimage.distance_transform_edt(gap, return_indices=True)
                thin = gap & (dist <= 2)
                mask[thin] = mask[idx[0][thin], idx[1][thin]]

            Image.fromarray(mask).save(out_dir / "masks" / f"{name}.png")
            img.save(out_dir / "images" / f"{name}.jpg", quality=92)
            base = frame.astype(np.float32)
            over = base.copy()
            for cid, col in COLORS.items():
                sel = mask == cid
                over[sel] = 0.45 * base[sel] + 0.55 * np.array(col)
            for cid in (7, 8, 9, 10):
                sel = mask == cid
                if sel.any():
                    edge = ndimage.binary_dilation(sel, iterations=3) & ~sel
                    over[edge] = np.array(COLORS[cid])
            vis = Image.fromarray(over.astype(np.uint8))
            dd = ImageDraw.Draw(vis)
            shares = {k: round(100 * float((mask == v).mean()), 2) for k, v in IDS.items()}
            lab_pct = round(100 * float((mask != IGNORE).mean()), 1)
            dd.rectangle([4, 4, 470, 206], fill=(0, 0, 0))
            dd.text((12, 10), f"{name}  labelled {lab_pct}%", fill=(255, 255, 255))
            y = 28
            for k, v in IDS.items():
                dd.rectangle([12, y + 2, 26, y + 12], fill=COLORS[v])
                dd.text((32, y), f"{k}: {shares[k]}%", fill=(255, 255, 255))
                y += 16
            if n % preview_stride == 0:
                vis.save(out_dir / "preview" / f"{name}.jpg", quality=92)
            report.append(
                {
                    "name": name,
                    "camera": camera,
                    "slot": slot,
                    "labelled_pct": lab_pct,
                    "shares": shares,
                    "products_kept": len(keep) - len(refused),
                    "products_refused": [
                        (r["cls"], round(r["score"], 2), why) for r, why in refused
                    ],
                    "structure": [d for d in decisions if d["win"]],
                }
            )
            print(f"  {name}: labelled {lab_pct}%  {shares}")
        status.write(
            json.dumps(
                {
                    "camera": camera,
                    "slot": slot,
                    "stem": stem,
                    "frames": len(picks[:n_here]),
                    "seconds": round(time.time() - t_clip, 1),
                    "plate": str(plate_path),
                }
            )
            + "\n"
        )
        status.flush()
        del kept

    tag = f"{render[0][0]}_{render[0][2]}" if render else "empty"
    report_path = out_dir / f"report_{tag}.json"
    # A resumed unit skips the clips it already wrote, so `report` holds only the frames
    # THIS process rendered -- empty when every clip was skipped. Writing that straight
    # out replaces the original run's per-frame record with `[]`, which is how all 261
    # reports were lost on 2026-08-20. Merge on `name`: a re-rendered frame updates its
    # own entry and every frame nobody re-made survives.
    merged = {}
    if report_path.exists():
        try:
            merged = {r["name"]: r for r in json.loads(report_path.read_text())}
        except (ValueError, KeyError, TypeError):
            merged = {}
    merged.update({r["name"]: r for r in report})
    report_path.write_text(json.dumps(list(merged.values()), indent=1))
    print(
        f"\n{len(report)} frames written to {out_dir}; {report_path.name} holds {len(merged)}"
    )


if __name__ == "__main__":
    main()
