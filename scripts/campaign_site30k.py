#!/usr/bin/env python3
"""Site30k campaign orchestrator: teacher-annotate site frames under the campaign taxonomy.

    # phase 1 pilot (2 cameras, clips already on disk)
    nice -n 10 .venv/bin/python scripts/campaign_site30k.py annotate \\
        --out datasets/site30k --frames 25 \\
        datasets/studioa_clips/Kaohsiung-cam04/*.mp4 datasets/studioa_clips/Taichung-cam01/*.mp4

    # pet census: score distribution BEFORE choosing a threshold (rule 2)
    nice -n 10 .venv/bin/python scripts/campaign_site30k.py pet-census \\
        --out runs/site30k_qa/pet_census.json --sample 500 datasets/studioa_clips/*/*.mp4

    # QA counts over the finished dataset
    .venv/bin/python scripts/campaign_site30k.py qa --root datasets/site30k

TAXONOMY (ruled by the planner session)
---------------------------------------
Segmentation, 7 classes: void/floor/wall/column/display_table/shelf/person.
`fixture` is SPLIT: the prompt families below partition
`sam3_prompts_objects.BY_NAME["fixture"].prompts` into a table family and a shelving
family, asserted against the source list so a prompt added there cannot silently fall
through here. Three prompts -- `office furniture`, `trolley`, `trash can` -- belong to
neither family and are omitted: their pixels stay unclaimed (255) rather than being
guessed into one. `display_table` and `shelf` sit on the SAME layer, so where the two
families both fire the pixel becomes IGNORE via `compose`'s same-layer rule -- the
"on-table vs on-shelf" judgement stays a judgement, exactly the `table`/`display table`
precedent.

Detection, 2 classes: person (GDINO @0.35, the measured day/night gap) and product
(SAM 3 at the frame's native resolution -- the 352x240 lesson; both SAM 3 merchandise
families are merged into one `product` category, the source family kept per box).

`pet` was REMOVED from the campaign scope on 2026-08-19 (the user's decision, recorded in
`git show b7457c2:docs/journal/2026-08-19-small-batch-breaks-three-gates.md`
section 9). The census this
script wrote -- 2477 boxes over 336 frames, runs/site30k_qa/pet_census.json -- tops out at
0.5427 with p99 = 0.2018, so no threshold was ever defensible and the class would ship
with zero positives and never fire. The `pet-census` subcommand and the id-2 category are
kept so the census can be re-run if a site ever has pets, but nothing labels them and the
taxonomy does not carry the class.

TEACHERS AND MERGE
------------------
* SAM 3 static classes are consensus-voted across the clip (0.9): the camera does not
  move, so disagreement across frames is SAM 3 guessing. Machinery imported from
  syncai_bev3d.teachers (sam3 / gdino / boxes / photometry), not copied.
* `person` pixels are composited per frame after the vote (people move).
* A third opinion from runs/hydranet_retail_security_b03_gdino/best.pt vetoes
  floor/wall/column/person: where the model's terrain head disagrees with the voted
  mask, the pixel becomes 255 -- never a guess (the compose() precedent).
* Night-IR frames (pixel-tested, never filename-tested: luma/chroma per
  sam3_person_boxes.is_daylight) get person boxes only and an all-255 mask, so the
  seg_folder pairing stays complete while the frame contributes nothing to the seg loss.

Everything written is a teacher's opinion, not ground truth; instances_*.json says so.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
from PIL import Image

HERE = Path(__file__).resolve().parent
for candidate in (HERE.parent / "src", HERE / "src"):
    if candidate.is_dir():
        sys.path.insert(0, str(candidate))
sys.path.insert(0, str(HERE))

# The teachers live in the wheel as of 2026-08-20 (52eb141 SAM 3, 556582b Grounding
# DINO, 95d816a the clip-list refusals): scripts/sam3_prelabel.py and
# scripts/gdino_person_boxes.py are CLI front ends over these modules now and re-export
# only part of the surface, so this imports the package directly rather than through them.
from syncai_bev3d.teachers.boxes import drop_static  # noqa: E402
from syncai_bev3d.teachers.boxes import nms as gdino_nms  # noqa: E402
from syncai_bev3d.teachers.gdino import (  # noqa: E402
    PERSON_THRESHOLD,
    load_gdino,
)
from syncai_bev3d.teachers.gdino import detect as gdino_detect  # noqa: E402
from syncai_bev3d.teachers.photometry import is_daylight, luma_chroma  # noqa: E402
from syncai_bev3d.teachers.sam3 import (  # noqa: E402
    MAX_BOX_FRAC,
    consensus,
    frame_boxes,
    frame_masks,
    load_sam3,
)
from syncai_bev3d.teachers.sam3 import MODEL_ID as SAM3_MODEL_ID  # noqa: E402
from syncai_hydranet.data import sam3_prompts_objects as OBJ  # noqa: E402
from syncai_hydranet.data.frame_selection import describe, farthest_first  # noqa: E402
from syncai_hydranet.data.sam3_prompts import Concept  # noqa: E402
from syncai_hydranet.data.video import frames as decode_frames  # noqa: E402
from syncai_hydranet.data.video import (  # noqa: E402
    probe,
    session_names,
    validate_inputs,
)
from syncai_hydranet.labels import IGNORE  # noqa: E402
from syncai_hydranet.utils.device import pick_device  # noqa: E402

# ---------------------------------------------------------------------------------
# The campaign taxonomy. New ids, deliberately NOT the retail_objects ids: id 4 there
# is `fixture` and id 5 is `product`, and reusing the numbers for different meanings
# is how a mask gets read under the wrong table without an error.
SITE30K = {
    "void": 0,
    "floor": 1,
    "wall": 2,
    "column": 3,
    "display_table": 4,
    "shelf": 5,
    "person": 6,
}

# The fixture split. Families partition the source prompt list; asserted below.
# Table family: flat horizontal display surfaces and counters.
TABLE_FAMILY = (
    "display table",
    "display podium",
    "retail counter",
    "display counter",
    "checkout counter",
    "table",
)
# Shelving family: vertical storage and racking, including glazed cabinets/cases --
# enclosed vertical furniture is racking to a shopper, not a table.
SHELF_FAMILY = (
    "shelving unit",
    "shelf",
    "merchandise rack",
    "product display rack",
    "product display stand",
    "showcase cabinet",
    "display case",
)
# Neither family. Left unclaimed (-> 255) rather than guessed into one; recorded here
# so a later taxonomy has a place to start (the trolley note in sam3_prompts_objects).
OMITTED_FIXTURE_PROMPTS = ("office furniture", "trolley", "trash can")

DETECTION_CATEGORIES = (
    {"id": 1, "name": "person"},
    {"id": 2, "name": "pet"},
    {"id": 3, "name": "product"},
)

# The teacher's own threshold, imported rather than restated -- tools/site30k/boxes.py
# read this name while its neighbour box_pass.py read gdino's, so one directory held two
# routes to one number. Measured day/night gap: night IR tops out 0.326, day >= 0.35.
PERSON_TRAIN_THR = PERSON_THRESHOLD
FLOOR_TOL_M = 0.20  # on-plane tolerance; see the measured gap note in GeomTeacher
SCORE_FLOOR = 0.10  # kept in instances_all so both populations stay visible
CONSENSUS = 0.9  # the measured setting from sam3_prelabel
NMS_IOU = 0.55  # same as teachers.boxes.nms

THIRD_OPINION_RUN = HERE.parent / "runs/hydranet_retail_security_b03_gdino"
# b03 terrain head: 0 void, 1 floor, 2 wall, 3 column, 4 fixture, 5 person.
# Veto map: site id -> the b03 id that counts as agreement. `display_table` and
# `shelf` are absent on purpose -- the model cannot arbitrate a split it never saw.
VETO_SITE_TO_B03 = {1: 1, 2: 2, 3: 3, 6: 5}

PROGRESS_LOG = HERE.parent / "runs/site30k_qa/progress.log"


def log_progress(msg: str) -> None:
    stamp = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    PROGRESS_LOG.parent.mkdir(parents=True, exist_ok=True)
    with PROGRESS_LOG.open("a") as fh:
        fh.write(f"{stamp} {msg}\n")
    print(f"[progress] {msg}", flush=True)


def build_concepts() -> list[Concept]:
    """The campaign concept table, derived from the object table rather than retyped.

    The assertion is the contract: every fixture prompt is either a table, a shelf, or
    deliberately omitted, and a prompt added to the source list breaks the run here
    instead of silently annotating under the old split.
    """
    fixture = set(OBJ.BY_NAME["fixture"].prompts)
    claimed = set(TABLE_FAMILY) | set(SHELF_FAMILY) | set(OMITTED_FIXTURE_PROMPTS)
    assert claimed == fixture, (
        f"fixture prompt split out of date: unclaimed={fixture - claimed}, "
        f"phantom={claimed - fixture}"
    )
    assert not set(TABLE_FAMILY) & set(SHELF_FAMILY), "a prompt is in both families"
    col = OBJ.BY_NAME["column"]
    return [
        Concept("floor", OBJ.BY_NAME["floor"].prompts, OBJ.LAYER_GROUND, taxonomy=SITE30K),
        Concept("wall", OBJ.BY_NAME["wall"].prompts, OBJ.LAYER_GROUND, taxonomy=SITE30K),
        Concept(
            "column",
            col.prompts,
            OBJ.LAYER_THING,
            min_score=col.min_score,
            taxonomy=SITE30K,
            note="min_score inherited from the 48-camera sweep",
        ),
        # Same layer on purpose: overlap between the families -> IGNORE, not a guess.
        Concept("display_table", TABLE_FAMILY, OBJ.LAYER_THING, taxonomy=SITE30K),
        Concept("shelf", SHELF_FAMILY, OBJ.LAYER_THING, taxonomy=SITE30K),
        Concept("person", ("person",), OBJ.LAYER_PERSON, taxonomy=SITE30K),
    ]


# ---------------------------------------------------------------------------------
# Third opinion


class ThirdOpinion:
    """b03's terrain head, as a per-pixel veto for floor/wall/column/person."""

    def __init__(self, run_dir: Path, device: str):
        from syncai_hydranet.config import load_config
        from syncai_hydranet.models.hydranet import build_model
        from syncai_hydranet.utils.checkpoint import load_checkpoint, select_weights

        self.cfg = load_config(str(run_dir / "config.yaml"), [])
        self.device = device
        self.model = build_model(self.cfg).to(device).eval()
        ckpt = load_checkpoint(str(run_dir / "best.pt"))
        self.model.load_state_dict(select_weights(ckpt, "ema"))
        self.size = self.cfg["data"]["input_size"]
        self.letterbox = bool(self.cfg["data"].get("letterbox", False))

    @torch.no_grad()
    def terrain_ids(self, img: Image.Image) -> np.ndarray:
        from syncai_hydranet.utils.visualize import crop_box, preprocess

        x, _canvas, region = preprocess(img, self.size, self.letterbox)
        result = self.model.predict(x.to(self.device))
        terr = crop_box(result["terrain"][0].cpu().numpy(), region)
        out = Image.fromarray(terr.astype(np.uint8)).resize(img.size, Image.Resampling.NEAREST)
        return np.asarray(out)

    def veto(
        self, mask: np.ndarray, img: Image.Image, ids: dict[int, int] | None = None
    ) -> tuple[np.ndarray, dict]:
        """Where the model disagrees on a vetoed class, the pixel becomes IGNORE.

        ``ids`` narrows the veto set; the v2 recipe passes {person: person} only --
        floor and wall are resolved statically against the plate, and column is
        exempt because b03 is the documented site `column` failure.
        """
        b03 = self.terrain_ids(img)
        out = mask.copy()
        stats = {}
        for site_id, b03_id in (ids or VETO_SITE_TO_B03).items():
            claim = mask == site_id
            if not claim.any():
                continue
            dis = claim & (b03 != b03_id)
            out[dis] = IGNORE
            stats[site_id] = {"claimed": int(claim.sum()), "vetoed": int(dis.sum())}
        return out, stats


# ---------------------------------------------------------------------------------
# v2 recipe: SAM3 supplies shape, geometry supplies the class.
#
# The user's inspection ruling on the v1 pilot: person and product stand; floor and
# display_table are wrong too often. Root cause: SAM3's floor/furniture-type prompts
# are weak (the old prompt table ships floor off-by-default for a measured reason).
# v2 therefore:
#   floor          = b03(plate) floor channel  INTERSECT  geometric backprojection of
#                    runs/zones01/<cam> walkable_floor through runs/onboard01/<cam>
#                    calib (division-model undistort + ground plane). Either source
#                    alone -> IGNORE. SAM3 does not vote on floor.
#   display_table/ = SAM3 generic fixture mask (all 16 fixture-family prompts as ONE
#   shelf            concept) classified by DA-V2 height-above-ground on the plate:
#                    horizontal top surface in the table band -> display_table,
#                    above the shelf floor OR vertical-face-dominant -> shelf,
#                    between -> IGNORE. Band edges are swept on the pilot cameras
#                    first (rule 2) and passed in explicitly.
#   wall           = SAM3, kept only where b03(plate) agrees (doubt -> IGNORE).
#   column         = SAM3 + consensus, NO b03 veto (b03 is the documented site
#                    `column` failure and cannot be its examiner).
#   person px      = subtracted from the static vote input per frame (a still person
#                    otherwise enters the vote), then composited per frame as before,
#                    still b03-vetoed per frame.

FIXTURE_TMP = 7  # internal id for the generic fixture mask, classified before writing
SITE30K_V2 = {**SITE30K, "fixture": FIXTURE_TMP}


def build_concepts_v2() -> tuple[list[Concept], list[Concept]]:
    """(static concepts, moving concepts) for the v2 recipe. No floor concept."""
    col = OBJ.BY_NAME["column"]
    static = [
        Concept("wall", OBJ.BY_NAME["wall"].prompts, OBJ.LAYER_GROUND, taxonomy=SITE30K_V2),
        Concept(
            "column", col.prompts, OBJ.LAYER_THING, min_score=col.min_score, taxonomy=SITE30K_V2
        ),
        Concept(
            "fixture",
            OBJ.BY_NAME["fixture"].prompts,
            OBJ.LAYER_THING,
            taxonomy=SITE30K_V2,
            note="generic fixture; geometry names table/shelf",
        ),
    ]
    moving = [Concept("person", ("person",), OBJ.LAYER_PERSON, taxonomy=SITE30K_V2)]
    return static, moving


class GeomTeacher:
    """Per-camera geometry: floor backprojection and height-above-ground maps.

    Everything is derived from artifacts other sessions measured -- calib
    (`runs/onboard01` by default), zones (runs/zones01), the static plate -- through the
    package's own geometry code (undistort_points, pixel_to_ground, unproject). Heights
    are in metres via the calib's person-height scale; its stated systematic band is
    6-11%, which is why the class bands must come from a sweep, not from a guess.

    **`calib_root` is an argument because a second reader of it already existed and this
    one ignored it.** `tools/site30k/recipe.py`'s `CameraGeometry.CALIB_ROOT` is
    rebindable exactly so a caller can point at another sweep -- `masks_pass.py` exposes
    it as `--calib-root` -- but the rebinding stopped here, at a hardcoded
    `runs/onboard01`. The flag therefore worked for cameras that were in `onboard01`
    anyway and failed for the ones it existed for: `runs/onboard02`'s ten RTSP channels,
    and `dingpu-1f/test1`, which is where it surfaced on 2026-09-01.
    """

    def __init__(
        self,
        camera: str,
        third: ThirdOpinion,
        frame_hw=(1080, 1920),
        calib_root: Path | None = None,
    ):
        import math as _math

        from matplotlib.path import Path as MplPath

        from syncai_bev3d.plate_calibration import (
            run_depth,
            undistort_image,
        )
        from syncai_hydranet.geometry.ground import (
            Camera as GCamera,
        )
        from syncai_hydranet.geometry.ground import (
            GroundPlane,
            pixel_to_ground,
            undistort_points,
            unproject,
        )

        root = Path(calib_root) if calib_root is not None else HERE.parent / "runs/onboard01"
        calib = json.loads((root / f"{camera}.calib.json").read_text())
        zones = json.loads((HERE.parent / f"runs/zones01/{camera}.zones.json").read_text())
        if zones.get("units") != "m" or calib.get("scale") is None:
            raise SystemExit(f"{camera}: zones/calib not metric; v2 floor recipe needs both")
        poly = next(p for p in zones["proposals"] if p["name_suggestion"] == "walkable_floor")
        self.camera = camera
        self.third = third
        h, w = frame_hw
        vfov = float(calib["vfov_assumed_deg"])
        k1 = float(calib.get("k1_division_model") or 0.0)
        plane = GroundPlane(
            height=float(calib["height_m"]),
            pitch=_math.radians(calib["pitch_deg"]),
            roll=_math.radians(calib["roll_deg"]),
        )
        cam_full = GCamera.from_vfov(h, w, vfov)

        # Undistorted coordinate of every (distorted) frame pixel, frame scale.
        vv, uu = np.mgrid[0:h, 0:w].astype(np.float64)
        xy = np.stack([uu.ravel(), vv.ravel()], axis=1)
        und = undistort_points(xy, k1, (w / 2.0, h / 2.0), _math.hypot(h, w) / 2.0)
        uu_u = und[:, 0].reshape(h, w)
        vv_u = und[:, 1].reshape(h, w)

        # Geometric floor: ray -> ground point -> inside the walkable_floor polygon.
        gx, gz = pixel_to_ground(uu_u, vv_u, cam_full, plane)
        pts = np.stack([gx.ravel(), gz.ravel()], axis=1)
        finite = np.isfinite(pts).all(axis=1)
        inside = np.zeros(h * w, dtype=bool)
        inside[finite] = MplPath(np.asarray(poly["polygon_m"])).contains_points(pts[finite])
        geom_floor = inside.reshape(h, w)

        # b03 floor channel on the plate (static, person-light).
        plate = (
            Image.open(calib["plate_used"])
            .convert("RGB")
            .resize((w, h), Image.Resampling.LANCZOS)
        )
        b03_ids = third.terrain_ids(plate)
        self.b03_plate = b03_ids  # 0 void 1 floor 2 wall 3 column 4 fixture 5 person
        self._geom_floor = geom_floor  # kept for floor-diag attribution
        self.floor_mask = geom_floor & (b03_ids == 1)
        self.floor_sources = {
            "geom_only": int((geom_floor & (b03_ids != 1)).sum()),
            "b03_only": int(((~geom_floor) & (b03_ids == 1)).sum()),
            "intersection": int(self.floor_mask.sum()),
        }

        # Height-above-ground and surface verticality from DA-V2 on the undistorted
        # plate, in metres via the calib scale, sampled back into distorted frame space.
        plate_rgb = np.asarray(Image.open(calib["plate_used"]).convert("RGB"))
        ph, pw = plate_rgb.shape[:2]
        depth = run_depth(undistort_image(plate_rgb, k1)) * float(calib["scale"])
        cam_plate = GCamera.from_vfov(ph, pw, vfov)
        cam_pts = unproject(depth, cam_plate)
        level = cam_pts @ plane.rotation  # camera frame -> level frame (y down)
        height = plane.height - level[..., 1]
        d_dx = np.gradient(level, axis=1)
        d_dy = np.gradient(level, axis=0)
        n = np.cross(d_dx.reshape(-1, 3), d_dy.reshape(-1, 3)).reshape(ph, pw, 3)
        norm = np.linalg.norm(n, axis=-1)
        with np.errstate(invalid="ignore", divide="ignore"):
            horiz = np.abs(n[..., 1]) / np.where(norm > 1e-12, norm, np.nan)
        # sample at the undistorted coordinate of each frame pixel, plate scale
        su, sv = uu_u * (pw / w), vv_u * (ph / h)
        sui = np.clip(np.round(su).astype(int), 0, pw - 1)
        svi = np.clip(np.round(sv).astype(int), 0, ph - 1)
        self.height = height[svi, sui].astype(np.float32)
        self.horiz = horiz[svi, sui].astype(np.float32)  # 1 = horizontal top, 0 = vertical face

        # v3 floor core: b03 floor channel AND the on-plane test. The zones polygon
        # was measured (floor-diag, 2026-08-19) losing whole floor regions -- its
        # outer-contour trace drops BEV components split off by a counter (T-cam01's
        # right aisle, 7.1% of frame) and K-cam04's polygon reaches less than half of
        # b03's floor. |height-above-plane| is the independent metric witness: over
        # b03-floor pixels it sits at p95 0.066/0.143/0.204 m on the three pilot
        # cameras against a non-floor median of 0.74-0.78 m -- an order-of-magnitude
        # gap. 0.20 m is mid-gap and covers the worst-calibrated camera's tail.
        fin = np.isfinite(self.height)
        self.onplane_floor = (b03_ids == 1) & fin & (np.abs(self.height) <= FLOOR_TOL_M)
        self.floor_sources["onplane_core"] = int(self.onplane_floor.sum())

    def plate_labeling(self, plate_path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """v4 plate-first labeling: (plate_floor, dirty, plate_rgb_1080).

        On a person-free daytime plate the two floor sources are each high-precision,
        so the PLATE label is their UNION -- b03's floor channel OR the pure on-plane
        test -- minus the dirty region: wherever the model reads *person* on a plate,
        someone stood through most of the clip and the median is smeared
        (stable_infer's rule: a dirty plate is never trusted). Kaohsiung-cam04's
        19:27L plate measures 8.6% dirty, and the v3 pilot's person-shaped floor
        ghost came exactly from such a region.
        """
        from scipy import ndimage

        img = (
            Image.open(plate_path).convert("RGB").resize((1920, 1080), Image.Resampling.LANCZOS)
        )
        ids = self.third.terrain_ids(img)
        dirty = ndimage.binary_dilation(ids == 5, iterations=3)  # person + halo
        fin = np.isfinite(self.height)
        onplane = fin & (np.abs(self.height) <= FLOOR_TOL_M)
        plate_floor = ((ids == 1) | onplane) & ~dirty
        return plate_floor, dirty, np.asarray(img, dtype=np.int16)

    def classify_fixture(self, fixture: np.ndarray, args) -> tuple[np.ndarray, dict]:
        """Generic fixture pixels -> display_table / shelf / IGNORE by height band."""
        out = np.full(fixture.shape, IGNORE, dtype=np.uint8)
        h, hz = self.height, self.horiz
        horizontal = hz >= args.horiz_thr
        vertical = hz <= args.vert_thr
        table = fixture & horizontal & (h >= args.table_lo) & (h <= args.table_hi)
        # A vertical face counts as shelf only ABOVE the table band: the sweep shows
        # vertical fixture pixels are continuous from 0 to 2.1 m (table sides and
        # cabinet fronts share the low band with shelf faces), so an unconditioned
        # vertical->shelf rule would paint every table's side `shelf`. Above
        # vert_min_h no table top exists (measured gap), so verticality is decisive.
        shelf = fixture & ((h >= args.shelf_min) | (vertical & (h >= args.vert_min_h)))
        both = table & shelf  # a vertical/horizontal call cannot be both; height can
        out[table] = SITE30K["display_table"]
        out[shelf] = SITE30K["shelf"]
        out[both] = IGNORE
        stats = {
            "fixture_px": int(fixture.sum()),
            "table_px": int((out == 4).sum()),
            "shelf_px": int((out == 5).sum()),
            "ignored_px": int((fixture & (out == IGNORE)).sum()),
        }
        return out, stats


# ---------------------------------------------------------------------------------
# Clip handling


# The daylight verdict is a statistic over the whole frame, so it is read off every
# eighth pixel in each direction rather than all 2.07 M of them. This is the single
# largest cost in the campaign and it was hiding behind the decoder: on a 304-frame clip
# ffmpeg takes 1.0 s, while `is_daylight` at full resolution takes 23.7 s (47.5 s on an
# IR night clip, whose frames are less compressible for the same reason they are grey).
# Measured at stride 8: 0.35 s and 0.71 s, with the SAME verdict on every frame of both
# clips -- luma moves 134.64 -> 133.76 against a threshold of 40, and night chroma is
# 0.000 either way. The gate has three orders of magnitude of headroom; the sampling has
# none of the cost.
DAYLIGHT_STRIDE = 8


def load_clip(clip: str, sample_fps: float = 1.0):
    """Decoded frames at sample rate, with per-frame daylight verdicts."""
    w, h, _ = probe(clip)
    kept, day = [], []
    for frame in decode_frames(clip, w, h, sample_fps):
        kept.append(frame.copy())
        day.append(is_daylight(frame[::DAYLIGHT_STRIDE, ::DAYLIGHT_STRIDE], 40.0, 1.0))
    return kept, day


def coco_image(
    images: list, name: str, img: Image.Image, lu: float, ch: float, night: bool, camera: str
) -> int:
    image_id = len(images) + 1
    images.append(
        {
            "id": image_id,
            "file_name": name,
            "width": img.width,
            "height": img.height,
            "camera": camera,
            "luma": round(lu, 2),
            "chroma": round(ch, 2),
            "night": bool(night),
        }
    )
    return image_id


def add_boxes(store: list, boxes, image_id: int, cat_id: int, extra: dict | None = None):
    for b in boxes:
        store.append(
            {
                "id": len(store) + 1,
                "image_id": image_id,
                "category_id": cat_id,
                "bbox": [float(b[0]), float(b[1]), float(b[2] - b[0]), float(b[3] - b[1])],
                "area": float((b[2] - b[0]) * (b[3] - b[1])),
                "score": float(b[4]),
                "iscrowd": 0,
                **(extra or {}),
            }
        )


# ---------------------------------------------------------------------------------
# annotate


def cmd_annotate(args) -> int:
    validate_inputs(args.clips)
    # session name = <camera>__<clipstem>: the camera prefix is the collision lesson
    # from data.video.session_names, applied up front rather than re-learned.
    sessions = session_names(
        [str(Path(c).parent.name) + "__" + Path(c).stem for c in args.clips]
    )
    split_of = (
        json.loads(Path(args.split_json).read_text())["assign"] if args.split_json else {}
    )

    device = str(pick_device())
    if args.recipe in ("v2", "v3"):
        static, moving = build_concepts_v2()  # v3 uses only the person concept
    else:
        concepts = build_concepts()
        static = [c for c in concepts if c.name != "person"]
        moving = [c for c in concepts if c.name == "person"]
    sam3_proc, sam3_model = load_sam3(SAM3_MODEL_ID, device)
    gd_proc, gd_model = load_gdino(args.gdino_model, device)
    third = ThirdOpinion(THIRD_OPINION_RUN, device) if not args.no_third_opinion else None
    if args.recipe in ("v2", "v3") and third is None:
        raise SystemExit(f"recipe {args.recipe} needs the b03 third opinion")
    geom_cache: dict[str, GeomTeacher] = {}

    # frame_masks/frame_boxes read these off an args namespace; native resolution on
    # purpose (the 352x240 lesson is docs-recorded).

    root = Path(args.out)
    per_split: dict[str, dict] = defaultdict(
        lambda: {"images": [], "annotations": [], "all_annotations": []}
    )
    census = {"person": [], "pet": [], "product": []}
    manifest = {
        "teachers": {
            "seg": f"SAM 3 ({SAM3_MODEL_ID}) consensus@{CONSENSUS} "
            f"+ b03 third-opinion veto on floor/wall/column/person",
            "person": f"Grounding DINO ({args.gdino_model}) @{PERSON_TRAIN_THR}",
            "pet": f"Grounding DINO '{args.pet_prompt}' -- census first, threshold "
            f"{args.pet_thr if args.pet_thr is not None else 'NOT CHOSEN YET'}",
            "product": "SAM 3 DETECTION_CLASSES at native resolution, merged to `product`",
        },
        "taxonomy": SITE30K,
        "fixture_split": {
            "display_table": list(TABLE_FAMILY),
            "shelf": list(SHELF_FAMILY),
            "omitted": list(OMITTED_FIXTURE_PROMPTS),
        },
        "consensus": CONSENSUS,
        "clips": [],
    }
    pixel_totals: dict[str, Counter] = defaultdict(Counter)
    veto_totals: dict[int, Counter] = defaultdict(Counter)

    for clip, session in zip(args.clips, sessions, strict=True):
        t0 = time.time()
        camera = Path(clip).parent.name
        split = split_of.get(camera, args.default_split)
        kept, day = load_clip(clip, args.sample_fps)
        if not kept:
            log_progress(f"{session}: no frames decoded, skipped")
            manifest["clips"].append({"session": session, "skipped": "no frames"})
            continue
        night_clip = sum(day) < len(day) / 2
        pool = list(range(len(kept))) if night_clip else [i for i, d in enumerate(day) if d]
        if not pool:
            pool = list(range(len(kept)))
        descs = [describe(kept[i]) for i in pool]
        picks = [pool[i] for i in farthest_first(descs, args.frames)]

        img_dir = root / "images" / split / session
        ann_dir = root / "annotations" / split / session
        img_dir.mkdir(parents=True, exist_ok=True)
        ann_dir.mkdir(parents=True, exist_ok=True)
        bucket = per_split[split]

        clip_entry = {
            "session": session,
            "source": clip,
            "camera": camera,
            "split": split,
            "night": night_clip,
            "frames": len(picks),
            "picks": picks,
            "recipe": args.recipe,
        }

        shared = None
        static_share = None
        mv_list: list[np.ndarray] | None = None
        v4_ctx = None
        if not night_clip and args.recipe == "v4":
            # v4 plate-first (ruled after v3 review): label the person-free plate as
            # fully as the evidence allows (union of sources minus dirty region),
            # then propagate per frame: a pixel that has not moved off the plate --
            # judged against ITS OWN noise floor, static_plates' statistic -- inherits
            # the plate label; moved pixels are person (SAM3 contour) or IGNORE.
            # GDINO boxes no longer carve the mask at all.
            if camera not in geom_cache:
                geom_cache[camera] = GeomTeacher(camera, third)
            geom = geom_cache[camera]
            slot = Path(clip).stem.split("_")[1]
            cand = HERE.parent / f"datasets/studioa_static/{camera}/plate_{slot}.png"
            if cand.exists():
                plate_path = str(cand)  # the SAME clip's own median -- crispest diff
            else:
                calib_p = json.loads(
                    (HERE.parent / f"runs/onboard01/{camera}.calib.json").read_text()
                )
                plate_path = calib_p["plate_used"]
            plate_floor, dirty, plate_rgb = geom.plate_labeling(plate_path)
            # Per-pixel noise floor of |frame - plate|, static_plates style: a LOW
            # percentile of the pixel's own deviations over the clip (p10), never a
            # frame-wide constant, times the measured multiplier. The resampling
            # residual of the 960x540 plate is inside the floor by construction.
            sub = np.linspace(0, len(kept) - 1, min(48, len(kept))).astype(int)
            devs = np.stack(
                [
                    np.abs(kept[i].astype(np.int16) - plate_rgb).mean(axis=2).astype(np.uint8)
                    for i in sub
                ]
            )
            noise = np.maximum(np.percentile(devs, 10, axis=0), 1.0)
            del devs
            thr_map = noise * args.prop_mult
            mv_list = [
                frame_masks(sam3_proc, sam3_model, kept[i], moving, device=device)[1]
                for i in picks
            ]
            v4_ctx = {
                "plate_floor": plate_floor,
                "plate_rgb": plate_rgb,
                "thr_map": thr_map,
                "coverage": [],
            }
            clip_entry["plate"] = plate_path
            clip_entry["plate_dirty_pct"] = round(100 * float(dirty.mean()), 2)
            clip_entry["plate_floor_pct"] = round(100 * float(plate_floor.mean()), 2)
        elif not night_clip and args.recipe == "v3":
            # v3 (scope ruled 2026-08-19): floor + person ONLY, everything else
            # IGNORE. Floor is the two-strong-source core (b03 plate floor channel
            # AND the DA-V2 on-plane test); no SAM3 static pass at all.
            if camera not in geom_cache:
                geom_cache[camera] = GeomTeacher(camera, third)
            geom = geom_cache[camera]
            shared = np.full(kept[0].shape[:2], IGNORE, np.uint8)
            shared[geom.onplane_floor] = SITE30K["floor"]
            mv_list = [
                frame_masks(sam3_proc, sam3_model, kept[i], moving, device=device)[1]
                for i in picks
            ]
            clip_entry["floor_sources"] = geom.floor_sources
            clip_entry["floor_core_pct"] = round(100 * float(geom.onplane_floor.mean()), 2)
        elif not night_clip:
            per_frame = [
                frame_masks(sam3_proc, sam3_model, kept[i], static, device=device)
                for i in picks
            ]
            if args.recipe == "v2":
                # Person pixels are subtracted from the static vote INPUT: a person who
                # stands still for the whole clip otherwise enters the vote and takes a
                # static class with her (measured on the v1 pilot: a customer painted
                # `shelf`). Computed once and reused for the per-frame composite below.
                mv_list = [
                    frame_masks(sam3_proc, sam3_model, kept[i], moving, device=device)[1]
                    for i in picks
                ]
                for (_, sm), pm in zip(per_frame, mv_list, strict=True):
                    sm[pm != IGNORE] = IGNORE
            shared, static_share = consensus([m for _, m in per_frame], CONSENSUS)
            clip_entry["consensus_static_share"] = round(static_share, 4)
            if args.recipe == "v2":
                if camera not in geom_cache:
                    geom_cache[camera] = GeomTeacher(camera, third)
                geom = geom_cache[camera]
                # wall kept only where b03(plate) agrees; doubt -> IGNORE
                wall_bad = (shared == SITE30K["wall"]) & (geom.b03_plate != 2)
                shared[wall_bad] = IGNORE
                # generic fixture -> table/shelf/IGNORE by measured height bands
                fix = shared == FIXTURE_TMP
                cls, fstats = geom.classify_fixture(fix, args)
                shared[fix] = cls[fix]
                # floor: intersection of the two strong sources, on otherwise-unclaimed
                # pixels only; single-source floor stays IGNORE by construction
                shared[(shared == IGNORE) & geom.floor_mask] = SITE30K["floor"]
                clip_entry["fixture_classify"] = fstats
                clip_entry["wall_veto_px"] = int(wall_bad.sum())
                clip_entry["floor_sources"] = geom.floor_sources

        night_person: list[tuple[int, np.ndarray]] = []
        for n, idx in enumerate(picks):
            frame = kept[idx]
            img = Image.fromarray(frame)
            pboxes_pre = None  # v3 computes GDINO before the mask; others after
            stem = f"{n:04d}"
            lu, ch = luma_chroma(frame)
            img.save(img_dir / f"{stem}.jpg", quality=92)

            if night_clip:
                # person-only tranche: an all-IGNORE mask keeps the pairing complete
                # and contributes nothing to the seg loss.
                mask = np.full((img.height, img.width), IGNORE, np.uint8)
            else:
                if v4_ctx is not None:
                    dev_t = np.abs(frame.astype(np.int16) - v4_ctx["plate_rgb"]).mean(axis=2)
                    static_px = dev_t <= v4_ctx["thr_map"]
                    mask = np.full(dev_t.shape, IGNORE, np.uint8)
                    mask[static_px & v4_ctx["plate_floor"]] = SITE30K["floor"]
                else:
                    mask = shared.copy()
                if args.recipe == "v3":
                    # A person standing in this frame occupies pixels the static core
                    # calls floor. Where a GDINO person box overlaps the floor core,
                    # the floor claim is withdrawn (IGNORE); SAM3's person pixels then
                    # paint over whatever they cover. Floor survives only outside any
                    # person evidence -- precision first.
                    pboxes_pre = gdino_detect(
                        gd_proc, gd_model, img, "person", SCORE_FLOOR, device
                    )
                    strong = gdino_nms(
                        pboxes_pre[pboxes_pre[:, 4] >= PERSON_TRAIN_THR], NMS_IOU
                    )
                    bm = np.zeros(mask.shape, dtype=bool)
                    for b in strong:
                        x0, y0, x1, y1 = (
                            max(int(b[0]), 0),
                            max(int(b[1]), 0),
                            min(int(b[2]) + 1, mask.shape[1]),
                            min(int(b[3]) + 1, mask.shape[0]),
                        )
                        bm[y0:y1, x0:x1] = True
                    mask[bm & (mask == SITE30K["floor"])] = IGNORE
                if mv_list is not None:
                    mv = mv_list[n]
                else:
                    _, mv = frame_masks(sam3_proc, sam3_model, frame, moving, device=device)
                mask[mv != IGNORE] = mv[mv != IGNORE]
                if third is not None:
                    # v2/v3: per-frame veto narrows to person -- floor is resolved
                    # statically against the plate, column is exempt (b03 is the
                    # documented site `column` failure, not its examiner)
                    ids = {SITE30K["person"]: 5} if args.recipe in ("v2", "v3", "v4") else None
                    mask, vstats = third.veto(mask, img, ids)
                    for sid, st in vstats.items():
                        veto_totals[sid].update(st)
                if v4_ctx is not None:
                    # Acceptance metric: how much of the plate's visible floor this
                    # frame labelled, and what the holes are. Holes may only be person
                    # contours and genuinely changed pixels -- anything else is a bug.
                    pf = v4_ctx["plate_floor"]
                    p_total = int(pf.sum())
                    lab = int(((mask == SITE30K["floor"]) & pf).sum())
                    person_hole = int(((mask == SITE30K["person"]) & pf).sum())
                    changed_hole = int((pf & ~static_px & (mask != SITE30K["person"])).sum())
                    veto_hole = p_total - lab - person_hole - changed_hole
                    v4_ctx["coverage"].append(
                        {
                            "frame": stem,
                            "floor_labelled_pct_of_visible": round(
                                100 * lab / max(p_total, 1), 1
                            ),
                            "person_hole_pct": round(100 * person_hole / max(p_total, 1), 1),
                            "changed_hole_pct": round(100 * changed_hole / max(p_total, 1), 1),
                            "other_hole_pct": round(100 * veto_hole / max(p_total, 1), 1),
                        }
                    )
            Image.fromarray(mask).save(ann_dir / f"{stem}.png")
            pixel_totals[camera].update(mask.ravel().tolist())

            image_id = coco_image(
                bucket["images"], f"{session}/{stem}.jpg", img, lu, ch, night_clip, camera
            )

            # person: GDINO, floor kept in *_all, threshold+NMS into the train file.
            # Night clips: the thresholded boxes are DEFERRED and passed through a
            # within-clip recurrence gate after the loop -- measured on the night
            # smoke (Taichung-cam01 23:59, empty store): hanging packets score 0.62,
            # far above the 0.326 "measured gap" which came from a different camera.
            # A box recurring in place across a 5-minute closed-store clip is
            # furniture. This gate is night-only: by day it deletes standing staff
            # (measured, sam3_person_boxes docstring).
            pboxes = (
                pboxes_pre
                if pboxes_pre is not None
                else gdino_detect(gd_proc, gd_model, img, "person", SCORE_FLOOR, device)
            )
            census["person"].extend(float(b[4]) for b in pboxes)
            add_boxes(bucket["all_annotations"], pboxes, image_id, 1)
            kept_person = gdino_nms(pboxes[pboxes[:, 4] >= PERSON_TRAIN_THR], NMS_IOU)
            if night_clip:
                night_person.append((image_id, kept_person))
            else:
                add_boxes(bucket["annotations"], kept_person, image_id, 1)

            # pet: census always; boxes only once a threshold exists
            petboxes = gdino_detect(
                gd_proc, gd_model, img, args.pet_prompt, SCORE_FLOOR, device
            )
            census["pet"].extend(float(b[4]) for b in petboxes)
            add_boxes(bucket["all_annotations"], petboxes, image_id, 2)
            if args.pet_thr is not None:
                add_boxes(
                    bucket["annotations"],
                    gdino_nms(petboxes[petboxes[:, 4] >= args.pet_thr], NMS_IOU),
                    image_id,
                    2,
                )

            # product: SAM 3 at native resolution, day frames only (no merchandise
            # measurement exists for IR night and person-only is the tranche's label)
            if not night_clip:
                found, _dropped = frame_boxes(
                    sam3_proc,
                    sam3_model,
                    frame,
                    OBJ.DETECTION_CLASSES,
                    device=device,
                    max_box_frac=args.max_box_frac,
                )
                for b in found:
                    census["product"].append(float(b["score"]))
                    bucket["annotations"].append(
                        {
                            "id": len(bucket["annotations"]) + 1,
                            "image_id": image_id,
                            "category_id": 3,
                            "bbox": b["bbox"],
                            "area": b["area"],
                            "score": b["score"],
                            "iscrowd": 0,
                            "source_family": b["category"],
                            "prompt": b["prompt"],
                        }
                    )
                    bucket["all_annotations"].append(
                        dict(bucket["annotations"][-1], id=len(bucket["all_annotations"]) + 1)
                    )

        if v4_ctx is not None:
            clip_entry["floor_coverage"] = v4_ctx["coverage"]
            covs = [c["floor_labelled_pct_of_visible"] for c in v4_ctx["coverage"]]
            print(
                f"  visible-floor labelled per frame: min {min(covs):.1f}% "
                f"mean {sum(covs) / len(covs):.1f}% max {max(covs):.1f}%"
            )

        if night_person:
            kept_f, dropped_f = drop_static(
                [b for _, b in night_person], args.night_static_iou, args.night_static_share
            )
            n_drop = int(sum(len(b) for b in dropped_f))
            for (image_id, _), kb in zip(night_person, kept_f, strict=True):
                add_boxes(bucket["annotations"], kb, image_id, 1)
            clip_entry["night_person_dropped_static"] = n_drop
            clip_entry["night_person_kept"] = int(sum(len(b) for b in kept_f))

        manifest["clips"].append(clip_entry)
        share_str = static_share if static_share is None else round(static_share, 3)
        log_progress(
            f"{session}: split={split} night={night_clip} frames={len(picks)} "
            f"static_share={share_str} {time.time() - t0:.0f}s"
        )

    # write COCO files per split
    info = {
        "description": "site30k teacher annotations -- PRE-LABEL, NOT GROUND TRUTH",
        "person": f"GDINO @{PERSON_TRAIN_THR} (train file), floor {SCORE_FLOOR} (all file)",
        "pet": f"prompt '{args.pet_prompt}', threshold "
        f"{args.pet_thr if args.pet_thr is not None else 'not chosen: census only'}",
        "product": "SAM 3 native-resolution instances, families merged, family recorded",
    }
    for split, bucket in per_split.items():
        ann_root = root / "annotations"
        ann_root.mkdir(parents=True, exist_ok=True)
        for tag, anns in (("", bucket["annotations"]), ("all_", bucket["all_annotations"])):
            (ann_root / f"instances_{tag}{split}.json").write_text(
                json.dumps(
                    {
                        "info": info,
                        "categories": list(DETECTION_CATEGORIES),
                        "images": bucket["images"],
                        "annotations": anns,
                    }
                )
            )

    manifest["veto_totals"] = {str(k): dict(v) for k, v in veto_totals.items()}
    root.mkdir(parents=True, exist_ok=True)
    out_manifest = root / f"campaign_batch_{datetime.now():%Y%m%d-%H%M%S}.json"
    out_manifest.write_text(json.dumps(manifest, indent=2) + "\n")
    (root / "census_latest.json").write_text(
        json.dumps({k: sorted(v) for k, v in census.items()}) + "\n"
    )

    # per-camera pixel composition, printed with denominators
    id_to_name = {v: k for k, v in SITE30K.items()}
    print("\nper-camera pixel composition (share of labelled pixels)")
    for camera, tot in sorted(pixel_totals.items()):
        total = sum(tot.values())
        labelled = total - tot[IGNORE]
        parts = "  ".join(
            f"{id_to_name[i]}:{100 * tot[i] / max(labelled, 1):.1f}%"
            for i in sorted(id_to_name)
            if tot[i]
        )
        print(
            f"  {camera}: labelled {100 * labelled / max(total, 1):.1f}% of {total} px  {parts}"
        )
    for sid, st in sorted(veto_totals.items()):
        c, v = st["claimed"], st["vetoed"]
        print(f"  veto {id_to_name[sid]}: {v}/{c} px ({100 * v / max(c, 1):.1f}%) -> 255")
    print(f"\nwrote {root}; manifest {out_manifest}")
    return 0


# ---------------------------------------------------------------------------------
# floor diagnosis: why is the two-source floor core the size it is, per camera


def cmd_floor_diag(args) -> int:
    """Per-camera floor evidence audit: geometry vs b03, agreement and disagreement.

    Writes one overlay per camera on its plate -- green = both agree (the v3 core),
    blue = b03-only (geometry gap: polygon clipped/misprojected, or b03 painting floor
    where the polygon does not reach), yellow = geometry-only (b03 disagrees:
    fixtures/occlusion in b03's view, or a real misprojection landing the polygon on
    furniture). Misprojection shows as offset bands of blue and yellow along the same
    boundary; a too-small polygon shows as blue with no yellow twin.
    """
    device = str(pick_device())
    third = ThirdOpinion(THIRD_OPINION_RUN, device)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {}
    for camera in args.cameras:
        geom = GeomTeacher(camera, third)
        calib = json.loads((HERE.parent / f"runs/onboard01/{camera}.calib.json").read_text())
        plate = (
            Image.open(calib["plate_used"])
            .convert("RGB")
            .resize((1920, 1080), Image.Resampling.LANCZOS)
        )
        b03_floor = geom.b03_plate == 1
        gonly = geom._geom_floor & ~b03_floor
        bonly = b03_floor & ~geom._geom_floor
        core = geom.floor_mask
        base = np.asarray(plate).astype(np.float32)
        over = base.copy()
        for sel, rgb in (
            (core, (60, 200, 60)),
            (bonly, (60, 90, 230)),
            (gonly, (240, 210, 40)),
        ):
            over[sel] = 0.35 * base[sel] + 0.65 * np.array(rgb, np.float32)
        Image.fromarray(over.astype(np.uint8)).save(
            out_dir / f"floor_diag_{camera}.jpg", quality=90
        )
        total = core.size
        # The independent geometric witness: DA-V2 height-above-plane on the plate.
        # Quantiles of |h| over b03-floor px against b03-non-floor px tell whether an
        # on-plane test separates floor from everything else, and at what tolerance.
        h = geom.height
        fin = np.isfinite(h)
        h_floor = np.abs(h[b03_floor & fin])
        h_other = np.abs(h[(~b03_floor) & fin])
        qs = (0.5, 0.75, 0.9, 0.95)
        report[camera] = {
            "core_both_pct": round(100 * float(core.mean()), 2),
            "b03_only_pct": round(100 * float(bonly.mean()), 2),
            "geom_only_pct": round(100 * float(gonly.mean()), 2),
            "b03_floor_pct": round(100 * float(b03_floor.mean()), 2),
            "geom_floor_pct": round(100 * float(geom._geom_floor.mean()), 2),
            "abs_h_b03floor_q": {str(q): round(float(np.quantile(h_floor, q)), 3) for q in qs},
            "abs_h_other_q": {str(q): round(float(np.quantile(h_other, q)), 3) for q in qs},
            "onplane_pct_at": {
                str(t): round(100 * float((b03_floor & fin & (np.abs(h) <= t)).mean()), 2)
                for t in (0.1, 0.15, 0.2, 0.3)
            },
            "total_px": int(total),
            "calib_flags": calib.get("flags", []),
        }
        print(camera, report[camera])
    (out_dir / "floor_diag.json").write_text(json.dumps(report, indent=2) + "\n")
    return 0


# ---------------------------------------------------------------------------------
# fixture height sweep (rule 2: measure the distribution, then choose the band)


def cmd_sweep_fixture(args) -> int:
    """Height/verticality distribution over the consensus fixture mask, per camera.

    The band edges for display_table vs shelf must come out of a measured gap, so this
    runs the exact v2 static pass (SAM3 generic fixture, person-subtracted, 9/10 vote)
    on the pilot clips and prints, per camera, the height histogram of fixture pixels
    split by surface orientation. The chosen edges are then passed to `annotate
    --recipe v2` explicitly and recorded in the QA report with this table.
    """
    device = str(pick_device())
    static, moving = build_concepts_v2()
    sam3_proc, sam3_model = load_sam3(SAM3_MODEL_ID, device)
    third = ThirdOpinion(THIRD_OPINION_RUN, device)
    report = {}
    for clip in args.clips:
        camera = Path(clip).parent.name
        kept, day = load_clip(clip, 1.0)
        pool = [i for i, d in enumerate(day) if d] or list(range(len(kept)))
        descs = [describe(kept[i]) for i in pool]
        picks = [pool[i] for i in farthest_first(descs, args.frames)]
        per_frame = [
            frame_masks(sam3_proc, sam3_model, kept[i], static, device=device) for i in picks
        ]
        for i, (_, sm) in zip(picks, per_frame, strict=True):
            _, pm = frame_masks(sam3_proc, sam3_model, kept[i], moving, device=device)
            sm[pm != IGNORE] = IGNORE
        shared, share = consensus([m for _, m in per_frame], CONSENSUS)
        geom = GeomTeacher(camera, third)
        fix = shared == FIXTURE_TMP
        h, hz = geom.height[fix], geom.horiz[fix]
        finite = np.isfinite(h) & np.isfinite(hz)
        h, hz = h[finite], hz[finite]
        rows = []
        print(
            f"\n{camera}: {int(fix.sum())} consensus fixture px "
            f"(static share {share:.3f}); height x orientation histogram"
        )
        print(f"  {'height bin':>12} {'horiz>=0.8':>10} {'0.5-0.8':>9} {'vert<=0.5':>10}")
        for lo in np.arange(-0.2, 2.6, 0.1):
            hi = lo + 0.1
            sel = (h >= lo) & (h < hi)
            r = {
                "bin": f"{lo:.1f}-{hi:.1f}",
                "horizontal": int((sel & (hz >= 0.8)).sum()),
                "mid": int((sel & (hz > 0.5) & (hz < 0.8)).sum()),
                "vertical": int((sel & (hz <= 0.5)).sum()),
            }
            rows.append(r)
            if r["horizontal"] + r["mid"] + r["vertical"]:
                print(
                    f"  {r['bin']:>12} {r['horizontal']:>10} {r['mid']:>9} {r['vertical']:>10}"
                )
        q = {f"{p}": round(float(np.quantile(hz, p)), 3) for p in (0.1, 0.25, 0.5, 0.75, 0.9)}
        print(f"  horiz quantiles over fixture px: {q}")
        report[camera] = {
            "fixture_px": int(fix.sum()),
            "static_share": round(share, 4),
            "clip": clip,
            "frames": len(picks),
            "histogram": rows,
            "horiz_quantiles": q,
        }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2) + "\n")
    print(f"\nwrote {args.out}")
    return 0


# ---------------------------------------------------------------------------------
# pet census


def cmd_pet_census(args) -> int:
    device = str(pick_device())
    gd_proc, gd_model = load_gdino(args.gdino_model, device)
    clips = args.clips
    per_clip = max(1, args.sample // max(len(clips), 1))
    rows = []
    n_frames = 0
    for clip in clips:
        camera = Path(clip).parent.name
        try:
            w, h, _ = probe(clip)
        except Exception as exc:  # a truncated clip must not lose the other N-1
            print(f"{clip}: probe failed ({exc}), skipped")
            continue
        got = 0
        try:
            for i, frame in enumerate(decode_frames(clip, w, h, 0.2)):
                if got >= per_clip:
                    break
                img = Image.fromarray(frame)
                lu, ch = luma_chroma(frame)
                boxes = gdino_detect(
                    gd_proc, gd_model, img, args.pet_prompt, args.floor, device
                )
                for b in boxes:
                    rows.append(
                        {
                            "camera": camera,
                            "clip": Path(clip).stem,
                            "frame": i,
                            "score": round(float(b[4]), 4),
                            "bbox": [round(float(x), 1) for x in b[:4]],
                            "luma": round(lu, 1),
                            "chroma": round(ch, 1),
                        }
                    )
                got += 1
                n_frames += 1
        except Exception as exc:
            print(f"{clip}: decode stopped ({exc}); kept {got} frames")
        if n_frames >= args.sample:
            break
    scores = sorted(r["score"] for r in rows)
    out = {
        "prompt": args.pet_prompt,
        "floor": args.floor,
        "frames_scored": n_frames,
        "clips": len(clips),
        "boxes": len(rows),
        "scores_sorted": scores,
        "quantiles": {
            q: round(float(np.quantile(scores, float(q))), 4)
            for q in ("0.5", "0.9", "0.98", "0.99", "1.0")
        }
        if scores
        else {},
        "detections": rows,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2) + "\n")
    print(
        f"{len(rows)} '{args.pet_prompt}' boxes >= {args.floor} over {n_frames} frames "
        f"-> {args.out}"
    )
    if scores:
        print("score quantiles:", out["quantiles"])
    else:
        print("no detections at the floor at all")
    return 0


# ---------------------------------------------------------------------------------
# preview

# Colors chosen for maximal separation of the new pair: display_table is warm orange,
# shelf is magenta -- nothing else in the palette is near either.
PREVIEW_COLORS = {
    1: ((60, 180, 75), "floor"),
    2: ((0, 130, 200), "wall"),
    3: ((255, 225, 25), "column"),
    4: ((245, 130, 48), "display_table"),
    5: ((240, 50, 230), "shelf"),
    6: ((230, 25, 75), "person"),
}
BOX_COLORS = {"person": (230, 25, 75), "pet": (70, 240, 240), "product": (170, 255, 0)}


def render_preview(
    img: Image.Image, mask: np.ndarray, boxes: list[dict], title: str
) -> Image.Image:
    from PIL import ImageDraw

    # left: segmentation overlay + legend
    base = np.asarray(img).astype(np.float32)
    over = base.copy()
    counted = Counter(mask.ravel().tolist())
    for cid, (rgb, _name) in PREVIEW_COLORS.items():
        sel = mask == cid
        if sel.any():
            over[sel] = 0.42 * base[sel] + 0.58 * np.array(rgb, np.float32)
    left = Image.fromarray(over.astype(np.uint8))
    draw = ImageDraw.Draw(left)
    y = 8
    total = mask.size
    labelled = total - counted[IGNORE]
    draw.rectangle([4, 4, 330, 8 + 22 * (len(PREVIEW_COLORS) + 1)], fill=(0, 0, 0))
    draw.text((10, y), f"{title}  labelled {100 * labelled / total:.0f}%", fill=(255, 255, 255))
    y += 22
    for cid, (rgb, name) in PREVIEW_COLORS.items():
        draw.rectangle([10, y + 3, 26, y + 15], fill=rgb)
        share = 100 * counted[cid] / max(labelled, 1)
        draw.text((32, y), f"{name}  {share:.1f}% of labelled", fill=(255, 255, 255))
        y += 22

    # right: detection boxes with class + score
    right = img.copy()
    draw = ImageDraw.Draw(right)
    tally = Counter(b["category"] for b in boxes)
    for b in boxes:
        x, yb, w, h = b["bbox"]
        color = BOX_COLORS.get(b["category"], (255, 255, 255))
        draw.rectangle([x, yb, x + w, yb + h], outline=color, width=2)
        draw.text((x + 2, max(yb - 12, 0)), f"{b['category']} {b['score']:.2f}", fill=color)
    head = "  ".join(f"{k}:{v}" for k, v in sorted(tally.items())) or "no boxes"
    draw.rectangle([4, 4, 420, 24], fill=(0, 0, 0))
    draw.text((10, 8), f"boxes: {head}", fill=(255, 255, 255))

    out = Image.new("RGB", (left.width * 2 + 8, left.height), (30, 30, 30))
    out.paste(left, (0, 0))
    out.paste(right, (left.width + 8, 0))
    return out


def cmd_preview(args) -> int:
    root = Path(args.root)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for split_dir in sorted((root / "images").glob("*/")):
        split = split_dir.name
        inst = root / "annotations" / f"instances_{split}.json"
        by_image: dict[str, list[dict]] = defaultdict(list)
        if inst.exists():
            coco = json.loads(inst.read_text())
            name_of_img = {im["id"]: im["file_name"] for im in coco["images"]}
            cat_of = {c["id"]: c["name"] for c in coco["categories"]}
            for a in coco["annotations"]:
                by_image[name_of_img[a["image_id"]]].append(
                    {
                        "bbox": a["bbox"],
                        "score": a.get("score", 0.0),
                        "category": cat_of[a["category_id"]],
                    }
                )
        for jpg in sorted(split_dir.rglob("*.jpg")):
            rel = jpg.relative_to(split_dir)
            png = root / "annotations" / split / rel.with_suffix(".png")
            if not png.exists():
                continue
            img = Image.open(jpg).convert("RGB")
            mask = np.asarray(Image.open(png))
            panel = render_preview(img, mask, by_image.get(str(rel), []), f"{split}/{rel}")
            name = f"{str(rel).replace('/', '__')}".replace(
                ".jpg", f"_preview{args.suffix}.jpg"
            )
            panel.save(out_dir / name, quality=90)
            written.append(str(out_dir / name))
    print("\n".join(written))
    print(f"{len(written)} previews -> {out_dir}")
    return 0


# ---------------------------------------------------------------------------------
# compare: per-frame class shares of two labelings of the same frames


def cmd_compare(args) -> int:
    """Match frames across two dataset roots by JPEG content hash and diff the masks.

    Frame stems are NOT comparable across runs -- farthest-first over a different k
    renumbers the sorted picks -- but the saved JPEG bytes are deterministic for the
    same decoded frame, so the hash is the identity.
    """
    import hashlib

    def index(root: Path) -> dict[str, tuple[str, Path]]:
        out = {}
        for jpg in sorted(root.glob("images/*/**/*.jpg")):
            # Content identity for pairing a jpg with its png, not a security claim --
            # `usedforsecurity=False` says so to a reader and to bandit, and lets this
            # run on a FIPS build where the bare call raises. The hash that does carry
            # identity in this project is `data/fingerprint.py`'s, and it is sha256.
            digest = hashlib.md5(jpg.read_bytes(), usedforsecurity=False).hexdigest()
            rel = jpg.relative_to(root / "images")
            png = root / "annotations" / rel.with_suffix(".png")
            if png.exists():
                out[digest] = (str(rel), png)
        return out

    a, b = index(Path(args.old)), index(Path(args.new))
    common = sorted(set(a) & set(b), key=lambda k: a[k][0])
    ids = [1, 2, 3, 4, 5, 6, 255]
    hdr = ["floor", "wall", "col", "table", "shelf", "person", "ignore"]
    print(
        f"{len(common)} matched frames of {len(a)} old / {len(b)} new "
        f"(shares are % of the whole frame, old -> new)"
    )
    print(f"{'frame':55s} " + " ".join(f"{h:>14}" for h in hdr))
    rows = []
    for k in common:
        rel, png_a = a[k]
        _, png_b = b[k]
        ma = np.asarray(Image.open(png_a)).ravel()
        mb = np.asarray(Image.open(png_b)).ravel()
        row = {"frame": rel}
        cells = []
        for i, h in zip(ids, hdr, strict=True):
            pa, pb = 100 * (ma == i).mean(), 100 * (mb == i).mean()
            row[h] = [round(pa, 2), round(pb, 2)]
            cells.append(f"{pa:5.1f}->{pb:5.1f}")
        rows.append(row)
        print(f"{rel:55s} " + " ".join(f"{c:>14}" for c in cells))
    if args.out:
        Path(args.out).write_text(json.dumps(rows, indent=1) + "\n")
        print(f"wrote {args.out}")
    return 0


# ---------------------------------------------------------------------------------
# qa


def cmd_qa(args) -> int:
    root = Path(args.root)
    id_to_name = {v: k for k, v in SITE30K.items()}
    report = {"root": str(root), "splits": {}}
    for split_dir in sorted((root / "annotations").glob("*/")):
        split = split_dir.name
        per_cam: dict[str, Counter] = defaultdict(Counter)
        n_masks = 0
        for png in split_dir.rglob("*.png"):
            camera = png.parent.name.split("__")[0]
            arr = np.asarray(Image.open(png))
            per_cam[camera].update(arr.ravel().tolist())
            n_masks += 1
        boxes_by = defaultdict(Counter)
        inst = root / "annotations" / f"instances_{split}.json"
        n_images = 0
        if inst.exists():
            coco = json.loads(inst.read_text())
            n_images = len(coco["images"])
            cam_of = {
                im["id"]: im.get("camera", im["file_name"].split("__")[0])
                for im in coco["images"]
            }
            cat_of = {c["id"]: c["name"] for c in coco["categories"]}
            for a in coco["annotations"]:
                boxes_by[cam_of[a["image_id"]]][cat_of[a["category_id"]]] += 1
        cams = {}
        for camera in sorted(set(per_cam) | set(boxes_by)):
            tot = per_cam[camera]
            total = sum(tot.values())
            labelled = total - tot[IGNORE]
            cams[camera] = {
                "total_px": total,
                "labelled_px": labelled,
                "pixels": {id_to_name[i]: tot[i] for i in sorted(id_to_name) if tot[i]},
                "boxes": dict(boxes_by[camera]),
            }
        report["splits"][split] = {"masks": n_masks, "coco_images": n_images, "cameras": cams}
    out = Path(args.out) if args.out else root / "qa_counts.json"
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote {out}")
    return 0


# ---------------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    an = sub.add_parser("annotate", help="teacher-annotate clips into the site30k layout")
    an.add_argument("clips", nargs="+")
    an.add_argument("--out", required=True)
    an.add_argument("--frames", type=int, default=8, help="frames kept per clip")
    an.add_argument("--sample-fps", type=float, default=1.0)
    an.add_argument(
        "--split-json",
        default=str(HERE.parent / "datasets/retail_objects_batch03/split.json"),
        help="camera split assignments to INHERIT (R1/R2)",
    )
    an.add_argument(
        "--default-split",
        default="train",
        help="for a camera the inherited split does not name",
    )
    an.add_argument("--gdino-model", default="IDEA-Research/grounding-dino-base")
    an.add_argument("--pet-prompt", default="dog. cat.")
    an.add_argument(
        "--pet-thr",
        type=float,
        default=None,
        help="chosen from the census; None = census only, no pet boxes",
    )
    an.add_argument("--max-box-frac", type=float, default=MAX_BOX_FRAC)
    an.add_argument("--no-third-opinion", action="store_true")
    an.add_argument(
        "--recipe",
        choices=["v1", "v2", "v3", "v4"],
        default="v4",
        help="v2: SAM3 shape + geometry classes. v3: floor(intersection)+"
        "person. v4 (current ruling): plate-first -- label the person-free "
        "plate with the UNION of floor sources minus the dirty region, "
        "then propagate per frame by each pixel's own noise floor; moved "
        "pixels are person (SAM3 contour) or IGNORE",
    )
    an.add_argument(
        "--prop-mult",
        type=float,
        default=8.0,
        help="v4: a pixel has moved off the plate when |frame-plate| "
        "exceeds this multiple of its own p10 noise floor "
        "(static_plates' measured DYNAMIC_MULT)",
    )
    # v2 band edges: MUST come from a sweep-fixture gap (rule 2), passed explicitly.
    an.add_argument("--table-lo", type=float, default=0.6)
    an.add_argument("--table-hi", type=float, default=1.2)
    an.add_argument("--shelf-min", type=float, default=1.4)
    an.add_argument(
        "--horiz-thr",
        type=float,
        default=0.8,
        help="|n_up| at/above which a surface counts as a horizontal top",
    )
    an.add_argument(
        "--vert-thr",
        type=float,
        default=0.5,
        help="|n_up| at/below which a surface counts as a vertical face",
    )
    an.add_argument(
        "--night-static-iou",
        type=float,
        default=0.7,
        help="night clips only: IoU at which a recurring box is the same box",
    )
    an.add_argument(
        "--night-static-share",
        type=float,
        default=0.5,
        help="night clips only: drop a person box recurring in this share "
        "of the clip's frames. Measured: the Taichung-cam01 packet scores "
        "0.62 and recurs in 6 of 9 other frames (0.67) -- 0.8 missed it; "
        "0.5 catches it, and a night-time real person walks through in "
        "1-3 frames of a 5-minute clip",
    )
    an.add_argument(
        "--vert-min-h",
        type=float,
        default=1.0,
        help="a vertical face is shelf only at/above this height; below it "
        "table sides and shelf faces are one population (measured)",
    )
    an.set_defaults(fn=cmd_annotate)

    sw = sub.add_parser(
        "sweep-fixture", help="height/orientation distribution over consensus fixture px"
    )
    sw.add_argument("clips", nargs="+")
    sw.add_argument("--out", required=True)
    sw.add_argument("--frames", type=int, default=10)
    sw.set_defaults(fn=cmd_sweep_fixture)

    pc = sub.add_parser("pet-census", help="GDINO pet score distribution, threshold-free")
    pc.add_argument("clips", nargs="+")
    pc.add_argument("--out", required=True)
    pc.add_argument("--sample", type=int, default=500)
    pc.add_argument("--floor", type=float, default=0.05)
    pc.add_argument("--pet-prompt", default="dog. cat.")
    pc.add_argument("--gdino-model", default="IDEA-Research/grounding-dino-base")
    pc.set_defaults(fn=cmd_pet_census)

    pv = sub.add_parser("preview", help="side-by-side seg overlay + detection panels")
    pv.add_argument("--root", required=True)
    pv.add_argument("--out", required=True)
    pv.add_argument("--suffix", default="", help="appended to preview names, e.g. _v2")
    pv.set_defaults(fn=cmd_preview)

    fd = sub.add_parser("floor-diag", help="per-camera floor evidence audit + overlays")
    fd.add_argument("cameras", nargs="+")
    fd.add_argument("--out", required=True)
    fd.set_defaults(fn=cmd_floor_diag)

    cp = sub.add_parser("compare", help="per-frame class shares, old vs new labeling")
    cp.add_argument("--old", required=True)
    cp.add_argument("--new", required=True)
    cp.add_argument("--out", default=None)
    cp.set_defaults(fn=cmd_compare)

    qa = sub.add_parser("qa", help="per-class per-camera pixel and box counts")
    qa.add_argument("--root", required=True)
    qa.add_argument("--out", default=None)
    qa.set_defaults(fn=cmd_qa)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
