#!/usr/bin/env python3
"""NTU's 3D actions, seen from our cameras. The projection IS the domain adaptation.

    uv run python tools/temporal/ntu_project.py --out runs/ntu_project01

PLAN §2.3 and step 8: `sit`, `crouch` and `fall` come from a temporal model over keypoint
sequences, trained on public 3D action data **projected through the camera parameters we
measured**, because in-store staging is ruled out. The model consumes keypoints and not
pixels, so re-imposing our viewpoint on a 3D skeleton is the whole of the adaptation --
there is no appearance gap left to cross.

This does that projection, and then asks the question that has to be answered **before**
a training pipeline is built around it: **do `analytics/pose_sequence.py`'s features
separate the classes at all, once our viewpoint is imposed?** That module has been in the
tree, tested, with zero consumers, so nothing has ever measured what it can tell apart.
Building a trainer first and discovering the features were the problem is the expensive
order.

---------------------------------------------------------------------------
WHAT THE PROJECTION HAS TO GET RIGHT, AND HOW EACH IS CHECKED

**Gravity, not the sensor.** A Kinect's +y is up only if that Kinect was level, and NTU's
setups are not; the vertical is estimated per clip from the subject's own standing pose,
which sits a median 13.8 deg off the camera axis (§7.14). Taking sensor y would tilt every
projected shopper by that camera's mounting.

**Yaw is sampled, not fixed.** A shopper faces any direction, and a model trained on one
heading learns that heading. Each clip is placed at several yaws, which is augmentation
that costs nothing because the data is 3D.

**Position is sampled inside the view.** Foreshortening is not uniform across the frame,
and PLAN's own `column` failure is what happens when a model meets a geometry it never
saw. Placements that put a joint outside the frame are dropped rather than clipped: a
clipped skeleton is a different posture, not the same one at the edge.

**The pinhole is the shipped one.** `geometry.ground.ground_to_pixel` projects floor
points; a skeleton needs points *above* the floor, so `_project` generalises it by one
term -- and an unconditional self-check on startup asserts the generalisation
reproduces `ground_to_pixel` exactly at zero height, rather than assuming a one-line
change is safe. (There is no `--self-check` flag; the check always runs.)
"""

from __future__ import annotations

import argparse
import json
import sys
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from syncai_hydranet.analytics.pose_sequence import sequence_features  # noqa: E402
from syncai_hydranet.data.ntu_skeletons import JOINTS, members, read_clip  # noqa: E402
from syncai_hydranet.geometry.ground import Camera, GroundPlane, ground_to_pixel  # noqa: E402

# The camera `hm3d_cctv` renders at and PLAN §2.3 names, so a sequence projected here and
# a frame rendered there describe the same room.
CAM_HEIGHT_M = 2.38
CAM_PITCH_DEG = 50.2
CAM_VFOV_DEG = 70.4
FRAME_W, FRAME_H = 1920, 1080

# COCO-17 <- NTU-25. The five face keypoints have no NTU counterpart and are emitted at
# zero confidence; none of `pose_sequence.LIMBS` uses them, so the features are complete.
COCO_FROM_NTU = {
    0: "head",
    5: "l_shoulder",
    6: "r_shoulder",
    7: "l_elbow",
    8: "r_elbow",
    9: "l_wrist",
    10: "r_wrist",
    11: "l_hip",
    12: "r_hip",
    13: "l_knee",
    14: "r_knee",
    15: "l_ankle",
    16: "r_ankle",
}
CLASSES = {43: "fall", 6: "pick_up", 8: "sit_down", 1: "stand_still", 42: "stagger"}
# NTU records at 30 fps and this fleet's analytics runs at 5 (PLAN §7.4). `sequence_
# features` takes a **per-frame** difference for its velocity term, so the frame rate is
# inside the features: a 2.4 s fall is 72 frames at 30 and 12 at 5, and a model trained
# on one and run on the other sees a sixfold different dynamics. Resampled at the source
# rather than corrected later, because the velocity is computed from whatever arrives.
NTU_FPS = 30.0
FEET = ("l_foot", "r_foot", "l_ankle", "r_ankle")


def camera() -> tuple[Camera, GroundPlane]:
    fy = (FRAME_H / 2) / np.tan(np.radians(CAM_VFOV_DEG) / 2)
    cam = Camera(fx=fy, fy=fy, cx=FRAME_W / 2, cy=FRAME_H / 2)
    return cam, GroundPlane(height=CAM_HEIGHT_M, pitch=np.radians(CAM_PITCH_DEG))


def _project(pts_level: np.ndarray, cam: Camera, plane: GroundPlane) -> np.ndarray:
    """Level-frame points (x lateral, height above floor, z forward) -> pixels.

    `ground_to_pixel` is this with the height fixed at zero; the only change is that the
    camera-frame y becomes `plane.height - height` instead of `plane.height`. Asserted
    against it by `self_check`, which runs unconditionally on startup.
    """
    p = np.stack(
        [pts_level[..., 0], plane.height - pts_level[..., 1], pts_level[..., 2]], axis=-1
    )
    cp = p @ plane.rotation.T
    with np.errstate(divide="ignore", invalid="ignore"):
        u = cam.fx * (cp[..., 0] / cp[..., 2]) + cam.cx
        v = cam.fy * (cp[..., 1] / cp[..., 2]) + cam.cy
    return np.stack([u, v, cp[..., 2]], axis=-1)


def gravity_frame(body: np.ndarray, opening: int = 10) -> np.ndarray:
    """A rotation taking the Kinect frame to one whose +y is the subject's own up."""
    n = min(opening, len(body))
    head = np.nanmedian(body[:n, JOINTS["head"]], axis=0)
    foot = np.nanmedian(np.stack([body[:n, JOINTS[f]] for f in FEET]).reshape(-1, 3), axis=0)
    up = head - foot
    up = up / max(float(np.linalg.norm(up)), 1e-6)
    tmp = np.array([1.0, 0.0, 0.0])
    if abs(up @ tmp) > 0.9:
        tmp = np.array([0.0, 0.0, 1.0])
    right = np.cross(tmp, up)
    right /= max(float(np.linalg.norm(right)), 1e-6)
    fwd = np.cross(up, right)
    return np.stack([right, up, fwd])  # rows: the new basis, so R @ v is v in it


def to_coco(body_level: np.ndarray, cam: Camera, plane: GroundPlane) -> np.ndarray | None:
    """(T, 25, 3) placed in the level frame -> (T, 17, 3) COCO pixels with confidence."""
    out = np.zeros((len(body_level), 17, 3), dtype=np.float32)
    for coco_i, name in COCO_FROM_NTU.items():
        uvz = _project(body_level[:, JOINTS[name]], cam, plane)
        out[:, coco_i, :2] = uvz[:, :2]
        out[:, coco_i, 2] = 1.0
        behind = ~np.isfinite(uvz[:, 2]) | (uvz[:, 2] <= 0.1)
        out[behind, coco_i, 2] = 0.0
    live = out[:, list(COCO_FROM_NTU), :]
    if (live[..., 2] == 0).any():
        return None
    u, v = live[..., 0], live[..., 1]
    # A clipped skeleton is a different posture, not the same one at the edge.
    if u.min() < 0 or v.min() < 0 or u.max() >= FRAME_W or v.max() >= FRAME_H:
        return None
    return out


def place(body: np.ndarray, rot: np.ndarray, yaw: float, x_m: float, z_m: float) -> np.ndarray:
    """Gravity-align, spin to `yaw`, and stand the subject at (x_m, z_m) on our floor."""
    g = body @ rot.T  # (T, 25, 3) in a frame whose +y is up
    c, s = np.cos(yaw), np.sin(yaw)
    spin = np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])
    g = g @ spin.T
    feet = np.stack([g[:, JOINTS[f]] for f in FEET]).reshape(-1, 3)
    floor_y = float(np.nanmedian(feet[:, 1]))
    centre = np.nanmedian(g[:, JOINTS["spine_base"]], axis=0)
    g = g - np.array([centre[0], floor_y, centre[2]])
    return g + np.array([x_m, 0.0, z_m])


def self_check(cam: Camera, plane: GroundPlane) -> None:
    rng = np.random.default_rng(0)
    x = rng.uniform(-4, 4, 500)
    z = rng.uniform(1.0, 10.0, 500)
    u0, v0, _ = ground_to_pixel(x, z, cam, plane)  # type: ignore[misc]
    ours = _project(np.stack([x, np.zeros_like(x), z], axis=-1), cam, plane)
    assert np.allclose(ours[:, 0], u0, atol=1e-9), "generalised projection disagrees in u"
    assert np.allclose(ours[:, 1], v0, atol=1e-9), "generalised projection disagrees in v"
    print("self-check: the generalised projection matches ground_to_pixel at zero height")


def _logreg(x: np.ndarray, y: np.ndarray, iters: int = 60, l2: float = 1e-2) -> np.ndarray:
    xb = np.c_[x, np.ones(len(x))]
    w = np.zeros(xb.shape[1])
    sw = np.where(y == 1, 0.5 / max(y.sum(), 1), 0.5 / max((1 - y).sum(), 1)) * len(y)
    for _ in range(iters):
        p = 1 / (1 + np.exp(-xb @ w))
        g = xb.T @ (sw * (p - y)) / len(y) + l2 * w
        h = xb.T @ ((sw * p * (1 - p))[:, None] * xb) / len(y) + l2 * np.eye(len(w))
        w -= np.linalg.solve(h + 1e-6 * np.eye(len(w)), g)
    return w


def separability(rows: list[dict]) -> dict[str, float]:
    """Can a linear model on these features tell the classes apart, held out by performer?

    The question step 8 has to answer before a trainer is written. **By performer, not by
    clip**: NTU repeats each action per subject and a clip-level split puts the same body
    on both sides, which would report a number the deployment can never reproduce. The
    placements of one clip all move together for the same reason.

    Time-pooled mean and max under a linear model, deliberately: this is a floor. A
    temporal model is what step 8 builds, and a floor that already clears the geometric
    rule is what says the features carry the information rather than the model inventing
    it.
    """
    x = np.array([r["x"] for r in rows], dtype=np.float64)
    x = np.nan_to_num((x - x.mean(0)) / (x.std(0) + 1e-6))
    lab = np.array([r["label"] for r in rows])
    who = np.array([r["performer"] for r in rows])
    out: dict[str, float] = {}
    print(f"\n{'pair':30s} {'balanced acc':>13s} {'n':>6s} {'performers':>11s}")
    pairs = [("fall", "pick_up"), ("fall", "sit_down"), ("fall", "stand_still"),
             ("fall", "stagger"), ("pick_up", "sit_down")]  # fmt: skip
    for pos, neg in pairs:
        m = (lab == pos) | (lab == neg)
        if not m.any():
            continue
        xs, ys, gs = x[m], (lab[m] == pos).astype(float), who[m]
        yt, yp = [], []
        for person in np.unique(gs):
            te = gs == person
            if te.all() or ys[~te].std() == 0:
                continue
            w = _logreg(xs[~te], ys[~te])
            yp.append(1 / (1 + np.exp(-np.c_[xs[te], np.ones(te.sum())] @ w)))
            yt.append(ys[te])
        if not yt:
            continue
        yt_a, yp_a = np.concatenate(yt), np.concatenate(yp)
        pred = (yp_a >= 0.5).astype(int)
        rec = ((pred == 1) & (yt_a == 1)).sum() / max((yt_a == 1).sum(), 1)
        spec = ((pred == 0) & (yt_a == 0)).sum() / max((yt_a == 0).sum(), 1)
        acc = 0.5 * (rec + spec)
        out[f"{pos} vs {neg}"] = float(acc)
        print(f"{pos + ' vs ' + neg:30s} {acc:13.3f} {len(yt_a):6d} {len(np.unique(gs)):11d}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--zip", nargs="+",
        default=["datasets/_incoming/ntu_rose/nturgbd_skeletons_s001_to_s017.zip"],
    )  # fmt: skip
    ap.add_argument("--per-class", type=int, default=80)
    ap.add_argument("--placements", type=int, default=4, help="yaw/position samples per clip")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--fps", type=float, default=5.0,
        help="resample NTU to the rate the analytics runs at; the velocity feature is a "
        "per-frame difference, so this is not cosmetic",
    )  # fmt: skip
    ap.add_argument("--sequences", help="npz of the per-frame features, for the trainer")
    ap.add_argument("--out")
    a = ap.parse_args()

    cam, plane = camera()
    self_check(cam, plane)
    rng = np.random.default_rng(a.seed)

    kept: list[dict] = []
    for path in a.zip:
        zf = zipfile.ZipFile(path)
        for action, label in CLASSES.items():
            have = sum(1 for r in kept if r["label"] == label)
            for member in members(zf, actions={action})[: max(0, a.per_class - have)]:
                clip = read_clip(zf, member)
                body = clip.joints[:, 0]
                if np.isnan(body).any():
                    continue
                step = max(1, round(NTU_FPS / a.fps))
                body = body[::step]
                if len(body) < 4:
                    continue
                rot = gravity_frame(body)
                for _ in range(a.placements):
                    placed = place(
                        body, rot,
                        yaw=float(rng.uniform(0, 2 * np.pi)),
                        x_m=float(rng.uniform(-2.0, 2.0)),
                        z_m=float(rng.uniform(2.5, 6.0)),
                    )  # fmt: skip
                    coco = to_coco(placed, cam, plane)
                    if coco is None:
                        continue
                    feats = sequence_features(coco)
                    kept.append(
                        {
                            "label": label,
                            "performer": clip.performer,
                            "member": member.rsplit("/", 1)[-1],
                            # mean and max over time: the model will be temporal, this is
                            # only asking whether the information is present at all
                            "x": np.concatenate([feats.mean(0), feats.max(0)]).tolist(),
                            "seq": feats.astype(np.float32),
                        }
                    )

    separability(kept)

    counts: dict[str, int] = {}
    for r in kept:
        counts[r["label"]] = counts.get(r["label"], 0) + 1
    print(f"projected sequences: {len(kept)}  {counts}")
    if a.out:
        out = Path(a.out)
        out.mkdir(parents=True, exist_ok=True)
        (out / "projected.json").write_text(
            json.dumps([{k: v for k, v in r.items() if k != "seq"} for r in kept]) + "\n"
        )
        print(f"-> {out}/projected.json")
    if a.sequences and kept:
        t_max = max(len(r["seq"]) for r in kept)
        x = np.zeros((len(kept), t_max, kept[0]["seq"].shape[1]), dtype=np.float32)
        length = np.zeros(len(kept), dtype=np.int32)
        for i, r in enumerate(kept):
            x[i, : len(r["seq"])] = r["seq"]
            length[i] = len(r["seq"])
        np.savez_compressed(
            a.sequences,
            x=x,
            length=length,
            label=np.array([r["label"] for r in kept]),
            performer=np.array([r["performer"] for r in kept]),
            fps=np.array([a.fps]),
        )
        print(f"-> {a.sequences}  {x.shape}, lengths {length.min()}-{length.max()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
