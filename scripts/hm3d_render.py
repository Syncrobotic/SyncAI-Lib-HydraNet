#!/usr/bin/env python3
"""Render HM3D scenes at a *measured* camera pose: RGB, metric depth, semantic ids.

    python3 scripts/hm3d_render.py --scene datasets/hm3d/00800-TEEsavR23oF \\
        --preset cctv --n 8 --out runs/hm3d/cctv01

Every public indoor dataset is shot from where a person holds a camera. Neither of this
repo's two products looks at the world from there: product B is a store CCTV on a ceiling
bracket, and product A is a quadruped with a lens 25 cm off the floor. That mismatch is
not a detail -- `scripts/eprep_teacher_nyuv2.py` measured what a domain change costs a
"metric" model, and it was a flat 15% of scale.

HM3D is 1,000 building-scale scans, so it can be rendered from *any* pose, including the
one `scripts/fit_camera_from_people.py` actually measured on Taichung-cam01:

    k1 = -0.225   vfov 70.4 deg   pitch 50.2 deg   height 2.38 m

That is the `cctv` preset. Rendering training data at the pose the deployment camera was
measured at is the one thing no downloaded dataset can offer.

---------------------------------------------------------------------------
WHY RAY CASTING AND NOT A GL RENDERER

habitat-sim is the obvious tool and it cannot be installed here: it ships only from the
`aihabitat` channel on conda.anaconda.org, which this network resets at TLS handshake,
and no mirror carries it (checked: prefix.dev, TUNA, BFSU, USTC, NJU, SJTU all 404, no
conda-forge feedstock, PyPI has a macOS-only wheel). Rather than wait on a firewall, this
casts rays with embree.

That turns out to suit the job better than a rasteriser would. One cast returns the hit
distance *and* the triangle, so depth and semantics come out of the same query with no
second pass and no z-fighting between them. It needs no GL context, which matters on a
box whose GPU is busy training.

**Depth is the z-component, not the ray length.** A rasteriser's depth buffer holds
distance along the optical axis, and every published depth metric assumes that. Ray
casting naturally gives Euclidean distance to the hit, which at the corner of a 70-degree
frame is ~20% larger. Getting this backwards produces a depth map that is wrong in a
smooth, plausible, radially-symmetric way -- the kind of error that trains a model
happily and never looks like a bug.

---------------------------------------------------------------------------
THE TWO MESHES, AND WHY BOTH

HM3D ships `<scene>.basis.glb` for rendering and `<scene>.semantic.glb` for labels, with
identical topology. **The basis one is unusable here**: its textures are Basis Universal,
which only Magnum decodes, and trimesh reads 0 of 209 materials from it. The plain
`<scene>.glb` from `hm3d-<split>-glb-v0.2.tar` reads 209 of 209, so that is the RGB
source. Semantics come from the semantic mesh's own texture -- a plain 2048x2048 PNG
whose colours index `<scene>.semantic.txt`.

**Semantic colours are snapped, not looked up.** Sampling that texture returns blend
pixels along every UV seam: 20 geometries yielded 1,576 distinct colours against 660 real
labels. Nearest-neighbour sampling avoids most of it and a nearest-colour snap with a
distance cut handles the rest; anything further than the cut becomes `unlabeled` rather
than being forced onto whichever label happens to be closest in RGB.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import NamedTuple

import numpy as np
import trimesh
from PIL import Image

from syncai_hydranet.geometry.ground import Camera, fit_ground_plane

UNLABELED = 0
SNAP_MAX = 12.0  # RGB L2 distance; beyond this a sampled colour is a seam, not a label


class Pose(NamedTuple):
    """A camera preset. A NamedTuple rather than a dict because the fields have different
    types -- three floats and a size pair -- and a dict of both leaves every read typed
    `float | tuple[int, int]`, which is true and useless."""

    height: float  # metres above the floor point below the camera
    pitch: float  # degrees, positive looks down
    vfov: float
    size: tuple[int, int]  # H, W


PRESETS = {
    # Taichung-cam01 as fitted by scripts/fit_camera_from_people.py -- the tile-grid fit,
    # not the people-based one, for the reason that file's docstring gives.
    "cctv": Pose(height=2.38, pitch=50.2, vfov=70.4, size=(480, 640)),
    # The Lite3's forward camera. Height is from the first robot capture, where the
    # forward cone put floor at 0.34 m; pitch is the mount tilt hydra_infer.py assumes.
    "robot": Pose(height=0.25, pitch=18.0, vfov=58.0, size=(384, 512)),
}


def load_label_table(txt: Path) -> tuple[np.ndarray, list[str]]:
    """`<scene>.semantic.txt` maps a hex colour to a category name."""
    colours: list[tuple[int, int, int]] = [(0, 0, 0)]
    names: list[str] = ["unlabeled"]
    for line in txt.read_text().splitlines()[1:]:
        parts = line.split(",")
        if len(parts) < 3:
            continue
        h = parts[1].strip().upper()
        if len(h) != 6:
            continue
        colours.append((int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)))
        names.append(parts[2].strip().strip('"'))
    return np.asarray(colours, dtype=np.float64), names


class TexturedScene:
    """A glb flattened into one ray-castable mesh that remembers its textures.

    trimesh's per-geometry meshes cannot be cast against as a unit, and concatenating
    them throws the materials away. So faces are stacked with an index recording which
    geometry each came from, and texture lookup happens after the hit.
    """

    def __init__(self, path: Path):
        scene = trimesh.load(path, process=False)
        # `trimesh.load` is typed as returning `Geometry`, which has no `.geometry`; an
        # HM3D glb is always a multi-geometry Scene, and a single mesh would still need
        # the per-geometry texture bookkeeping below, so this refuses rather than guesses.
        if not isinstance(scene, trimesh.Scene):
            raise TypeError(f"{path} loaded as {type(scene).__name__}, expected a Scene")
        geoms = list(scene.geometry.values())
        verts, faces, uvs, owner = [], [], [], []
        self.textures: list[np.ndarray | None] = []
        offset = 0
        for i, m in enumerate(geoms):
            img = getattr(getattr(m.visual, "material", None), "baseColorTexture", None)
            self.textures.append(np.asarray(img.convert("RGB")) if img is not None else None)
            uv = getattr(m.visual, "uv", None)
            verts.append(np.asarray(m.vertices, dtype=np.float64))
            faces.append(np.asarray(m.faces, dtype=np.int64) + offset)
            uvs.append(
                np.zeros((len(m.vertices), 2))
                if uv is None
                else np.asarray(uv, dtype=np.float64)
            )
            owner.append(np.full(len(m.faces), i, dtype=np.int32))
            offset += len(m.vertices)
        self.vertices = np.concatenate(verts)
        self.faces = np.concatenate(faces)
        self.uv = np.concatenate(uvs)
        self.face_owner = np.concatenate(owner)
        self.mesh = trimesh.Trimesh(self.vertices, self.faces, process=False)

    def sample(self, face_idx: np.ndarray, bary: np.ndarray) -> np.ndarray:
        """Texture colour at each hit, via barycentric UV. Nearest, never bilinear --
        on the semantic mesh a blended texel is a colour no label owns."""
        out = np.zeros((len(face_idx), 3), dtype=np.uint8)
        tri_uv = self.uv[self.faces[face_idx]]  # [N, 3, 2]
        uv = (tri_uv * bary[:, :, None]).sum(axis=1)
        for gid in np.unique(self.face_owner[face_idx]):
            tex = self.textures[gid]
            if tex is None:
                continue
            m = self.face_owner[face_idx] == gid
            h, w = tex.shape[:2]
            # glTF's v axis runs opposite to image rows.
            px = np.clip((uv[m, 0] % 1.0) * (w - 1), 0, w - 1).astype(np.int32)
            py = np.clip((1.0 - uv[m, 1] % 1.0) * (h - 1), 0, h - 1).astype(np.int32)
            out[m] = tex[py, px]
        return out


def snap_labels(sampled: np.ndarray, table: np.ndarray) -> np.ndarray:
    """Sampled texture colours -> label ids: nearest within `SNAP_MAX`, else unlabeled.

    Done over the frame's *distinct* colours, not its pixels. The direct version compares
    every hit against every label -- a 304k x 661 x 3 array, 1.6 GB of temporary -- and
    profiling put it at **6.71 s of a 7.2 s frame**, against 0.14 s for the raycast that
    does the actual work. A 480x640 frame holds on the order of a thousand distinct
    colours, so collapsing to those first gives the identical answer for ~1% of the cost.

    Worth stating plainly because it is the general shape: this was never a job that
    wanted a bigger machine. An idle GPU would have accelerated the 0.14 s.
    """
    flat = sampled.reshape(-1, 3).astype(np.int32)
    packed = (flat[:, 0] << 16) | (flat[:, 1] << 8) | flat[:, 2]
    uniq, inverse = np.unique(packed, return_inverse=True)
    rgb = np.stack([(uniq >> 16) & 255, (uniq >> 8) & 255, uniq & 255], axis=1).astype(
        np.float64
    )
    d = np.linalg.norm(rgb[:, None, :] - table[None, :, :], axis=2)
    near = d.argmin(axis=1)
    near[d[np.arange(len(near)), near] > SNAP_MAX] = UNLABELED
    return near[inverse].astype(np.uint16)


def up_axis(mesh: trimesh.Trimesh) -> int:
    """Which axis is vertical, decided from the mesh rather than assumed.

    HM3D is nominally Y-up, but a scene's bounds do not say so on their own -- a tall
    narrow house and a wide flat one disagree about which extent is 'height'. Face
    normals do: a building's floors and ceilings are its largest flat area, so the axis
    most normals align with is up. Getting this wrong silently renders the ceiling.
    """
    n = mesh.face_normals * mesh.area_faces[:, None]
    return int(np.argmax(np.abs(n).sum(axis=0)))


def camera_rays(size, vfov_deg: float, position, yaw_deg: float, pitch_deg: float, up: int):
    """Pinhole ray directions plus the forward axis, in world coordinates.

    The basis is built directly against the scene's own up axis rather than by rendering
    in a canonical y-up frame and permuting afterwards. The permutation version rendered
    a 50-degree *downward* CCTV pitch as 45% ceiling: swapping two axes flips handedness,
    which silently negates pitch. An explicit orthonormal basis cannot do that, and it
    works for whichever axis `up_axis` found without a special case.
    """
    h, w = size
    f = (h / 2) / np.tan(np.radians(vfov_deg) / 2)

    u = np.zeros(3)
    u[up] = 1.0
    ax, bx = (i for i in range(3) if i != up)
    ea, eb = np.zeros(3), np.zeros(3)
    ea[ax], eb[bx] = 1.0, 1.0

    y, p = np.radians(yaw_deg), np.radians(pitch_deg)
    horizon = np.cos(y) * ea + np.sin(y) * eb
    # +pitch looks down, which is how a CCTV bracket is described and how
    # fit_camera_from_people.py reports it.
    fwd = np.cos(p) * horizon - np.sin(p) * u
    fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, u)
    right /= np.linalg.norm(right)
    cam_up = np.cross(right, fwd)

    xs = (np.arange(w) + 0.5) - w / 2
    ys = (np.arange(h) + 0.5) - h / 2
    xx, yy = np.meshgrid(xs, ys)
    dirs = xx[..., None] * right + (-yy)[..., None] * cam_up + f * fwd
    dirs = (dirs / np.linalg.norm(dirs, axis=-1, keepdims=True)).reshape(-1, 3)
    origins = np.broadcast_to(np.asarray(position, dtype=np.float64), dirs.shape)
    return origins, dirs, fwd


def render(
    rgb: TexturedScene, sem: TexturedScene | None, table, position, yaw, cfg: Pose, up: int
) -> dict:
    h, w = cfg.size
    origins, dirs, fwd = camera_rays(cfg.size, cfg.vfov, position, yaw, cfg.pitch, up)
    idx_tri, idx_ray, locs = rgb.mesh.ray.intersects_id(
        origins, dirs, multiple_hits=False, return_locations=True
    )
    depth = np.zeros(h * w, dtype=np.float32)
    colour = np.zeros((h * w, 3), dtype=np.uint8)
    labels = np.zeros(h * w, dtype=np.uint16)
    if len(idx_ray):
        # Depth along the optical axis, not |hit - eye|. See the module docstring.
        depth[idx_ray] = ((locs - origins[idx_ray]) @ fwd).astype(np.float32)
        tri = rgb.faces[idx_tri]
        bary = trimesh.triangles.points_to_barycentric(rgb.vertices[tri], locs)
        colour[idx_ray] = rgb.sample(idx_tri, bary)
        # Same topology in both meshes, so the hit triangle index carries across.
        if sem is not None:
            labels[idx_ray] = snap_labels(sem.sample(idx_tri, bary), table)
    return {
        "rgb": colour.reshape(h, w, 3),
        "depth": depth.reshape(h, w),
        "semantic": labels.reshape(h, w),
        "hit_frac": float(len(idx_ray)) / (h * w),
    }


def floor_points(rgb: TexturedScene, up: int, n: int, rng, need_clear: float = 0.0):
    """Candidate standing positions: centres of near-horizontal faces low in the scene.

    A stand-in for habitat's navmesh, which is a Recast binary this cannot read. It is
    weaker -- it does not know a surface is reachable, only that it is flat and low -- so
    it will occasionally pick a table top. Rendered frames are cheap and a bad one is
    visible; a wrong navmesh would not be.

    **`need_clear` exists because HM3D is houses and the CCTV preset is a shop.** A store
    camera sits at 2.38 m, which in a residential scan is the ceiling: dropping the eye
    that far above a floor point buries it in the ceiling void, and the frame comes back
    as a close-up of joists. One ray straight up per candidate measures the real headroom
    and rejects those points, which is far cheaper than rendering them and discovering it.
    """
    n_faces = rgb.mesh.face_normals
    horizontal = np.abs(n_faces[:, up]) > 0.95
    centres = rgb.mesh.triangles_center[horizontal]
    if not len(centres):
        return np.empty((0, 3))
    lo = np.percentile(centres[:, up], 20)
    floor = centres[centres[:, up] < lo + 0.3]
    if not len(floor):
        floor = centres

    if need_clear > 0 and len(floor):
        origin = floor + np.eye(3)[up] * 0.05  # lift off the surface so it is not self-hit
        direction = np.tile(np.eye(3)[up], (len(floor), 1))
        _tri, idx_ray, locs = rgb.mesh.ray.intersects_id(
            origin, direction, multiple_hits=False, return_locations=True
        )
        headroom = np.zeros(len(floor))
        headroom[idx_ray] = locs[:, up] - origin[idx_ray, up]
        # No hit at all means open sky above (a scan boundary), which is fine to shoot from.
        headroom[headroom == 0] = np.inf
        floor = floor[headroom > need_clear]
        if not len(floor):
            return np.empty((0, 3))

    pick = rng.choice(len(floor), size=min(n, len(floor)), replace=False)
    return floor[pick]


def pick_yaw(rgb: TexturedScene, eye: np.ndarray, up: int, rng, n_probe: int = 24) -> float:
    """Aim the camera at open space instead of spinning it at random.

    A random yaw in a furnished house points at a wall most of the time, and a frame with
    no floor in it is dropped later at full rendering cost. Under the plausibility gate
    that cost 60-70 of every 70 placements in the worst scenes and left a validation split
    of ten frames.

    So probe the horizon first: one ray per candidate yaw, then choose among the roomiest.
    Randomised over the top third rather than taking the maximum, because always facing
    the longest sightline would make every frame a corridor shot and quietly narrow the
    dataset in a different way.
    """
    yaws = np.linspace(0, 360, n_probe, endpoint=False)
    ax, bx = (i for i in range(3) if i != up)
    dirs = np.zeros((n_probe, 3))
    dirs[:, ax] = np.cos(np.radians(yaws))
    dirs[:, bx] = np.sin(np.radians(yaws))
    origins = np.broadcast_to(eye, dirs.shape)
    _tri, idx_ray, locs = rgb.mesh.ray.intersects_id(
        origins, dirs, multiple_hits=False, return_locations=True
    )
    clear = np.full(n_probe, np.inf)
    if len(idx_ray):
        clear[idx_ray] = np.linalg.norm(locs - origins[idx_ray], axis=1)
    top = np.argsort(clear)[-max(1, n_probe // 3) :]
    return float(yaws[rng.choice(top)])


def floor_label(depth: np.ndarray, cfg: Pose, inlier_m: float = 0.05) -> dict | None:
    """The frame's true pose relative to the floor, measured from its own ground truth.

    **Not the same as the pose it was rendered at, and that difference was a real bug.**
    `floor_points` places the eye a fixed height above whatever flat surface it sampled,
    and some of those are counters, steps and platforms: across eight test frames three
    sat 0.48-0.65 m up, and for each one RANSAC on the ground-truth depth returned exactly
    `placed height + platform height`. The fit was right and the nominal label was wrong,
    by half a metre, on three frames in eight -- and the loss would have fallen anyway.

    So the label is whatever the geometry says, computed with the same function that will
    later run on *predicted* depth, which makes the two comparable by construction.

    `inlier_frac` is how much of the frame the fitted plane actually explains, and it
    replaces an earlier check that counted semantic `floor` pixels. Two reasons it is
    better: it measures what the fit stands on rather than what the annotation calls
    floor, and **it needs no semantic mesh** -- which matters because HM3D annotates only
    a subset of scenes. Six of ten minival scenes have no `.semantic.glb` at all, and
    gating on semantics silently discarded every one of them.
    """
    d = depth.astype(np.float64).copy()
    d[d <= 0] = np.nan
    h, w = d.shape
    plane, residual = fit_ground_plane(d, Camera.from_vfov(h, w, cfg.vfov))
    if plane is None:
        return None
    finite = np.isfinite(residual)
    inlier = float((np.abs(residual[finite]) <= inlier_m).mean()) if finite.any() else 0.0
    return {
        "height": round(plane.height, 4),
        "pitch": round(math.degrees(plane.pitch), 3),
        "roll": round(math.degrees(plane.roll), 3),
        "inlier_frac": round(inlier, 4),
    }


# A downward-looking camera standing on a floor. Deliberately far wider than any jitter
# this script applies, so these reject physically impossible fits without narrowing the
# label distribution the model has to learn.
# Separate constants rather than one dict: mixing a pair and a scalar in a dict types
# every read as their union, which is true and unusable -- the same shape that made
# PRESETS a NamedTuple above.
FLOOR_HEIGHT_M = (0.5, 5.0)
FLOOR_PITCH_DEG = (5.0, 85.0)
FLOOR_MAX_ROLL_DEG = 15.0


def plausible_floor(label: dict, min_inlier: float) -> bool:
    """Is the fitted plane a floor, or is it a wall RANSAC liked the look of?

    An inlier fraction alone does not answer this, and trusting it produced a dataset
    whose pitch labels ran from -71 to +85 degrees against a rendered range of 38 to 62,
    with heights down to 0.01 m. Those are ceilings and walls: any large flat surface has
    inliers, and a plane fitted to the ceiling is every bit as tight as one fitted to the
    floor.

    What separates them is not fit quality but geometry -- a floor is below the camera and
    the camera looks down at it. Bounds are wide enough that they cannot be doing the
    labelling; they only remove fits that no camera placement could have produced.
    """
    lo, hi = FLOOR_HEIGHT_M
    plo, phi = FLOOR_PITCH_DEG
    return (
        label["inlier_frac"] >= min_inlier
        and lo <= label["height"] <= hi
        and plo <= label["pitch"] <= phi
        and abs(label["roll"]) <= FLOOR_MAX_ROLL_DEG
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scene", type=Path, help="one HM3D scene directory")
    ap.add_argument("--scenes", type=Path, help="a root of HM3D scene directories")
    # Pose jitter, and it is not decoration. Rendering every frame at the preset's exact
    # height and pitch makes the pose a constant of the dataset, and a model that ignores
    # the image entirely then "recovers" it perfectly. Varying it is what turns pose
    # recovery into something the picture has to answer.
    ap.add_argument("--jitter-height", type=float, default=0.0, help="+/- metres")
    ap.add_argument("--jitter-pitch", type=float, default=0.0, help="+/- degrees")
    ap.add_argument("--jitter-vfov", type=float, default=0.0, help="+/- degrees")
    # A frame with no floor in it cannot teach floor recovery. Two of eight test frames
    # came back with 0.0% floor pixels -- the camera boxed in facing a wall -- and RANSAC
    # duly fitted that wall at 0.19 m and -32 degrees.
    ap.add_argument(
        "--min-floor-frac",
        type=float,
        default=0.25,
        help="drop frames whose fitted ground plane explains less of the frame than this",
    )
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--preset", choices=sorted(PRESETS), default="cctv")
    ap.add_argument("--n", type=int, default=4, help="frames to render")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    base = PRESETS[args.preset]
    args.out.mkdir(parents=True, exist_ok=True)
    if args.scenes:
        scenes = sorted(d for d in args.scenes.iterdir() if d.is_dir() and "-" in d.name)
    elif args.scene:
        scenes = [args.scene]
    else:
        raise SystemExit("give --scene or --scenes")

    rng = np.random.default_rng(args.seed)
    manifest: list[dict] = []
    names: list[str] = []
    i = 0
    dropped = 0
    for scene in scenes:
        stem = scene.name.split("-", 1)[1]
        if not (scene / f"{stem}.glb").exists():
            continue
        rgb = TexturedScene(scene / f"{stem}.glb")
        # Semantics are optional. HM3D ships annotations for a subset only -- six of ten
        # minival scenes have no semantic mesh -- and depth supervision does not need
        # them. Requiring them threw away those scenes with a stack trace.
        sem_path = scene / f"{stem}.semantic.glb"
        has_sem = sem_path.exists() and (scene / f"{stem}.semantic.txt").exists()
        sem = TexturedScene(sem_path) if has_sem else None
        table, names = (
            load_label_table(scene / f"{stem}.semantic.txt")
            if has_sem
            else (np.zeros((1, 3)), ["unlabeled"])
        )
        up = up_axis(rgb.mesh)
        print(
            f"{stem}: {len(rgb.faces):,} faces, up-axis={'xyz'[up]}, "
            f"{len(names) if has_sem else 'no'} labels"
        )
        # Headroom for the tallest jittered eye, plus a little, so a frame is not
        # rejected for a placement the jitter happened to push into the ceiling.
        spots = floor_points(
            rgb, up, args.n, rng, need_clear=base.height + args.jitter_height + 0.15
        )
        for spot in spots:
            cfg = Pose(
                height=base.height + float(rng.uniform(-1, 1)) * args.jitter_height,
                pitch=base.pitch + float(rng.uniform(-1, 1)) * args.jitter_pitch,
                vfov=base.vfov + float(rng.uniform(-1, 1)) * args.jitter_vfov,
                size=base.size,
            )
            eye = spot.copy()
            eye[up] += cfg.height
            yaw = pick_yaw(rgb, eye, up, rng)
            r = render(rgb, sem, table, eye, yaw, cfg, up)
            label = floor_label(r["depth"], cfg)
            if label is None or not plausible_floor(label, args.min_floor_frac):
                dropped += 1
                continue
            Image.fromarray(r["rgb"]).save(args.out / f"{i:05d}_rgb.png")
            # Depth as 16-bit millimetres: lossless to 1 mm over the 0-65 m this needs,
            # and readable by anything, where a float32 .npy is neither.
            Image.fromarray((r["depth"] * 1000).clip(0, 65535).astype(np.uint16)).save(
                args.out / f"{i:05d}_depth.png"
            )
            if has_sem:
                Image.fromarray(r["semantic"]).save(args.out / f"{i:05d}_semantic.png")
            seen = np.unique(r["semantic"])
            d = r["depth"][r["depth"] > 0]
            manifest.append(
                {
                    "frame": i,
                    "scene": stem,
                    "preset": args.preset,
                    "eye": eye.tolist(),
                    "yaw_deg": yaw,
                    # The measured pose, which is the label. `*_nominal` is where the eye
                    # was put, kept only so a surprising gap between them is visible.
                    **label,
                    "height_nominal": round(cfg.height, 4),
                    "pitch_nominal": round(cfg.pitch, 3),
                    "vfov": cfg.vfov,
                    "size": list(cfg.size),
                    "hit_frac": round(r["hit_frac"], 4),
                    "depth_m": [round(float(d.min()), 3), round(float(d.max()), 3)]
                    if len(d)
                    else None,
                    "labels": [names[j] for j in seen[:12]] if has_sem else [],
                }
            )
            i += 1
        print(f"  -> {i} kept, {dropped} dropped so far")

    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (args.out / "labels.json").write_text(json.dumps(names, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
