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
from pathlib import Path
from typing import NamedTuple

import numpy as np
import trimesh
from PIL import Image

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
    rgb: TexturedScene, sem: TexturedScene, table, position, yaw, cfg: Pose, up: int
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
        sem_rgb = sem.sample(idx_tri, bary).astype(np.float64)
        d = np.linalg.norm(sem_rgb[:, None, :] - table[None, :, :], axis=2)
        near = d.argmin(axis=1)
        near[d[np.arange(len(near)), near] > SNAP_MAX] = UNLABELED
        labels[idx_ray] = near.astype(np.uint16)
    return {
        "rgb": colour.reshape(h, w, 3),
        "depth": depth.reshape(h, w),
        "semantic": labels.reshape(h, w),
        "hit_frac": float(len(idx_ray)) / (h * w),
    }


def floor_points(rgb: TexturedScene, up: int, n: int, rng) -> np.ndarray:
    """Candidate standing positions: centres of near-horizontal faces low in the scene.

    A stand-in for habitat's navmesh, which is a Recast binary this cannot read. It is
    weaker -- it does not know a surface is reachable, only that it is flat and low -- so
    it will occasionally pick a table top. Rendered frames are cheap and a bad one is
    visible; a wrong navmesh would not be.
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
    pick = rng.choice(len(floor), size=min(n, len(floor)), replace=False)
    return floor[pick]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scene", type=Path, required=True, help="an HM3D scene directory")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--preset", choices=sorted(PRESETS), default="cctv")
    ap.add_argument("--n", type=int, default=4, help="frames to render")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    stem = args.scene.name.split("-", 1)[1]
    cfg = PRESETS[args.preset]
    args.out.mkdir(parents=True, exist_ok=True)

    rgb = TexturedScene(args.scene / f"{stem}.glb")
    sem = TexturedScene(args.scene / f"{stem}.semantic.glb")
    table, names = load_label_table(args.scene / f"{stem}.semantic.txt")
    up = up_axis(rgb.mesh)
    print(f"{stem}: {len(rgb.faces):,} faces, up-axis={'xyz'[up]}, {len(names)} labels")

    rng = np.random.default_rng(args.seed)
    spots = floor_points(rgb, up, args.n, rng)
    manifest = []
    for i, spot in enumerate(spots):
        eye = spot.copy()
        eye[up] += cfg.height
        yaw = float(rng.uniform(0, 360))
        r = render(rgb, sem, table, eye, yaw, cfg, up)
        Image.fromarray(r["rgb"]).save(args.out / f"{i:03d}_rgb.png")
        # Depth as 16-bit millimetres: lossless to 1 mm over the 0-65 m this needs, and
        # readable by anything, where a float32 .npy is neither.
        Image.fromarray((r["depth"] * 1000).clip(0, 65535).astype(np.uint16)).save(
            args.out / f"{i:03d}_depth.png"
        )
        Image.fromarray(r["semantic"]).save(args.out / f"{i:03d}_semantic.png")
        seen = np.unique(r["semantic"])
        d = r["depth"][r["depth"] > 0]
        manifest.append(
            {
                "frame": i,
                "scene": stem,
                "preset": args.preset,
                "eye": eye.tolist(),
                "yaw_deg": yaw,
                "height": cfg.height,
                "pitch": cfg.pitch,
                "vfov": cfg.vfov,
                "size": list(cfg.size),
                "hit_frac": round(r["hit_frac"], 4),
                "depth_m": [round(float(d.min()), 3), round(float(d.max()), 3)]
                if len(d)
                else None,
                "labels": [names[j] for j in seen[:12]],
            }
        )
        print(
            f"  {i:03d} hit={r['hit_frac']:.3f} "
            f"depth={d.min():.2f}-{d.max():.2f}m labels={len(seen)}"
            if len(d)
            else f"  {i:03d} EMPTY"
        )
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (args.out / "labels.json").write_text(json.dumps(names, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
