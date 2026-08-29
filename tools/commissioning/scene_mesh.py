"""CLI for the commissioning scene render -- writes the PNG, the OBJ and the GLB.

    uv run python tools/commissioning/scene_mesh.py <camera> [...] [--gif] [--ragged]

**The scene itself lives in `syncai_bev3d.scene_mesh`, and moved there on 2026-08-29.** Four
tools were importing this file by bare module name, which resolves only because Python
puts the entry script's own directory on `sys.path`, and
`tests/test_scripts_are_not_libraries.py` went red on the fourth. What is left here is
what a script is for: argument parsing, the export formats, and the orbit gif.
"""

import sys
from pathlib import Path

import numpy as np
from PIL import Image

from syncai_bev3d.meshes import _merge, to_obj
from syncai_bev3d.scene_mesh import (
    PALETTE,
    build_scene,
    build_scene_regular,
    implausible,
    render,
)

ROOT = Path(__file__).resolve().parents[2]


def export_glb(camera, items):
    """A-path: the same scene as a GLB any glTF viewer can orbit."""
    import trimesh

    scene = trimesh.Scene()
    for i, (mesh, key, alpha, _s) in enumerate(items):
        verts, faces = mesh
        if len(faces) == 0:
            continue
        tm = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
        rgba = [*PALETTE[key], alpha]
        tm.visual = trimesh.visual.TextureVisuals(
            material=trimesh.visual.material.PBRMaterial(
                baseColorFactor=[c / 255 for c in rgba],
                metallicFactor=0.05,
                roughnessFactor=0.85,
                alphaMode="BLEND" if alpha < 255 else "OPAQUE",
                doubleSided=True,
            )
        )
        scene.add_geometry(tm, node_name=f"{key}_{i}")
    path = ROOT / "runs/commission01" / camera / "scene.glb"
    scene.export(path)
    return path


def export_obj(camera, items):
    solids = [m for m, k, a, _ in items if a == 255]
    walls = [m for m, k, a, _ in items if a == 105]
    obj = to_obj(_merge(*solids, *walls), name=camera)
    path = ROOT / "runs/commission01" / camera / "scene.obj"
    path.write_text(obj)
    return path


def main():
    argv = [a for a in sys.argv[1:] if not a.startswith("--")]
    gif = "--gif" in sys.argv[1:]
    # **The regular path is the default, and the flag now opts OUT of it.** Every real
    # consumer -- `demo_video`, `heads_video`, `scene_overlay` -- has called
    # `build_scene_regular` for some time; `main()` was the last caller of the ragged one,
    # and `main()` is what writes `assets/commission_mesh_*.png` and what the social
    # preview card was cut from. So the two most widely seen images in the project were
    # the only ones still built by the older path.
    #
    # It matters because the paths differ in the axis they align to. `rect_decompose`
    # tiles the cell grid with rectangles axis-aligned to the **world** x/z, while
    # `build_scene_regular` fits each fixture in the **store** frame from `store_yaw`. A
    # shop standing 30 deg off the world axes therefore came out of the ragged path as
    # staircases of small world-aligned blocks -- read by a reviewer as "cabinets at 45
    # degrees", which is what it looks like and is not what the reconstruction believed.
    ragged = "--ragged" in sys.argv[1:]
    for camera in argv:
        built = (build_scene if ragged else build_scene_regular)(camera)
        _cf, items, heights = built[:3]
        for line in implausible(built[3] if len(built) > 3 else []):
            print(f"  {camera}: implausible {line}")
        out = ROOT / f"assets/commission_mesh_{camera}.png"
        render(camera, items, heights, out)
        obj = export_obj(camera, items)
        glb = export_glb(camera, items)
        print(f"{camera}: {out.name}, {obj.name}, {glb.name}")
        if gif:
            xs = np.concatenate([m[0][:, 0] for m, *_ in items])
            zs = np.concatenate([m[0][:, 2] for m, *_ in items])
            cx_m, cz_m = float(np.median(xs)), float(np.median(zs))
            frames = []
            for ang in np.linspace(0, 2 * np.pi, 28, endpoint=False):
                eye = [cx_m + 9.2 * np.cos(ang), 5.8, cz_m + 9.2 * np.sin(ang)]
                tmp = ROOT / "assets" / "_orbit_tmp.png"
                render(camera, items, heights, tmp, eye=eye, target=[cx_m, 0.5, cz_m])
                frames.append(Image.open(tmp).convert("P", palette=Image.ADAPTIVE))
            tmp.unlink()
            gif_path = ROOT / f"assets/commission_mesh_{camera}_orbit.gif"
            frames[0].save(
                gif_path, save_all=True, append_images=frames[1:], duration=140, loop=0
            )
            print(f"{camera}: {gif_path.name}")


if __name__ == "__main__":
    main()
