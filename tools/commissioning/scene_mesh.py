"""CLI for the commissioning scene render -- writes the PNG, the OBJ and the GLB.

    uv run python tools/commissioning/scene_mesh.py <camera> [...] [--gif] [--ragged]

**The scene itself lives in `syncai_bev3d.scene_mesh`, and moved there on 2026-08-29.** Four
tools were importing this file by bare module name, which resolves only because Python
puts the entry script's own directory on `sys.path`, and
`tests/test_scripts_are_not_libraries.py` went red on the fourth. What is left here is
what a script is for: argument parsing, the export formats, and the orbit gif.
"""

import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

from syncai_bev3d.meshes import _merge, to_obj
from syncai_bev3d.render_provenance import sha256
from syncai_bev3d.scene_audit import audit_export, capture_inputs
from syncai_bev3d.scene_mesh import (
    build_scene,
    build_scene_regular,
    colour_of,
    implausible,
    render,
)

ROOT = Path(os.environ.get("SYNCAI_ROOT", Path(__file__).resolve().parents[2]))


def export_glb(camera, items, *, out=None):
    """A-path: the same scene as a GLB any glTF viewer can orbit."""
    import trimesh
    import trimesh.visual

    scene = trimesh.Scene()
    for i, (mesh, key, alpha, _s) in enumerate(items):
        verts, faces = mesh
        if len(faces) == 0:
            continue
        tm = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
        rgba = [*colour_of(key), alpha]
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
    path = out if out is not None else ROOT / "runs/commission01" / camera / "scene.glb"
    scene.export(path)
    return path


def export_obj(camera, items, *, out=None):
    solids = [m for m, k, a, _ in items if a == 255]
    walls = [m for m, k, a, _ in items if a == 105]
    obj = to_obj(_merge(*solids, *walls), name=camera)
    path = out if out is not None else ROOT / "runs/commission01" / camera / "scene.obj"
    path.write_text(obj)
    return path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("cameras", nargs="+")
    ap.add_argument("--gif", action="store_true")
    ap.add_argument("--ragged", action="store_true")
    ap.add_argument("--root", type=Path, default=ROOT, help="commissioning input checkout")
    ap.add_argument(
        "--out", type=Path, help="new review directory; preserve commissioned outputs"
    )
    ap.add_argument(
        "--opening-controls",
        type=Path,
        help="directory of CAMERA.json source-bound controls; requires --out",
    )
    args = ap.parse_args()
    if args.opening_controls is not None and (args.out is None or args.ragged):
        ap.error("--opening-controls requires --out and the regular scene path")
    root = args.root.resolve()
    if args.out is not None:
        args.out.mkdir(parents=True, exist_ok=False)
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
    ragged = args.ragged
    for camera in args.cameras:
        if Path(camera).name != camera or camera in {".", ".."}:
            ap.error("camera must be a single camera identifier")
        controls = (
            (args.opening_controls / f"{camera}.json").resolve()
            if args.opening_controls is not None
            else None
        )
        before = capture_inputs(root, camera, opening_controls=controls)
        destination = (
            args.out / camera if args.out is not None else root / f"runs/commission01/{camera}"
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=f".{camera}-", dir=destination.parent))
        object_report = []
        support_report = []
        surface_report = []
        built = (
            build_scene(camera, root)
            if ragged
            else build_scene_regular(
                camera,
                root,
                object_report=object_report,
                support_report=support_report,
                surface_report=surface_report,
                opening_controls=controls,
            )
        )
        _cf, items, heights = built[:3]
        for line in implausible(built[3] if len(built) > 3 else []):
            print(f"  {camera}: implausible {line}")
        out = staging / "scene.png"
        render(camera, items, heights, out, shapes=built[3] if len(built) > 3 else ())
        obj = export_obj(camera, items, out=staging / "scene.obj")
        glb = export_glb(camera, items, out=staging / "scene.glb")
        if not ragged:
            report_path = staging / "scene.objects.json"
            report_path.write_text(json.dumps(object_report, indent=2) + "\n")
            report_path.with_name("scene.supports.json").write_text(
                json.dumps(support_report, indent=2) + "\n"
            )
        audit = audit_export(
            root,
            camera,
            glb,
            surface_report,
            object_report,
            support_report,
            staging,
            regular=not ragged,
        )
        (staging / "scene.surfaces.json").write_text(
            json.dumps(audit, indent=2, allow_nan=False) + "\n"
        )
        if args.gif:
            xs = np.concatenate([m[0][:, 0] for m, *_ in items])
            zs = np.concatenate([m[0][:, 2] for m, *_ in items])
            cx_m, cz_m = float(np.median(xs)), float(np.median(zs))
            frames = []
            for ang in np.linspace(0, 2 * np.pi, 28, endpoint=False):
                eye = [cx_m + 9.2 * np.cos(ang), 5.8, cz_m + 9.2 * np.sin(ang)]
                tmp = staging / "_orbit_tmp.png"
                render(camera, items, heights, tmp, eye=eye, target=[cx_m, 0.5, cz_m])
                frames.append(Image.open(tmp).convert("P", palette=Image.Palette.ADAPTIVE))
            tmp.unlink()
            gif_path = staging / "scene.orbit.gif"
            frames[0].save(
                gif_path, save_all=True, append_images=frames[1:], duration=140, loop=0
            )
        if capture_inputs(root, camera, opening_controls=controls) != before:
            raise RuntimeError(
                f"{camera}: inputs or code changed during build; "
                f"unpublished candidate at {staging}"
            )
        manifest = {
            "schema": 1,
            "camera": camera,
            "status": "complete",
            "provenance": before,
            "outputs": {p.name: sha256(p) for p in sorted(staging.iterdir()) if p.is_file()},
        }
        (staging / "scene.manifest.json").write_text(
            json.dumps(manifest, indent=2, allow_nan=False) + "\n"
        )
        if args.out is not None:
            staging.rename(destination)
        else:
            destination.mkdir(parents=True, exist_ok=True)
            # A partially installed generation must not retain a complete old manifest.
            (destination / "scene.manifest.json").unlink(missing_ok=True)
            for p in sorted(staging.iterdir(), key=lambda p: p.name == "scene.manifest.json"):
                p.replace(destination / p.name)
            staging.rmdir()
            assets = root / "assets"
            assets.mkdir(exist_ok=True)
            shutil.copy2(destination / "scene.png", assets / f"commission_mesh_{camera}.png")
            if args.gif:
                shutil.copy2(
                    destination / "scene.orbit.gif",
                    assets / f"commission_mesh_{camera}_orbit.gif",
                )
        print(
            f"{camera}: {destination} ({obj.name}, {glb.name}, scene.manifest.json)", flush=True
        )


if __name__ == "__main__":
    main()
