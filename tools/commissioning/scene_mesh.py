"""CLI for the commissioning scene render -- writes the PNG, the OBJ and the GLB.

    uv run python tools/commissioning/scene_mesh.py <camera> [...] [--gif] [--ragged]

**The scene itself lives in `syncai_bev3d.scene_mesh`, and moved there on 2026-08-29.** Four
tools were importing this file by bare module name, which resolves only because Python
puts the entry script's own directory on `sys.path`, and
`tests/test_scripts_are_not_libraries.py` went red on the fourth. What is left here is
what a script is for: argument parsing, the export formats, and the orbit gif.
"""

import argparse
import html
import json
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

ROOT = Path(__file__).resolve().parents[2]


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


def export_html(glb, out):
    """Use trimesh's bundled offline viewer for the same exported object scene."""
    import trimesh
    from trimesh.viewer.notebook import scene_to_html

    scene = trimesh.load_scene(glb)
    centre = scene.bounds.mean(0)
    span = max(float(np.linalg.norm(np.ptp(scene.bounds, axis=0))), 2.0)
    eye = centre + np.array([0.7, 0.7, -0.7]) * span
    forward = (eye - centre) / np.linalg.norm(eye - centre)
    right = np.cross([0, 1, 0], forward)
    right /= np.linalg.norm(right)
    transform = np.eye(4)
    transform[:3, :3] = np.column_stack([right, np.cross(forward, right), forward])
    transform[:3, 3] = eye
    scene.camera_transform = transform
    page = scene_to_html(scene)
    fallback = (
        '<main style="max-width:1200px;margin:24px auto;font:18px sans-serif">'
        "<p>此瀏覽器無法啟用互動 3D。可查看下方場景圖或下載 GLB 模型。</p>"
        f'<p><a href="{html.escape(glb.name, quote=True)}">下載 GLB 模型</a></p>'
        '<img style="width:100%" '
        f'src="{html.escape(glb.with_suffix(".png").name, quote=True)}" '
        'alt="物件式 3D 場景"></main>'
    )
    handler = (
        '<script>window.addEventListener("error", function () {'
        f"document.body.innerHTML = {json.dumps(fallback)};"
        "});</script>"
    )
    page = page.replace("<head>", "<head>" + handler, 1)
    out.write_text(page)
    return out


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
    ap.add_argument(
        "--model-run", type=Path, help="completed StudioA scene training run; requires --out"
    )
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = ap.parse_args()
    if args.model_run is not None and (
        args.out is None or args.ragged or args.opening_controls
    ):
        ap.error(
            "--model-run requires --out and the regular scene path without opening controls"
        )
    if args.opening_controls is not None and (args.out is None or args.ragged):
        ap.error("--opening-controls requires --out and the regular scene path")
    root = args.root.resolve()
    if args.out is not None:
        args.out.mkdir(parents=True, exist_ok=False)
    source_root = root
    if args.model_run is not None:
        from syncai_hydranet.cli.scene_evidence import prepare_model_evidence

        root = args.out.resolve() / "inputs"
        for camera in args.cameras:
            if Path(camera).name != camera or camera in {".", ".."}:
                ap.error("camera must be a single camera identifier")
            prepare_model_evidence(
                args.model_run, source_root, camera, root, device=args.device
            )
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
        export_html(glb, staging / "scene.html")
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
        if args.model_run is not None:
            audit["reference_scope"] = (
                "student semantic masks, not independent truth; "
                "floor extent inherits commissioned walkable footprint"
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
        if args.model_run is not None:
            evidence = root / f"runs/commission01/{camera}/masks"
            shutil.copy2(evidence / "model_evidence.json", staging / "model_evidence.json")
            shutil.copy2(
                evidence / "prediction.overlay.png", staging / "prediction.overlay.png"
            )
            from syncai_hydranet.geometry.camera_json import CameraFile

            source_camera = CameraFile.load(root / f"runs/commission01/{camera}.camera.json")
            shutil.copy2(root / str(source_camera.plate_file), staging / "source.png")
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
