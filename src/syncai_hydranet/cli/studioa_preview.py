"""Render new StudioA semantics on source-bound, existing depth geometry."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from syncai_bev3d.geometry_review import cache_ground_error, load_geometry_cache
from syncai_hydranet.data.studioa_review import digest, write_json
from syncai_hydranet.data.studioa_supervision import CLASSES
from syncai_hydranet.geometry.camera_json import CameraFile


def visible_mesh(arrays, labels, confidence, *, stride=10, threshold=0.35):
    """Triangulate visible depth samples; never fill unseen or discontinuous surfaces."""
    if labels.shape != arrays["height"].shape or confidence.shape != labels.shape:
        raise ValueError("prediction and geometry shapes differ")
    if stride < 1 or not 0 <= threshold <= 1:
        raise ValueError("invalid preview sampling settings")
    if not set(np.unique(labels)) <= set(CLASSES.values()):
        raise ValueError("unknown semantic IDs")
    floor = labels == CLASSES["floor"]
    xyz = np.stack(
        [
            np.where(floor, arrays["gx"], arrays["lx"]),
            np.where(floor, 0, arrays["height"]),
            np.where(floor, arrays["gz"], arrays["lz"]),
        ],
        -1,
    )[::stride, ::stride]
    valid = arrays["geom_ok"][::stride, ::stride] & (
        confidence[::stride, ::stride] >= threshold
    )
    valid &= np.isfinite(xyz).all(-1)
    valid &= (abs(xyz[..., 0]) < 12) & (xyz[..., 2] > 0) & (xyz[..., 2] < 16)
    valid &= (xyz[..., 1] >= -0.15) & (xyz[..., 1] <= 4)
    height, width = valid.shape
    grid = np.arange(height * width).reshape(height, width)
    a, b, c, d = grid[:-1, :-1], grid[:-1, 1:], grid[1:, :-1], grid[1:, 1:]
    faces = np.concatenate(
        [np.stack([a, b, c], -1).reshape(-1, 3), np.stack([b, d, c], -1).reshape(-1, 3)]
    )
    points = xyz.reshape(-1, 3)
    keep = valid.ravel()[faces].all(1)
    for i, j in [(0, 1), (1, 2), (2, 0)]:
        keep &= np.linalg.norm(points[faces[:, i]] - points[faces[:, j]], axis=1) < 0.65
    faces = faces[keep]
    if len(faces) == 0:
        raise ValueError("no valid visible triangles")
    used, inverse = np.unique(faces, return_inverse=True)
    return points[used], inverse.reshape(-1, 3), used, valid.shape


def render_view(run: Path, camera_root: Path, camera: str, out: Path, *, device="cuda"):
    """Use only the selected new model for semantic colours, with explicit depth provenance."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import torch
    import torch.nn.functional as functional
    import trimesh
    from matplotlib.patches import Patch
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    from syncai_hydranet.models import build_model
    from syncai_hydranet.preprocessing import IMAGENET_MEAN, IMAGENET_STD
    from syncai_hydranet.utils.checkpoint import load_checkpoint
    from syncai_hydranet.utils.visualize import letterbox, terrain_palette

    report = json.loads((run / "report.json").read_text())
    checkpoint = run / "model/best.pt"
    if report["status"] != "completed" or report["outputs"]["model/best.pt"] != digest(
        checkpoint
    ):
        raise ValueError("render requires a completed, bound new model")
    cp = load_checkpoint(checkpoint)
    if cp.get("studioa_job_sha256") != digest(run / "job.json"):
        raise ValueError("foreign model checkpoint")
    cf = CameraFile.load(camera_root / camera / "camera.json")
    plate_path = camera_root / camera / "plate.png"
    geometry_path = camera_root / camera / "geometry.npz"
    arrays = load_geometry_cache(geometry_path, cf, plate_path=plate_path)
    if "plate_sha256" not in arrays or "geometry_signature" not in arrays:
        raise ValueError("preview requires source-bound geometry and plate")
    cfg = json.loads((run / "config.json").read_text())
    cfg["model"]["backbone"]["pretrained"] = False
    model = build_model(cfg).to(device).eval()
    model.load_state_dict(cp["model"])
    image = Image.open(plate_path).convert("RGB")
    height, width = arrays["height"].shape
    image = image.resize((width, height), Image.Resampling.BILINEAR)
    canvas, (x0, y0, w, h) = letterbox(image, cfg["data"]["input_size"])
    data = (np.asarray(canvas, dtype=np.float32) / 255 - IMAGENET_MEAN) / IMAGENET_STD
    tensor = torch.from_numpy(data.transpose(2, 0, 1).copy())[None].to(device)
    with torch.inference_mode():
        logits = model(tensor)["scene"][:, :, y0 : y0 + h, x0 : x0 + w]
        logits = functional.interpolate(
            logits, size=(height, width), mode="bilinear", align_corners=False
        )
        confidence, labels = logits.softmax(1).max(1)
        labels = labels[0].cpu().numpy().astype(np.uint8)
        confidence = confidence[0].cpu().numpy()
    out.mkdir(parents=True, exist_ok=False)
    Image.fromarray(labels).save(out / "prediction.png")
    np.save(out / "confidence.npy", confidence.astype(np.float16), allow_pickle=False)
    vertices, faces, used, _ = visible_mesh(arrays, labels, confidence)
    palette = terrain_palette(list(CLASSES), len(CLASSES))
    rgb = np.asarray(image)
    vertex_rgb = rgb[::10, ::10].reshape(-1, 3)[used]
    vertex_labels = labels[::10, ::10].ravel()[used]
    colors = palette[vertex_labels]
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_colors=colors, process=False)
    mesh.export(out / "semantic_scene.glb")
    mesh = trimesh.Trimesh(
        vertices=vertices, faces=faces, vertex_colors=vertex_rgb, process=False
    )
    mesh.export(out / "textured_scene.glb")
    overlay = (rgb * 0.55 + palette[labels] * 0.45).astype(np.uint8)
    overlay[confidence < 0.35] = rgb[confidence < 0.35]
    fig = plt.figure(figsize=(16, 11), facecolor="#101827")
    for index, pixels, title in [
        (1, rgb, "CCTV source frame"),
        (2, overlay, "New model: 19 semantic classes"),
    ]:
        ax = fig.add_subplot(2, 2, index)
        ax.imshow(pixels)
        ax.set_title(title, color="white")
        ax.axis("off")
    plotting = vertices[:, [0, 2, 1]]
    lo, hi = plotting.min(0), plotting.max(0)
    for index, paint, title in [
        (3, vertex_rgb, "Visible 3D surfaces: source texture"),
        (4, colors, "Visible 3D surfaces: new model semantics"),
    ]:
        ax = fig.add_subplot(2, 2, index, projection="3d", facecolor="#101827")
        collection = Poly3DCollection(
            plotting[faces],
            facecolors=paint[faces].mean(1) / 255,
            linewidths=0,
            rasterized=True,
        )
        ax.add_collection3d(collection)
        ax.set(xlim=(lo[0], hi[0]), ylim=(lo[1], hi[1]), zlim=(min(0, lo[2]), max(2.5, hi[2])))
        ax.set_box_aspect(np.maximum(hi - lo, [1, 1, 2.5]))
        ax.view_init(elev=35, azim=-65)
        ax.set_title(title, color="white")
        ax.tick_params(colors="#b8c7d9", labelsize=7)
        ax.set_xlabel("X (calibrated units)", color="white", fontsize=8)
        ax.set_ylabel("Z (calibrated units)", color="white", fontsize=8)
        ax.set_zlabel("height", color="white", fontsize=8)
    fig.suptitle(f"{camera} | new model {digest(checkpoint)[:12]}", color="white", fontsize=18)
    fig.text(
        0.5,
        0.025,
        "Model semantics + existing calibrated depth; visible surfaces only. "
        "Absolute scale and hidden geometry are not independently verified.",
        ha="center",
        color="#c9d5e4",
        fontsize=10,
    )
    fig.legend(
        handles=[Patch(color=palette[i] / 255, label=name) for name, i in CLASSES.items()],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.055),
        ncol=7,
        fontsize=8,
        facecolor="#101827",
        labelcolor="white",
    )
    fig.subplots_adjust(top=0.91, bottom=0.16, wspace=0.04, hspace=0.13)
    fig.savefig(out / "world.png", dpi=130, facecolor=fig.get_facecolor())
    plt.close(fig)
    provenance = {
        "camera": camera,
        "model_run": str(run),
        "checkpoint_sha256": digest(checkpoint),
        "camera_sha256": digest(camera_root / camera / "camera.json"),
        "plate_sha256": digest(plate_path),
        "geometry_sha256": digest(geometry_path),
        "sampled_floor_drift_m": cache_ground_error(cf, arrays),
        "semantic_source": "new model forward pass; no teacher masks or old scene objects",
        "geometry_source": "existing source-bound calibrated depth; visible surfaces only",
        "coordinate_frame": "per-camera: x lateral, y height, z forward; no cross-store frame",
        "absolute_metric_accuracy_verified": False,
        "confidence_threshold": 0.35,
        "confidence_retained_fraction": float((confidence >= 0.35).mean()),
        "vertices": len(vertices),
        "triangles": len(faces),
        "classes": CLASSES,
        "outputs": {p.name: digest(p) for p in out.iterdir() if p.is_file()},
    }
    write_json(out / "provenance.json", provenance)
    return provenance
