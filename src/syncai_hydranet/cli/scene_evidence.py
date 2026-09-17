"""Adapt a trained scene head to the existing commissioning object modeller.

This is an input adapter, not a renderer. Calibration/depth and instance proposals
remain explicit inputs; the selected student supplies the semantic masks.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

from syncai_bev3d.geometry_review import load_geometry_cache
from syncai_bev3d.object_instances import ObjectInstance, save_instances
from syncai_bev3d.scene_audit import capture_inputs
from syncai_hydranet.data.studioa_review import digest, write_json
from syncai_hydranet.data.studioa_supervision import CLASSES
from syncai_hydranet.geometry.camera_json import CameraFile

# Asset families in the existing modeller. Counter/table share construction geometry,
# while their separate predicted semantic IDs remain in prediction.png and the report.
MASK_CLASSES = {
    "walkable": ("floor",),
    "floor": ("floor",),
    "wall": ("wall",),
    "column": ("column",),
    "display_table": ("display_table", "counter"),
    "display_shelf": ("display_cabinet",),
    "glass": ("glass_panel",),
    "door": ("door",),
    "product_boxed_stock": ("boxed_stock",),
    "product_macbook": ("laptop",),
    "product_ipad": ("tablet",),
    "product_iphone": ("phone",),
}


def structural_instances(labels, confidence, proposals, *, threshold=0.5, min_pixels=200):
    """Split student regions using existing instance proposals, never their class labels.

    Proposal IDs only partition touching objects. Uncovered connected components get
    their own IDs. Low confidence pixels do not acquire a semantic class from a teacher.
    """
    if labels.shape != confidence.shape or labels.shape != proposals.shape:
        raise ValueError("semantic predictions and instance proposals must align")
    if not np.isfinite(confidence).all() or ((confidence < 0) | (confidence > 1)).any():
        raise ValueError("invalid semantic confidence")
    if not set(np.unique(labels)) <= set(CLASSES.values()):
        raise ValueError("unknown semantic IDs")
    objects = np.zeros(labels.shape, np.int32)
    index = 0
    for names in [("wall",), ("column",), ("display_table", "counter"), ("display_cabinet",)]:
        mask = np.isin(labels, [CLASSES[n] for n in names]) & (confidence >= threshold)
        components, count = ndimage.label(mask)
        for component in range(1, count + 1):
            region = components == component
            if region.sum() < min_pixels:
                continue
            ids, sizes = np.unique(proposals[region & (proposals > 0)], return_counts=True)
            seeds = np.where(
                region & np.isin(proposals, ids[sizes >= min_pixels]), proposals, 0
            )
            if seeds.any():
                # Nearest proposal partitions this component; it never expands its mask.
                nearest = ndimage.distance_transform_edt(
                    seeds == 0, return_distances=False, return_indices=True
                )
                owners = seeds[tuple(nearest)]
                for oid in np.unique(seeds[seeds > 0]):
                    index += 1
                    objects[region & (owners == oid)] = index
            else:
                index += 1
                objects[region] = index
    return objects


def prepare_model_evidence(run: Path, root: Path, camera: str, target: Path, *, device="cpu"):
    """Freeze candidate inputs and run the student; leave commissioned inputs intact."""
    import torch
    import torch.nn.functional as functional

    from syncai_hydranet.models import build_model
    from syncai_hydranet.preprocessing import IMAGENET_MEAN, IMAGENET_STD
    from syncai_hydranet.utils.checkpoint import load_checkpoint
    from syncai_hydranet.utils.visualize import letterbox, terrain_palette

    run, root, target = run.resolve(), root.resolve(), target.resolve()
    report = json.loads((run / "report.json").read_text())
    checkpoint = run / "model/best.pt"
    if report["status"] != "completed" or report["outputs"]["model/best.pt"] != digest(
        checkpoint
    ):
        raise ValueError("scene requires a completed, bound checkpoint")
    job = json.loads((run / "job.json").read_text())
    if report["job_sha256"] != digest(run / "job.json"):
        raise ValueError("completion report belongs to another training job")
    if job["files"].get("config.json") != digest(run / "config.json"):
        raise ValueError("model configuration changed after training")
    cp = load_checkpoint(checkpoint)
    if cp.get("studioa_job_sha256") != digest(run / "job.json"):
        raise ValueError("checkpoint belongs to another training job")
    cf_path = root / f"runs/commission01/{camera}.camera.json"
    cf = CameraFile.load(cf_path)
    plate = root / str(cf.plate_file)
    arrays = load_geometry_cache(
        root / f"runs/site30k_qa/geometry_cache/{camera}.npz", cf, plate_path=plate
    )
    if "plate_sha256" not in arrays or "geometry_signature" not in arrays:
        raise ValueError("scene requires source-bound geometry")
    before = capture_inputs(root, camera)
    folder = target / f"runs/commission01/{camera}/masks"
    folder.mkdir(parents=True, exist_ok=False)
    # Copy only source-bound camera/depth/plate/calibration. Do not smuggle teacher
    # merchandise, support refinements or opening masks into a student candidate.
    for relative in [
        cf_path.relative_to(root),
        plate.relative_to(root),
        Path(f"runs/site30k_qa/geometry_cache/{camera}.npz"),
        Path(f"runs/onboard01/{camera}.calib.json"),
    ]:
        src, dst = root / relative, target / relative
        if src.is_file():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
    for relative in ["src", "tools/commissioning", "pyproject.toml", "uv.lock"]:
        src, dst = root / relative, target / relative
        if not dst.exists():
            if src.is_dir():
                shutil.copytree(src, dst, ignore=shutil.ignore_patterns("__pycache__"))
            else:
                shutil.copy2(src, dst)
    cfg = json.loads((run / "config.json").read_text())
    if cfg["data"].get("terrain_classes") != list(CLASSES):
        raise ValueError("checkpoint semantic class order differs from StudioA")
    cfg["model"]["backbone"]["pretrained"] = False
    model = build_model(cfg).to(device).eval()
    model.load_state_dict(cp["model"])
    image = Image.open(plate).convert("RGB")
    width, height = cf.image_size_px
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
    Image.fromarray(labels).save(folder / "prediction.png")
    np.save(folder / "confidence.npy", confidence, allow_pickle=False)
    palette = terrain_palette(list(CLASSES), len(CLASSES))
    rgb = np.asarray(image)
    overlay = (rgb * 0.55 + palette[labels] * 0.45).astype(np.uint8)
    overlay[confidence < 0.5] = rgb[confidence < 0.5]
    Image.fromarray(overlay).save(folder / "prediction.overlay.png")
    proposals = np.zeros(labels.shape, np.int32)
    proposal_path = cf.mask_files.get("objects")
    if proposal_path:
        proposal_file = root / "runs/commission01" / proposal_path
        with Image.open(proposal_file) as im:
            proposals = np.asarray(im.resize((width, height), Image.Resampling.NEAREST)).astype(
                np.int32
            )
    objects = structural_instances(
        labels, confidence, proposals, min_pixels=max(40, int(width * height * 0.0002))
    )
    Image.fromarray(objects.astype(np.uint16)).save(folder / "objects.png")
    mapping = {}
    for name, entities in MASK_CLASSES.items():
        mask = np.isin(labels, [CLASSES[n] for n in entities]) & (confidence >= 0.5)
        Image.fromarray(mask.astype(np.uint8) * 255).save(folder / f"{name}.png")
        mapping[name] = f"{camera}/masks/{name}.png"
    # Room extent is a calibrated commissioning prior. Student floor evidence remains
    # separate in floor.png; occluding furniture must not punch holes in the room slab.
    floor_prior = root / "runs/commission01" / cf.mask_files["walkable"]
    shutil.copy2(floor_prior, folder / "walkable.png")
    mapping["objects"] = f"{camera}/masks/objects.png"
    cf_data = json.loads(cf_path.read_text())
    cf_data["mask_files"] = mapping
    write_json(target / cf_path.relative_to(root), cf_data)
    instances = []
    for category in ["laptop", "tablet", "phone", "chair"]:
        components, count = ndimage.label((labels == CLASSES[category]) & (confidence >= 0.5))
        for index in range(1, count + 1):
            mask = components == index
            if mask.sum() >= max(12, width * height * 0.000025):
                instances.append(
                    ObjectInstance(
                        category,
                        mask,
                        float(confidence[mask].mean()),
                        "student semantic connected component",
                    )
                )
    save_instances(
        folder / "object_instances.npz",
        instances,
        shape=labels.shape,
        source=target / plate.relative_to(root),
        categories=["laptop", "tablet", "phone", "chair"],
    )
    if capture_inputs(root, camera) != before:
        raise ValueError("commissioned evidence changed while preparing candidate")
    provenance = {
        "camera": camera,
        "checkpoint_sha256": digest(checkpoint),
        "run": str(run),
        "source_inputs": before["inputs"],
        "confidence_threshold": 0.5,
        "semantic_source": "student forward pass",
        "instance_source": "student regions partitioned by existing instance proposals",
        "geometry_source": (
            "existing camera calibration/depth; existing parametric scene modeller"
        ),
        "unsupported_asset_classes": [
            "ceiling",
            "cardboard_box",
            "speaker",
            "poster",
            "fire_equipment",
            "person",
        ],
        "absolute_metric_accuracy_verified": False,
        "floor_extent_source": (
            "retained commissioned walkable footprint; not student prediction"
        ),
        "adapter_sha256": digest(Path(__file__)),
        "classes": CLASSES,
        "outputs": {p.name: digest(p) for p in folder.iterdir() if p.is_file()},
    }
    write_json(folder / "model_evidence.json", provenance)
    return provenance
