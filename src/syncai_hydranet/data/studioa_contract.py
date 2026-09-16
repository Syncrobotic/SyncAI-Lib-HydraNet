"""StudioA annotation vocabulary; deliberately not a deployed model-head label map.

Entity IDs belong to review proposals. A human-approved annotation and an explicit
head-specific export are required before they can become training supervision.
"""

from __future__ import annotations

import hashlib
import json

import numpy as np

from .label_maps_retail_objects import RETAIL_OBJECTS, RETAIL_SURFACES
from .label_maps_site30k import SITE30K

VERSION = "studioa.scene.v1"
IGNORE = 255
# IDs are stable annotation IDs, not existing student output channels. Zero is reserved.
ENTITY_NAMES = (
    "floor",
    "wall",
    "ceiling",
    "column",
    "door",
    "glass_panel",
    "display_cabinet",
    "display_table",
    "counter",
    "laptop",
    "phone",
    "tablet",
    "boxed_stock",
    "cardboard_box",
    "speaker",
    "poster",
    "fire_equipment",
    "chair",
    "person",
)
ENTITY_IDS = {name: index for index, name in enumerate(ENTITY_NAMES, 1)}
ZH = (
    "地面",
    "牆壁",
    "天花板",
    "柱子",
    "門",
    "固定玻璃",
    "展示櫃",
    "展示桌",
    "櫃檯",
    "筆電",
    "手機",
    "平板",
    "盒裝物品",
    "紙箱",
    "喇叭",
    "海報",
    "消防設備",
    "椅子",
    "人物",
)
# Public views can overlap: a glass door is one door instance, not a second object.
VIEWS = {
    **{
        name: {"entity": name}
        for name in ENTITY_NAMES
        if name not in {"counter", "glass_panel"}
    },
    "glass": {"entity": "glass_panel"},
    "glass_door": {"entity": "door", "glazed": True},
    "checkout_counter": {"entity": "counter", "role": "checkout"},
}
MATERIALS = ("glass", "metal", "wood", "plastic", "paper", "cardboard", "fabric", "other")
STAGES = {
    "stage0": "calibration, scale evidence and coordinate-frame identity",
    "stage1": "scene entities, visible geometry, material/role and person observations",
    "stage2": "tracks, observation quality and supported appearance attributes",
    "stage3": "temporal events and person-object state transitions",
    "stage4": "aggregation, deduplication scope and evidence replay",
}
SOURCE_CLASSES = {
    "site30k_native": SITE30K,
    "retail_objects_native": RETAIL_OBJECTS,
    "retail_surfaces_native": RETAIL_SURFACES,
}
# Preserve only distinctions the source actually made; source wall is a merged shell.
_SOURCE_EXACT = {
    "site30k_native": (
        "floor",
        "column",
        "display_table",
        "person",
        "laptop",
        "tablet",
        "phone",
        "boxed_stock",
    ),
    "retail_objects_native": ("floor", "column", "person"),
    "retail_surfaces_native": ("floor", "column", "person"),
}
RELABEL_REASONS = {
    "wall": "merged wall/ceiling/door/glazing; cannot recover the original entity",
    "shelf": "open shelving is not necessarily a display cabinet",
    "fixture": "merged tables/cabinets/shelves/counters; role is not observed",
    "product": "merged merchandise; device and packaging distinctions are missing",
}


def contract() -> dict:
    """Serializable source of truth, consumed by review packages and validators."""
    return {
        "version": VERSION,
        "purpose": "annotation and review; not a training-head configuration",
        "ignore_id": IGNORE,
        "void": "unlabelled pixels map to 255; never an entity or a negative example",
        "entities": [
            {"id": ENTITY_IDS[name], "name": name, "zh": zh}
            for name, zh in zip(ENTITY_NAMES, ZH, strict=True)
        ],
        "views": VIEWS,
        "materials": list(MATERIALS),
        "stage_roles": STAGES,
        "fire_scope": "physical fire equipment; subtype unconfirmed; excludes smoke/flame",
        "migration": {
            source: {
                name: {
                    "target_id": ENTITY_IDS[name] if name in _SOURCE_EXACT[source] else IGNORE,
                    "reason": "candidate only; requires human review"
                    if name in _SOURCE_EXACT[source]
                    else "ignore"
                    if name == "void"
                    else RELABEL_REASONS[name],
                }
                for name in classes
            }
            for source, classes in SOURCE_CLASSES.items()
        },
    }


def contract_sha256() -> str:
    payload = json.dumps(contract(), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def migrate_mask(mask: np.ndarray, source: str) -> tuple[np.ndarray, dict]:
    """Return review proposals plus explicit lost distinctions; never infer finer labels."""
    if source not in SOURCE_CLASSES:
        raise ValueError(f"unsupported source taxonomy: {source}")
    if mask.ndim != 2 or not np.issubdtype(mask.dtype, np.integer):
        raise ValueError("source mask must be a two-dimensional integer label image")
    classes = SOURCE_CLASSES[source]
    ids, counts = np.unique(mask, return_counts=True)
    unknown = set(ids.tolist()) - set(classes.values()) - {IGNORE}
    if unknown:
        raise ValueError(f"unknown source labels: {sorted(unknown)}")
    result = np.full(mask.shape, IGNORE, dtype=np.uint8)
    preserved, relabel = {}, {}
    for name, source_id in classes.items():
        count = int(counts[ids == source_id].sum())
        if name in _SOURCE_EXACT[source]:
            result[mask == source_id] = ENTITY_IDS[name]
            preserved[name] = count
        elif name != "void":
            relabel[name] = {"pixels": count, "reason": RELABEL_REASONS[name]}
    return result, {
        "candidate_pixels": preserved,
        "requires_relabel": relabel,
        "ignored_pixels": int((result == IGNORE).sum()),
        "human_reviewed": False,
    }


def entity_views(entity: dict) -> set[str]:
    """Validate entity attributes and resolve user-facing views without double instances."""
    name = entity.get("entity")
    if name not in ENTITY_IDS:
        raise ValueError(f"unknown entity: {name}")
    glazed = entity.get("glazed")
    if glazed is not None and (type(glazed) is not bool or name != "door"):
        raise ValueError("glazed must be a boolean door observation or null")
    role = entity.get("role", "unknown")
    if role not in {"unknown", "checkout", "display"}:
        raise ValueError("unknown role")
    if role != "unknown":
        if name not in {"counter", "display_table", "display_cabinet"}:
            raise ValueError("functional role requires a fixture")
        if role == "checkout" and name != "counter":
            raise ValueError("checkout role requires a counter")
        evidence = entity.get("role_evidence")
        if not isinstance(evidence, str) or not evidence.strip():
            raise ValueError("functional role requires role_evidence")
    materials = entity.get("materials", [])
    if not isinstance(materials, list) or any(m not in MATERIALS for m in materials):
        raise ValueError("unsupported materials; use an empty list for unknown")
    if glazed is True and "glass" not in materials:
        raise ValueError("glazed door requires an observed glass material")
    return {
        view for view, rule in VIEWS.items() if all(entity.get(k) == v for k, v in rule.items())
    }
