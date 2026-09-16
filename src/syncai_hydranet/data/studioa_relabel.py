"""Second AI instrument for source-bound candidate review, never human ground truth."""

from __future__ import annotations

import json
from collections import Counter
from copy import deepcopy
from typing import Any, TypedDict

import numpy as np
from PIL import Image, ImageDraw, ImageOps

from .studioa_autolabel import Candidate, annotate, coverage, decode, overlap
from .studioa_contract import ENTITY_NAMES, entity_views

MODEL = "nvidia/Cosmos-Reason1-7B"
REVISION = "3210bec0495fdc7a8d3dbb8d58da5711eab4b423"
EXTRA_PROMPTS = {
    "display_cabinet": ("upright product display rack", "wall-mounted merchandise display"),
    "display_table": ("round display table", "long display table"),
    "counter": ("checkout counter", "customer service desk"),
    "cardboard_box": ("cardboard shipping box",),
    "fire_equipment": ("fire extinguisher",),
}


class ReviewCache(TypedDict):
    job_sha256: str
    frame_id: str
    groups: list[list[dict]]
    decisions: list[dict]
    source_annotation_sha256: str
    extra_candidates: int


PROMPT = """Classify the isolated segmented object/surface from an electronics store.
Everything outside its mask is replaced by gray; gray is not part of the target.
Identify the visible target itself. Answer with exactly ONE class name. No explanation.
Allowed CLASS: floor, wall, ceiling, column, door, glass_panel, display_cabinet,
display_table, counter, laptop, phone, tablet, boxed_stock, cardboard_box, speaker,
poster, fire_equipment, chair, person, unknown.
Definitions: display_table = round/long demo table or podium holding demonstration
devices, including its base/cabinet doors. display_cabinet = upright shelving/racks
or cabinets holding stock, including open wall-mounted retail racks. counter =
customer service/checkout work desk, supported by its service equipment and context.
door = architectural room/entrance door, NEVER furniture doors or panels.
boxed_stock = individual packaged merchandise, NOT a whole rack or unboxed device.
cardboard_box = shipping/storage carton. speaker = loudspeaker, NOT headphones.
phone/tablet/laptop = actual physical devices, not product-box pictures or posters.
person = actual person, not a poster/reflection. glass_panel = independent fixed glass,
not a cabinet part. Use unknown for unsupported, mixed-object, wrong-scale masks or
objects outside these classes. Do not infer a customer's identity or attributes.
Output only the class name, without JSON or any explanation.
"""

FIXTURE_PROMPT = """Identify the main furniture in this electronics-store crop.
Choose one class and output only its class name, without JSON or explanation:
display_table = a round/cylindrical demo pedestal or long table carrying demo devices.
A solid cylindrical base is still a display_table, NEVER an upright display_cabinet.
A long slab with demonstration laptops is a display_table even if its base has cupboards.
display_cabinet = an UPRIGHT rack with multiple stacked shelves or vertical hanging rows
of boxed merchandise. The merchandise storage is vertical, not just a tabletop.
counter = a SERVICE/CHECKOUT work desk with visible service equipment, printers,
paperwork or register context; do not call every white display table a counter.
unknown = ambiguous, mixed objects, or not one of these three furniture types.
Use the items on the furniture to distinguish its function. Output one class name only.
"""


def relabel_policy() -> dict:
    return {
        "model": MODEL,
        "revision": REVISION,
        "prompt": PROMPT,
        "fixture_prompt": FIXTURE_PROMPT,
        "extra_prompts": {name: list(prompts) for name, prompts in EXTRA_PROMPTS.items()},
        "group_iou": 0.75,
        "max_new_tokens": 16,
        "min_pixels": 128 * 28 * 28,
        "max_pixels": 512 * 28 * 28,
        "batch_size": 16,
        "presentation": "tight original fixture crop; isolated mask for other objects",
        "cross_family_policy": "limit family changes; preserve accepted room semantics",
        "source_semantics": "preserve accepted SAM room surfaces; trim occlusion only",
        "quality": "AI reclassification, not independently verified accuracy",
        "occlusion": "person > goods > fixtures > openings/columns > room surfaces",
    }


def candidate_groups(rows: list[dict]) -> list[list[dict]]:
    """Group near-identical masks only; no containment-based merging of distinct objects."""
    groups: list[list[dict]] = []
    representatives: list[Candidate] = []
    for row in sorted(rows, key=lambda item: (-item["score"], item["id"])):
        candidate = Candidate(row["entity"], decode(row["segmentation"]), row["score"])
        index = next(
            (
                i
                for i, other in enumerate(representatives)
                if overlap(candidate, other)[0] > 0.75
            ),
            None,
        )
        if index is None:
            groups.append([row])
            representatives.append(candidate)
        else:
            groups[index].append(row)
    return groups


def review_panel(image: Image.Image, row: dict) -> Image.Image:
    """Isolate the mask so a nearby large fixture cannot answer for a small target."""
    mask = decode(row["segmentation"])
    rgb = np.asarray(image.convert("RGB")).copy()
    fixture = row["entity"] in {"display_table", "display_cabinet", "counter", "other_shelf"}
    if not fixture:
        rgb[~mask] = (100, 100, 100)
    x, y, w, h = row["bbox_xywh"]
    margin = max(12 if fixture else 2, round(max(w, h) * (0.12 if fixture else 0.04)))
    crop = Image.fromarray(rgb).crop(
        (
            max(0, x - margin),
            max(0, y - margin),
            min(image.width, x + w + margin),
            min(image.height, y + h + margin),
        )
    )
    panel = Image.new("RGB", (448, 448), (100, 100, 100))
    resized = ImageOps.contain(crop, (448, 420))
    panel.paste(resized, ((448 - resized.width) // 2, 28 + (420 - resized.height) // 2))
    draw = ImageDraw.Draw(panel)
    draw.text((8, 8), "FURNITURE CROP" if fixture else "ISOLATED TARGET", fill="white")
    return panel


def parse_answer(raw: str) -> dict:
    label = raw.strip().strip('`"').strip()
    if label in (*ENTITY_NAMES, "unknown"):
        return {
            "entity": label,
            "reason": "Local VLM classification of the supplied target image",
            "raw": raw,
            "response_format": "class_only",
        }
    try:
        start, end = raw.index("{"), raw.rindex("}") + 1
        answer = json.loads(raw[start:end])
        if answer.get("entity") not in (*ENTITY_NAMES, "unknown"):
            raise ValueError("class outside vocabulary")
        if not isinstance(answer.get("reason"), str) or not answer["reason"].strip():
            raise ValueError("missing visible evidence")
        return {"entity": answer["entity"], "reason": answer["reason"], "raw": raw}
    except (ValueError, TypeError, AttributeError):
        return {
            "entity": "unknown",
            "reason": "invalid or incomplete model response",
            "raw": raw,
        }


def constrain_decision(group: list[dict], decision: dict) -> dict:
    """Do not let a tiny product mask become furniture based on nearby context."""
    surfaces = {row["entity"] for row in group}
    if (
        len(surfaces) == 1
        and surfaces <= {"floor", "wall", "ceiling"}
        and any(not row.get("reasons") for row in group)
    ):
        entity = next(iter(surfaces))
        return {
            **decision,
            "entity": entity,
            "model_entity": decision["entity"],
            "basis": "source_semantic_preservation",
            "reason": "Preserved source semantic surface with foreground occlusion trimmed",
        }
    fixtures = {"display_table", "display_cabinet", "counter", "other_shelf"}
    goods = {
        "laptop",
        "phone",
        "tablet",
        "boxed_stock",
        "cardboard_box",
        "speaker",
        "other_monitor",
    }
    allowed = {"unknown"}
    for row in group:
        entity = row["entity"]
        if entity in fixtures:
            allowed.update(fixtures - {"other_shelf"})
        elif entity in goods:
            allowed.update({r["entity"] for r in group} & (goods - {"other_monitor"}))
        elif entity in {"door", "glass_panel"}:
            allowed.update(fixtures - {"other_shelf"})
            allowed.update({"door", "glass_panel"})
        else:
            allowed.add(entity)
    if decision["entity"] in allowed:
        return decision
    return {
        **decision,
        "entity": "unknown",
        "model_entity": decision["entity"],
        "constraint": "unsupported cross-family change",
        "reason": "Source instruments disagree on object family: " + decision["reason"],
    }


def validate_visual_decisions(document: dict, frames: list[dict], annotations: dict) -> dict:
    """Bind assistant visual classifications to exact source images and candidate IDs."""
    if document.get("reviewer_kind") != "ai":
        raise ValueError("visual decisions must identify AI provenance")
    index = {f["id"]: f for f in frames}
    result: dict[str, dict[str, dict]] = {}
    for item in document["frames"]:
        frame = index.get(item["frame_id"])
        if frame is None or any(
            item.get(key) != frame[key] for key in ("image_sha256", "annotation_sha256")
        ):
            raise ValueError("visual decision source hash mismatch")
        annotation = annotations[frame["id"]]
        rows = {
            r["id"]: r for r in annotation["entities"] + annotation["unresolved_candidates"]
        }
        target = result.setdefault(frame["id"], {})
        for decision in item["decisions"]:
            identity = decision["candidate_id"]
            if identity not in rows or identity in target:
                raise ValueError("unknown or duplicate visual candidate")
            if decision["entity"] not in ENTITY_NAMES or not decision.get("reason"):
                raise ValueError("invalid visual classification")
            target[identity] = decision
    return result


def apply_visual_decision(group: list[dict], decision: dict, overrides: dict) -> dict:
    matches = [overrides[row["id"]] for row in group if row["id"] in overrides]
    if not matches:
        return decision
    if len({item["entity"] for item in matches}) != 1:
        raise ValueError("conflicting visual decisions in one group")
    return {
        **decision,
        "entity": matches[0]["entity"],
        "reason": matches[0]["reason"],
        "model_entity": decision.get("model_entity", decision["entity"]),
        "basis": "assistant_visual_review",
        "source_candidate_ids": [item["candidate_id"] for item in matches],
    }


def needs_isolated_recheck(group: list[dict], decision: dict) -> bool:
    fixtures = {"display_table", "display_cabinet", "counter", "other_shelf"}
    if decision.get("basis") == "assistant_visual_review":
        return False
    return decision["entity"] in fixtures and not {row["entity"] for row in group} <= fixtures


def revise_group_decisions(raw: dict, reviews: list[dict]) -> list[dict]:
    """Apply explicit AI inspection decisions; caller verifies the raw file hash."""
    decisions = deepcopy(raw["decisions"])
    if len(decisions) != len(raw["groups"]):
        raise ValueError("cannot review incomplete inference")
    seen = set()
    for review in reviews:
        index = review["group"]
        if not isinstance(index, int) or not 0 <= index < len(decisions) or index in seen:
            raise ValueError("invalid or duplicate review group")
        if review["entity"] not in (*ENTITY_NAMES, "unknown") or not review.get("reason"):
            raise ValueError("invalid group classification")
        seen.add(index)
        previous = decisions[index]
        decisions[index] = {
            **previous,
            "entity": review["entity"],
            "reason": review["reason"],
            "basis": review.get("review_source", "assistant_visual_group_review"),
            "review_decision": review,
            "previous_decision": previous,
        }
    return decisions


def layer(entity: str) -> int:
    if entity == "person":
        return 5
    if entity in {"laptop", "phone", "tablet", "boxed_stock", "cardboard_box", "speaker"}:
        return 4
    if entity in {
        "display_table",
        "display_cabinet",
        "counter",
        "chair",
        "poster",
        "fire_equipment",
    }:
        return 3
    if entity in {"door", "glass_panel", "column"}:
        return 2
    return 1


def reannotate(groups: list[list[dict]], decisions: list[dict], shape: tuple[int, int]) -> dict:
    if len(groups) != len(decisions):
        raise ValueError("decision/group count mismatch")
    candidates, unknown = [], []
    raw_masks = []
    for index, (group, decision) in enumerate(zip(groups, decisions, strict=True)):
        entity = decision["entity"]
        if entity not in (*ENTITY_NAMES, "unknown") or not decision.get("reason"):
            raise ValueError("invalid AI decision")
        representative = group[0]
        if entity == "unknown":
            row = deepcopy(representative)
            row.update(
                id=f"vlm-unknown-{index:04d}",
                reasons=["vision_language_model_abstention"],
                possible_entities=sorted(
                    {
                        "display_cabinet" if r["entity"] == "other_shelf" else r["entity"]
                        for r in group
                    }
                ),
                ai_review=decision,
            )
            unknown.append(row)
            continue
        mask = decode(representative["segmentation"])
        raw_masks.append({"group": index, "entity": entity, "original_area": int(mask.sum())})
        candidates.append(
            Candidate(
                entity,
                mask,
                representative["score"],
                [
                    f"AI review group {index}: {decision['reason']}",
                    *(
                        prompt
                        for row in group
                        if row["entity"] == entity
                        for prompt in row["prompts"]
                    ),
                ],
            )
        )
    # Only remove pixels supported by a more foreground, positively classified mask.
    # Same-layer conflicts remain unresolved. Keep counts plus full raw source masks.
    foreground = {level: np.zeros(shape, dtype=bool) for level in range(1, 6)}
    for candidate in candidates:
        foreground[layer(candidate.entity)] |= candidate.mask
    visible = []
    for candidate, record in zip(candidates, raw_masks, strict=True):
        mask = candidate.mask.copy()
        for level in range(layer(candidate.entity) + 1, 6):
            mask &= ~foreground[level]
        record["visible_area"] = int(mask.sum())
        if mask.any():
            visible.append(
                Candidate(candidate.entity, mask, candidate.score, candidate.prompts)
            )
    data = annotate(visible, shape)
    data["unresolved_candidates"].extend(unknown)
    data["coverage"] = coverage(data["entities"], data["unresolved_candidates"])
    data["counts_by_view"] = dict(
        Counter(view for row in data["entities"] for view in entity_views(row))
    )
    data["ai_reclassification"] = {
        "model": MODEL,
        "revision": REVISION,
        "groups": len(groups),
        "unknown_groups": len(unknown),
        "visible_pixel_changes": raw_masks,
    }
    return data


class LocalReviewer:
    def __init__(self, device="cuda"):
        import torch
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        self.device = device
        self.processor = AutoProcessor.from_pretrained(
            MODEL,
            revision=REVISION,
            local_files_only=True,
            min_pixels=128 * 28 * 28,
            max_pixels=512 * 28 * 28,
        )
        self.processor.tokenizer.padding_side = "left"
        # Transformers' decorated generation protocol is narrower than its runtime
        # Qwen class. Keep this optional third-party API behind the tested adapter.
        self.model: Any = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            MODEL,
            revision=REVISION,
            local_files_only=True,
            dtype=torch.bfloat16 if device == "cuda" else torch.float32,
            attn_implementation="sdpa",
        )
        torch.nn.Module.to(self.model, device)
        self.model.eval()

    def classify(
        self, panels: list[Image.Image], prompts: list[str] | None = None
    ) -> list[dict]:
        import torch

        prompts = prompts if prompts is not None else [PROMPT] * len(panels)
        if len(prompts) != len(panels):
            raise ValueError("prompt/image count mismatch")
        texts = []
        for prompt in prompts:
            messages = [
                {
                    "role": "system",
                    "content": "Classify the target image. Reply only with one class name.",
                },
                {
                    "role": "user",
                    "content": [{"type": "image"}, {"type": "text", "text": prompt}],
                },
            ]
            texts.append(
                self.processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            )
        inputs = self.processor(
            text=texts, images=panels, padding=True, return_tensors="pt"
        ).to(self.device)
        with torch.inference_mode():
            output = self.model.generate(**inputs, max_new_tokens=16, do_sample=False)
        answers = self.processor.batch_decode(
            output[:, inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )
        return [parse_answer(answer) for answer in answers]
