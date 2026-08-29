"""Which channel of a detection head means `person`, and the refusal when none does.

Lifted out of `scripts/track_review.py` on 2026-08-28 when a second script needed the
same refusal. `tests/test_scripts_are_not_libraries.py` states the rule and the reason:
code two scripts share sits **outside the wheel, outside the type ratchet and outside the
coverage floor**, so the thing every caller depends on is the thing nothing checks. A
refusal is the worst possible candidate for that, because the failure it exists to catch
is silent — a checkpoint whose label 0 is shelf stock produces a complete, well-formed
`tracks.json` and a review sheet full of crops, and nothing downstream can tell.

`track_review` re-exports both names, so its own tests and `scripts/offline_tracks.py`
reach them exactly where they always did.
"""

from __future__ import annotations

from syncai_hydranet.data.coco_subsets import COCO_NAMES
from syncai_hydranet.data.label_maps_retail_security import get_det_vocab

PERSON = COCO_NAMES.index("person")


def head_class_names(cfg: dict) -> tuple[str, ...] | None:
    """What the detection head's channels are called, or `None` if the config cannot say.

    Three sources, in the order a config states them: the head's own `classes` list, a
    `det_vocab` shared with the datasets, and -- only for a head of exactly 80 channels --
    the COCO space, which this repo's 80-class configs name by omission rather than by
    listing. `None` means "unknown", which the caller must treat as a refusal: a head whose
    channels have no names is exactly the head nobody can check.
    """
    head = cfg.get("model", {}).get("heads", {}).get("detection", {})
    if head.get("classes"):
        return tuple(head["classes"])
    datasets = cfg.get("data", {}).get("datasets", [])
    vocabs = {d["det_vocab"] for d in datasets if d.get("det_vocab")}
    if len(vocabs) == 1:
        return tuple(get_det_vocab(next(iter(vocabs))).classes)
    if head.get("num_classes") == len(COCO_NAMES):
        return tuple(COCO_NAMES)
    return None


def check_person_class_space(n_classes: int, names: tuple[str, ...] | None = None) -> None:
    """Refuse a detection head in which label `PERSON` does not mean a person.

    Raises rather than warns. A warning is the wrong instrument here: the failure produces
    a complete, well-formed `tracks.json` and a review sheet full of crops, so nothing
    downstream can tell the difference and the reviewer is looking at the crops rather
    than at the scrollback.

    `names` defaults to the COCO space so the historical single-argument call still means
    what it did: "this head has `n_classes` channels and they are COCO's".
    """
    names = tuple(COCO_NAMES) if names is None else names
    if len(names) != n_classes:
        raise SystemExit(
            f"The config names {len(names)} detection classes and the checkpoint's head "
            f"has {n_classes} channels. One of them is not the model you meant; the "
            f"tracks would be labelled by a taxonomy the weights were not trained on."
        )
    if len(names) <= PERSON or names[PERSON] != "person":
        got = names[PERSON] if len(names) > PERSON else "nothing"
        raise SystemExit(
            f"In this detection head label {PERSON} is {got!r}, not 'person'. Nothing "
            f"would error; the tracks would just not be people. Its classes are "
            f"{', '.join(names)} -- use a checkpoint whose head detects people."
        )
