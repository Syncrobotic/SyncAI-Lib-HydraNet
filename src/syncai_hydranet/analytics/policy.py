"""A store's policy: the thresholds the event layer fires on, as a file with a provenance.

`events.zones_from_camera` carries geometry only, on purpose -- every policy field comes
back at its default, so a commissioned camera on its own fires nothing. Until this module
existed the policy was three argument defaults in `scripts/step6_events.py`, which is a
place a store manager cannot edit and a serving process cannot read. This is the file
both read, so the offline log and the live one fire on the same numbers.

The shape follows `serving/camera.py`'s threshold book: a schema tag, a required
``provenance`` sentence saying who set these values and on what grounds, and a refusal
for every silent failure the file could otherwise carry. A policy with no stated basis
cannot be told apart from numbers tuned until the alerts stopped, which is docs/PLAN.md
section 9.2's failure one artefact later.

What a rule may set is exactly what `events.Zone` carries -- `loiter_seconds`,
`max_occupancy`, `restricted` -- keyed by the camera file's zone *kind*, because a kind is
the only thing a policy can name before the camera is commissioned: a store has tills and
display tables, and which polygon is which is `camera.json`'s to say.

`open_hours` has one consumer today, `scripts/step6_reread.py`, which splits a log into
trading and closed time. An after-hours rule (any person in a shut shop is an event) is
deliberately not a field yet: the event layer has no producer for it, and a policy field
nothing reads is the "guard that half-covers" docs/PLAN.md section 2.1g records.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import yaml

from syncai_hydranet.analytics.events import Zone, zones_from_camera
from syncai_hydranet.geometry.camera_json import ZONE_KINDS

POLICY_SCHEMA = "store_policy/1"

#: The kinds a rule may key on: every polygon kind. `entrance_line` is a line and reaches
#: the event layer through `events.counting_lines`, not through a `Zone`.
RULE_KINDS = frozenset(ZONE_KINDS - {"entrance_line"})

#: The fields a rule may set, and the `events.Zone` field each one lands on.
RULE_FIELDS = ("loiter_seconds", "max_occupancy", "restricted")


@dataclass(frozen=True)
class ZoneRule:
    """What the store says about every zone of one kind."""

    kind: str
    loiter_seconds: float | None = None
    max_occupancy: int | None = None
    restricted: bool = False

    def __post_init__(self) -> None:
        if self.kind not in RULE_KINDS:
            raise ValueError(
                f"zone kind {self.kind!r} is not one of {sorted(RULE_KINDS)}; a rule on an "
                "unknown kind would match no polygon and never fire"
            )
        if self.loiter_seconds is None and self.max_occupancy is None and not self.restricted:
            raise ValueError(f"the rule for {self.kind!r} sets nothing, which is not a rule")
        if self.loiter_seconds is not None and not self.loiter_seconds > 0:
            raise ValueError(
                f"{self.kind}: loiter_seconds must be > 0, got {self.loiter_seconds}"
            )
        if self.max_occupancy is not None and not self.max_occupancy >= 1:
            raise ValueError(
                f"{self.kind}: max_occupancy must be >= 1, got {self.max_occupancy}"
            )


@dataclass(frozen=True)
class StorePolicy:
    """One store's rules. Frozen: a policy is loaded, applied, and recorded, never edited."""

    store: str
    provenance: str
    timezone_hours: float
    open_hours: tuple[int, int]
    min_seconds: float
    rules: dict[str, ZoneRule]
    source: str  # the file this came from, for the report's settings block

    @property
    def tz(self) -> timezone:
        return timezone(timedelta(hours=self.timezone_hours))

    def is_open(self, when: datetime) -> bool:
        """Trading hours by the store's clock. A naive datetime is refused."""
        if when.tzinfo is None:
            raise ValueError("is_open needs an aware datetime; the corpus stamps UTC")
        hour = when.astimezone(self.tz).hour
        lo, hi = self.open_hours
        return lo <= hour < hi

    def zones_for(self, cam_file: Any) -> list[Zone]:
        """The camera's polygons with this store's thresholds on them, by kind.

        A kind with no rule keeps its defaults and so never fires, which is the
        camera-only behaviour and is correct: a store that has said nothing about its
        stockroom door has not asked to be told about it.
        """
        kind_of = {z.name: z.kind for z in cam_file.zones}
        out = []
        for z in zones_from_camera(cam_file):
            rule = self.rules.get(kind_of[z.name])
            if rule is None:
                out.append(z)
                continue
            out.append(
                replace(
                    z,
                    loiter_seconds=rule.loiter_seconds,
                    max_occupancy=rule.max_occupancy,
                    restricted=rule.restricted,
                )
            )
        return out


def load_policy(path: str | Path) -> StorePolicy:
    """Read a policy file, refusing everything that would fail silently downstream."""
    p = Path(path)
    raw = yaml.safe_load(p.read_text()) or {}
    if raw.get("schema") != POLICY_SCHEMA:
        raise ValueError(f"{p}: unexpected policy schema {raw.get('schema')!r}")
    provenance = str(raw.get("provenance", "")).strip()
    if not provenance:
        raise ValueError(
            f"{p}: no `provenance`. A policy with no stated basis cannot be told apart "
            "from numbers tuned until the alerts stopped."
        )
    store = str(raw.get("store", "")).strip()
    if not store:
        raise ValueError(f"{p}: no `store`")
    hours = raw.get("open_hours")
    if (
        not isinstance(hours, list | tuple)
        or len(hours) != 2
        or not all(isinstance(h, int) and 0 <= h <= 24 for h in hours)
        or not hours[0] < hours[1]
    ):
        raise ValueError(f"{p}: open_hours must be [start, end] in whole hours, got {hours!r}")
    min_seconds = float(raw.get("min_seconds", 1.0))
    if not min_seconds >= 0:
        raise ValueError(f"{p}: min_seconds must be >= 0, got {min_seconds}")
    rules_raw = raw.get("zones") or {}
    if not isinstance(rules_raw, dict):
        raise ValueError(f"{p}: `zones` must map a zone kind to its rule")
    rules: dict[str, ZoneRule] = {}
    for kind, fields in rules_raw.items():
        if not isinstance(fields, dict):
            raise ValueError(f"{p}: rule for {kind!r} must be a mapping, got {fields!r}")
        unknown = set(fields) - set(RULE_FIELDS)
        if unknown:
            raise ValueError(
                f"{p}: rule for {kind!r} sets {sorted(unknown)}, which the event layer does "
                f"not read; a field nothing reads is a rule that silently does nothing"
            )
        rules[str(kind)] = ZoneRule(kind=str(kind), **fields)
    return StorePolicy(
        store=store,
        provenance=provenance,
        timezone_hours=float(raw.get("timezone_hours", 0)),
        open_hours=(int(hours[0]), int(hours[1])),
        min_seconds=min_seconds,
        rules=rules,
        source=str(p),
    )
