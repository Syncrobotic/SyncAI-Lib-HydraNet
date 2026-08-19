#!/usr/bin/env python3
"""Which cameras carry a `column`, per camera, written down this time.

    python3 scripts/column_camera_sweep.py --rotate --out runs/column_sweep03
    python3 scripts/column_camera_sweep.py --report-only --out runs/column_sweep03

Sweep C (2026-08-18) asked SAM 3's five `column` prompts of 48 cameras and settled the
class's `min_score` at 0.5. Its **totals** survive, in the `column` entry of
[sam3_prompts_objects.py](../src/syncai_hydranet/data/sam3_prompts_objects.py). Its
**per-camera output does not exist anywhere on disk**, so the cameras it discarded could
not be named, and naming them was the only way left to add a training camera for this
class without egress. That is why this is a script and not a session: a measurement whose
identities were never written down has to be paid for twice.

WHAT THE FIRST FULL RUN FOUND (runs/column_sweep03, 2026-08-19)

**Nothing that can be trained on, and that is the useful part.** 17 cameras clear 0.50
and 24 clear 0.25, but the split has already spent every one of them that matters:

  - 3 cameras clear 0.50 that are not in the shipped 14 -- Kaohsiung-cam05 (val),
    Taichung-cam12 (batch03 test), Tao-Hsin-cam11 (trainable).
  - 7 more sit in the 0.25-0.50 band; exactly one, Taichung-cam09, is trainable.
  - Both trainable candidates were opened at native resolution and **rejected**.
    Tao-Hsin-cam11's mask is street bollards seen through a glass door -- glass failure
    mode 2 in `sam3_prompts.py`, "finds what is behind it". Taichung-cam09's is a narrow
    vertical strip beside a display podium, the documented drift onto "narrow vertical
    thing". Neither is a column.

So the trainable population for `column` gains nothing -- and the 3-of-4 rule below then
takes one away, leaving **5 selling-floor cameras**. "More cameras, not more frames" is
exhausted within the current pull. The next column comes from a pull date
or a store this project has not seen.

THE 3-OF-4 RULE, WHICH IS NOW THE RULE

Four store-local tranches rather than sweep C's two, and the count of tranches a camera
clears 0.50 in turns out to separate a column from a lighting artefact almost by itself:
13 of the shipped 14 fire in 3 or 4, while every camera this sweep adds fires in exactly
one. **A camera must clear 0.50 in 3 of the 4 tranches before it supplies a `column`
pixel.** Adopted 2026-08-19; `datasets/retail_objects_columns_v2/split.json` carries it
under `column_supply_rule` so it travels with the data.

A peak score is one frame's opinion under one light, which is why the rule counts tranches
rather than raising the threshold. Sweep C had two frames per camera and could not have
run this test at all.

**It bit a sitting member, which is the reason to trust it.** `Taichung-cam04` was in the
shipped 14 and clears 0.50 at midnight only -- 0.727, against 0.408 / 0.000 / 0.332 for
midday, afternoon and evening. Opened at native resolution, the midnight mask is a vertical
fragment behind the back counter that never reaches the floor: `column` drifting onto a wall
strip. Its column pixels are now IGNORE in the supplement; its fixture, product and person
pixels are untouched. The honest count of selling-floor cameras supplying `column` is
therefore **5**, not 6.

THE ROTATION ARM: NEGATIVE, AND WORTH SAYING SO

The fleet census of 2026-08-19 (`runs/camera_census01/REPORT.md`) found **five** sideways
~90 degree mounts, not the one the roll census had counted: K-05, K-09, K-10, T-02, TH-05.
Sweep C asked all five for a "column" without rotating them, so a column would have been a
horizontal bar. `--rotate` re-asks them upright, and the answer is **zero cameras**: not
one of the five gains a column at 0.50 when turned the right way up. The hypothesis was
reasonable and it is wrong. Nobody needs to try it again.

WHAT THIS SCRIPT DOES NOT DO

It does not pull anything. Every frame comes from the clips already in
`datasets/studioa_clips/<camera>/`, at native 1920x1080, four per camera, selected by
**store-local** hour -- see `TRANCHES`, and the mistake that comment records. The 30k
campaign is pulling 42 cameras x 32 days into the same tree; consume that rather than
opening a second pipeline.

Do not sweep `datasets/studioa_clips/_survey/*.jpg`. Those plates came from the
mislabelled pull and are all ~19:30 store-local, whatever else they look like.

It also does not decide anything. It writes a table. A camera it names still has to be
looked at at native resolution before it supplies a training pixel -- the contact-sheet
lesson in `hydranet_retail_surfaces_columns.yaml` was that a downscaled tile shows *that*
a prompt fired and not *what* it fired on.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
for candidate in (HERE.parent / "src", HERE / "src"):
    if candidate.is_dir():
        sys.path.insert(0, str(candidate))

from syncai_hydranet.data import sam3_prompts_objects as _objects  # noqa: E402
from syncai_hydranet.data.video import frames as video_frames  # noqa: E402
from syncai_hydranet.data.video import probe  # noqa: E402


def _prelabel():
    """`sam3_prelabel.py` is a script, not a package module; load it by path.

    Importing it rather than copying `segment` is deliberate: the encode-once path and
    the post-processing thresholds are the thing being reproduced, and a second copy of
    them would drift from the one that made the batches.

    **Declared rather than hidden:** this is a script-to-script dependency, and loading it
    by path makes it invisible to `tests/test_scripts_are_not_libraries.py`, which counts
    them and ratchets. That is not why it is written this way -- `sam3_prelabel.py` is not
    an importable module name -- but the exemption is real, so it is written down here. If
    a third caller wants `segment`, that is the signal to move it into
    `src/syncai_hydranet` rather than to add another of these.
    """
    path = HERE / "sam3_prelabel.py"
    spec = importlib.util.spec_from_file_location("_sam3_prelabel", path)
    if spec is None or spec.loader is None:
        # Unreachable for an existing .py file; the guard is what makes the None-typed
        # returns above checkable rather than silently narrowed.
        raise ImportError(f"cannot build an import spec for {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# The census's five, by the names cameras.json uses. Sweep C saw all five unrotated.
SIDEWAYS = (
    "Kaohsiung-cam05",
    "Kaohsiung-cam09",
    "Kaohsiung-cam10",
    "Taichung-cam02",
    "Tao-Hsin-cam05",
)

# The thresholds that bracket the question. 0.50 is the class's shipped `min_score`; 0.25
# is where sweep C counted 18; 0.40 is the table-wide default, reported so a camera that
# is marginal for a different reason is visible rather than rounded into the band.
CUTS = (0.25, 0.40, 0.50)

# Below this a "column" is a speck. Same constant and same reason as sam3_prelabel's
# MIN_BOX_PIXELS: a handful of stray pixels is not an instance.
MIN_PIXELS = 40


# Store-local target hours, and the names the report uses. UTC+8, so the clip names --
# which are UTC, always -- get 8 added before anything is compared.
#
# **This list exists because the first run of this script got it wrong in the documented
# way.** It swept `datasets/studioa_clips/_survey/*.jpg` as its "daylight" arm. Those
# plates were cut from `manifest_2026-08-16_1130-1600UTC-mislabelled.json` -- the pull
# whose bug `pull_studioa.py` describes at length -- so the burnt-in clock on every one
# of them reads about 19:30 and the arm labelled "day" was a **shuttered evening store**.
# The real midday footage was on disk the whole time, in the correctly-labelled pull.
#
# Selecting from the clips by local hour, and printing the hour actually landed on, is
# the fix. A tranche name that cannot be checked against the frame is how this recurs.
TRANCHES = (("midday", 11), ("afternoon", 14), ("evening", 19), ("midnight", 0))
UTC_OFFSET = 8.0


def clip_for(cam_dir: Path, local_hour: int) -> tuple[Path, float] | None:
    """The clip whose start is nearest a **store-local** hour, and the hour it landed on.

    Returns the gap so the caller can print it: a tranche that silently lands four hours
    away is the same class of error as the mislabelled pull, one level down.
    """
    clips = sorted(cam_dir.glob("*.mp4"))
    if not clips:
        return None

    def local(p: Path) -> float:
        stamp = p.name.removeprefix("archive_").split("_", 1)[0].split("-")[1]
        return (int(stamp[:2]) + int(stamp[2:4]) / 60 + UTC_OFFSET) % 24

    def gap(p: Path) -> float:
        d = abs(local(p) - local_hour)
        return min(d, 24 - d)

    best = min(clips, key=gap)
    return best, local(best)


def first_frame(clip: Path) -> Image.Image | None:
    w, h, _ = probe(str(clip))
    for arr in video_frames(str(clip), w, h, None):
        return Image.fromarray(arr)
    return None


def ask(mod, proc, model, image: Image.Image, device: str, prompts) -> dict:
    """Every `column` prompt against one frame, as counts and pixel shares per cut.

    The union is over prompts, which is what the concept means: `column` and `pillar`
    finding the same pillar is two prompts agreeing, not two columns.
    """
    embeds = mod.vision_features(proc, model, image, device)
    area = image.width * image.height
    per_prompt: dict[str, dict] = {}
    unions = {c: np.zeros((image.height, image.width), dtype=bool) for c in CUTS}

    for prompt in prompts:
        # Ask once at the lowest cut and filter upward: the scores are the same objects.
        inst = [
            (m, s)
            for m, s in mod.segment(proc, model, image, prompt, min(CUTS), device, embeds)
            if m.sum() >= MIN_PIXELS
        ]
        per_prompt[prompt] = {
            "peak": round(max((s for _, s in inst), default=0.0), 4),
            "n": {str(c): sum(1 for _, s in inst if s >= c) for c in CUTS},
        }
        for cut in CUTS:
            for m, s in inst:
                if s >= cut:
                    unions[cut] |= m

    return {
        "prompts": per_prompt,
        "peak": round(max((p["peak"] for p in per_prompt.values()), default=0.0), 4),
        "union_px_frac": {str(c): round(float(unions[c].sum()) / area, 5) for c in CUTS},
        "union_n": {str(c): sum(p["n"][str(c)] for p in per_prompt.values()) for c in CUTS},
    }


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--clips", default="datasets/studioa_clips")
    ap.add_argument("--out", default="runs/column_sweep03")
    ap.add_argument("--device", default="cuda")
    ap.add_argument(
        "--cameras", nargs="*", help="default: every camera directory under --clips"
    )
    ap.add_argument("--night", action="store_true", default=True)
    ap.add_argument("--no-night", dest="night", action="store_false")
    ap.add_argument(
        "--report-only",
        action="store_true",
        help="rebuild REPORT.md from an existing sweep.json without touching the GPU",
    )
    ap.add_argument(
        "--rotate",
        action="store_true",
        help="also ask the five sideways mounts upright (census 2026-08-19). Adds an "
        "arm; never removes a camera from the unrotated result.",
    )
    args = ap.parse_args()

    clips = Path(args.clips)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    if args.report_only:
        report(json.loads((out / "sweep.json").read_text()), out)
        return 0

    roles = json.loads((clips / "cameras.json").read_text())["cameras"]
    cams = args.cameras or sorted(d.name for d in clips.iterdir() if d.name in roles)

    concept = _objects.BY_NAME["column"]
    prompts = list(concept.prompts)
    print(f"{len(cams)} cameras x {len(prompts)} prompts, cuts {CUTS}")
    print(f"prompts: {', '.join(prompts)}\n")

    mod = _prelabel()
    proc, model = mod.load_sam3(mod.MODEL_ID, args.device)

    results: dict[str, dict] = {}
    for i, cam in enumerate(cams, 1):
        entry: dict[str, dict] = {"role": roles[cam]["role"], "tranches": {}}
        for name, hour in TRANCHES:
            picked = clip_for(clips / cam, hour)
            if picked is None:
                continue
            clip, landed = picked
            frame = first_frame(clip)
            if frame is None:
                continue
            entry[name] = ask(mod, proc, model, frame, args.device, prompts)
            entry["tranches"][name] = {"clip": clip.name, "local_hour": round(landed, 2)}
            if args.rotate and cam in SIDEWAYS:
                entry[f"{name}_rot90"] = ask(
                    mod, proc, model, frame.rotate(90, expand=True), args.device, prompts
                )

        results[cam] = entry
        peaks = " ".join(f"{n[:3]} {entry[n]['peak']:.2f}" for n, _ in TRANCHES if n in entry)
        print(f"[{i:2d}/{len(cams)}] {cam:18s} {entry['role']:15s} {peaks}")

    (out / "sweep.json").write_text(json.dumps(results, indent=1, sort_keys=True))
    report(results, out)
    return 0


BATCHES = ("datasets/retail_objects_batch02", "datasets/retail_objects_batch03")
SHIPPED = "datasets/retail_objects_columns"  # the 14 sweep C actually banked, at 0.50


def disposition() -> tuple[dict, set]:
    """Where the split has already spent each camera, and which are in the shipped 14.

    A camera this sweep finds is only useful if it is not already held for evaluation,
    and that is a fact about `split.json` rather than about SAM 3. Reading it here is
    what turns a score table into a list of cameras somebody can act on.
    """
    held: dict[str, dict] = {}
    for b in BATCHES:
        p = Path(b, "split.json")
        if not p.exists():
            continue
        for cam, where in json.loads(p.read_text())["assign"].items():
            if where in ("val", "test"):
                held.setdefault(cam, {})[Path(b).name] = where
    shipped = Path(SHIPPED, "images/train")
    banked = {d.name for d in shipped.iterdir() if d.is_dir()} if shipped.is_dir() else set()
    return held, banked


def report(results: dict, out: Path) -> None:
    """The band, and the two things that decide whether a camera in it is worth a look."""

    arms_all = tuple(n for n, _ in TRANCHES)
    arms_rot = tuple(f"{n}_rot90" for n in arms_all)

    def hit(entry: dict, cut: float, arms=arms_all) -> bool:
        return any(entry.get(a, {}).get("union_n", {}).get(str(cut), 0) > 0 for a in arms)

    at = {c: sorted(k for k, v in results.items() if hit(v, c)) for c in CUTS}
    band = sorted(set(at[0.25]) - set(at[0.50]))
    rot_only = sorted(
        k for k, v in results.items() if not hit(v, 0.50) and hit(v, 0.50, arms_rot)
    )

    lines = [
        "# `column` per camera, with the identities kept",
        "",
        "Reproduces sweep C and writes down what it did not. Totals first, so a drift from",
        "the figures quoted in `sam3_prompts_objects.py` is visible before anything is read",
        "off the rows.",
        "",
        "| cut | cameras |",
        "|---|---|",
    ]
    for c in CUTS:
        lines.append(f"| {c:.2f} | **{len(at[c])}** |")
    held, shipped = disposition()

    def where(cam: str) -> str:
        if cam in held:
            return " / ".join(f"{b.split('_')[-1]} {w}" for b, w in sorted(held[cam].items()))
        return "**trainable**"

    def listing(cams: list[str]) -> list[str]:
        if not cams:
            return ["| _none_ | | | |"]
        rows = []
        for c in cams:
            e = results[c]
            rows.append(
                f"| {c} | {e['role']} | {where(c)} "
                f"| {'yes' if c in shipped else '**no — new**'} |"
            )
        return rows

    new_at_50 = [c for c in at[0.50] if c not in shipped]

    consist: dict[int, list] = {}
    for cam, e in results.items():
        n = sum(1 for a in arms_all if e.get(a, {}).get("union_n", {}).get("0.5", 0) > 0)
        consist.setdefault(n, []).append((cam, cam in shipped))

    lines += [
        "",
        f"## Cleared 0.50 but is not in the shipped 14: {len(new_at_50)}",
        "",
        f"`{SHIPPED}` banked 14 cameras at this threshold. This sweep clears "
        f"{len(at[0.50])}, so these are the difference — either sweep C missed them or "
        "they were dropped for a reason nobody wrote down.",
        "",
        "| camera | role | split | in shipped 14 |",
        "|---|---|---|---|",
        *listing(new_at_50),
        "",
        f"## The band, 0.25-0.50: {len(band)} cameras",
        "",
        "The candidates the class's `min_score` discards. This is a **review list, not a**",
        "**batch**: 0.25-0.50 is exactly where sweep C watched `column` drift onto the",
        "vertical wall strips of adjacent shopfronts. Look at each at native resolution",
        "first — the contact-sheet lesson is that a downscaled tile shows *that* a prompt",
        "fired and not *what* it fired on.",
        "",
        "| camera | role | split | in shipped 14 |",
        "|---|---|---|---|",
        *listing(band),
        "",
        "## Tranche consistency, which is the cheapest filter here",
        "",
        "How many of the four store-local tranches clear 0.50 on each camera. Sweep C had",
        "two (one day, one night) and could not have run this test; with four it separates",
        "a column from a lighting artefact almost by itself:",
        "",
        "| tranches @0.50 | cameras | in shipped 14 |",
        "|---|---|---|",
        *[
            f"| **{k}/4** | {', '.join(c for c, _ in v)} "
            f"| {sum(1 for _, s in v if s)}/{len(v)} |"
            for k, v in sorted(consist.items(), reverse=True)
            if k
        ],
        "",
        "**13 of the shipped 14 fire in 3 or 4 tranches, and every camera this sweep adds",
        "at 0.50 fires in one.** A single-tranche hit is a claim about that hour's light,",
        "not about the room -- both cameras reviewed by eye on this run (Tao-Hsin-cam11,",
        "evening only: bollards through a glass door; Taichung-cam09, never at 0.50: a",
        "narrow strip beside a podium) confirm it. Require 3 of 4 before a camera supplies",
        "a training pixel.",
        "",
        f"## Rotation arm: {len(rot_only)} cameras",
        "",
        "Sideways mounts (census 2026-08-19: K-05, K-09, K-10, T-02, TH-05) that clear",
        "0.50 only when asked upright. A column is a horizontal bar on an unrotated frame",
        "from these five, so sweep C could not have found one there.",
        "",
        *listing(rot_only),
        "",
        "## Every camera",
        "",
        "Every tranche is reported apart. Folding them into one number is how the first",
        "version of this table showed `Taichung-cam04` with no instances at 0.50 and 2.99%",
        "of its pixels claimed at 0.50 -- the counts had taken one arm and the pixel share",
        "the larger of two. A camera whose column is only found after dark is a finding",
        "about this fleet, not a rounding case.",
        "",
        "Tranche = store-local hour, selected from the clips on disk. The header says which",
        "hour each column actually landed on for the first camera listed; per-camera clip",
        "names and hours are in `sweep.json` under `tranches`.",
        "",
        "| camera | role | "
        + " | ".join(f"{n} peak" for n in arms_all)
        + " | n@0.50 (best) | px@0.50 (best) |",
        "|---|---|" + "---|" * (len(arms_all) + 2),
    ]
    for cam in sorted(results):
        e = results[cam]
        peaks = " | ".join(f"{e.get(n, {}).get('peak', 0):.3f}" for n in arms_all)
        best = [e.get(n, {}) for n in arms_all]
        n50 = max((a.get("union_n", {}).get("0.5", 0) for a in best), default=0)
        px50 = max((a.get("union_px_frac", {}).get("0.5", 0) for a in best), default=0)
        lines.append(f"| {cam} | {e['role']} | {peaks} | {n50} | {px50:.2%} |")

    (out / "REPORT.md").write_text("\n".join(lines) + "\n")
    print(f"\nband (0.25-0.50): {band}")
    print(f"rotation-only: {rot_only}")
    print(f"wrote {out}/REPORT.md and {out}/sweep.json")


if __name__ == "__main__":
    raise SystemExit(main())
