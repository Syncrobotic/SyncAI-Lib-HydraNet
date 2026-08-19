# 2026-08-19 — 0.35 does not hold at night fleet-wide, and the two teachers fail on different cameras

Worker-session record, closing the named caveat on the `person` teacher swap. The swap to
Grounding DINO at 0.35 was decided the same day, and
[its journal entry](2026-08-19-security-retail-teachers-and-methodology.md) named its own
gap in the same breath: the night half rested on **one empty-store clip on one camera**, and
nine of 48 cameras had returned SAM 3 night hallucinations, so 0.35 had to be re-measured
across the fleet before "works at night" was a claim.

It has been. **It does not hold.**

Artefacts: `runs/gdino_night_fleet`, `runs/sam3_night_fleet`,
`runs/static_gdino_night_fleet`, `runs/night_person_verdict` (table, verdicts, frames).
Reproduce with `scripts/night_person_verdict.py`.

---

## 1. The measurement

42 live cameras × 12 frames from the ~23:58 store-local clip already on disk — 504 frames,
every one IR monochrome (`chroma_median` 0.00 across the fleet), so "night" is a property of
the pixels rather than of the filename. Three passes over the *same* frames:

| pass | what | why the frames are identical |
|---|---|---|
| GDINO, floor 0.10 | the teacher under test | — |
| SAM 3, its own 0.50 | the failure being replaced | `gdino_person_boxes.py` and `sam3_person_boxes.py` share a sampling rule to the line (`every = 300*5/frames`, decode at 5 fps, break at `--frames`); running the second with `--min-luma 0 --min-chroma 0` puts both on the same pixels with no new sampler to disagree |
| static share | packet vs person | `datasets/studioa_static` has a ~00:00 plate for all 42 cameras, cut from these very clips, so the reference illumination matches |

**The pipeline validates against the original measurement before anything else is read off
it.** On Taichung-cam09 — the one camera the swap was argued on — this run returns GDINO max
**0.323** against the recorded 0.326, and SAM 3 **229** boxes against the recorded 229. The
differences elsewhere are therefore about the cameras, not about the method.

## 2. The verdict

**28 of 42 cameras hold. 3 have a real person. 11 are counter-examples.**

A camera over threshold is not automatically a failure — a cleaner or a passer-by at midnight
is a real person and finding them is the threshold working. The static share separates the
two, and it was checked in both directions rather than assumed:

- `Tao-Hsin-cam04`, 0.551, static **0.27** → a dark upright figure at the glass entrance,
  moving. Correctly detected. Verdict `person present`.
- `Taichung-cam01`, 0.594, static **0.98** → boxed merchandise on the wall display of an
  empty, shuttered store. Verdict `COUNTER-EXAMPLE`.

**The worst false score is 0.594 — 1.7× the threshold it is supposed to sit below.**

## 3. What the eleven actually are

Opened at native pixels, cropped rather than downscaled (`frames/counterexamples.jpg`), because
the contact-sheet lesson of the same morning is that a downscaled tile shows *that* a box
fired and not *what* it fired on:

| morphology | cameras | note |
|---|---|---|
| hanging stock / pegboard packets | Taichung-cam01 (0.594), Kaohsiung-cam07 (0.364) | **the identical failure SAM 3 was replaced for** |
| printed human figures — poster, easel, framed wall panel | Taichung-cam03 (0.415), Taichung-cam06 (0.397) | a picture of a person is a person to both models |
| a parked scooter | Kaohsiung-cam01 (0.442) | the storefront arcade the census re-identified |
| dark objects on furniture | Kaohsiung-cam13 (0.483), Taichung-cam05 (0.358) | a bag, a chair back |
| low-contrast / blown-out IR regions | Tao-Hsin-cam06, Tao-Hsin-cam07, Kaohsiung-cam12, Kaohsiung-cam10 | named with less confidence than the rows above; the object is not identifiable by eye, only the fact that it is static |

The first row is the finding that matters most: **the teacher swap did not fix the hanging-packet
failure, it moved it.** SAM 3 read pegboard packets as people on Taichung-cam09; GDINO reads
them as people on Taichung-cam01 and Kaohsiung-cam07 instead.

## 4. The two teachers fail on different cameras, and consensus is not free

Box-level agreement at night is **24%** — of 84 GDINO boxes ≥0.35, only 20 have a SAM 3 box
at IoU ≥ 0.3 on the same frame. The failures are close to disjoint:

| camera | GDINO max | SAM 3 @0.50 |
|---|---|---|
| Taichung-cam01 | **0.594** | **0** |
| Taichung-cam09 | 0.323 | **229** |

So neither teacher is safe alone at night. The obvious move — require both — was measured
rather than assumed, and it is a genuine trade:

- it removes the false box on **9 of the 11** counter-example cameras;
- it fails on the other two, **Taichung-cam03** and **Taichung-cam05**, which is exactly what
  the morphology predicts: a printed human figure looks like a person to both models, so
  agreement between them is not evidence;
- and it **costs real people** — Kaohsiung-cam02 loses 6 of 8 real-person boxes (75%),
  Tao-Hsin-cam09 loses 2 of 2 (100%). Only Tao-Hsin-cam04 survives intact.

Deleting the people a security model exists to find is the failure `sam3_person_boxes.py`'s
static gate was already switched off for. Consensus is not a fix; it is the same trade with a
different name.

## 5. Consequences

**`person`-only night annotation cannot go in unreviewed.** That was the question this run was
asked to answer for the campaign's night tranches, and the answer is no: 11 of 42 cameras
would inject false people at up to 0.594, on frames where the store is empty and every box is
therefore wrong. Day frames are unaffected by anything measured here.

**0.35 stays as the day threshold and is not a night threshold.** It is not a measured gap at
night — the night population reaches 0.594 on this fleet, and the "gap" was one camera's.
Raising it is not obviously the fix either: 0.594 is above where day people would still need
to be found, so a single fleet-wide number may not exist. What the data supports today:

1. **Night `person` needs a per-camera decision, not a fleet threshold** — 28 cameras are
   clean at 0.35 and could be used unreviewed; the 11 are named above and cannot.
2. **The static share is the discriminator that works**, in both directions, on all 42
   cameras, with a plate that already exists. It is not a threshold on the detector at all.
3. Human review of the 11, or their exclusion from night tranches, is the cheap path.

## 6. What this does not settle

The static share is a measurement of the *pixels*, not ground truth about people: it says a
box did not move, and "did not move" is inferred to mean "not a person" from a separation
measured on two cameras. A person asleep or standing perfectly still for the whole clip would
be classified as furniture by it, and nothing here would notice. The three `person present`
cameras were confirmed by eye; the 28 `holds` were not opened, because a camera with no box
above threshold has nothing to open.

And both teachers remain teachers. Day counts agreeing between them is agreement, not
accuracy, exactly as the swap entry said.
