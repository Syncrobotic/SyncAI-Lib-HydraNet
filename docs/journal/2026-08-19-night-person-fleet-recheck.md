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

**28 of 42 cameras hold. 1 has a real person. 13 are counter-examples.**

> **Corrected the same day, and the correction is the more useful half of this
> entry — see §7.** The first pass of this table said 3 and 11. Two cameras were
> filed as `person present` on a *low* static share, and opening them showed
> parked scooters and a hanging accessory. A veto's quiet side is not evidence.

A camera over threshold is not automatically a failure — a cleaner or a passer-by at midnight
is a real person and finding them is the threshold working. What separates the two is the
frame, opened at native pixels. The static share proposes the split and cannot settle it:

- `Tao-Hsin-cam04`, 0.551, static **0.27** → a dark upright figure at the glass entrance.
  Opened: **a person**. The one genuine detection in the fleet's midnight hour.
- `Taichung-cam01`, 0.594, static **0.98** → opened: **boxed merchandise** on the wall
  display of an empty, shuttered store. Counter-example.
- `Kaohsiung-cam02`, 0.501, static **0.09** → the low share said "moving". Opened: **two
  parked scooters**. Counter-example, and the reason §7 exists.

**The worst false score is 0.594 — 1.7× the threshold it is supposed to sit below.**

## 3. What the thirteen actually are

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
| **parked scooters, outdoors** | Kaohsiung-cam02 (0.501) | added by §7; the static plate could not see it |
| **hanging accessory, near-nadir rack** | Tao-Hsin-cam09 (0.362) | added by §7; same reason |

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
- and the recall cost it looked like it had is **not** real: Kaohsiung-cam02 and
  Tao-Hsin-cam09, the two cameras it appeared to strip of people, turned out to hold no
  people at all (§7). Tao-Hsin-cam04's 12 genuine boxes survive it intact.

So consensus is cheaper than the first pass claimed, and still does not work: it leaves the
two poster cameras, and it makes the pipeline depend on keeping a second teacher loaded for
a class it is not the teacher of. The static veto in §8 does the same job with a plate that
already exists.

## 5. Consequences

**`person`-only night annotation cannot go in unreviewed.** That was the question this run was
asked to answer for the campaign's night tranches, and the answer is no: 13 of 42 cameras
would inject false people at up to 0.594, on frames where the store is empty and every box is
therefore wrong. Day frames are unaffected by anything measured here.

**0.35 stays as the day threshold and is not a night threshold.** It is not a measured gap at
night — the night population reaches 0.594 on this fleet, and the "gap" was one camera's.
Raising it is not obviously the fix either: 0.594 is above where day people would still need
to be found, so a single fleet-wide number may not exist. What the data supports today:

1. **Night `person` needs a per-camera decision, not a fleet threshold** — 28 cameras are
   clean at 0.35 and could be used unreviewed; the 13 are named above and cannot.
2. **The static share is the discriminator that works — as a veto, one-directionally.** A
   high share removes a box safely on all 42 cameras with a plate that already exists. A low
   share proves nothing, which §7 is the cost of learning.
3. §8 is what was built from that: the veto, plus two cameras excluded by name.

## 6. What this does not settle

The static share is a measurement of the *pixels*, not ground truth about people: it says a
box did not move, and "did not move" is inferred to mean "not a person" from a separation
measured on this fleet. A person asleep or standing perfectly still for the whole clip would
be classified as furniture by it, and nothing here would notice. Every camera with a box over
threshold has now been opened by eye; the 28 `holds` were not, because a camera with no box
above threshold has nothing to open.

And both teachers remain teachers. Day counts agreeing between them is agreement, not
accuracy, exactly as the swap entry said.

---

## 7. CORRECTION: two cameras were filed as `person present` and hold no people

The first pass of §2 reported 3 cameras with a real person and 11 counter-examples. Opening
all of them at native pixels — which the acceptance set for the veto recipe required, and
which the first pass had done for one camera of the three — makes it **1 and 13**:

| camera | first pass | what the box is | static |
|---|---|---|---|
| Tao-Hsin-cam04 | person | **a person**, dark figure at the glass entrance, 12 boxes | 0.25–0.28 |
| Kaohsiung-cam02 | person | **two parked scooters** in a street bay, identical across four frames | 0.00–0.09 |
| Tao-Hsin-cam09 | person | **a hanging accessory** on a near-nadir product rack | 0.23–0.29 |

**The mechanism is worth more than the correction.** The static share is a measurement of
*pixels changing*, and it was used in two directions when it only supports one:

- **high share ⇒ the box never moved ⇒ not a person.** Sound, and it is what a veto needs.
- **low share ⇒ the box moved ⇒ a person.** Not sound. Where the plate itself is noisy,
  everything reads as moving. Kaohsiung-cam02 is an outdoor street bay whose headlights and
  IR gain change the whole scene between plate samples; Tao-Hsin-cam09 is a dark near-nadir
  rack where noise dominates the signal. Neither box moved; the reference did.

That is the same shape as this project's other recurring error — a number that is valid in
the direction it was measured and invented in the other. It also means §2's own framing was
too generous to itself: a three-way verdict computed from one measurement was reported as if
it were two independent ones.

**What it does not change.** The counter-example count went up, not down, so the conclusion
of §2 — that 0.35 does not hold at night fleet-wide — is stronger than first reported, not
weaker. The 28 `holds` cameras are unaffected; a camera with no box over threshold has
nothing to misread.

## 8. What to do instead, built and accepted

`src/syncai_hydranet/data/night_person.py`, with
`scripts/night_person_filter.py` as the CLI a batch pipeline can call.

**GDINO at 0.35, then drop any box whose static share against that camera's own midnight
plate is ≥ 0.50.** The threshold is not delicate — swept against the eye-verified set, every
value from 0.30 to 0.75 removes the same 62 of 72 false boxes and loses none of the 12 real
ones. `DROP_ABOVE` is the centre of that 0.45-wide plateau.

**Two cameras are excluded by name rather than gated.** Kaohsiung-cam02 and Tao-Hsin-cam09
score *below* the verified person population, so no threshold separates them; §7 is why.
Excluding them is honest, and running them through a gate that cannot see them would not be.

Accepted against three criteria, all measured on the 42-camera fleet rather than asserted:

| criterion | result |
|---|---|
| every eye-verified person box survives | **12 of 12** on Tao-Hsin-cam04 |
| the veto does not fire on a clean camera | **0 boxes dropped** across the 28 |
| the false-positive cameras are emptied | **72 of 72** removed — 62 by the veto, 10 by exclusion |

`tests/test_night_person_veto.py` holds these, plus the behaviours that make it a veto rather
than a classifier: no plate is a refusal and not a pass, a box smaller than the plate's
resolution is kept rather than dropped, and the two excluded cameras refuse with the reason
attached.

`static_person_filter.py` now imports `static_share` from the package instead of keeping its
own copy — the report that chose the threshold and the gate that applies it must not be able
to disagree about a box. Re-run after the change, its 8,616 boxes are bit-identical.

## 9. The residual risk, which no labelling gate can close

**A person who holds perfectly still for the whole clip is furniture to this veto.** The
share measures pixels, not people. Raising the threshold does not help: a motionless person
and a motionless packet are the same measurement, and the veto has no way to tell them apart
even in principle.

That is an argument for a human on night *events* in the alerting layer, not something the
annotation layer can solve — and it is the honest boundary of what was built here. The
narrower version: a person standing in front of a static display inherits some of its
stillness. The measured margin absorbs it on this fleet (person max 0.282 against a
threshold of 0.50), but that margin rests on 12 boxes from one camera and should be
re-measured when the campaign's night tranches land.
