# Person attributes, and the order they have to be built in

The product ask is age, gender and action per shopper. This file records what the
measurements decide about that, which public datasets are worth their download, and the
one ordering constraint that makes the difference between a report and a wrong report.

> **Updated 2026-08-18.** This file opened with "nothing here is built yet". Two of the
> three pieces have since landed and are described in place below: `analytics/reid_metrics.py`
> is the association metric, and `runs/crop_encoder01` is a first attribute encoder. The
> **ordering constraint below is unchanged and is still the point of the document** —
> association is still not fixed, and the encoder was trained on PA-100K rather than on the
> RAP v2 this page argues for first, because RAP's licence question is still open.

`analytics/tracker.py` is IoU association over detection boxes and `analytics/dwell.py`
turns tracks into dwell. `analytics/track_attributes.py` and `analytics/reid_metrics.py`
have since joined them; there is still no re-identification *model*, and association is
still unfixed.

## What the site measurements already decide

Measured on `assets/archive_*.mp4` and the three site clips with the shipped detection head:

| finding | number | how |
|---|---|---|
| person box height | **244–336 px** median, 85–98% above 128 px | boxes at score 0.20, mapped back through the letterbox |
| head region | **31–42 px** | body height / 8, an upper bound under a 50° down-pitch |
| track length | median **9–16 frames** at 5 fps | `scripts/retail_flow.py` |
| tracks per clip | **1234** in one 4.6-minute clip | same |

Two consequences, and they point in opposite directions:

**Body-crop attributes are feasible.** A 244–336 px crop is a comfortable input for a
second-stage classifier. Nothing about the resolution is the problem.

**Face-based age is not.** At 31–42 px a face carries no usable age signal, and any
pipeline that routes age through a face detector will report confidently on noise. Age
has to come from the body crop or not at all.

## The ordering constraint

**Track association comes before attributes. This is not a preference about what is more
interesting to build.**

A 4.6-minute clip fragments into 1234 tracks. Attributes computed per track on that
fragmentation produce an age/gender distribution over 1234 "people" who are perhaps
thirty actual shoppers, each counted a variable number of times and each count weighted
equally. The output looks like a demographic report and is one only if the denominator is
people. It currently is not.

The failure is worse than noisy, because it is *biased*: a shopper who lingers fragments
into more tracks than one who walks through, so dwell-weighted traits are over-counted by
exactly the amount dwell analytics exist to measure.

So the second-stage crop encoder should be built to serve **association first, attributes
second** — one model, and the embedding it learns is what re-links fragments. The same
encoder later carries action and product sub-class.

> `ARCHITECTURE_REVIEW.md` rules tracking and re-identification **"do not build"**, and
> that ruling stands as written: it is about the *exported graph*, where cross-frame state
> would break the no-dynamic-control-flow property that makes TensorRT conversion work
> first time. A host-side crop encoder is a different thing and does not conflict with it.

## The three public attribute datasets

Assessed against this project's cameras, not against each other.

| dataset | size | attributes | domain | verdict |
|---|---|---|---|---|
| **RAP v2** | 84k crops | 72 | **indoor mall CCTV** | first, and check its re-ID subset |
| **PA-100K** | 100k crops | 26 | outdoor street | volume only, kept a minority by *step share* |
| **PETA** | 19k crops | 61 | ten sub-datasets, mixed | benchmark only, verify the split first |

### RAP v2 — first, and the reason is distribution, not size

Real indoor shopping-mall surveillance: the same mounting height, down-pitch, indoor
lighting and compression as the 23 selling-floor cameras. It is 16% *smaller* than PA-100K and
that does not matter.

This project has already paid for the alternative view. ADE20K's `column` is architectural
pillars in atriums and lobbies; the model scored **0.40–0.51 on val and predicted 0.00% of
pixels** across four daytime store cameras. Size did not rescue a distribution mismatch and
will not here.

Its **viewpoint and occlusion annotations may be worth more than the 72 attributes**,
because occlusion is what fragments the tracks and no other dataset here labels it.

**Check before planning around it:** whether its identity / re-ID subset is usable. If it
is, RAP is the only one of the three that serves *both* jobs, which moves it from "first"
to "decisive". It also requires a signed agreement, so confirm the licence permits the
intended use before committing a schedule to it.

### PA-100K — volume, and the mixing rule that is not optional

100k outdoor crops, 26 attributes. Useful as pre-training mass; wrong as the main source.

**Run anyway, 2026-08-18, because it was the one on disk.** `runs/crop_encoder01`, 8 epochs:
Female recall 0.839 / precision 0.856 on 36,492 training crops, AgeLess18 0.639 / 0.919 on
4,152, and **AgeOver60 0.119 / 0.357 on 1,127 — degrading as training ran**, which is a
1.41% attribute once the loss can afford to say no. Unasked-for and more useful: the same
embedding scores **mAP 0.0543 / rank-1 0.1689** on Market-1501's protocol with no identity
label anywhere, against a 0.0318 ImageNet floor. Read that as a floor cleared, not as
association solved — and note it does not change the verdict in the table: an outdoor-street
encoder is still the wrong main source for indoor CCTV.

**Mixing it in by file count is a mistake this project has already made and measured.**
ADE20K at 90.2% of segmentation steps, containing zero `product` pixels, held that class at
IoU 0.000 for 22 consecutive epochs — not dilution, but every batch supplying the class as
a labelled negative. PA-100K's 26 attributes would suppress RAP's other 46 by the same
mechanism.

Keep the out-of-domain set a minority **by step share**, which is not the same as by file
count, and see `METHODOLOGY.md` §3 for the full account and the direction of the fix —
lower the abundant set's ratio rather than raise the scarce one.

### PETA — a benchmark, not a source of truth

Assembled from ten sub-datasets, so "does it match our distribution" has no single answer.

More importantly, **its standard split is reported to share identities between train and
test.** Verify that before quoting any number from it. This is the same shape as the three
site runs that used `site_sam3:test` as their validation split and so selected `best.pt` on
the set they were measured by — a mistake caught here on 2026-08-17, and not less of one
for being in someone else's dataset.

## What to do first

1. **Confirm RAP v2's licence and whether its re-ID subset is usable.** One question,
   and it changes the plan more than anything else on this page.
2. **Fix track association**, measured on ID switches and track length rather than on
   attribute accuracy. **The metric now exists** — `analytics/reid_metrics.py`, landed in
   `078f0e9` — but only half of it is live: `idf1` and `id_switches` need ground-truth
   tracks and no site clip is labelled, so the next thing on this line is **one labelled
   clip**, not a model.
3. **Then attributes**, on the same encoder.

Reversing 2 and 3 produces a demographic report whose denominator is fragments. It will
look finished.
