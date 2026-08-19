# Project methodology

How a team picks this project up and runs it: who owns what, what data has to be collected,
how models are trained, and how they are evaluated and tested before anything reaches a
store.

This is the *process* document. For the concepts behind multi-head training, read
[TRAINING_GUIDE.md](TRAINING_GUIDE.md) first — it is the shorter of the two and this one
assumes it.

---

## 0. Where the project actually stands

Read this before planning anything, because it determines the priorities below.

The model design is ahead of the data. The architecture works, exports cleanly to TensorRT,
and the parameter budget confirms the shared-trunk premise (84.4% trunk, 15.6% heads). What
limits the system today is not the network:

> **Revised 2026-08-19.** Four rows of the table below were rewritten because the
> measurements behind them were superseded: the COCO row's collapse was an artefact of
> scoring on the split that also selected the checkpoint (see level 2), and site data
> arrived in the meantime. The shape of the conclusion did not change — the constraint is
> still data — but the numbers that argued it did.

| Constraint | Status |
|---|---|
| **No site number is an accuracy** | The binding constraint, and it outranks every class-level problem below. See the next row |
| A *field* number exists and is an **agreement**, not an accuracy | Site splits are built and reserved by camera ([RETAIL_DATA.md](RETAIL_DATA.md)), but every mask in them is SAM 3 output with no human pass, so a model trained on SAM 3 and scored against SAM 3 shares its errors. `split.json`'s `test_provenance` says so with the data |
| Detection is trained, and the number depends entirely on which set | `releases/v1` scores **mAP 0.335 on COCO val2017** (0.3246 on the 25 indoor categories). The retail line, which is the only one scored on site boxes at all, gets **0.07–0.11 on `site_boxes`** from a different family of runs. A web number is not evidence about a store, and the two are not comparable. Export refuses any head no dataset supervises |
| The COCO share is a monotonic trade, not a collapse | Swept at four ratios and scored on **test** (`journal/2026-08-14-experiments-and-geometry.md`): at `sample_ratio: 0.1` every segmentation metric matches or beats the segmentation-only baseline and detection arrives free; above it, segmentation falls monotonically. There is no sweet spot to find and no blocker here — only an exchange rate that gets worse as you climb |
| Real-world accuracy is still unmeasured | Site footage is now in training (`retail_objects_batch02`/`batch03`, 23 selling-floor cameras) — what is missing is a *human-corrected* test split, not the data |
| Rare classes are memorised per camera, not learned | `column` scores 0.86–0.88 on cameras a run trained on and 0.00–0.51 on cameras it never saw. More *cameras* fixes this; more frames from the same ones does not |
| `person` is trained on web photography | COCO alone, eye level and uncompressed, against ceiling-mounted h.264. Site boxes now exist as a teacher's opinion (1,141 across 23 cameras) with no human pass, and **night is not covered at all** |
| Terrain metrics are not stable across dataset changes | `terrain_mIoU` averages over the classes *present*. Read `mIoU_classes` and the per-class `support` before reading `mIoU` |
| The indoor taxonomy's gaps are inherited, not active | `caution` is capped by its sourcing — 3 of its 4 terrain classes have zero examples, and it scores 0.33 on test. `hydranet_indoor.yaml` still trains it; the retail line does not, and derives free space from `terrain` instead |

**The work that moves this project is data work.** Plan accordingly: a team that assigns four
engineers to model tuning and none to annotation will produce a better-tuned model with the
same ceiling.

---

## 1. How to divide the work

Four workstreams. They are separable — each has a defined interface with the others — so they
can run in parallel and be owned by different people.

### A. Data and annotation — *the critical path*

Owns everything from camera to a directory the training code can read.

- Capture planning: sites, lighting, camera height, sessions
- Annotation: running it, writing the spec, quality control
- Split discipline: deciding which sessions are train, val and test
- Publishing datasets into `seg_folder` layout

**Interface out:** a dataset directory plus a one-line config entry. Nothing about the model
concerns this stream.

**Staff this first and most heavily.** It is the binding constraint, it has the longest lead
time, and it is the one thing that cannot be parallelised away later.

### B. Training and evaluation

Owns the model, the runs and the numbers.

- Config discipline, run hygiene, checkpoint selection
- Reading results, deciding what is a real improvement
- Loss balancing, schedules, resolution and batch decisions
- Keeping the evaluation protocol honest

**Interface in:** dataset directories from A. **Interface out:** a `best.pt` plus a
`meta.json` that says exactly what produced it.

One person can hold this. It is the *least* parallelisable of the four — two people tuning
the same model in different directions produce results neither can interpret.

### C. Deployment

Owns everything from `best.pt` to a running engine on the inference box.

- ONNX export and its validation
- TensorRT engine builds, FP16 now, INT8 when the model settles
- The C++/Python runtime and post-processing (decode, NMS — and argmax, unless the export
  folded it into the graph, which on one board halved the frame)
- Pre-processing parity between training and the deployed engine
- Latency budgets on the target board

**Interface in:** a checkpoint plus the config that made it. See
[DEPLOY.md](DEPLOY.md).

Can start immediately and in parallel — the ONNX contract is already stable, so this stream
does not have to wait for a good model to build out the runtime.

### D. Platform

Owns the things that make the other three repeatable.

- Environments, drivers, `uv.lock`, CI
- Run storage, and a shared view of results once training spans machines
- Release and versioning discipline

Part-time, but real. Deferring it is what produces "it works on my machine" at the worst
moment.

### Coordination rules

- **One person decides `primary_metric`.** It defines what "better" means; if two people
  disagree about it, no comparison between their runs is meaningful.
- **The annotation spec is a shared contract**, not a note in someone's head. It lives in the
  repo, and the terrain→traversability policy table is part of it.
- **Nobody edits a config value and a dataset in the same change.** When two variables move,
  a difference in results tells you nothing.
- **All four streams read `meta.json`.** It is the common language: which code, which data,
  which settings produced a result.

---

## 2. What data to collect

### Rank by what no other instrument can answer

> **This section was rewritten on 2026-08-19.** It used to rank annotation by *danger x how
> blind the LiDAR is*, for a quadruped that carried one. The quadruped line was removed and
> never had a LiDAR anyway — the platform was a monocular camera and two ultrasound returns,
> a correction that had already invalidated the ranking before the line went. The old order
> (`glass`, `wet_slippery`, `floor_metal`, `threshold_ramp`, `stairs`) belongs to the indoor
> taxonomy, which `hydranet_indoor.yaml` still trains; it is not what the shipped line needs.
> The principle survived the platform: **rank by what no other instrument can answer.**

On a fixed store camera there is no second sensor at all, so the ranking is not against
another instrument — it is against the **teacher models** that can label without a human, and
against what a fixed viewpoint gives away for free. Three things follow, and they decide the
order below:

| | a teacher can supply it | a human must |
|---|---|---|
| masks for large edge-defined classes (`floor`, `wall`, `fixture`) | **yes** — SAM 3, prompted by text | review, not draw |
| `person` boxes | **yes** — Grounding DINO at a measured threshold | verify a sample |
| **is any of it right** | **no** | **this is the whole gap** |

That last row is the one that orders everything. Every site number this project can produce
today is an **agreement with a teacher**, not an accuracy: the training masks and the test
masks come from the same model, so they share its systematic errors and cancel exactly where
both are wrong. Annotation budget buys the difference between those two words.

### Priority order

| Priority | What | Why | State |
|---|---|---|---|
| **1** | **Human-corrected test masks**, ~72 frames on the reserved cameras | Until they exist this line has **no site number at all** — not a low one. Everything below is unmeasurable without it, which is why it outranks classes that are visibly worse. | SAM 3 pre-labels plus one automated pass; `split.json`'s `test_provenance` says so with the data |
| **2** | **One labelled clip of person tracks** | `idf1` and `id_switches` have no ground truth, so the tracker's 1234 fragments per clip can be described and not scored. Offline non-causal tracking has cut the human cost to minutes. | one human step from existing |
| **3** | **Night `person`** | The teacher swap fixed the day; an empty IR store used to return 229 false people. A store camera is most often asked a security question at night, and the current batch is daylight-only. | day covered, night uncovered |
| 4 | `column` review | Sourced, and *memorised per camera* rather than learned: 0.86–0.88 on cameras a run trained on, 0.00–0.51 on cameras it never saw. More cameras, not more frames from the same ones. | 10 of 23 selling-floor cameras carry it |
| 5 | `product` / `fixture` review | Both have teacher data and want *judging* rather than drawing — keep / drop / relabel per connected component. That bounds precision honestly and costs clicks instead of brushwork. | `fixture` is a sink: 92.6% recall, 56.1% precision |
| 6 | **One known length per store** | Not a mask. The geometry recipe recovers orientation from a background plate and **cannot recover scale**; the metre has to come from a tape measure once per camera. Without it every event polygon is a shape without a size. | one camera of 48 is calibrated |

Everything else — `floor`, `wall` — is adequately covered by the teachers plus ADE20K. Do not
spend annotation budget there.

### Teacher models are the annotation lever

Geometric classes on the old robot line were going to be pre-labelled from a point cloud. The
retail equivalent is a **promptable segmentation model**, and it is strictly better in one
respect and worse in another.

Better: it can propose a class it has never seen. `scripts/sam3_prelabel.py` takes text, so a
display podium that no public dataset contains comes back as a mask. A label drawn from *this
project's own model* can only reinforce what the model already believes — self-training on
site frames moved `display_fixture` by **−0.0096** for exactly that reason.

Worse: **a teacher is a second model with its own errors, not an oracle.** Everything it
writes is a pre-label, and `hydranet-annotation check` still gates it. The two limits are
structural and neither is fixable by prompting harder: consensus across frames removes only
*random* error, and a fixture the teacher consistently misses survives untouched; and it
improves precision while saying nothing about recall.

### A fixed camera is free validation

The old robot line's version of this was "where the model says *go* and the point cloud shows a
20 cm drop, that is a failure case found without anyone annotating anything". A fixed camera
gives the same thing from a different direction, and it costs nothing but compute:

**The scene does not move, so a pixel that changes class between frames is one the model was
unsure about.** No labels are needed to ask the question — only whether the answer was
stable. Over 610 frames of one camera, 12.7% of the frame was settled floor and **16.7%
flickered**, concentrated on the brighter, more specular near-field tiles. That is a
measurement of the model's confusion obtained from unlabelled footage, and no metric computed
on the validation split was going to report it.

Two uses follow. `engine/consensus.py` turns it into a target-domain check that needs no
ground truth. And the same property makes the annotation cheap in the other direction: on a
camera that never moves the floor is **one polygon, correct for every frame that camera will
ever produce** — per camera, not per frame.


### How much, and the unit is cameras

**Count cameras, not frames.** On a fixed installation a frame is nearly free and adds
almost nothing: the 189th clip of a day from one camera tells you very little its 1st did
not, because the shelves, the floor tiles and the mount have not moved. What varies is the
camera.

The measurement that settles it: resampling one run's validation score over **cameras**
rather than over images gave `column` an interval of 0.000–0.512, because one of the three
val cameras carried no `column` at all. **The camera is the dominant term.** Adding frames to
the same three cameras would have attacked the smaller one.

As an initial target per class that needs work:

- the class present on **at least two test cameras**, or the per-class number is not reported
  at all — it is that camera's number, and it moves for reasons unrelated to the model
- **all three stores**, and both viewpoint families: wall-mounted wide-angle and near-nadir
  over a desk are different problems
- the **richest** camera for a rare class stays in *training*; the test only needs enough
  pixels for a stable number, and where those compete training wins

The one axis where more frames of the same camera does buy something is **time of day**.
`person` and `product` genuinely vary across slots; `column` does not, and night IR is a
different *appearance* of the same fixed scene rather than more of it.

### How to capture

The training data has to look like what the deployed camera sees, or the model learns a
domain it will never run in.

- **The viewpoint is the mount, and it is already fixed.** Nothing here is captured by a
  person walking around — it is pulled from the recorder. ADE20K's weakness is exactly this:
  human-height, human-framed photography against ceiling-mounted, down-pitched, h.264 store
  cameras.
- **Cover the trading day, including after it.** Open-store slots for shoppers and stock, and
  the night IR clips — which are not a nuisance case but the ones security is asked about.
- **Include the hard negatives.** Specular reflections on a polished floor that read as open
  aisle, storefront glass with the street visible through it, and the pegboard of hanging
  packets that a person detector called 229 people on an empty night frame.
- **Check the camera is worth pulling before pulling it.** A 48-camera site does not have 48
  usable cameras: 6 are dead, 19 are back of house, and the ways one fails are independent —
  container metadata catches a never-1080p camera, pixel statistics catch a black or
  greyscale one, and only the pre-labels catch a stockroom pointed at racking.
- **Record which camera and which slot.** Camera identity is what makes an honest split
  possible later; frames without it cannot be split safely.

### Annotation spec

The full contract — the id table, the export format, the two commands that check it, and
the background-remap trap that is the sharpest edge in the pipeline — is §3's
[annotation spec](#the-annotation-spec). Two points belong here because they are planning
decisions rather than drawing ones:

- **Pin the policy table in the spec.** If annotators and training disagree about where
  `fixture` ends and `product` begins, the labels are silently wrong and nothing will
  surface it. That boundary is the hardest one in this taxonomy and the one a coarser
  scheme cannot recover once lost.
- **Pick the scheme before the first mask is drawn.** A store annotated under the indoor
  taxonomy comes back with no `display_fixture` in it at all — the masks validate, they
  train, and the model learns that shelving is furniture. Nothing downstream reports it,
  because a class absent from the data looks exactly like a class that is rare.

### Split rules

**Split by camera. Never randomly, and never by clip.** Adjacent frames are near-identical, and
on a fixed camera *every* frame is near-identical — a random split puts the same shelf on both
sides and validation then measures memorisation. The score will look excellent and mean
nothing. The gap has been measured: one run scored mIoU 0.778 on cameras it trained on and
0.641 on cameras it never saw.

A test camera never supplies a training frame. Not "just the night clips from it" — the moment
it does, every comparison across batches needs a footnote explaining which model saw what, and
in practice nobody writes that footnote.

Three splits, with distinct jobs:

| Split | Job | Who may look at it |
|---|---|---|
| `train` | Fit the weights | Everyone, constantly |
| `val` | Select the checkpoint, tune settings | Training stream, every epoch |
| `test` | Report the number | **Once, at a release decision** |

Configure `split_test` explicitly. It is deliberately not defaulted, because a silent fallback
to val produces an official-looking number that is quietly circular.

---

## 3. Producing the labels

Merged in from `ANNOTATION_SETUP.md` on 2026-08-19. §2 above says *what* to collect and
in what order; this section is the machinery that turns footage into a dataset the trainer
will accept, and the measurements that decide how much of it a machine may do.

CVAT runs on a Google Cloud VM with **no public IP**. It is reached through an IAP tunnel,
so there is no login page exposed to the internet and no TLS certificate to maintain — the
tunnel is the encrypted channel, and Google IAM is the authentication.

Site footage is likely to show customer premises, which is why it is set up this way.

### What exists

| | |
|---|---|
| VM | `hydranet-annotation`, `asia-east1-b`, `e2-standard-2`, 100 GB |
| Project | `syncrobotic-aisw` |
| Public IP | none — IAP only |
| Firewall | `allow-iap-ssh-hydranet`: tcp:22 from `35.235.240.0/20` (Google's IAP range) to tagged hosts only |
| Storage | `gs://syncai-hydranet`, reachable from the VM over Private Google Access |
| Service account | `syncai-hydranet@…`, scoped to that one bucket |
| Deployment | [`tools/annotation/`](../tools/annotation/) — docker compose, upstream checkout at `/opt/cvat/src` |

### Connecting

Each annotator needs the `gcloud` CLI and IAM access to the project.

```bash
gcloud auth login
gcloud config set project syncrobotic-aisw

# open the tunnel; leave this running
gcloud compute ssh hydranet-annotation --zone=asia-east1-b \
    --tunnel-through-iap -- -L 8080:localhost:8080
```

Then open <http://localhost:8080> in a browser. The connection is local to your machine;
CVAT itself is never exposed.

The IAM roles an annotator needs are `roles/iap.tunnelResourceAccessor` and
`roles/compute.instanceAdmin.v1` (or a narrower custom role granting
`compute.instances.get` and `compute.instances.osLogin`).

### Running the stack

CVAT is a nineteen-container application, and upstream's `docker-compose.yml` is the only
honest description of it. So this repo does not reimplement it: it pins a release tag and
layers one override file on top.

```
tools/annotation/
  cvat.env                     CVAT_VERSION — the single pin
  docker-compose.override.yml  local policy: loopback-only port
  cvat.sh                      up / admin / backup / status / logs / down
```

```bash
gcloud compute ssh hydranet-annotation --zone=asia-east1-b --tunnel-through-iap
cd /path/to/SyncAI-Lib-HydraNet/tools/annotation
./cvat.sh up
```

`up` clones or moves `/opt/cvat/src` to the pinned tag, refuses to run if that checkout has
been edited, and starts the stack with the override applied. Upgrading CVAT is therefore a
one-line change to `cvat.env` — and `./cvat.sh backup` first, because a newer CVAT migrates
the database on start and the migration is not reversible.

Two things the override changes, both deliberate:

- **The port binds to `127.0.0.1` only.** Upstream publishes 8080 on every interface. The VM
  has no public IP and the only way in is a tunnel to loopback, so binding wider buys
  nothing and would expose site footage the day a VPC route or peering appears.
- **`restart: always` on the server**, so a job does not die with the annotator's tunnel.

> **The live instance predates this.** It was brought up by hand from CVAT's `develop`
> branch (`7064c1b`, 2026-08-13) on the floating `cvat/server:dev` tag, which means a
> restart can land on a different CVAT than the one before it. It holds no annotations
> yet, so the switch is free — run `./cvat.sh up` once and it moves onto the pin. After
> real annotation exists, back up first.

### The admin account

CVAT ships without one, and creating it is the step every bring-up forgets:

```bash
./cvat.sh admin paul            # prompts twice for the password, never echoes it
```

Store the password in a password manager. Annotator accounts are then created from CVAT's
own admin UI at <http://localhost:8080/admin>.

### Backups

Annotation is the most expensive thing in this project per hour, and it is the only
artefact here that cannot be regenerated by rerunning something.

```bash
./cvat.sh backup                # ~1 min: stops the app containers, dumps, restarts
./cvat.sh backup --online       # no outage, but a task being edited can be caught torn
```

Each run writes `tools/annotation/backups/<timestamp>/` — a `pg_dump` of the database,
tarballs of the `cvat_data` and `cvat_keys` volumes, and a `MANIFEST` naming the CVAT
version — then uploads it to `gs://syncai-hydranet/annotation-backups/`.

To restore, bring the stack up on **the CVAT version in the MANIFEST** (an older database
cannot be read by a newer server until it migrates, and a newer one cannot be read by an
older server at all), then:

```bash
./cvat.sh down
docker volume rm cvat_cvat_data cvat_cvat_keys cvat_cvat_db
./cvat.sh up
./cvat.sh down cvat_server                       # let the db initialise, then stop the app
for v in cvat_data cvat_keys; do
  docker run --rm -v cvat_$v:/to -v "$PWD/backups/<stamp>:/from" alpine:3.20 \
      tar xzf /from/$v.tgz -C /to
done
gunzip -c backups/<stamp>/cvat_db.sql.gz | docker exec -i cvat_db psql -U root -d cvat
./cvat.sh up
```

### Stop the VM when nobody is annotating

Annotation is bursty, and a stopped VM costs only its disk.

```bash
gcloud compute instances stop  hydranet-annotation --zone=asia-east1-b   # ~$10/month
gcloud compute instances start hydranet-annotation --zone=asia-east1-b   # ~$59/month running
```

CVAT restarts with the machine; the containers are configured to come back up. Give it a
couple of minutes before the API answers.

### The annotation spec

Everything below is a contract between annotators and training. If the two disagree, the
labels are silently wrong and no metric will reveal it. Two commands hold up the half of
that contract a machine can check:

```bash
hydranet-annotation labels --out cvat_labels.json   # paste into CVAT: project → Raw labels
hydranet-annotation check  datasets/site_a          # exit 1 on any violation below

# annotating a shop: the indoor 12 plus display_fixture
hydranet-annotation labels --scheme retail --out cvat_labels_retail.json
hydranet-annotation check  datasets/store_a --scheme retail
```

**Pick the scheme before the first mask is drawn.** A store annotated under the indoor
taxonomy comes back with no `display_fixture` in it at all — the masks validate, they
train, and the model learns that shelving is furniture. Nothing downstream reports that,
because a class that is absent from the data looks exactly like a class that is rare.

`labels` generates the class list from the same `label_maps_indoor.py` that training reads,
with the same colours the inference overlays use — so nobody types an id twice, and a mask
drawn in CVAT is the colour its prediction will be. `check` reads a finished dataset with
training's own pairing code and refuses it if it breaks the rules below. Run it before the
first training run on new labels, and again whenever a batch lands.

- **Label the 12 indoor terrain classes**, ids exactly as in
  [`label_maps_indoor.py`](../src/syncai_hydranet/data/label_maps_indoor.py):

  | id | class | | id | class |
  |---|---|---|---|---|
  | 0 | `void` | | 6 | `threshold_ramp` |
  | 1 | `floor_hard` | | 7 | `wall` |
  | 2 | `floor_soft` | | 8 | `glass` |
  | 3 | `floor_metal` | | 9 | `door` |
  | 4 | `wet_slippery` | | 10 | `obstacle_furniture` |
  | 5 | `stairs` | | 11 | `person` |

- **Do not annotate traversability.** It is derived from terrain through the policy table
  in the same file. One annotation pass supervises both heads.
- **Ambiguous pixels get 255 (ignore), never a guess.** An ignore region costs nothing in
  training; a wrong label is trained on.
- Export as **single-channel PNG** where the pixel value *is* the class id, into the
  `seg_folder` layout, and set `label_map: indoor_native` in the config — the ids need no
  translation. An RGB export trains on the red channel and calls it a class.
- **Remap the exported background from 0 to 255.** This is the sharpest edge in the whole
  pipeline. Most tools write unlabelled pixels as 0; under `indoor_native`, 0 is `void`,
  a *trained* class — the loss ignores 255 and only 255. Export the background as 0 and
  every unlabelled pixel teaches the model to predict `void` there — a confident label over
  exactly the regions nobody looked at. `check` fails on this.
- **One capture session per directory**: `images/train/lobby-2026-08-20-a/000123.png`. The
  directory is what carries session identity into the split, and `check` uses it to prove
  no session appears in two splits.

### A clip that half-decodes now stops the run

Every script that reads video goes through `data/video.frames()`, which reads fixed-size
frames off a rawvideo pipe until a read comes back short. A finished clip ends that way.
So does a truncated one — and until 2026-08-19 nothing looked at ffmpeg's exit status, so
the two were the same event.

The consequence was one class of silent wrong, spread over fifteen call sites: a clip that
died a third of the way through produced a generator that stopped cleanly. `site_events`
wrote its `events.json`, `sam3_prelabel` wrote its pre-labels, `annotation_batch` wrote its
batch — each over a third of the footage, each reporting success, and none leaving any
artefact recording which frames never existed. On a 3 TB bucket pulled over the network,
that is not a hypothetical.

`frames()` now raises `DecodeError` when *it* reached the end of the stream and ffmpeg had
exited non-zero, or when the last read was a partial frame. The message carries the whole-frame
count, the exit status, and ffmpeg's own stderr, because "moov atom not found" and "Invalid
NAL unit size" call for different actions.

A caller stopping early — `--max-frames`, or the `if i >= n` in half the scripts — is
deliberate and stays silent; ffmpeg is terminated rather than waited on. If a run stops with
`DecodeError`, re-pull the clip before re-running: the frames already written are a prefix,
and appending to them is how a partial pull becomes a permanent hole in a dataset.

### Pre-labelling the missing classes with SAM 3

```bash
uv pip install -e '.[annotate]'
hf auth login                       # facebook/sam3 is a gated repo

python3 scripts/sam3_prelabel.py --out datasets/retail_sam3_batch01 \
    --frames 40 /path/to/hls-record/cam*/archive_*.mp4
hydranet-annotation check datasets/retail_sam3_batch01 --scheme retail
```

There are now two pre-labellers, and they are not alternatives:

| | label source | what it can do |
|---|---|---|
| `scripts/annotation_batch.py` | the model itself | correct what it already half-knows |
| `scripts/sam3_prelabel.py` | SAM 3, promptable by text | **propose a class it has never seen** |

The second capability is the whole point, and it rests on one property.
[RETAIL_DATA.md](RETAIL_DATA.md) §5 measured self-training on this footage: it moved
`display_fixture` by **−0.0096**, because the podiums came back unlabelled and the model
was then taught that shop fixtures are not fixtures. A label drawn from the model can
only reinforce what the model already believes. SAM 3's knowledge is **external**, so a
podium it segments is new information rather than an echo — which also means it is a
second model with its own errors, not an oracle. Everything it writes is a pre-label and
`check` still gates it.

Prompts live in [`data/sam3_prompts.py`](../src/syncai_hydranet/data/sam3_prompts.py),
next to the label maps and importing their ids rather than restating them. Measured on
eight site frames, 2026-08-15:

- **`display_fixture` needs eight prompts, not one.** `display table` finds the island
  podiums, `shelving unit` the wall runs, `retail counter` the service desk — each misses
  the other two. One class here is three objects to SAM 3. `gondola shelf` scored 0/8
  despite being the trade term: prompts should read like a caption, not a spec sheet.
- **`table` and `display table` fight over the same pixels on 6/8 frames.** Which is
  right depends on what the shop uses the table for, which is not visible from the
  ceiling. Those pixels are written as 255 for an annotator to settle — the same reason
  `label_maps_retail.py` refuses to map ADE20K's `table` to `display_fixture`.
- **`glass` segments what is *behind* the pane.** On the storefront camera it returned
  the pavement, a truck and two pedestrians along with the glass. Accepted as-is that
  teaches the model a street is glass, in the one class whose failure mode is walking
  into it. Redraw every glass pre-label; the raised threshold buys fewer, not better.
- **`stairs`, `threshold_ramp`, `wet_slippery`, `floor_metal` all returned 0/8**, and the
  reason splits in half. `stairs` and `threshold_ramp` are solid object concepts SAM 3
  would very likely find — this site is a single flat floor and has neither, so that is a
  **capture gap** and the fix is filming a site that has them. `wet_slippery` and
  `floor_metal` are *materials*: a wet floor is the same object as a dry floor, and a
  promptable concept model has no handle on it. They stay hand-drawn, exactly as the
  priority list above says.
- **The floor is off by default.** Not because SAM 3 fails on it (`tiled floor` hits 4/8)
  but because §5 measured the cheaper answer: on a fixed camera the floor is one static
  region, so one polygon per camera beats a pre-label per frame. `--include floor_hard`
  if a camera is not in fact fixed.

### When there is no annotation budget

Everything above assumes a human corrects the pre-label. Where nobody is available that
assumption does not degrade gracefully — an unchecked pre-label *is* a label, and SAM 3's
mistakes go into the weights unopposed.

The instinct in that situation is to add classes anyway and hope. Do not: `wet_slippery`,
`floor_metal` and `threshold_ramp` have been in the taxonomy since the beginning and are
all still 0.000, because **a class with nobody to draw it is a permanently empty channel**.
Adding more produces more of those.

What works instead is to spend compute where labour is not available.
`--consensus 0.9` votes each pixel across a clip and keeps only what agrees:

```bash
python3 scripts/sam3_prelabel.py --out datasets/site --consensus 0.9 \
    --frames 12 --upscale 1.0 /path/to/cam*/clip.mp4
```

The camera does not move, so a shelf is the same pixels in every frame; a pixel labelled
differently between two frames is one SAM 3 was guessing at. Nobody has to say which
frame was right — the question is only whether the answer was stable, and that is free to
ask. Over 10 frames per camera on the three pilot stores:

| camera | ≥60% | ≥80% | **≥90%** | 100% |
|---|---|---|---|---|
| Kaohsiung-cam08 | 63.5% | 62.1% | **51.2%** | 43.6% |
| Taichung-cam01 | 56.2% | 45.8% | **37.4%** | 23.0% |
| Tao-Hsin-cam05 | 62.8% | 48.7% | **36.8%** | 11.5% |

At 0.9 the surviving pixels are 56% `floor_hard`, 25% `wall`, 19% `display_fixture`.
`person` disappears entirely, which is correct — it moves, so it is excluded from the vote
and composited per frame instead. That split is also what makes long runs affordable: the
static pass is paid once per camera, and further frames cost one prompt each.

Two limits, both structural. It removes only *random* error; a fixture SAM 3 consistently
misses and pavement it consistently calls glass are perfectly stable and survive untouched.
And it improves precision while saying nothing about recall.

**The one thing this cannot produce is a test split.** Scoring a model against SAM 3's own
masks measures agreement with SAM 3, and a consensus filter does not change that — it makes
the labels more self-consistent, not more true. If only a couple of hours of human time
exist, spend all of it hand-labelling ~10 held-out frames and none of it on training data:
those frames are what decides whether any later number means anything.

### Which cameras are worth pre-labelling at all

A 48-camera site does not have 48 usable cameras, and the ways one fails are independent
— each check below passed cameras the others rejected, so running one and calling it a
survey overstates the fleet. Measured across the three stores. **The full census landed
2026-08-18** in `datasets/studioa_clips/cameras.json`: **23 selling floor, 19 back of
house, 6 dead**, with a role and a note per camera. Read that file rather than re-deriving
a count from the checks below.

| check | reads | caught | misses |
|---|---|---|---|
| container metadata | `ffprobe` width/height | 7 never-1080p cameras | everything about the image |
| **pixel level** | per-channel mean/std | 4 cameras outputting **pure black** — recorder running, timestamps live, no image; 2 stuck in **greyscale during trading hours** (`R_mean == G_mean == B_mean`, both bright, textured and 1080p) | what the camera is pointed at |
| **scene type** | the pre-labels themselves | a **stockroom** (boxes and racking, floor 1.5%) and a **top-down repair workbench** (ignore 74%, no fixtures) — neither is a room view, so neither can supervise traversability | — |

The black cameras pass nothing but a container check; the monochrome pair passes both
container *and* blackness and fails only on chroma; the stockroom passes all three and
fails only on what it shows. A low floor share is the signal for the last one — but read
it as a question, not a verdict, since it is also what a genuinely crowded frame looks
like.

Store-level exposure is worth knowing before tuning anything per-camera: across 16
cameras Taichung runs ~28 luma brighter and ~12 std flatter than Kaohsiung, and four of
the five lowest-contrast cameras on the whole site are Taichung's. That is a store
setting, not per-camera noise.

Two consequences to plan around. A batch with the floor off is **100% `blocked`** in its
labelled pixels — it supervises what is occupied and says nothing about the free space, so
pair it with the per-camera floor polygon before reading any number derived from the floor. And **never pre-label the test split with SAM 3**: a model scored against SAM 3's
own masks measures agreement with SAM 3. §6 already requires building that split by hand,
before the first training run.

### Which classes to draw first

**The order is in §2**, and it is ranked by what no teacher model can supply. What belongs
here is the measured detail behind three of its rows, because it decides *how* to draw them
rather than whether:

1. **`column`** — was zero examples and unrecoverable from anything already drawn, because
   every site mask put columns inside `wall`. Also the cheapest class in the taxonomy: the
   cameras are fixed, so it is **one polygon per camera**, correct for every frame that
   camera will ever produce. **Sourced 2026-08-17** — SAM 3 finds those columns at 7.14% of
   labelled pixels, and 10 of the 23 selling-floor cameras carry it at 3% or more. Sourcing
   it did not teach it: the model memorises it per camera rather than learning it.
2. **`product`** — was zero examples with no public dataset and no measurement of whether a
   teacher could find it. **Measured 2026-08-17 and the prompts work**: 19.28% of labelled
   pixels in train and 17.23% in test, present in every frame, plus 10,524 instance boxes.
   The earlier "0 instances" figure was a property of 352×240 footage and one prompt of
   eight, not of the footage.
3. **`fixture`** — has data and needs a *review* rather than fresh drawing: the migration
   merges `obstacle_furniture` and `display_fixture` pixel-exactly, so what arrives is as
   good as those two classes were, which for `display_fixture` was IoU 0.336. It is also a
   sink — 92.6% recall against 56.1% precision, everything leaks in — so the review is
   mostly *removals*.

`glass` leaves the list entirely: the object taxonomy folds it into `wall`, so its failure
stops being a class-level hazard. The pixels are still wrong; they are now wrong inside a
class rather than being their own class.

---

## 4. How to train

### Before the first run

1. `uv sync --group dev --extra export`, then `uv run pytest -q` — 1,300+ tests, no dataset
   needed. If these fail, stop; nothing downstream will be interpretable.
2. `uv run pre-commit install --hook-type pre-commit --hook-type commit-msg` — both types, see
   [CONTRIBUTING.md](../CONTRIBUTING.md).
3. Confirm the device: the log's first lines print `device=`, the AMP dtype and the backend
   flags. On CUDA see [the CUDA move](journal/2026-08-12-mps-to-cuda.md); on Apple Silicon see
   [DEPLOY.md](DEPLOY.md).

### Config discipline

`--set` is for experiments. **Anything you intend to keep goes in a config file.**

```bash
# experiment
uv run hydranet-train --config configs/hydranet_indoor.yaml --set train.lr=4.0e-4

# keeping it: copy the config, edit, commit
cp configs/hydranet_indoor.yaml configs/indoor_site_a.yaml
```

Typos are caught: the config is validated as a whole after overrides, and unknown keys, wrong
types, a `supervises` naming a non-existent head, or a `terrain_classes` count that disagrees
with the head all abort with the full list of problems.

That now includes `--set`'s one structural gap. It cannot index lists, and
`data.datasets[2].sample_ratio=0.1` used to create a useless key named `datasets[2]` in
silence; the schema rejects it and suggests `datasets`. The override still cannot be
expressed that way — override the whole list, or copy the file.

### Choose `primary_metric` deliberately, before the run

It is the single number that selects `best.pt`, which makes it the run's real objective. The
default `traversability_mIoU` is a balanced choice. When a specific failure mode matters more,
say so:

```yaml
primary_metric: IoU/traversability/00_blocked
```

### The run itself

- **Start from the settings that are already known to work** for the hardware — section 6 of
  [the CUDA move](journal/2026-08-12-mps-to-cuda.md) for CUDA — rather than inventing a configuration.
- **Change one thing at a time.** With two variables moving, a difference tells you nothing.
- **Let it run.** Intermediate epochs are noisy: this project's own baseline run measured
  `caution` at 0.158 on epoch 14 and 0.229 by epoch 27. Judging a run mid-flight is how teams
  abandon configurations that were working.
- **Resume rather than restart** when a run is interrupted: `--resume runs/.../last.pt`
  continues the schedule and reports any config drift from the checkpoint.

### What to watch, in order of value

1. **`val_pred/*` images.** The highest-value signal available. A falling loss is entirely
   compatible with the model calling a whole floor a wall; you see that instantly in a
   picture and never in a curve.
2. **Per-class IoU** for the rare, dangerous classes — `glass`, `stairs`, the new ones.
3. **`task_weight/*`.** A head's weight collapsing toward zero means it has stopped learning
   while the total loss still looks healthy.
4. **img/s.** Well below what the hardware should do means the data pipeline is the
   bottleneck, which is a different fix entirely.

### When to stop tuning

When the failing class has no data. If `caution` will not move, the question is not "which
learning rate" — it is "how many examples of it exist?" Route that back to workstream A. This
is the single most common way time gets wasted on this project.

---

## 5. How to evaluate and test

Four levels, cheapest first. Each answers a different question, and none substitutes for
another.

### Level 1 — Does the code work? (every commit)

```bash
uv run pytest -q          # 1,300+ tests, no dataset required
```

Includes `test_overfit.py`, which memorises one synthetic batch to >95% pixel accuracy. Shape
tests pass on a model wired backwards; this is what separates "it runs" from "it trains".
CI runs the same on Python 3.10 and 3.12 and fails below 80% coverage.

### Level 2 — Is the model learning? (every epoch)

Automatic. Validation writes to `metrics.jsonl` and TensorBoard, and `best.pt` is updated on
`primary_metric`. This is for steering, not for reporting.

**Know what mIoU is averaging.** `terrain_mIoU` is the mean over classes *present in the
validation set* — currently 8 of 12. Two consequences:

- It is **not comparable** to a published 12-class mIoU.
- It will likely **drop** when you add the missing classes, because harder classes enter the
  average. That is not a regression. Record the class count alongside the number, and compare
  per-class IoU when the dataset changes.

The retail-objects line is the worked example, and it is the reason to read `mIoU_classes`
before reading `mIoU`. The 60-epoch run reports **0.7668 over five classes** on ADE20K val;
the site runs report **~0.56–0.60 over six**, on a val set that also contains real store
frames. Nothing got worse: a sixth and much harder class entered the mean, and the mean is
now taken against harder data. **The number went down and the measurement got better** — at
the same epoch the newer run is ahead, 0.5973 against 0.5585.

**And a per-class IoU needs its support, which is the same argument one level down.**
`support/<head>/NN_<name>` is emitted beside every `IoU/<head>/NN_<name>` for exactly this:
`column` scored 0.40–0.51 on **0.66% of ADE20K val pixels**, 22 of 285 images, and then
predicted **0.00% of pixels** across 240 frames of four daytime store cameras. A number and a
well-evidenced number are formatted identically, and only one of them is a measurement. Below
`evaluator.THIN_SUPPORT` (1% of labelled pixels) the log says so and the honest reading is
"not measured".

**A mean can hide a class that stopped learning entirely.** This is the sharper failure.
The instance this section was written from is **withdrawn**, and both halves are kept
because the difference between them is the lesson. Adding COCO to supervise the detection
head appeared to move `traversability_mIoU` from 0.6765 to 0.6303 — a 7% drop, easy to
accept as the cost of a second task. Underneath, on **val**:

| Class | Seg-only (ships) | With COCO (ships) |
|---|---|---|
| `blocked` | 0.9547 | 0.9536 |
| `go` | 0.8455 | 0.8463 |
| **`caution`** | **0.2294** | **0.0908** |

> **Withdrawn 2026-08-14: these are val numbers, and val also selected both checkpoints.**
> Scored on the held-out **test** split, the same checkpoints are equivalent on `caution` —
> 0.3252 against 0.3346 — and the four-point ratio sweep in
> [`journal/2026-08-14-experiments-and-geometry.md`](journal/2026-08-14-experiments-and-geometry.md)
> shows the real shape: segmentation falls **monotonically** with COCO share, and at 0.1 it
> comes out slightly *ahead* of the segmentation-only baseline. There was no collapse at 0.1
> to explain. (Take each sweep point's ratio from its run's `config.yaml`, never from the
> directory name: **`coco10` is share 1.0**, `coco03` is 0.3, and the 0.1 point is
> `hydranet_indoor_det60`. The mapping and its cross-checks are in
> [ARCHITECTURE.md](ARCHITECTURE.md).) `best.pt` is chosen on val, so each run was cut at whatever epoch that rare
> class happened to spike, and the selection manufactured the variance it was then measured
> with. The comparison also changed two things at once — COCO was added *and* the run was cut
> from 60 epochs to 30 — which is the other reason it could not have attributed anything.
>
> **What survives is the reading habit, not the number.** A mean really can hide a class
> pinned at zero; the denominator never changes, so the class count would not catch it. Read
> per-class IoU as a time series. Just do not read a rare class off the split that selected
> the checkpoint — see "Report rare classes on the test split only" in
> [RETAIL.md](RETAIL.md) §8.

The starvation mechanism is still worth understanding, because a *large* COCO share does
cost segmentation and the sweep measures it: COCO has 118,287 images against ADE20K's 5,998,
so the ratio decides how many optimiser steps the segmentation heads get at all. `stairs`
occupies 0.3% of pixels, and since three of `caution`'s four constituent classes have no data
at all, `caution` is effectively `stairs` and moves with it. At `sample_ratio: 1.0` on test,
`glass` goes 0.5021 → 0.0692. **Share is the lever; 0.1 is the floor, not an optimum.**

**It recurred on 2026-08-17 as `product`, and one part of the account is established while
another is not. Both are recorded here, because this is the process reference and a reader
should be able to tell which is which.**

*Established.* `product` sat at exactly **0.000 for 22 consecutive epochs**. ADE20K was
**90.2% of segmentation steps and contains zero `product` pixels**, so in nine batches out of
ten every pixel was a labelled *negative* for that channel. That is a stronger statement than
starvation: a starved class learns slowly, and this one did not move at all. **Sign, not
share, is what explains a channel pinned at exactly zero.**

*Not established: that sign is the whole account.* Rebalancing to 41.9% site did not hold the
class. `runs/hydranet_retail_objects_site_balanced` is the control and the shape is the point
— read it rather than the claim:

| epoch | 2 | 3 | 10 | 15 | 25 | 39 |
|---|---|---|---|---|---|---|
| `IoU/terrain/05_product/site_sam3` | **0.4821** | 0.0395 | 0.1030 | 0.0300 | 0.0677 | 0.0526 |

**It spiked and collapsed.** If removing the negative evidence were sufficient, it should have
held. Rebalancing *delayed* the collapse rather than preventing it.

Selection behaved correctly here, and it is worth reading why. That run's `primary_metric` is
`IoU/terrain/05_product/site_sam3` — the class under investigation, not `terrain_mIoU` — so
`best.pt` holds **epoch 2 at 0.4821**, the peak itself. The log records exactly two `new best`
lines, both inside the first two minutes, and nothing across the remaining 37 epochs. **A
checkpoint pinned at epoch 2 of 39 is not a selection failure; it is the finding.** It says
the run's best answer for that class arrived before it had learned anything else, which is
the same information the table shows and harder to miss.

The competing account is that this is a *contradiction* rather than an absence:
`ADE20K_ID_TO_RETAIL_OBJECTS` sends **15 source ids to `fixture` and 0 to `product`**, so a
shelf of goods is one `fixture` region under ADE20K and a `fixture` + `product` split under
the site masks — the same pixels with two targets. **That account is unmeasured, not
refuted**, and the reason is worth knowing: `retail_objects_batch01` leaves floor and wall at
255 (57.5% of its pixels are ignored, `floor` and `wall` are both 0.00%), so `fixture`
over-predicting into the floor is invisible to every metric computed on that split.
`retail_objects_batch02` asserts them — floor 29.8%, wall 21.1%, 11.7% ignored — which is why
it exists.

**The fix has a direction, and the obvious one is wrong.** Lower the abundant dataset's
`sample_ratio`; do not raise the scarce one. Both reach the same balance, but raising the
scarce one gets there by showing the same few images many times an epoch, which trades a
suppressed channel for a memorised one. That direction holds whichever account is right.

`config_schema.unsourced_terrain_classes` names classes *no* dataset can produce, and
`minority_sourced_terrain_classes` names the ones only a smaller dataset can — both before a
GPU-hour is spent. Neither can see pixels, so both are candidates to confirm rather than
conclusions; see the identity-map caveat in the second one's docstring.

**So: never read a mIoU without the per-class numbers next to it**, and when a dataset mix
changes, check that the rare classes still have a pulse before accepting the mean.

### Level 3 — How good is it, honestly? (release candidates)

```bash
uv run hydranet-eval --config ... --checkpoint runs/.../best.pt \
    --split test --json reports/rc1_test.json
```

Rules that make this number worth anything:

- **Run it once**, at a release decision. Every extra look leaks test into your choices, and
  it degrades into a second val split.
- **Report per-class**, not just mIoU. A mean that hides `glass` at 0.2 is not a summary.
- **Use the same `--weights`** as deployment will (EMA by default).
- **Never tune against it.** If a test result is disappointing, the response is more data or
  a different approach — not a new checkpoint chosen on that same test set.

### Level 4 — Does it work on the store's own footage? (before deployment)

The only level that measures the actual system. The three below it can all pass on a model
that fails here.

- **Field footage inference.** Run `hydranet-infer-video` on real site recordings and watch
  the overlay. Polished-floor reflections, storefront glass, and night IR are what to look
  for — and watch a *held-out* camera, or you are watching the training set.
- **Frame consensus on a fixed camera**, which needs no labels at all: a pixel that changes
  class across a static scene is one the model was unsure about. See §2.
- **Export parity.** Confirm the ONNX/TensorRT output matches PyTorch on the same input.
  A silent mismatch here — usually pre-processing — is the classic deployment bug, and the
  contract is carried by the input binding's *name*: `image_rgb_255` means the graph
  normalises and the runtime hands over **raw 0–255 RGB**, while `images` means the old
  convention where the runtime owns mean/std. Normalising against the first does it twice,
  which costs accuracy and raises nothing. A BGR feed handed over unconverted costs 5–10
  points just as quietly. [DEPLOY.md](DEPLOY.md) §1 and §5 have both halves.
- **Latency on the target board**, not on the workstation.

### Acceptance gates before deployment

Written when every one was unmet. **Two have since been met** (2026-08-19, cutting
`releases/v1`): export parity was confirmed at worst relative divergence 7.95e-06 across 17
outputs, and the export guard that refuses an unsupervised head is what let the bundle build
at all. The rest still stand, and [RELEASE.md](RELEASE.md) §3 carries the current list:

| Gate | Requirement |
|---|---|
| Every head is supervised | Nothing exports with initial weights. Detection needs COCO, or the head must be removed from the export |
| A `test` split exists | Configured, populated, **split by camera**, never trained on — and human-corrected, or the number is an agreement with a teacher rather than an accuracy |
| Per-class floors met | Each rare class above an agreed threshold, reported with its support — set it with whoever owns the product, not by whoever is training |
| Field footage reviewed | A human has watched the overlay on real site video |
| Export parity confirmed | TensorRT output matches PyTorch within tolerance |
| Provenance intact | `meta.json` identifies code, config and dataset fingerprint for the exact checkpoint being shipped |

---

## 6. Keeping runs answerable

Every run writes `meta.json` (commit, dirty flag, environment, dataset fingerprints, resolved
config), `config.yaml`, `uncommitted.patch` when the tree was dirty, `metrics.jsonl`,
`train.log` and `tb/`. A second run into an occupied directory gets a timestamped sibling
rather than overwriting `best.pt`.

```bash
uv run hydranet-report runs/*  --diff     # rank runs, and show what differed
```

Two habits make this pay off:

- **Never delete a run directory that produced a shipped model.** The dataset fingerprint in
  it is the only record of which data that model saw.
- **Know what a history rewrite does to run metadata.** Runs record the commit they came
  from, and the Conventional Commits rewrite changed every hash — so the baseline run's
  `meta.json` names `ba30fa88`, which `git` can no longer reach. **No code was lost:** the
  rewrite only edited commit messages, so the old and new commits have byte-identical
  trees (`git diff ba30fa88 aa07bbe` is empty). Recover the mapping by matching the commit
  *message* rather than the hash — the rewrite lower-cased the subject and added a type
  prefix, so `ba30fa88` is `aa07bbe`. [CONTRIBUTING.md](../CONTRIBUTING.md) carries the
  current mapping. If a future rewrite ever changes trees rather than messages, tag the
  commits a run directory references *before* doing it.

Workstream D was told to revisit the "no tracking server" call once comparing runs stopped
being possible by eye, and it has. `scripts/mlflow_sync.py` mirrors `meta.json` and
`metrics.jsonl` into a local MLflow store under `runs/mlflow/`, idempotently, so cron or
systemd can call it.

**The files remain the truth** — nothing in training depends on the sync and deleting
`runs/mlflow/` loses nothing it cannot rebuild. The trigger was not "training spans
machines"; it was the `coco10` incident of 2026-08-19, where the COCO share sat in every
`config.yaml` and still nearly produced a false contradiction, because nothing put many
runs' parameters side by side unless someone thought to ask.

### A run is not a version, and the gap between them was 54 to 0

Counted on 2026-08-19: **54 run directories, 0 releases.** Of the 49 carrying `meta.json`,
**31 were trained from a dirty tree** — their exact source exists only in
`uncommitted.patch`, so they cannot be released at all, and the check that would have said
so runs at release time, long after the eight hours were spent.

That ratio is the argument for two habits, and neither is about tooling:

- **Refuse a dirty tree at the start of training, not at release.** The warning the trainer
  prints today is read after the fact or not at all. The cost of ignoring it is a whole run.
- **Cut the release when the model ships, not when someone remembers.** `releases/v1` was
  cut months after `hydranet_joint_coco10` was first deployed, and cutting it immediately
  surfaced that the shipped checkpoint scores 0.360 terrain mIoU where the project's own
  notes had been quoting 0.491 — a per-head maximum from a different epoch. A model that is
  running somewhere with no frozen record is a model whose numbers drift in retelling.

[RELEASE.md](RELEASE.md) has the mechanism and the three gates. The point here is only that
the mechanism existed, was correct, and was worth nothing until it was run once.

---

## 7. Suggested order for a team starting today

1. **Week 1 — correct the test masks, before anything else.** Not capture: the footage is
   already on disk and the teachers have already pre-labelled it. What does not exist is a
   number anyone can defend, and §2's priority 1 is the reason. ~72 frames on the reserved
   cameras, corrected by hand. Workstream C builds the Orin runtime against the existing
   ONNX contract in parallel; Workstream B reproduces the current baseline so there is a
   known-good reference.
2. **Week 2 — re-measure everything against it.** Every site number this project has quoted
   becomes checkable for the first time, and some of them will move. Expect that, and
   expect it to be the most valuable week: a number that changes when it is finally
   measured properly is a number that was steering decisions while wrong.
3. **Week 3 — the labelled clip, then the classes.** One clip of person tracks gives `idf1`
   and `id_switches` their first real denominators. Then `column` on more cameras and the
   `fixture` review, in §2's order. Compare per-class with its support, not mIoU, for the
   reason in level 2.
4. **Then, and only then**, decide whether the model needs changing. It probably does not: at
   84.4% of parameters in a shared trunk exporting cleanly to TensorRT, the architecture is
   not what is holding this back.
