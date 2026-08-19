# 2026-08-19 — planner session handoff

Handoff from the planner session that ran the security/retail pivot day. The decision
record with every measurement is the companion entry
(`2026-08-19-security-retail-teachers-and-methodology.md`) — read it first; this file
carries only the live coordination state a successor cannot reconstruct from it.

## The day in one paragraph

Goal fixed: security+retail CCTV model is the primary line (robot line removed from the
tree). The person teacher swapped to GDINO@0.35 and retrained through three seeds at no
cost to other heads; two human-verified track clips exist; output stabilisation (EMA +
track rendering) is decided into the main path; per-camera calibration is an
install-time artifact (orientation automatic, metre from priors ±5–8%); the 96-stream
target has 2.23x compute margin with copies off the critical path (16-stream pilot
holds 14.6/15 fps); the disposition lake, zone proposals, MLflow, and the night
person veto (13 counter-example cameras, one-directional static share) all shipped.
The site30k data campaign is mid-flight, gated on a one-frame-at-a-time floor recipe
the user is personally approving.

## Live state (the part that is not in the decision record)

### The floor gate — THE active loop with the user
- Process rule (user-imposed, in memory): ONE frame → user confirms → ~10-frame small
  batch → scale. v1–v3 were rejected at batch size; never accumulate versions again.
- Current: v4.1 single frame (K04 frame 0002) passed my check and user review with one
  remaining complaint: **fragmentation at edges and in the far field**. v4.2 directive
  is with the campaign agent: superpixel edge-snapping + regularisation moved into BEV
  metric space (perspective-correct kernels) + depth-scaled on-plane tolerance.
  Deliverable: same frame + edge/far-field before/after crops. **Verify it personally
  (every pixel defensible), then hand to the user. Do not proceed to the small batch
  without their 確認.**

### The site30k campaign agent (very long context — consider replacing it after the
floor recipe is approved; its recipe knowledge is all in runs/site30k_qa/progress.log)
- GCS pull: ~17/31 days done at last report, 0 failures, ~46 GB expected. Auth works
  (user re-logged admin@syncrobotic.ai). Bucket surveyed: 48 cams × 32 days × 24 h,
  7.14 TB total, live ingest.
- Phase 3 annotation LOCKED until the floor gate passes. Then classes one at a time
  (user rule), detection taxonomy: person / pet(schema-only, zero real pets measured) /
  laptop / tablet / phone / boxed_stock. Sideways cameras (5, verified by eye) rotate
  first; night tranche uses scripts/night_person_filter.py (13 named counter-example
  cameras; K02/TH09 excluded by name).
- Known campaign debt: annotate() overwrites instances_*.json per invocation —
  incremental merge needed before scale-out.

### Other sessions on this checkout
- syncai-lib-hydranet-13: closed out the column line (5 trainable cameras, two-camera
  pixel concentration, 3/4 time-slot rule), leak guard (clean_against digests), and the
  night veto. Standby or gone; its work is all committed.
- The a0 audit session (doc-sync duty) is gone; **doc-sync baton is with the planner**.
  A "2c" session existed, mandate unknown — do not conscript without asking the user.
- Working-tree hygiene rules that earned their place today: never `git add -A`; verify
  ownership before touching any uncommitted file (the writer set is larger than the
  session list); pkill patterns must not match your own command line.

### Pending user decisions (do not re-ask, just don't decide for them)
- RetailS licence inquiry email (draft was provided; user sends it).
- AgeOver60: no data source exists anywhere (PA-100K degrading, RAP v2 zero positives
  in the identity split) — keep in product spec only if data acquisition is planned.
- CCTV person-attribute inference (age/gender) legal check before it enters the SDK.
- README repositioned to CCTV-primary already; no further positioning question open.

### Queued work, in order (from the approved plan)
1. Floor gate → classes one-by-one → Phase 3 (30k+) → the big retrain with the 7-class
   hardware taxonomy (display_table/shelf split; product-on-table by composition).
2. Serving increment 2: NVDEC (the named bottleneck: CPU decode + Python post), post
   scaling (EMA on-GPU), static-composite skip measured on TRT, per-class thresholds
   fed from sweeps, engine provenance.
3. Attributes round 2 prerequisites (all three before GPU): identity loss beside BCE,
   rap_* columns wired (rap_Employee 6.15% = staff's first source), in-domain retrieval
   protocol. The BCE-only retrain was REJECTED (embedding fell below the untrained
   floor).
4. Pose residency (ViTPose as selective L1 stage), metric-space fall semantics via
   calib (image-plane angles are viewpoint-broken), zone confirm-UI.
5. R3 human-judged test pass (keep/drop per component) — the only route from
   "agreement" to "accuracy" until operator dispositions accumulate.

## Standing user rules (all in MEMORY.md — the index is current; read it)

The ones that bite hardest: interrupt on design flaws immediately; verify every camera
view personally; one sample until confirmed; 3-minute demo videos, frame-checked;
source code in English, replies in Traditional Chinese; docs updated the moment
decisions land; warn before context handoff (this file is that warning's product).

## Final state at handoff (v4.2 delivered)

The v4.2 single frame (`runs/site30k_qa/preview/K04_0002_v42_single.jpg` + edge/far
before-after crops) passed the planner's own check and is **with the user for 確認**.
Both complaints measurably addressed: BEV metric-space regularisation + depth-scaled
tolerance fixed the far field (coverage 99→105% of the v4.1 far-field baseline), SLIC
snapping fixed edge fragmentation; attribution stays complete (OTHER 0.0%,
regularisation cost named at 4.3%). One named residual for review: thin pillar-skirt
collars ≤ tol(r) where both teachers blur the seam at 6 m — visible in
`_v42_pillar_zoom.jpg`, structurally bounded. Successor: on user approval run the
~10-frame small batch (same recipe, three pilot cameras), verify every frame
personally, get approval again, then unlock Phase 3 floor-first annotation.
