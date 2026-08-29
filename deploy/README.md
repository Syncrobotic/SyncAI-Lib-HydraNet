# deploy/ — deployment surfaces

A **deployment surface** is where a trained model becomes a running thing in its target
environment, plus the ops to get it there. It is the *downstream* end of the pipeline.

This repo is **one core, one product**. It carried two until 2026-08-19, when the quadruped
line was removed:

| surface | product | environment | state |
|---|---|---|---|
| [`retail-security/`](retail-security/) | retail + security analytics | server-side, fixed store CCTV: L0 inference → L1 tracks → L2 events | **not assembled** — README only, no runtime here yet |

**Read that state column.** The only line this repository now has does not yet have code in
its deployment surface. That was worth saying when there were two surfaces and it is worth
saying more now: what ships is still an ONNX and a hand-run engine build, not a service.

The surface consumes the **core** (`src/syncai_hydranet`: the HydraNet model, `geometry/`,
training, ONNX export). The core never imports a surface. That one-way arrow is what let two
products be *consumers of one codebase* rather than two copies that drift, and it is why
removing one of them took no core changes at all — the strongest evidence the boundary was
real.

**What is not here:** *data production*. Annotation (CVAT) produces the training labels for
both products and sits *upstream* of training, so it lives in [`../tools/annotation/`](../tools/annotation/),
not under `deploy/`. Putting it here would confuse "how we make labels" with "where the
model runs."

**When to split this into separate repos** is currently a question with no subject: there
is one surface, and one surface cannot be split from anything. The criterion is kept
because it is what the directory is shaped around, and it becomes live again the moment a
second surface arrives — split only on a real forcing function, meaning surfaces that gain
separate teams, separate release cadences, or a security/IP boundary (one ships to a cloud
you run, another to customer hardware). Until then the argument for the monorepo is the one
above it: the core is consumed, never consuming, and that boundary was measured rather than
asserted when removing an entire product line took no core changes at all.
