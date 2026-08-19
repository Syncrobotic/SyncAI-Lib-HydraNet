# tools/ — data production and dev tooling (upstream of training)

Everything here sits *upstream* of training: it makes the data and checks the inputs a run
depends on. It is not a deployment surface — that is [`../deploy/`](../deploy/), the
downstream end. Filing these under `deploy/` would confuse an input with an output.

## [`annotation/`](annotation/) — the CVAT stack

A version-pinned CVAT deployment (`cvat.sh` + an override over upstream's compose) that
produces the training labels for **both** products — the robot and retail-security train on
the same annotation pipeline. Full operator guide: [`docs/METHODOLOGY.md`](../docs/METHODOLOGY.md).
Site footage is customer premises, which is why the stack binds to loopback and is reached
through a tunnel, never a published port.

**Taxonomies the tooling actually gates.** `hydranet-annotation labels|check` (the
machine-checked half of the setup doc, `src/syncai_hydranet/cli/annotation.py`) validates
against `--scheme` = `indoor` / `retail` / `retail_objects`. Product B's 6-class
retail-security taxonomy is absent from that list **and that is the design, not a gap** —
an earlier version of this paragraph called it a gap on the strength of a claim from the
annotation session that turned out to be wrong, and the correction is worth keeping
because the wrong version is the intuitive one.

Nothing is ever annotated in six classes. Masks are drawn in the seven-class
`retail_objects` taxonomy, gated as `retail_objects`, and read down to six at load time by
the `retail_surfaces_from_objects` label map — which has **seven** entries, because it is a
reader of object masks rather than a taxonomy anything authors. `hydranet_retail_surfaces`
and every config under it point `site_seg` at `datasets/retail_objects_batch02`; there is
no six-class dataset on disk and there should not be.

The direction matters and only works one way. Six is derivable from seven by folding
`product` into `fixture`; seven is not recoverable from six, because the boundary between
merchandise and the fixture holding it is the hardest one in the taxonomy and cannot be
guessed back. So annotating at the finer level and deriving the coarser is strictly better
than the reverse, and a `retail_surfaces` scheme would let someone draw the lossy version
by mistake.
