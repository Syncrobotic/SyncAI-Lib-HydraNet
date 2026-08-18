# tools/ — data production and dev tooling (upstream of training)

Everything here sits *upstream* of training: it makes the data and checks the inputs a run
depends on. It is not a deployment surface — that is [`../deploy/`](../deploy/), the
downstream end. Filing these under `deploy/` would confuse an input with an output.

## [`annotation/`](annotation/) — the CVAT stack

A version-pinned CVAT deployment (`cvat.sh` + an override over upstream's compose) that
produces the training labels for **both** products — the robot and retail-security train on
the same annotation pipeline. Full operator guide: [`docs/ANNOTATION_SETUP.md`](../docs/ANNOTATION_SETUP.md).
Site footage is customer premises, which is why the stack binds to loopback and is reached
through a tunnel, never a published port.

**Taxonomies the tooling actually gates.** `hydranet-annotation labels|check` (the
machine-checked half of the setup doc, `src/syncai_hydranet/cli/annotation.py`) validates
against `--scheme` = `indoor` / `retail` / `retail_objects`. There is **no `retail_surfaces`
scheme yet**, so product B's 6-class retail-security taxonomy cannot currently be annotated
or gated through this path — a known gap, not a supported case. Do not read "annotation
supports our taxonomies" into this directory until that scheme exists.
