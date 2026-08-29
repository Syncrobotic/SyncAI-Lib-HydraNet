"""How deep the `_base_` chains are allowed to get, ratcheted rather than gated.

`config._MAX_BASE_DEPTH` is 8, so the loader merges at most nine files for one config. The
deepest family here is already eight — `hydranet.yaml <- retail_objects <- retail_products
<- retail_surfaces <- retail_security <- b03 <- b03_cw <- seed7` — and the cost has already
been paid once in the open: `hydranet_retail_security_b03_cw_hires_seed13.yaml` says in its
own header that it cannot inherit its sibling because that would reach nine, so it restates
`input_size`. A value copied to get around an inheritance limit is the thing inheritance
exists to prevent.

**A ratchet and not a gate**, for the reason `scripts/ty_ratchet.sh` gives about turning a
check red on all of the debt at once. The depth is a measured fact that may fall and must
not rise; flattening the family is the real fix and its own file says to do it once the
comparison it belongs to is finished, not in the middle of it.

What this converts: the limit is enforced by `load_config`, which means today it is found
by a training run refusing to start. That is a slow way to learn it, and the person who
learns it is not the person who added the file.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from syncai_hydranet.config import _MAX_BASE_DEPTH

REPO = Path(__file__).resolve().parent.parent
CONFIGS = REPO / "configs"

# The deepest chain as measured on 2026-08-19. Lower this when the family is flattened;
# never raise it — a rise means the next config in that family cannot be written at all.
BASELINE_CHAIN = 8


def _parents() -> dict[str, str]:
    out: dict[str, str] = {}
    for f in CONFIGS.rglob("*.yaml"):
        base = (yaml.safe_load(f.read_text(encoding="utf-8")) or {}).get("_base_")
        if base:
            out[f.name] = Path(base if isinstance(base, str) else base[0]).name
    return out


def _chain(name: str, parents: dict[str, str]) -> list[str]:
    out = [name]
    seen = {name}
    while name in parents:
        name = parents[name]
        if name in seen:  # the loader raises on this; do not spin here
            break
        seen.add(name)
        out.append(name)
    return out


def _deepest() -> tuple[int, list[str]]:
    parents = _parents()
    chains = [_chain(f.name, parents) for f in CONFIGS.rglob("*.yaml")]
    longest = max(chains, key=len)
    return len(longest), longest


def test_no_config_chain_is_deeper_than_the_baseline():
    depth, chain = _deepest()
    assert depth <= BASELINE_CHAIN, (
        f"a config now merges {depth} files, up from {BASELINE_CHAIN}:\n  "
        + " <- ".join(chain)
        + f"\nThe loader merges at most {_MAX_BASE_DEPTH + 1}, so this family is "
        f"{_MAX_BASE_DEPTH + 1 - depth} file(s) from being unable to express another "
        "variant. Flatten it rather than raising this number — and see "
        "hydranet_retail_security_b03_cw_hires_seed13.yaml for what the limit already "
        "cost once."
    )


def test_the_baseline_is_not_left_stale_after_a_flattening():
    """A ratchet nobody lowers stops measuring anything. This is the other direction."""
    depth, _ = _deepest()
    assert depth >= BASELINE_CHAIN, (
        f"the deepest chain is now {depth} files, below the pinned {BASELINE_CHAIN}. "
        "Good — lower BASELINE_CHAIN to lock the improvement in."
    )


def test_the_loader_still_refuses_deeper_than_it_claims():
    """The number this file ratchets against is only meaningful if the loader enforces it."""
    assert _MAX_BASE_DEPTH == 8, (
        "_MAX_BASE_DEPTH moved. If it was raised to fit a deeper family, that is the "
        "change this ratchet exists to make visible: the limit is not the problem."
    )
