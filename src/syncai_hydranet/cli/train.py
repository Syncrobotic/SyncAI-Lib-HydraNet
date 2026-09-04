"""Training entry point.

    hydranet-train --config configs/hydranet_retail.yaml

    # override any config key with a dot path
    hydranet-train --config configs/hydranet_indoor.yaml \
        --set train.batch_size=8 model.neck.name=fpn 'data.input_size=[384,512]'

    # resume
    hydranet-train --config ... --resume runs/hydranet_indoor/last.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ..config import load_config
from ..engine.trainer import Trainer
from ..utils.runmeta import git_state


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="hydranet-train", description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--resume", default=None, help="checkpoint to resume from")
    ap.add_argument("--load", default=None, help="load weights only, do not resume optimizer")
    ap.add_argument(
        "--set", nargs="*", default=[], metavar="KEY=VALUE", help="dot-path config overrides"
    )
    ap.add_argument(
        "--allow-dirty",
        action="store_true",
        help="train from a working tree with uncommitted changes under src/ or this "
        "config; the run will not be releasable",
    )
    return ap


# Paths whose contents decide what the weights become. Everything else in the tree --
# docs, scripts, other configs -- can be dirty without changing this run's model.
WEIGHT_BEARING = ("src/",)


def refuse_dirty_tree(config_path: str, allow_dirty: bool) -> None:
    """Stop before training if the code that produces the weights is uncommitted.

    `release_bundle.sh` already refuses such a run, but it does so at release time --
    which on this project meant discovering after eight hours of GPU that the exact
    source exists only in `uncommitted.patch`. Counted on 2026-08-19: 31 of the 49 runs
    carrying `meta.json` were trained from a dirty tree and are unreleasable. This moves
    the same check to the one moment it costs nothing.

    **Scoped, not blanket, and deliberately so.** This repository is a shared checkout
    with several sessions working in it at once, so the tree is dirty most of the time
    for reasons that have nothing to do with the model. A dirty `docs/` cannot change a
    weight. `src/` and the config being trained can, so those are what is checked.

    `src/` is deliberately broader than the truth. A peer's uncommitted
    `cli/infer_video.py` cannot alter a weight either, and it will still block you --
    that happened on the first run of this check. Enumerating the genuinely weight-bearing
    modules would refuse less often and would eventually be wrong, letting a real change
    through silently. Over-refusing is the recoverable error, and `--allow-dirty` is the
    one-flag answer when you can see that the dirt is somebody else's.
    """
    git = git_state()
    if not git.get("available") or not git.get("dirty"):
        return
    cfg_rel = str(Path(config_path)).lstrip("./")
    offending = [
        f for f in git.get("dirty_files", []) if f.startswith(WEIGHT_BEARING) or f == cfg_rel
    ]
    if not offending:
        return
    listing = "\n  ".join(offending[:10])
    more = f"\n  ... and {len(offending) - 10} more" if len(offending) > 10 else ""
    if allow_dirty:
        print(
            f"warning: training from uncommitted changes in {len(offending)} "
            f"weight-bearing file(s); this run cannot be released:\n  {listing}{more}"
        )
        return
    raise SystemExit(
        f"refusing to train: {len(offending)} uncommitted file(s) decide what these "
        f"weights become, so the exact source would exist only in uncommitted.patch "
        f"and `release_bundle.sh` would refuse the result:\n  {listing}{more}\n\n"
        f"Commit them, or pass --allow-dirty to train a run you accept is unreleasable."
    )


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    # Before the config is even resolved: a refusal here costs a second, the same
    # refusal at release time costs the whole run.
    refuse_dirty_tree(args.config, args.allow_dirty)
    cfg = load_config(args.config, args.set)
    trainer = Trainer(cfg, resuming=bool(args.resume))
    if args.resume:
        trainer.load(args.resume, resume=True)
    elif args.load:
        trainer.load(args.load, resume=False)
    trainer.train()


if __name__ == "__main__":
    main()
