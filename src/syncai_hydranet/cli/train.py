"""Training entry point.

    hydranet-train --config configs/hydranet_regnet800mf.yaml

    # override any config key with a dot path
    hydranet-train --config configs/hydranet_indoor.yaml \
        --set train.batch_size=8 model.neck.name=fpn 'data.input_size=[384,512]'

    # resume
    hydranet-train --config ... --resume runs/hydranet_indoor/last.pt
"""

from __future__ import annotations

import argparse

from ..config import load_config
from ..engine.trainer import Trainer


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="hydranet-train", description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--resume", default=None, help="checkpoint to resume from")
    ap.add_argument("--load", default=None, help="load weights only, do not resume optimizer")
    ap.add_argument(
        "--set", nargs="*", default=[], metavar="KEY=VALUE", help="dot-path config overrides"
    )
    return ap


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    cfg = load_config(args.config, args.set)
    trainer = Trainer(cfg)
    if args.resume:
        trainer.load(args.resume, resume=True)
    elif args.load:
        trainer.load(args.load, resume=False)
    trainer.train()


if __name__ == "__main__":
    main()
