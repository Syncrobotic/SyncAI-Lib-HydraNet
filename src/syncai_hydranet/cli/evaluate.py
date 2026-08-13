"""Run validation on a checkpoint.

    hydranet-eval --config configs/hydranet_indoor.yaml \
        --checkpoint runs/hydranet_indoor/best.pt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..config import load_config
from ..data.datasets import SPLITS, build_dataset
from ..engine.evaluator import evaluate
from ..models.hydranet import build_model
from ..utils.checkpoint import load_checkpoint
from ..utils.device import pick_device
from ..utils.logger import get_logger
from ..utils.runmeta import git_state


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="hydranet-eval", description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument(
        "--weights",
        choices=["ema", "model"],
        default="ema",
        help="EMA weights need enough training steps to be meaningful; see docs/TRAIN_MACOS.md",
    )
    ap.add_argument(
        "--split",
        choices=SPLITS,
        default="val",
        help=(
            "val is what training selected best.pt on, so it is optimistic; report "
            "final numbers on test, which no part of training ever reads"
        ),
    )
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    ap.add_argument(
        "--json",
        default=None,
        metavar="PATH",
        help="also write the metrics as JSON, for comparing runs without parsing logs",
    )
    return ap


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    cfg = load_config(args.config, args.set)
    device = pick_device(cfg.get("device"))
    logger = get_logger("eval")
    logger.info(f"device={device}, split={args.split}")

    model = build_model(cfg).to(device).eval()
    ckpt = load_checkpoint(args.checkpoint)
    state = ckpt.get("ema") if (args.weights == "ema" and ckpt.get("ema")) else ckpt["model"]
    model.load_state_dict(state)

    input_size = cfg["data"]["input_size"]
    lb = bool(cfg["data"].get("letterbox", False))
    eval_sets = [
        (d["name"], build_dataset(d, input_size, args.split, letterbox=lb))
        for d in cfg["data"]["datasets"]
    ]
    metrics = evaluate(model, eval_sets, cfg, device, logger)
    for k, v in metrics.items():
        logger.info(f"{k}: {v:.4f}")

    if args.json:
        record = {
            "checkpoint": args.checkpoint,
            "epoch": ckpt.get("epoch"),
            "weights": args.weights,
            "split": args.split,
            "git": git_state(),
            **metrics,
        }
        out = Path(args.json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(record, indent=2, default=float) + "\n")
        logger.info(f"wrote {out}")


if __name__ == "__main__":
    main()
