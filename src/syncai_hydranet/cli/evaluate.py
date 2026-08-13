"""Run validation on a checkpoint.

    hydranet-eval --config configs/hydranet_indoor.yaml \
        --checkpoint runs/hydranet_indoor/best.pt
"""

from __future__ import annotations

import argparse

import torch

from ..config import load_config
from ..data.datasets import build_dataset
from ..engine.evaluator import evaluate
from ..models.hydranet import build_model
from ..utils.device import pick_device
from ..utils.logger import get_logger


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
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    return ap


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    cfg = load_config(args.config, args.set)
    device = pick_device(cfg.get("device"))
    logger = get_logger("eval")
    logger.info(f"device={device}")

    model = build_model(cfg).to(device).eval()
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state = ckpt.get("ema") if (args.weights == "ema" and ckpt.get("ema")) else ckpt["model"]
    model.load_state_dict(state)

    input_size = cfg["data"]["input_size"]
    lb = bool(cfg["data"].get("letterbox", False))
    val_sets = [
        (d["name"], build_dataset(d, input_size, False, letterbox=lb))
        for d in cfg["data"]["datasets"]
    ]
    metrics = evaluate(model, val_sets, cfg, device, logger)
    for k, v in metrics.items():
        logger.info(f"{k}: {v:.4f}")


if __name__ == "__main__":
    main()
