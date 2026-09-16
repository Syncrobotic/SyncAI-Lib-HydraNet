#!/usr/bin/env python3
"""Export partial StudioA semantic targets and check real-batch loss wiring offline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from syncai_hydranet.data.store_split import STORES
from syncai_hydranet.data.studioa_review import digest, write_json
from syncai_hydranet.data.studioa_supervision import (
    CLASSES,
    IGNORE,
    StudioAPartialDataset,
    check_supervision,
    export_supervision,
)


def smoke(root: Path, held_out: str, out: Path) -> dict:
    import torch

    from syncai_hydranet.data.multitask import collate
    from syncai_hydranet.models.hydranet import build_model

    out.mkdir(parents=True, exist_ok=False)
    write_json(out / "status.json", {"status": "running"})
    try:
        torch.set_num_threads(4)
        torch.manual_seed(42)
        ds = StudioAPartialDataset(root, held_out, "train", input_size=(128, 192))
        indices = [i for i, frame in enumerate(ds.frames) if frame["valid_pixels"] > 0][:2]
        if len(indices) < 2:
            raise ValueError("need two train frames with supervised pixels for smoke check")
        batch = collate([ds[i] for i in indices])
        target = batch["targets"]["scene"]
        if not (target != IGNORE).any():
            raise ValueError("no supervision remains at smoke resolution")
        cfg = {
            "model": {
                "backbone": {"name": "resnet18", "pretrained": False},
                "neck": {"name": "fpn", "out_channels": 32, "num_levels": 5},
                "heads": {
                    "scene": {
                        "type": "semantic_fpn",
                        "num_classes": len(CLASSES),
                        "channels": 32,
                    }
                },
                "loss_balancing": "fixed",
                "fixed_weights": {"scene": 1.0},
            }
        }
        model = build_model(cfg).train()
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
        classifier = dict(model.named_parameters())["seg_heads.scene.classifier.weight"]
        before = classifier.detach().clone()
        outputs = model(batch["image"])
        logits = outputs["scene"]
        logits.retain_grad()
        loss, _ = model.compute_losses(outputs, batch["targets"], batch["supervises"])
        if not torch.isfinite(loss):
            raise ValueError("nonfinite loss")
        loss.backward()
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        if not grads or not all(torch.isfinite(g).all() for g in grads):
            raise ValueError("missing or nonfinite parameter gradients")
        assert logits.grad is not None
        ignored = logits.grad.permute(0, 2, 3, 1)[target == IGNORE]
        if ignored.numel() and torch.count_nonzero(ignored):
            raise ValueError("ignored pixels affected loss gradients")
        optimizer.step()
        delta = (classifier.detach() - before).norm().item()
        if delta == 0:
            raise ValueError("optimizer did not update classifier")
        repository = Path(__file__).resolve().parents[2]
        code = [*(repository / "src").rglob("*.py"), Path(__file__).resolve()]
        report = {
            "status": "passed",
            "kind": "one_step_wiring_check_not_trained_model",
            "dataset_manifest_sha256": digest(root / "manifest.json"),
            "held_out_store": held_out,
            "partition": "train",
            "seed": 42,
            "frames": [ds.frames[i]["id"] for i in indices],
            "model_config": cfg,
            "input_size": [128, 192],
            "device": "cpu",
            "torch": torch.__version__,
            "loss": loss.item(),
            "classifier_update_norm": delta,
            "valid_pixels": int((target != IGNORE).sum()),
            "ignored_pixels": int((target == IGNORE).sum()),
            "ignored_logit_gradients_nonzero": 0,
            "accuracy_claim": False,
            "checkpoint_written": False,
            "code": {str(p.relative_to(repository)): digest(p) for p in sorted(code)},
        }
        write_json(out / "report.json", report)
        write_json(out / "status.json", {"status": "completed"})
        return report
    except Exception as exc:
        write_json(out / "status.json", {"status": "failed", "error": str(exc)})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    p = sub.add_parser("export")
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p = sub.add_parser("check")
    p.add_argument("--data", type=Path, required=True)
    p = sub.add_parser("smoke")
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--held-out", choices=STORES, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.action == "export":
        result = export_supervision(args.source, args.out)
        print(json.dumps({k: v for k, v in result.items() if k != "outputs"}))
    elif args.action == "check":
        result = check_supervision(args.data)
        print(
            json.dumps(
                {
                    "frames": len(result["frames"]),
                    "folds": {
                        name: {k: v for k, v in fold.items() if k != "assignments"}
                        for name, fold in result["folds"].items()
                    },
                }
            )
        )
    else:
        result = smoke(args.data, args.held_out, args.out)
        print(json.dumps({k: v for k, v in result.items() if k != "code"}))


if __name__ == "__main__":
    main()
