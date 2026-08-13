"""Export ONNX for conversion to a TensorRT engine on Jetson Orin.

    hydranet-export-onnx --config configs/hydranet_indoor.yaml \
        --checkpoint runs/hydranet_indoor/best.pt --output hydranet.onnx

The forward graph holds only convolution, resize and exp: no NMS and no dynamic control
flow, so trtexec fuses it in one piece. NMS and argmax stay in the host post-processing
code (see docs/DEPLOY_JETSON.md).
"""

from __future__ import annotations

import argparse

import torch
import torch.nn as nn

from ..config import load_config
from ..models.hydranet import build_model
from ..utils.checkpoint import load_checkpoint


class ExportWrapper(nn.Module):
    """Flatten the dict/list output into a fixed-order tuple of tensors, as ONNX needs."""

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.seg_names = list(model.seg_heads.keys())

    def forward(self, images):
        out = self.model(images)
        flat = [out[n] for n in self.seg_names]
        if self.model.det_head is not None:
            flat += list(out["det_cls"]) + list(out["det_reg"]) + list(out["det_ctr"])
        return tuple(flat)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="hydranet-export-onnx", description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--output", default="hydranet.onnx")
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--weights", choices=["ema", "model"], default="ema")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    return ap


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    cfg = load_config(args.config, args.set)
    model = build_model(cfg).eval()
    if args.checkpoint:
        ckpt = load_checkpoint(args.checkpoint)
        state = (
            ckpt.get("ema") if (args.weights == "ema" and ckpt.get("ema")) else ckpt["model"]
        )
        model.load_state_dict(state)

    h, w = cfg["export"]["input_size"]
    dummy = torch.randn(args.batch, 3, h, w)
    wrapper = ExportWrapper(model)

    out_names = list(wrapper.seg_names)
    if model.det_head is not None:
        n_lv = len(model.det_head.in_levels)
        out_names += [f"det_cls_p{i + 3}" for i in range(n_lv)]
        out_names += [f"det_reg_p{i + 3}" for i in range(n_lv)]
        out_names += [f"det_ctr_p{i + 3}" for i in range(n_lv)]

    torch.onnx.export(
        wrapper,
        dummy,
        args.output,
        input_names=["images"],
        output_names=out_names,
        opset_version=int(cfg["export"].get("opset", 17)),
        do_constant_folding=True,
        dynamo=False,
    )
    print(f"exported {args.output}")
    print("output nodes:", out_names)

    try:
        import onnx

        m = onnx.load(args.output)
        onnx.checker.check_model(m)
        print("ONNX check passed")
        try:
            from onnxsim import simplify

            m, ok = simplify(m)
            if ok:
                onnx.save(m, args.output)
                print("onnxsim simplification done")
        except ImportError:
            pass
    except ImportError:
        print("(onnx not installed, skipping check)")

    print("\nOn Jetson Orin, build the TensorRT engine with:")
    print(f"  trtexec --onnx={args.output} --saveEngine=hydranet.engine --fp16")


if __name__ == "__main__":
    main()
