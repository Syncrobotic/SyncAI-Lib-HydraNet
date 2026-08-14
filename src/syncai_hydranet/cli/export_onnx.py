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
from ..config_schema import unsupervised_heads
from ..data.transforms import IMAGENET_MEAN, IMAGENET_STD
from ..models.hydranet import build_model
from ..utils.checkpoint import load_checkpoint

# The input name *is* the contract. A graph that normalises internally takes raw RGB in
# 0-255; one that does not takes an already-normalised tensor. Naming them differently
# means a runtime written for one fails to find its binding in the other, loudly, instead
# of feeding the wrong range and producing plausible nonsense.
INPUT_RAW = "image_rgb_255"
INPUT_NORMALISED = "images"


class ExportWrapper(nn.Module):
    """Flatten the dict/list output into a fixed-order tuple of tensors, as ONNX needs.

    With ``embed_preprocessing`` the graph also does its own normalisation, taking raw
    RGB in 0-255 rather than an ImageNet-normalised tensor.

    Why that is worth two extra operators: pre-processing parity between training and the
    robot is listed in METHODOLOGY.md as a deployment responsibility, and the repository
    was implementing it twice -- ``data/transforms.py`` for training and a hand-copied
    mean/std in ``scripts/bench_camera_orin.py`` for the Jetson. Nothing connects the two.
    Change one and no test fails, no error appears, and the model on the robot is simply
    worse in a way that gets blamed on quantisation or on the camera.

    Folded into the graph it stops being a discipline problem. The constants ship with the
    weights, TensorRT fuses them into the first convolution, and the robot's only job is
    to hand over pixels in the order and range the input name states.
    """

    def __init__(self, model, embed_preprocessing: bool = True):
        super().__init__()
        self.model = model
        self.seg_names = list(model.seg_heads.keys())
        self.embed_preprocessing = embed_preprocessing
        if embed_preprocessing:
            # Training divides by 255 and then normalises in 0-1 units. Scaling the
            # constants by 255 instead makes those two steps one, exactly.
            mean = torch.tensor(IMAGENET_MEAN, dtype=torch.float32).view(1, 3, 1, 1) * 255.0
            std = torch.tensor(IMAGENET_STD, dtype=torch.float32).view(1, 3, 1, 1) * 255.0
            self.register_buffer("pre_mean", mean)
            self.register_buffer("pre_std", std)
        # nn.Module defaults to training=True, and torch.onnx.export propagates the
        # wrapper's mode down the tree -- so without this the inner model comes back from
        # export in train mode, with BatchNorm switched to batch statistics. The exported
        # graph is traced in eval regardless, but anything that uses the model afterwards
        # in the same process silently gets different numbers.
        self.eval()

    @property
    def input_name(self) -> str:
        return INPUT_RAW if self.embed_preprocessing else INPUT_NORMALISED

    def forward(self, images):
        if self.embed_preprocessing:
            images = (images - self.pre_mean) / self.pre_std
        out = self.model(images)
        flat = [out[n] for n in self.seg_names]
        if self.model.det_head is not None:
            flat += list(out["det_cls"]) + list(out["det_reg"]) + list(out["det_ctr"])
        return tuple(flat)


def check_heads_are_trained(cfg: dict, allow: bool) -> None:
    """Refuse to export a head that no dataset ever supervised.

    An unsupervised head reaches the engine with its initial random weights and still
    produces output: the detection head emits boxes, and nothing downstream can tell
    them from real ones. Training only warns about this, because assembling datasets
    incrementally is normal. Export is the point where it becomes a deployed defect, so
    it is an error here.
    """
    stranded = unsupervised_heads(cfg)
    if not stranded or allow:
        if stranded:
            print(
                f"WARNING: exporting untrained head(s) {', '.join(sorted(stranded))} "
                f"because --allow-untrained-heads was given. Their outputs are random."
            )
        return
    datasets = [
        d.get("name")
        for d in (cfg.get("data") or {}).get("datasets") or []
        if isinstance(d, dict)
    ]
    raise SystemExit(
        f"refusing to export: head(s) {', '.join(sorted(stranded))} are supervised by "
        f"none of the configured datasets ({', '.join(str(d) for d in datasets) or 'none'}).\n"
        f"They would ship with initial random weights and emit output that nothing "
        f"downstream can distinguish from a real prediction.\n"
        f"Fix one of:\n"
        f"  - add a dataset that supervises them (detection needs COCO)\n"
        f"  - drop them from model.heads so they are not in the graph at all\n"
        f"  - pass --allow-untrained-heads for a deliberate shape-only export"
    )


def check_parity(wrapper, dummy, path: str, out_names: list[str], tol: float = 1e-4) -> bool:
    """Compare the exported graph against PyTorch on the same input.

    Compares **relative** error per output, not absolute. The outputs span three orders of
    magnitude -- centerness peaks near 2, while ``det_reg`` at P6 carries pixel distances
    up to ~480 -- so one absolute threshold either fires spuriously on the regression maps
    or is loosened until it can no longer see a real error in the logits.

    This catches a graph that exported wrongly. It does not catch a pre-processing
    mismatch on the robot, which is the other classic deployment failure; for that, feed a
    real frame through both paths.
    """
    try:
        import onnxruntime as ort
    except ImportError:
        print("(onnxruntime not installed: pip install 'syncai-hydranet[export]')")
        return True

    with torch.no_grad():
        ref = [t.numpy() for t in wrapper(dummy)]
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    got = sess.run(None, {sess.get_inputs()[0].name: dummy.numpy()})

    print(f"\nparity vs PyTorch ({len(out_names)} outputs, relative tolerance {tol:g})")
    worst, worst_name, failures = 0.0, "", []
    for name, a, b in zip(out_names, ref, got, strict=True):
        scale = max(float(abs(a).max()), 1e-9)
        rel = float(abs(a - b).max()) / scale
        if rel > worst:
            worst, worst_name = rel, name
        if rel > tol:
            failures.append((name, rel))
    for name, rel in failures:
        print(f"  FAIL {name:16s} relative {rel:.2e}")
    verdict = "PASS" if not failures else "FAIL"
    print(f"  worst: {worst_name} {worst:.2e}  -> {verdict}")
    return not failures


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="hydranet-export-onnx", description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--output", default="hydranet.onnx")
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--weights", choices=["ema", "model"], default="ema")
    ap.add_argument(
        "--allow-untrained-heads",
        action="store_true",
        help="export heads no dataset supervised; their outputs are random, so this is "
        "for shape-only exports and latency benchmarking, never for deployment",
    )
    ap.add_argument(
        "--check-parity",
        action="store_true",
        help="run the exported graph against PyTorch on the same input and fail if they "
        "disagree; the deployment acceptance gate, needs onnxruntime",
    )
    ap.add_argument(
        "--no-embed-preprocessing",
        action="store_true",
        help="export a graph that expects an already-normalised tensor, as before. The "
        "input is then named 'images' rather than 'image_rgb_255', and the runtime owns "
        "the mean/std -- which is the parity risk this defaults away from",
    )
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    return ap


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    cfg = load_config(args.config, args.set)
    check_heads_are_trained(cfg, args.allow_untrained_heads)
    model = build_model(cfg).eval()
    if args.checkpoint:
        ckpt = load_checkpoint(args.checkpoint)
        state = (
            ckpt.get("ema") if (args.weights == "ema" and ckpt.get("ema")) else ckpt["model"]
        )
        model.load_state_dict(state)

    # The export section is optional: without it, exporting at the training resolution
    # is the sane default, and a TensorRT engine built for the wrong input size is a
    # much later and much more confusing failure than a missing config key.
    ecfg = cfg.get("export") or {}
    h, w = ecfg.get("input_size") or cfg["data"]["input_size"]
    print(f"exporting at {h}x{w}")
    embed = not args.no_embed_preprocessing
    wrapper = ExportWrapper(model, embed_preprocessing=embed)
    # Exercise the graph over the range it will actually see. Feeding a 0-255 graph
    # standard-normal noise would make the parity check pass on inputs no camera produces.
    dummy = (
        torch.rand(args.batch, 3, h, w) * 255.0 if embed else torch.randn(args.batch, 3, h, w)
    )
    print(
        f"input '{wrapper.input_name}': "
        + ("raw RGB 0-255, normalisation is inside the graph" if embed else "pre-normalised")
    )

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
        input_names=[wrapper.input_name],
        output_names=out_names,
        opset_version=int(ecfg.get("opset", 17)),
        do_constant_folding=True,
        dynamo=False,
    )
    print(f"exported {args.output}")
    print("output nodes:", out_names)

    try:
        import onnx

        m = onnx.load(args.output)
        onnx.checker.check_model(m)
        # The input name carries the contract into the TensorRT engine, which does not
        # keep these properties; they are here for anyone reading the ONNX itself.
        for key, value in {
            "preprocessing": "embedded" if embed else "external",
            "input_range": "0-255" if embed else "imagenet-normalised",
            "input_layout": "NCHW",
            "channel_order": "RGB",
        }.items():
            entry = m.metadata_props.add()
            entry.key, entry.value = key, value
        onnx.save(m, args.output)
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

    if args.check_parity and not check_parity(wrapper, dummy, args.output, out_names):
        raise SystemExit("export parity check FAILED: see the per-output table above")

    if embed:
        print(
            "\nThe robot feeds raw RGB in 0-255, NCHW. It must NOT subtract a mean: the "
            "graph does that, and doing it twice is the failure this export prevents."
        )
    print("\nOn Jetson Orin, build the TensorRT engine with:")
    print(f"  trtexec --onnx={args.output} --saveEngine=hydranet.engine --fp16")


if __name__ == "__main__":
    main()
