"""Export ONNX for conversion to a TensorRT engine on Jetson Orin.

    hydranet-export-onnx --config configs/hydranet_indoor.yaml \
        --checkpoint runs/hydranet_indoor/best.pt --output hydranet.onnx

The forward graph holds only convolution, resize and exp: no NMS and no dynamic control
flow, so trtexec fuses it in one piece. NMS stays in the host post-processing code (see
docs/DEPLOY_JETSON.md).

Two flags move work off the host, and both change the output contract rather than the
model -- the weights are untouched and nothing is retrained:

* ``--argmax-seg`` folds the segmentation argmax into the graph. It was the largest single
  item in the frame on a GB10, larger than the engine itself.
* ``--detection-classes`` slices the class channels the deployment does not read, which is
  where the host's sigmoid cost lives.

Together, measured end to end: 6.69 -> 3.29 ms, and 2.25 ms with a CUDA graph. Both rename
the bindings they change, so a runtime written for the old contract fails to find them
rather than misreading them.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn

from ..config import load_config
from ..config_schema import unsupervised_heads
from ..data.coco_subsets import (
    COCO_NAMES,
    EXPORT_SUBSETS,
    head_order,
    narrow_indices,
    resolve_export_subset,
)
from ..models.hydranet import build_model
from ..preprocessing import IMAGENET_MEAN, IMAGENET_STD
from ..utils.checkpoint import load_checkpoint, select_weights

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

    ``argmax_seg`` does the same thing at the other end of the graph, and is the larger
    win of the two. Measured on a GB10 at 512x640, single-thread, real decode, median of
    three, in milliseconds:

        build                    infer   d2h  terrain  detect  TOTAL     fps
        shipped                   2.09  0.38     2.53    1.70   6.69  144-150
        --argmax-seg              2.71  0.23     0       1.71   4.00  203-250
        + --detection-classes     2.71  0.17     0       0.64   3.29  284-304
        + CUDA graph              1.49  0.15     0       0.62   2.25  381-444

    **The host argmax was the largest single item in the frame, larger than the engine**,
    and it was on nobody's lever list -- DEPLOY_JETSON.md §4 lists four ways to make the
    engine smaller and the engine was 31% of the problem. The D2H transfer falls with it,
    from 9.18 MB of float logits to 0.33 MB of uint8 class ids.

    **Read the `infer` column before benchmarking this.** Folding the argmax in makes the
    *engine* slower -- 2.09 to 2.71 ms -- because the work did not vanish, it moved onto
    the GPU where it is cheap. Anyone who measures with ``trtexec`` alone sees a 25%
    latency regression and reverts the flag, correctly by their measurement and wrongly by
    two milliseconds a frame. **This is a frame win measured end to end, and it cannot be
    seen from the engine.** Quote it with the whole table or not at all.

    What it costs beyond that is the logits: no confidence, no soft blend, no per-class
    probability.

    What it costs is the logits. Anything wanting a confidence, a soft blend or a
    per-class probability has to export without this. That is why it is a flag and not the
    default, and why the output binding is renamed rather than silently changing dtype and
    rank underneath a host that would keep running.
    """

    def __init__(self, model, embed_preprocessing: bool = True, argmax_seg: bool = False):
        super().__init__()
        self.model = model
        self.seg_names = list(model.seg_heads.keys())
        self.embed_preprocessing = embed_preprocessing
        self.argmax_seg = argmax_seg
        if argmax_seg:
            wide = {
                name: head.num_classes
                for name, head in model.seg_heads.items()
                if head.num_classes > 256
            }
            if wide:
                raise SystemExit(
                    "cannot fold argmax into the graph for head(s) "
                    + ", ".join(f"{n} ({c} classes)" for n, c in wide.items())
                    + ": the class map is uint8, which tops out at 256. Export without "
                    "--argmax-seg and argmax on the host."
                )
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

    @property
    def seg_output_names(self) -> list[str]:
        """Binding names for the segmentation heads.

        Suffixed when the graph emits class ids rather than logits. A host that keeps
        calling ``argmax`` on what it thinks are logits gets a wrong-rank array out of a
        uint8 class map -- sometimes a crash, sometimes a picture. Renaming turns it into
        a missing binding, which is neither.
        """
        suffix = "_argmax" if self.argmax_seg else ""
        return [f"{n}{suffix}" for n in self.seg_names]

    def forward(self, images):
        if self.embed_preprocessing:
            images = (images - self.pre_mean) / self.pre_std
        out = self.model(images)
        # argmax after the head's own bilinear upsample, not before it: taking it at P3
        # and resizing the class map with nearest would be a different answer at every
        # boundary, and a cheaper one nobody asked for. This way the graph computes
        # exactly what the host was computing.
        flat = [
            out[n].argmax(dim=1).to(torch.uint8) if self.argmax_seg else out[n]
            for n in self.seg_names
        ]
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


def trained_detection_classes(cfg: dict, num_classes: int) -> list[str]:
    """The class list the detection head's channels actually mean, in channel order.

    Read from the config's COCO block rather than assumed, because a head narrowed at
    *training* time via ``data.datasets[].classes`` has channel 0 meaning whatever that
    list's first class is in category-id order -- not ``person``. Assuming COCO's 80 there
    would rename every box and nothing would look wrong.
    """
    for ds in (cfg.get("data") or {}).get("datasets") or []:
        if not isinstance(ds, dict) or ds.get("type") != "coco":
            continue
        if "detection" not in (ds.get("supervises") or []):
            continue
        subset = ds.get("classes")
        names = head_order(subset) if subset else list(COCO_NAMES)
        break
    else:
        names = list(COCO_NAMES)

    if len(names) != num_classes:
        raise SystemExit(
            f"cannot name the detection head's channels: the config's COCO block implies "
            f"{len(names)} classes but model.heads.detection.num_classes is {num_classes}. "
            f"Narrowing would map names onto the wrong channels, silently. Fix the config "
            f"so the two agree before exporting a narrowed engine."
        )
    return names


def narrow_detection_head(model, keep_idx: list[int]) -> None:
    """Slice the classification convolution down to ``keep_idx``, in place.

    Slicing the weights rather than gathering channels in the graph: the exported ONNX
    then has a smaller convolution and no extra operator at all, so the saving is in the
    head's own arithmetic as well as in the host's sigmoid. A Gather would have cost a
    node and saved only the latter.

    Done after ``load_state_dict``, and it mutates the model -- which is fine here because
    nothing in this process uses it again, and would not be fine anywhere else.
    """
    conv = model.det_head.cls_pred
    narrowed = nn.Conv2d(
        conv.in_channels,
        len(keep_idx),
        kernel_size=conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
    )
    idx = torch.tensor(keep_idx, dtype=torch.long)
    with torch.no_grad():
        narrowed.weight.copy_(conv.weight[idx])
        narrowed.bias.copy_(conv.bias[idx])
    model.det_head.cls_pred = narrowed
    model.det_head.num_classes = len(keep_idx)


def det_output_names(n_levels: int, narrowed_to: int | None) -> list[str]:
    """Output binding names, with the class count in the ``det_cls`` name when narrowed.

    Same trick as ``INPUT_RAW`` / ``INPUT_NORMALISED``, on the output side. A host decoder
    written for the 80-class engine looks up ``det_cls_p3``, and against a narrowed engine
    it does not find it -- rather than finding 8 channels where it expects 80 and reporting
    ``zebra`` for a customer. ``det_reg`` and ``det_ctr`` keep their names because their
    shapes do not change, so a runtime that only wants boxes is unaffected.
    """
    tag = "" if narrowed_to is None else str(narrowed_to)
    return (
        [f"det_cls{tag}_p{i + 3}" for i in range(n_levels)]
        + [f"det_reg_p{i + 3}" for i in range(n_levels)]
        + [f"det_ctr_p{i + 3}" for i in range(n_levels)]
    )


def sidecar_path(output: str) -> Path:
    p = Path(output)
    return p.with_suffix(".classes.json") if p.suffix else Path(f"{output}.classes.json")


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
        if a.dtype.kind in "iu":
            # A class map. Relative error on label ids is meaningless -- ids 2 and 3 are
            # not "1 apart" in any sense the tolerance was set for -- so compare the only
            # thing that means anything, which is how many pixels disagree. Exported
            # argmax should be exact; a tie broken differently by two implementations
            # would show here rather than as a slightly-off float.
            rel = float((a != b).mean())
            label = "disagreeing pixels"
        else:
            scale = max(float(abs(a).max()), 1e-9)
            rel = float(abs(a - b).max()) / scale
            label = "relative"
        if rel > worst:
            worst, worst_name = rel, f"{name} ({label})"
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
    ap.add_argument(
        "--argmax-seg",
        action="store_true",
        help="emit uint8 class maps for the segmentation heads instead of float logits, "
        "folding the host's argmax into the graph. Measured on a GB10 at 512x640: the "
        "host argmax was 2.53 ms of a 6.70 ms frame -- larger than the engine -- and "
        "folding it in gives 4.00 ms, with --detection-classes 3.29 ms and with a CUDA "
        "graph 2.25 ms (444 fps). WARNING: this makes the ENGINE slower, 2.09 to 2.71 ms; "
        "it is a whole-frame win and trtexec alone shows it as a 25%% regression. Costs "
        "the logits, so anything needing a confidence or a soft blend must export without "
        "it. The bindings are renamed '<head>_argmax'",
    )
    ap.add_argument(
        "--detection-classes",
        default=None,
        metavar="SUBSET|name,name,...",
        help="narrow the detection head's class channels at export. A named subset ("
        + ", ".join(sorted(EXPORT_SUBSETS))
        + ") or a comma-separated list of COCO names. The checkpoint still trained on all "
        "80 -- this only stops the engine emitting, and the host decoding, channels the "
        "deployment does not read. The 'det_cls' bindings gain the class count, and the "
        "names are written to a <output>.classes.json sidecar",
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
        model.load_state_dict(select_weights(ckpt, args.weights))

    kept_names: list[str] | None = None
    kept_idx: list[int] = []
    trained: list[str] = []
    n_lv = 0 if model.det_head is None else len(model.det_head.in_levels)
    if args.detection_classes:
        if model.det_head is None:
            raise SystemExit(
                "--detection-classes was given but this model has no detection head; "
                "nothing to narrow."
            )
        trained = trained_detection_classes(cfg, model.det_head.num_classes)
        try:
            kept_names = resolve_export_subset(args.detection_classes)
            kept_idx = narrow_indices(kept_names, trained)
        except ValueError as exc:
            raise SystemExit(f"--detection-classes: {exc}") from exc
        narrow_detection_head(model, kept_idx)
        print(
            f"detection narrowed {len(trained)} -> {len(kept_names)} classes: "
            f"{', '.join(kept_names)}"
        )

    # The export section is optional: without it, exporting at the training resolution
    # is the sane default, and a TensorRT engine built for the wrong input size is a
    # much later and much more confusing failure than a missing config key.
    ecfg = cfg.get("export") or {}
    h, w = ecfg.get("input_size") or cfg["data"]["input_size"]
    print(f"exporting at {h}x{w}")
    embed = not args.no_embed_preprocessing
    wrapper = ExportWrapper(model, embed_preprocessing=embed, argmax_seg=args.argmax_seg)
    # Exercise the graph over the range it will actually see. Feeding a 0-255 graph
    # standard-normal noise would make the parity check pass on inputs no camera produces.
    dummy = (
        torch.rand(args.batch, 3, h, w) * 255.0 if embed else torch.randn(args.batch, 3, h, w)
    )
    print(
        f"input '{wrapper.input_name}': "
        + ("raw RGB 0-255, normalisation is inside the graph" if embed else "pre-normalised")
    )

    out_names = list(wrapper.seg_output_names)
    if model.det_head is not None:
        out_names += det_output_names(n_lv, len(kept_names) if kept_names else None)

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

    if args.argmax_seg:
        logits = sum(h.num_classes for h in model.seg_heads.values()) * h * w * 4
        ids = len(model.seg_heads) * h * w
        print(
            f"segmentation D2H drops from {logits / 1e6:.2f} MB of float logits to "
            f"{ids / 1e6:.2f} MB of uint8 class ids, and the host no longer argmaxes"
        )

    if kept_names is not None:
        # A TensorRT engine keeps its binding names and nothing else, so the ONNX
        # metadata below never reaches the board. The sidecar is what does -- it is the
        # only record of which COCO class each remaining channel means, and an engine
        # shipped without it is a set of unlabelled numbers.
        with torch.no_grad():
            cls_out = wrapper(dummy)[len(wrapper.seg_names) : len(wrapper.seg_names) + n_lv]
        positions = sum(int(t.shape[-1] * t.shape[-2]) for t in cls_out)
        side = sidecar_path(args.output)
        side.write_text(
            json.dumps(
                {
                    "detection_classes": kept_names,
                    "source_indices": kept_idx,
                    "cls_outputs": out_names[len(wrapper.seg_names) :][:n_lv],
                    "input_size": [h, w],
                },
                indent=2,
            )
            + "\n"
        )
        before = positions * len(trained)
        after = positions * len(kept_names)
        print(f"wrote {side}")
        print(
            f"host sigmoid drops from {before:,} to {after:,} values per frame "
            f"({before / after:.1f}x) at {h}x{w}, over {positions:,} positions"
        )

    try:
        import onnx

        m = onnx.load(args.output)
        onnx.checker.check_model(m)
        # The input name carries the contract into the TensorRT engine, which does not
        # keep these properties; they are here for anyone reading the ONNX itself.
        props = {
            "preprocessing": "embedded" if embed else "external",
            "input_range": "0-255" if embed else "imagenet-normalised",
            "input_layout": "NCHW",
            "channel_order": "RGB",
        }
        if kept_names is not None:
            props["detection_classes"] = ",".join(kept_names)
        props["segmentation_output"] = "class_ids_uint8" if args.argmax_seg else "logits"
        for key, value in props.items():
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
