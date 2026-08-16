"""A run's lifecycle: build it, step it, validate it, checkpoint it.

The three separable ideas that used to sit in this file are next door -- `ema.py` for the
weight average, `optim.py` for the parameter-group policy and the schedule. They left
because they are not about the loop: two test modules were already importing them past
`Trainer` to use on their own, which is the tell.

What stays is genuinely one thing. `Trainer` holds model, optimiser, scheduler, scaler,
EMA and the best-so-far metric, and every method here reads several of them at once;
splitting that further would move the coupling into signatures rather than remove it.
"""

from __future__ import annotations

import time
from pathlib import Path

import torch
import torch.nn as nn

from ..config import diff_config
from ..config_schema import check_config
from ..data.datasets import build_dataset
from ..data.fingerprint import fingerprint_dataset
from ..data.multitask import MultiTaskLoader
from ..models.hydranet import build_model
from ..utils.checkpoint import CKPT_FORMAT, load_checkpoint
from ..utils.device import (
    pick_device,
    supports_amp,
    supports_bfloat16,
    supports_pinned_memory,
)
from ..utils.logger import get_logger
from ..utils.runmeta import append_metrics, resolve_out_dir, write_run_meta
from ..utils.seeding import (
    apply_channels_last,
    configure_backends,
    model_memory_format,
    needs_grad_scaler,
    resolve_amp_dtype,
    seed_everything,
)
from ..utils.visualize import TRAV_COLORS, prediction_grid, terrain_palette
from .ema import ModelEMA
from .evaluator import build_val_loaders, evaluate, select_metric
from .optim import WarmupCosine, build_optimizer

DEFAULT_PRIMARY_METRIC = "traversability_mIoU"


def _targets_to_device(targets: dict, device) -> dict:
    out = {}
    for k, v in targets.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device, non_blocking=True)
        elif isinstance(v, list):
            out[k] = [t.to(device, non_blocking=True) for t in v]
        else:
            out[k] = v
    return out


def _build_datasets(dcfg, input_size):
    """Every dataset in the config, split into what trains and what selects.

    Returns `(train_sets, val_sets, names, ratios, val_names)`. Kept as a function
    returning values rather than a method assigning attributes, so the order the
    constructor does things in stays visible where it matters -- the loaders below need
    these, and the step count needs the loaders.
    """
    lb = bool(dcfg.get("letterbox", False))
    aug = dcfg.get("augment")
    train_sets, val_sets, names, ratios, val_names = [], [], [], [], []
    for ds in dcfg["datasets"]:
        train_sets.append(build_dataset(ds, input_size, "train", letterbox=lb, augment=aug))
        # A dataset may contribute training signal without joining checkpoint selection:
        # omit `split_val` and it is trained on but never validated on.
        #
        # This is not a convenience. The evaluator accumulates one confusion matrix per
        # head across every val set, so anything listed here lands in the number that
        # picks best.pt -- and on 2026-08-14 eight pseudo-labelled frames were enough to
        # read `display_fixture` 0.4224 against a 0.3612 baseline, which meant nothing
        # because the model was being scored against its own predecessor. A second
        # labelled domain is worth far more as a split nothing selects on than as
        # another voice in the selection.
        if ds.get("split_val"):
            # Validation never augments, so it takes no augment argument.
            val_sets.append(build_dataset(ds, input_size, "val", letterbox=lb))
            val_names.append(ds["name"])
        names.append(ds["name"])
        ratios.append(float(ds.get("sample_ratio", 1.0)))
    if not val_sets:
        raise ValueError(
            "no dataset declares split_val, so nothing can select a checkpoint; "
            "at least one is required"
        )
    return train_sets, val_sets, names, ratios, val_names


def _resolve_amp(tcfg, device):
    """Returns `(enabled, dtype, scaler)`, refusing a dtype this device cannot do."""
    amp = bool(tcfg.get("amp", True)) and supports_amp(device)
    # bfloat16, not float16, because that is what every run this project has produced
    # was actually trained in -- all of them passing it on the command line while the
    # default here said otherwise.
    amp_dtype = resolve_amp_dtype(str(tcfg.get("amp_dtype", "bfloat16")))
    if amp and amp_dtype is torch.bfloat16 and not supports_bfloat16(device):
        raise ValueError(
            "train.amp_dtype=bfloat16 needs an Ampere-or-newer CUDA device; this "
            f"one ({torch.cuda.get_device_name(device)}) does not support it. "
            "Set train.amp_dtype=float16 explicitly -- no run in this project has "
            "used fp16, so treat its first results as unvalidated."
        )
    return amp, amp_dtype, torch.amp.GradScaler(enabled=needs_grad_scaler(amp, amp_dtype))


def _make_ema(model, tcfg, total_iters: int, logger):
    """The EMA copy validation will run on, or None, plus the warning it may deserve."""
    if not tcfg.get("ema", True):
        return None
    decay = float(tcfg.get("ema_decay", 0.9998))
    warmup = int(tcfg.get("ema_warmup_steps", 2000))
    ema = ModelEMA(model, decay, warmup)
    # The ramp makes this rare rather than routine, but a run shorter than the ramp
    # itself can still validate on weights that are mostly noise, and that failure looks
    # exactly like a model that did not learn.
    residual = ema.residual_init_fraction(total_iters)
    if residual > 0.05:
        logger.warning(
            f"EMA warning: {total_iters} optimizer steps at decay={decay} "
            f"(warmup {warmup}) leave {residual:.0%} of the random "
            f"initialisation in the EMA weights used for validation. Scores "
            f"will be understated. Lower train.ema_warmup_steps or set "
            f"train.ema=false for runs this short."
        )
    return ema


def _open_tensorboard(out_dir: Path):
    """A SummaryWriter under `out_dir/tb`, or None if tensorboard is not importable.

    Best-effort on purpose: curves are worth having and never worth failing a run over,
    and `Trainer._log_images` and `_log_task_weights` both check for None.
    """
    try:
        from torch.utils.tensorboard import SummaryWriter

        return SummaryWriter(str(out_dir / "tb"))
    except Exception:
        return None


def _log_code_version(git: dict, out_dir: Path, logger) -> None:
    """Say which commit produced this run, or why that cannot be said.

    Datasets live outside git and checkpoints outlive branches, so "which code produced
    this" has to be answered at write time or not at all.
    """
    if not git.get("available"):
        logger.warning("not a git checkout: this run's code version is unrecorded")
    elif git["dirty"]:
        logger.warning(
            f"working tree is dirty at {git['commit'][:8]}; the exact code is only "
            f"recoverable via {out_dir / 'uncommitted.patch'}"
        )
    else:
        logger.info(f"code version: {git['commit'][:8]} ({git['branch']})")


class Trainer:
    def __init__(self, cfg, resuming: bool = False):
        self.cfg = cfg
        self.device = pick_device(cfg.get("device"))
        self.out_dir = resolve_out_dir(Path(cfg["output_dir"]), resuming=resuming)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.logger = get_logger("hydranet", self.out_dir / "train.log")
        if str(self.out_dir) != str(cfg["output_dir"]):
            self.logger.warning(
                f"{cfg['output_dir']} already holds a run; writing to {self.out_dir} "
                f"instead. Use --resume to continue the existing one."
            )
        for w in check_config(cfg):
            self.logger.warning(f"config: {w}")

        seed = int(cfg.get("seed", 42))
        seed_everything(seed)
        configure_backends(
            self.device,
            deterministic=bool(cfg["train"].get("deterministic", False)),
            cudnn_benchmark=bool(cfg["train"].get("cudnn_benchmark", True)),
            tf32=bool(cfg["train"].get("tf32", True)),
            logger=self.logger,
        )

        self.model = build_model(cfg).to(self.device)
        # Before the EMA copy is taken below: it deep-copies the model, so converting
        # afterwards would leave the weights validation actually runs on in NCHW.
        apply_channels_last(
            self.model,
            bool(cfg["train"].get("channels_last", False)),
            logger=self.logger,
        )
        self.memory_format = model_memory_format(self.model)
        tcfg = cfg["train"]
        dcfg = cfg["data"]
        input_size = dcfg["input_size"]

        train_sets, val_sets, names, ratios, val_names = _build_datasets(dcfg, input_size)
        self.train_loader = MultiTaskLoader(
            train_sets,
            names,
            ratios,
            batch_size=int(tcfg["batch_size"]),
            workers=int(dcfg.get("workers", 4)),
            seed=seed,
            pin_memory=supports_pinned_memory(self.device),
        )
        self.val_sets = list(zip(val_names, val_sets, strict=True))
        val_size = {n: len(v) for n, v in self.val_sets}
        skipped = [n for n in names if n not in val_size]
        if skipped:
            self.logger.info(
                f"trained on but not validated on: {', '.join(skipped)} "
                "(no split_val, so they take no part in choosing best.pt)"
            )
        # Built once: validation runs every epoch, and a DataLoader's worker pool is
        # expensive to create and pointless to throw away.
        self.val_loaders = build_val_loaders(self.val_sets, cfg, self.device)

        self.epochs = int(tcfg["epochs"])
        # Accumulation decouples the batch the optimiser sees from the batch that has to
        # fit in memory, so a laptop and an A100 can train the same effective batch
        # without retuning the LR.
        self.accum_steps = max(int(tcfg.get("grad_accum_steps", 1)), 1)
        self.steps_per_epoch = len(self.train_loader) // self.accum_steps
        if self.steps_per_epoch == 0:
            raise ValueError(
                f"grad_accum_steps={self.accum_steps} exceeds the "
                f"{len(self.train_loader)} batches in an epoch: no optimizer step "
                f"would ever run."
            )
        # The schedule advances once per optimizer step, not once per micro-batch.
        total_iters = self.epochs * self.steps_per_epoch
        self.optimizer = build_optimizer(self.model, tcfg)
        self.scheduler = WarmupCosine(
            self.optimizer, int(tcfg.get("warmup_iters", 500)), total_iters
        )

        self.amp, self.amp_dtype, self.scaler = _resolve_amp(tcfg, self.device)
        self.grad_clip = float(tcfg.get("grad_clip", 0.0))
        if self.accum_steps > 1:
            self.logger.info(
                f"gradient accumulation: {self.accum_steps} x batch "
                f"{tcfg['batch_size']} = effective batch "
                f"{self.accum_steps * int(tcfg['batch_size'])}, "
                f"{self.steps_per_epoch} optimizer steps/epoch"
            )
        if self.amp:
            self.logger.info(f"mixed precision: {self.amp_dtype}")

        self.ema = _make_ema(self.model, tcfg, total_iters, self.logger)

        # torch.compile, held as a *separate handle* rather than replacing self.model.
        #
        # `torch.compile(m)` returns an OptimizedModule wrapping m. It shares m's
        # parameters, so the optimizer and EMA see the same tensors either way -- but
        # its `state_dict()` prefixes every key with `_orig_mod.`. Assigning it to
        # self.model would therefore write checkpoints that neither `hydranet-eval`,
        # `hydranet-export-onnx` nor an older checkpoint loader can read, and the
        # breakage would surface at export time rather than here.
        #
        # So: self.model stays eager and owns state_dict, EMA and compute_losses;
        # self.forward_module is what the training step calls. Validation runs on the
        # EMA copy, which is eager by construction.
        self.forward_module = self.model
        if bool(tcfg.get("compile", False)):
            mode = str(tcfg.get("compile_mode", "default"))
            self.forward_module = torch.compile(self.model, mode=mode)
            self.logger.info(
                f"torch.compile enabled (mode={mode}); the first steps of epoch 1 "
                "include graph capture and will look stalled"
            )

        self.log_interval = int(tcfg.get("log_interval", 50))
        self.val_interval = int(tcfg.get("val_interval", 1))
        # One named metric decides best.pt. For a robot the honest choice is a
        # traversability number, not an average across heads: mistaking a wall for
        # floor is not compensated by a good mAP on chairs.
        self.primary_metric = str(tcfg.get("primary_metric", DEFAULT_PRIMARY_METRIC))
        self.logger.info(f"model selection: primary_metric={self.primary_metric}")
        self.tb = _open_tensorboard(self.out_dir)
        self.global_step = 0
        self.best_metric = -1.0
        self.start_epoch = 0  # last completed epoch; --resume advances it

        n_params = sum(p.numel() for p in self.model.parameters())
        meta = write_run_meta(
            self.out_dir,
            cfg,
            device=self.device,
            steps_per_epoch=self.steps_per_epoch,
            total_iters=total_iters,
            parameters=n_params,
            datasets=[
                {
                    "name": n,
                    "train_size": len(t),
                    # None, not 0: "not validated on" and "validated on nothing" are
                    # different facts and the run meta should not blur them.
                    "val_size": val_size.get(n),
                    # Datasets live outside git; without this, "which data produced
                    # this checkpoint" has no answer six months later.
                    **fingerprint_dataset(ds),
                }
                for n, t, ds in zip(names, train_sets, dcfg["datasets"], strict=True)
            ],
        )
        _log_code_version(meta["git"], self.out_dir, self.logger)

    # ------------------------------------------------------------------
    def train(self):
        if self.start_epoch >= self.epochs:
            self.logger.warning(
                f"checkpoint is already at epoch {self.start_epoch} of {self.epochs}: "
                f"nothing to train. Raise train.epochs to continue."
            )
            return
        self.logger.info(
            f"training: epochs {self.start_epoch + 1}..{self.epochs}, "
            f"{self.steps_per_epoch} optimizer steps/epoch, device={self.device}"
        )
        for epoch in range(self.start_epoch + 1, self.epochs + 1):
            self.train_one_epoch(epoch)
            if epoch % self.val_interval == 0:
                self.record_epoch(epoch, self.validate(epoch))
        self.logger.info("training complete")

    def record_epoch(self, epoch: int, metrics: dict) -> bool:
        """Update the best score, then write the checkpoints. Returns True on a new best.

        The order matters: best_metric has to be updated *before* last.pt is written,
        or last.pt carries the previous epoch's best and a resume from it re-accepts a
        worse model as the new best.
        """
        score = select_metric(metrics, self.primary_metric)
        is_best = score > self.best_metric
        if is_best:
            self.best_metric = score
        self.save("last.pt", epoch)
        if is_best:
            self.save("best.pt", epoch)
            self.logger.info(f"new best model ({self.primary_metric}={score:.4f})")
        return is_best

    def train_one_epoch(self, epoch: int):
        self.model.train()
        t0 = time.time()
        self.optimizer.zero_grad(set_to_none=True)
        micro_batches = self.steps_per_epoch * self.accum_steps  # drop the remainder
        for i, batch in enumerate(self.train_loader):
            if i >= micro_batches:
                # A partial accumulation group would take an optimizer step on fewer
                # samples than every other step, at a schedule position that assumed
                # otherwise. Cheaper to drop it than to explain it later.
                break
            # Layout is changed after the transfer, not during it: `.to(memory_format=)`
            # on the copy itself would give up the pinned-memory async path. When the
            # format is contiguous_format this returns the tensor unchanged.
            images = batch["image"].to(self.device, non_blocking=True)
            images = images.contiguous(memory_format=self.memory_format)
            targets = _targets_to_device(batch["targets"], self.device)
            with torch.amp.autocast(self.device.type, enabled=self.amp, dtype=self.amp_dtype):
                outputs = self.forward_module(images)
                loss, logs = self.model.compute_losses(outputs, targets, batch["supervises"])
            # Gradients sum across the group, so each micro-batch contributes its share
            # rather than a full step's worth.
            self.scaler.scale(loss / self.accum_steps).backward()

            if (i + 1) % self.accum_steps == 0:
                if self.grad_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
                self.scheduler.step()
                if self.ema:
                    self.ema.update(self.model)
                self.global_step += 1

            if (i + 1) % self.log_interval == 0:
                lr = self.optimizer.param_groups[0]["lr"]
                # The one place the loss tensors are read back. compute_losses returns
                # them detached but on-device precisely so this synchronisation happens
                # once per log_interval rather than three or four times per step.
                scalars = {k: float(v) for k, v in logs.items()}
                msg = " ".join(f"{k}={v:.3f}" for k, v in scalars.items())
                ips = self.log_interval * images.shape[0] / (time.time() - t0)
                t0 = time.time()
                self.logger.info(
                    f"E{epoch} [{i + 1}/{micro_batches}] "
                    f"ds={batch['dataset']} lr={lr:.2e} {msg} ({ips:.1f} img/s)"
                )
                if self.tb:
                    for k, v in scalars.items():
                        self.tb.add_scalar(f"train/{k}", v, self.global_step)
                    self.tb.add_scalar("train/lr", lr, self.global_step)
                    self.tb.add_scalar("train/img_per_sec", ips, self.global_step)
                    self._log_task_weights()

    # ------------------------------------------------------------------
    @torch.no_grad()
    def validate(self, epoch: int) -> dict:
        model = self.ema.ema if self.ema else self.model
        samples: dict | None = {} if self.tb else None
        metrics = evaluate(
            model,
            self.val_sets,
            self.cfg,
            self.device,
            self.logger,
            samples=samples,
            loaders=self.val_loaders,
        )
        append_metrics(
            self.out_dir,
            {
                "epoch": epoch,
                "global_step": self.global_step,
                "weights": "ema" if self.ema else "model",
                "primary_metric": self.primary_metric,
                **metrics,
            },
        )
        if self.tb:
            for k, v in metrics.items():
                # IoU/<head>/<class> already carries its own namespace.
                tag = k if k.startswith("IoU/") else f"val/{k}"
                self.tb.add_scalar(tag, v, self.global_step)
            self._log_images(samples)
        return metrics

    def _log_images(self, samples: dict | None) -> None:
        """Write "input | prediction | label" grids to TensorBoard.

        Curves only say the loss is falling. This says whether the model is calling a
        whole floor a wall, and how much of the frame is ignore padding.
        """
        # The one caller already guards on `self.tb`, and `samples` is only ever built
        # when it is set -- but that is the caller's invariant, not this method's. State
        # it here so the method is safe to call on its own and the writer is a writer.
        if self.tb is None:
            return
        classes = self.cfg["data"].get("terrain_classes")
        for head, (imgs, preds, gts) in (samples or {}).items():
            if head == "traversability":
                palette = TRAV_COLORS
            else:
                # n_classes is the fallback for a config that names no classes; a config
                # that does name them gets the palette built for that taxonomy.
                palette = terrain_palette(classes, self.model.seg_heads[head].num_classes)
            grid = prediction_grid(imgs, preds, gts, palette)
            self.tb.add_image(f"val_pred/{head}", grid, self.global_step, dataformats="HWC")

    def _log_task_weights(self) -> None:
        """Log ``exp(-s)`` from the uncertainty weighting: it shows which task is
        dominating the trunk's gradients. A weight collapsing toward zero means that
        head has effectively stopped learning."""
        bal = getattr(self.model, "balancer", None)
        log_vars = getattr(bal, "log_vars", None)
        if not (self.tb and log_vars):
            return
        for name, s in log_vars.items():
            self.tb.add_scalar(
                f"task_weight/{name}", float(torch.exp(-s.detach())), self.global_step
            )

    def state_dict(self, epoch: int) -> dict:
        """Everything needed to continue training as if it had never stopped.

        Weights alone are not enough: without the scheduler position a resumed run
        replays warmup and a whole cosine cycle at full LR, and without ``best_metric``
        the first validation overwrites ``best.pt`` with a worse model.
        """
        return {
            "format": CKPT_FORMAT,
            "model": self.model.state_dict(),
            "ema": self.ema.ema.state_dict() if self.ema else None,
            "ema_updates": self.ema.updates if self.ema else 0,
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "scaler": self.scaler.state_dict(),
            "epoch": epoch,
            "global_step": self.global_step,
            "best_metric": self.best_metric,
            "cfg": dict(self.cfg),
        }

    def save(self, name: str, epoch: int):
        torch.save(self.state_dict(epoch), self.out_dir / name)

    def load(self, path: str, resume: bool = False):
        ckpt = load_checkpoint(path)
        self.model.load_state_dict(ckpt["model"])
        if self.ema and ckpt.get("ema"):
            self.ema.ema.load_state_dict(ckpt["ema"])
        if resume:
            self.report_config_drift(ckpt)
            self.load_train_state(ckpt)
        self.logger.info(
            f"loaded checkpoint: {path} (epoch {ckpt.get('epoch')}, resume={resume})"
        )

    def report_config_drift(self, ckpt: dict) -> list[tuple[str, object, object]]:
        """Log how this run's config differs from the one the checkpoint was trained on.

        Changing ``train.epochs`` to extend a run is routine. Changing the learning rate,
        the class count or the datasets is usually a mistake, and the resulting run would
        otherwise record settings its weights never saw.
        """
        old = ckpt.get("cfg")
        if not isinstance(old, dict):
            return []
        drift = diff_config(old, dict(self.cfg))
        if not drift:
            return []
        structural = [d for d in drift if d[0].startswith(("model.", "data."))]
        for where, was, now in drift:
            self.logger.warning(f"config changed since the checkpoint: {where}: {was} -> {now}")
        if structural:
            self.logger.warning(
                f"{len(structural)} of those touch the model or the data. The weights "
                f"being loaded were not trained under them."
            )
        return drift

    def load_train_state(self, ckpt: dict) -> None:
        """Restore optimizer, schedule position and bookkeeping from a checkpoint."""
        if "optimizer" in ckpt:
            self.optimizer.load_state_dict(ckpt["optimizer"])
        # An empty dict is what a disabled GradScaler saves; feeding it to an enabled
        # one raises, which is exactly the CPU-run -> CUDA-run case.
        if ckpt.get("scaler"):
            self.scaler.load_state_dict(ckpt["scaler"])
        if self.ema:
            # Without this the decay ramp restarts and a mature average is dragged
            # back towards whatever the model happens to be at the resume point.
            self.ema.updates = int(ckpt.get("ema_updates", 0))
        self.start_epoch = int(ckpt.get("epoch", 0))
        self.global_step = int(ckpt.get("global_step", 0))
        self.best_metric = float(ckpt.get("best_metric", -1.0))

        if ckpt.get("scheduler") is not None:
            self.scheduler.load_state_dict(ckpt["scheduler"])
        else:
            # Pre-CKPT_FORMAT-2 checkpoints predate scheduler state. Reconstructing the
            # position from the epoch count is approximate but far better than the
            # alternative, which is replaying warmup at full LR.
            it = self.start_epoch * self.steps_per_epoch
            self.scheduler.load_state_dict({"it": it})
            self.logger.warning(
                f"checkpoint has no scheduler state; schedule position estimated as "
                f"{it} iters from epoch {self.start_epoch}."
            )
        lr = self.optimizer.param_groups[0]["lr"]
        self.logger.info(
            f"resuming after epoch {self.start_epoch}: step={self.global_step}, "
            f"lr={lr:.2e}, best_metric={self.best_metric:.4f}"
        )
