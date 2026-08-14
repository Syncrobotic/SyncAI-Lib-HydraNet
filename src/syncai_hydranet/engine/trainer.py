"""Training engine: AMP, warmup + cosine schedule, EMA, checkpoints, TensorBoard."""

from __future__ import annotations

import copy
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from ..config import diff_config
from ..config_schema import check_config
from ..data.datasets import build_dataset
from ..data.fingerprint import fingerprint_dataset
from ..data.multitask import MultiTaskLoader
from ..models.hydranet import build_model
from ..utils.checkpoint import CKPT_FORMAT, load_checkpoint
from ..utils.device import pick_device, supports_amp, supports_pinned_memory
from ..utils.logger import get_logger
from ..utils.runmeta import append_metrics, resolve_out_dir, write_run_meta
from ..utils.seeding import (
    configure_backends,
    needs_grad_scaler,
    resolve_amp_dtype,
    seed_everything,
)
from ..utils.visualize import TERRAIN_COLORS, TRAV_COLORS, prediction_grid
from .evaluator import build_val_loaders, evaluate, select_metric

DEFAULT_PRIMARY_METRIC = "traversability_mIoU"


class ModelEMA:
    """Exponential moving average of the weights, with a warmed-up decay.

    The average starts from the model's *initial* random weights. At a fixed decay of
    0.9998 that initialisation is still 45% of the EMA after 160 steps, and validation
    -- which runs on the EMA -- reported 0.16 mIoU for a model whose raw weights scored
    0.95. The failure is silent and looks exactly like a model that did not learn.

    The decay is therefore ramped: ``decay * (1 - exp(-updates / warmup_steps))``, so
    the first updates copy the model almost outright and the smoothing strengthens as
    the average acquires real history. This is the standard fix (YOLOv5, timm) and it
    makes EMA safe on short runs instead of merely warned about.
    """

    def __init__(self, model: nn.Module, decay: float = 0.9998, warmup_steps: int = 2000):
        self.ema = copy.deepcopy(model).eval()
        for p in self.ema.parameters():
            p.requires_grad_(False)
        self.decay = decay
        self.warmup_steps = max(int(warmup_steps), 0)
        self.updates = 0

    def decay_at(self, updates: int) -> float:
        if self.warmup_steps <= 0:
            return self.decay
        return self.decay * (1 - math.exp(-updates / self.warmup_steps))

    def residual_init_fraction(self, steps: int) -> float:
        """How much of the random initialisation survives after ``steps`` updates."""
        if steps <= 0:
            return 1.0
        if self.warmup_steps <= 0:
            return float(self.decay**steps)
        n = np.arange(1, steps + 1)
        decays = self.decay * (1 - np.exp(-n / self.warmup_steps))
        return float(np.exp(np.log(decays).sum()))

    @torch.no_grad()
    def update(self, model: nn.Module):
        self.updates += 1
        d = self.decay_at(self.updates)
        msd = model.state_dict()
        for k, v in self.ema.state_dict().items():
            if v.dtype.is_floating_point:
                v.mul_(d).add_(msd[k].detach(), alpha=1 - d)
            else:
                v.copy_(msd[k])


def build_optimizer(model: nn.Module, tcfg) -> torch.optim.Optimizer:
    """Lower LR for the backbone (transfer-learning convention); no weight decay on
    biases and norm parameters."""
    lr = float(tcfg["lr"])
    bb_mult = float(tcfg.get("backbone_lr_mult", 1.0))
    wd = float(tcfg.get("weight_decay", 0.05))
    decay, no_decay, bb_decay, bb_no_decay = [], [], [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        is_bb = name.startswith("backbone.")
        no_wd = p.ndim <= 1  # bias or norm
        target = (
            bb_no_decay
            if is_bb and no_wd
            else bb_decay
            if is_bb
            else no_decay
            if no_wd
            else decay
        )
        target.append(p)
    groups = [
        {"params": decay, "lr": lr, "weight_decay": wd},
        {"params": no_decay, "lr": lr, "weight_decay": 0.0},
        {"params": bb_decay, "lr": lr * bb_mult, "weight_decay": wd},
        {"params": bb_no_decay, "lr": lr * bb_mult, "weight_decay": 0.0},
    ]
    if tcfg.get("optimizer", "adamw") == "adamw":
        return torch.optim.AdamW(groups)
    return torch.optim.SGD(groups, momentum=0.9, nesterov=True)


class WarmupCosine:
    def __init__(self, optimizer, warmup_iters: int, total_iters: int):
        self.opt = optimizer
        self.warmup = max(warmup_iters, 1)
        self.total = total_iters
        self.base_lrs = [g["lr"] for g in optimizer.param_groups]
        self.it = 0

    def _factor(self, it: int) -> float:
        if it <= self.warmup:
            return it / self.warmup
        t = (it - self.warmup) / max(self.total - self.warmup, 1)
        return 0.5 * (1 + math.cos(math.pi * min(t, 1.0)))

    def _apply(self):
        f = self._factor(self.it)
        for g, base in zip(self.opt.param_groups, self.base_lrs, strict=True):
            g["lr"] = base * f

    def step(self):
        self.it += 1
        self._apply()

    def state_dict(self) -> dict:
        return {"it": self.it}

    def load_state_dict(self, state: dict) -> None:
        """Restore the schedule position and immediately re-apply it.

        Without the re-apply the first resumed step would run at the base LR, which for
        a run resumed near the end of cosine decay is orders of magnitude too high.
        """
        self.it = int(state["it"])
        self._apply()


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
        tcfg = cfg["train"]
        dcfg = cfg["data"]
        input_size = dcfg["input_size"]

        lb = bool(dcfg.get("letterbox", False))
        aug = dcfg.get("augment")
        train_sets, val_sets, names, ratios = [], [], [], []
        for ds in dcfg["datasets"]:
            train_sets.append(build_dataset(ds, input_size, "train", letterbox=lb, augment=aug))
            # Validation never augments, so it takes no augment argument.
            val_sets.append(build_dataset(ds, input_size, "val", letterbox=lb))
            names.append(ds["name"])
            ratios.append(float(ds.get("sample_ratio", 1.0)))
        self.train_loader = MultiTaskLoader(
            train_sets,
            names,
            ratios,
            batch_size=int(tcfg["batch_size"]),
            workers=int(dcfg.get("workers", 4)),
            seed=seed,
            pin_memory=supports_pinned_memory(self.device),
        )
        self.val_sets = list(zip(names, val_sets, strict=True))
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

        self.amp = bool(tcfg.get("amp", True)) and supports_amp(self.device)
        self.amp_dtype = resolve_amp_dtype(str(tcfg.get("amp_dtype", "float16")))
        self.scaler = torch.amp.GradScaler(enabled=needs_grad_scaler(self.amp, self.amp_dtype))
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

        ema_decay = float(tcfg.get("ema_decay", 0.9998))
        ema_warmup = int(tcfg.get("ema_warmup_steps", 2000))
        self.ema = (
            ModelEMA(self.model, ema_decay, ema_warmup) if tcfg.get("ema", True) else None
        )
        if self.ema:
            # The ramp makes this rare rather than routine, but a run shorter than the
            # ramp itself can still validate on weights that are mostly noise, and that
            # failure looks exactly like a model that did not learn.
            residual = self.ema.residual_init_fraction(total_iters)
            if residual > 0.05:
                self.logger.warning(
                    f"EMA warning: {total_iters} optimizer steps at decay={ema_decay} "
                    f"(warmup {ema_warmup}) leave {residual:.0%} of the random "
                    f"initialisation in the EMA weights used for validation. Scores "
                    f"will be understated. Lower train.ema_warmup_steps or set "
                    f"train.ema=false for runs this short."
                )

        self.log_interval = int(tcfg.get("log_interval", 50))
        self.val_interval = int(tcfg.get("val_interval", 1))
        # One named metric decides best.pt. For a robot the honest choice is a
        # traversability number, not an average across heads: mistaking a wall for
        # floor is not compensated by a good mAP on chairs.
        self.primary_metric = str(tcfg.get("primary_metric", DEFAULT_PRIMARY_METRIC))
        self.logger.info(f"model selection: primary_metric={self.primary_metric}")
        try:
            from torch.utils.tensorboard import SummaryWriter

            self.tb = SummaryWriter(str(self.out_dir / "tb"))
        except Exception:
            self.tb = None
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
                    "val_size": len(v),
                    # Datasets live outside git; without this, "which data produced
                    # this checkpoint" has no answer six months later.
                    **fingerprint_dataset(ds),
                }
                for n, t, v, ds in zip(
                    names, train_sets, val_sets, dcfg["datasets"], strict=True
                )
            ],
        )
        git = meta["git"]
        if not git.get("available"):
            self.logger.warning("not a git checkout: this run's code version is unrecorded")
        elif git["dirty"]:
            self.logger.warning(
                f"working tree is dirty at {git['commit'][:8]}; the exact code is only "
                f"recoverable via {self.out_dir / 'uncommitted.patch'}"
            )
        else:
            self.logger.info(f"code version: {git['commit'][:8]} ({git['branch']})")

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
            images = batch["image"].to(self.device, non_blocking=True)
            targets = _targets_to_device(batch["targets"], self.device)
            with torch.amp.autocast(self.device.type, enabled=self.amp, dtype=self.amp_dtype):
                outputs = self.model(images)
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
        palettes = {"traversability": TRAV_COLORS, "terrain": TERRAIN_COLORS}
        for head, (imgs, preds, gts) in (samples or {}).items():
            grid = prediction_grid(imgs, preds, gts, palettes.get(head, TERRAIN_COLORS))
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
