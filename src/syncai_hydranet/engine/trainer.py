"""Training engine: AMP, warmup + cosine schedule, EMA, checkpoints, TensorBoard."""

from __future__ import annotations

import copy
import math
import time
from pathlib import Path

import torch
import torch.nn as nn

from ..data.datasets import build_dataset
from ..data.multitask import MultiTaskLoader
from ..models.hydranet import build_model
from ..utils.checkpoint import CKPT_FORMAT, load_checkpoint
from ..utils.device import pick_device, supports_amp, supports_pinned_memory
from ..utils.logger import get_logger
from ..utils.visualize import TERRAIN_COLORS, TRAV_COLORS, prediction_grid
from .evaluator import evaluate


class ModelEMA:
    """Exponential moving average of the weights.

    Note the average starts from the model's *initial* weights, so after n steps a
    fraction ``decay**n`` of the random initialisation is still mixed in. Short runs
    must lower the decay or disable EMA; the Trainer warns when this bites.
    """

    def __init__(self, model: nn.Module, decay: float = 0.9998):
        self.ema = copy.deepcopy(model).eval()
        for p in self.ema.parameters():
            p.requires_grad_(False)
        self.decay = decay

    @torch.no_grad()
    def update(self, model: nn.Module):
        msd = model.state_dict()
        for k, v in self.ema.state_dict().items():
            if v.dtype.is_floating_point:
                v.mul_(self.decay).add_(msd[k].detach(), alpha=1 - self.decay)
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
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = pick_device(cfg.get("device"))
        self.out_dir = Path(cfg["output_dir"])
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.logger = get_logger("hydranet", self.out_dir / "train.log")
        torch.manual_seed(cfg.get("seed", 42))

        self.model = build_model(cfg).to(self.device)
        tcfg = cfg["train"]
        dcfg = cfg["data"]
        input_size = dcfg["input_size"]

        lb = bool(dcfg.get("letterbox", False))
        train_sets, val_sets, names, ratios = [], [], [], []
        for ds in dcfg["datasets"]:
            train_sets.append(build_dataset(ds, input_size, True, letterbox=lb))
            val_sets.append(build_dataset(ds, input_size, False, letterbox=lb))
            names.append(ds["name"])
            ratios.append(float(ds.get("sample_ratio", 1.0)))
        self.train_loader = MultiTaskLoader(
            train_sets,
            names,
            ratios,
            batch_size=int(tcfg["batch_size"]),
            workers=int(dcfg.get("workers", 4)),
            seed=cfg.get("seed", 42),
            pin_memory=supports_pinned_memory(self.device),
        )
        self.val_sets = list(zip(names, val_sets, strict=True))

        self.epochs = int(tcfg["epochs"])
        total_iters = self.epochs * len(self.train_loader)
        self.optimizer = build_optimizer(self.model, tcfg)
        self.scheduler = WarmupCosine(
            self.optimizer, int(tcfg.get("warmup_iters", 500)), total_iters
        )
        self.amp = bool(tcfg.get("amp", True)) and supports_amp(self.device)
        self.scaler = torch.amp.GradScaler(enabled=self.amp)
        self.grad_clip = float(tcfg.get("grad_clip", 0.0))

        ema_decay = float(tcfg.get("ema_decay", 0.9998))
        self.ema = ModelEMA(self.model, ema_decay) if tcfg.get("ema", True) else None
        if self.ema:
            residual = ema_decay ** max(total_iters, 1)
            if residual > 0.05:
                self.logger.warning(
                    f"EMA warning: ema_decay={ema_decay} over {total_iters} steps leaves "
                    f"{residual:.0%} of the random initialisation in the EMA weights used "
                    f"for validation. Scores will be badly understated. For short runs set "
                    f"train.ema=false, or lower ema_decay below "
                    f"{1 - 3 / max(total_iters, 1):.4f}."
                )

        self.log_interval = int(tcfg.get("log_interval", 50))
        self.val_interval = int(tcfg.get("val_interval", 1))
        try:
            from torch.utils.tensorboard import SummaryWriter

            self.tb = SummaryWriter(str(self.out_dir / "tb"))
        except Exception:
            self.tb = None
        self.global_step = 0
        self.best_metric = -1.0
        self.start_epoch = 0  # last completed epoch; --resume advances it

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
            f"{len(self.train_loader)} steps/epoch, device={self.device}"
        )
        for epoch in range(self.start_epoch + 1, self.epochs + 1):
            self.train_one_epoch(epoch)
            if epoch % self.val_interval == 0:
                metrics = self.validate(epoch)
                score = metrics.get("mean_score", 0.0)
                self.save("last.pt", epoch)
                if score > self.best_metric:
                    self.best_metric = score
                    self.save("best.pt", epoch)
                    self.logger.info(f"new best model (score={score:.4f})")
        self.logger.info("training complete")

    def train_one_epoch(self, epoch: int):
        self.model.train()
        t0 = time.time()
        for i, batch in enumerate(self.train_loader):
            images = batch["image"].to(self.device, non_blocking=True)
            targets = _targets_to_device(batch["targets"], self.device)
            with torch.amp.autocast(self.device.type, enabled=self.amp):
                outputs = self.model(images)
                loss, logs = self.model.compute_losses(outputs, targets, batch["supervises"])
            self.optimizer.zero_grad(set_to_none=True)
            self.scaler.scale(loss).backward()
            if self.grad_clip > 0:
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()
            if self.ema:
                self.ema.update(self.model)
            self.global_step += 1

            if (i + 1) % self.log_interval == 0:
                lr = self.optimizer.param_groups[0]["lr"]
                msg = " ".join(f"{k}={v:.3f}" for k, v in logs.items())
                ips = self.log_interval * images.shape[0] / (time.time() - t0)
                t0 = time.time()
                self.logger.info(
                    f"E{epoch} [{i + 1}/{len(self.train_loader)}] "
                    f"ds={batch['dataset']} lr={lr:.2e} {msg} ({ips:.1f} img/s)"
                )
                if self.tb:
                    for k, v in logs.items():
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
            model, self.val_sets, self.cfg, self.device, self.logger, samples=samples
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
            self.load_train_state(ckpt)
        self.logger.info(
            f"loaded checkpoint: {path} (epoch {ckpt.get('epoch')}, resume={resume})"
        )

    def load_train_state(self, ckpt: dict) -> None:
        """Restore optimizer, schedule position and bookkeeping from a checkpoint."""
        if "optimizer" in ckpt:
            self.optimizer.load_state_dict(ckpt["optimizer"])
        # An empty dict is what a disabled GradScaler saves; feeding it to an enabled
        # one raises, which is exactly the CPU-run -> CUDA-run case.
        if ckpt.get("scaler"):
            self.scaler.load_state_dict(ckpt["scaler"])
        self.start_epoch = int(ckpt.get("epoch", 0))
        self.global_step = int(ckpt.get("global_step", 0))
        self.best_metric = float(ckpt.get("best_metric", -1.0))

        if ckpt.get("scheduler") is not None:
            self.scheduler.load_state_dict(ckpt["scheduler"])
        else:
            # Pre-CKPT_FORMAT-2 checkpoints predate scheduler state. Reconstructing the
            # position from the epoch count is approximate but far better than the
            # alternative, which is replaying warmup at full LR.
            it = self.start_epoch * len(self.train_loader)
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
