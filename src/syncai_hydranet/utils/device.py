"""Device selection: CUDA (Jetson / servers) -> MPS (Apple Silicon) -> CPU.

On Apple Silicon, MPS is roughly 7x faster than CPU for this model (measured on an
M4 Pro: RegNetX-800MF at 384x512, batch 4, 705 ms/step on CPU vs 97 ms/step on MPS),
so local development must not silently fall back to CPU.

AMP and pinned memory are CUDA-only today, so those checks live here too rather than
being scattered across the trainer.
"""

from __future__ import annotations

import torch


def pick_device(prefer: str | None = None) -> torch.device:
    """Return the fastest available device, or ``prefer`` ('cuda' / 'mps' / 'cpu')."""
    if prefer:
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def supports_amp(device: torch.device) -> bool:
    """GradScaler and autocast are only fully supported on CUDA."""
    return device.type == "cuda"


def supports_pinned_memory(device: torch.device) -> bool:
    """MPS does not support pinned memory; setting it only emits a warning."""
    return device.type == "cuda"
