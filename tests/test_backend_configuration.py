"""Precision settings must survive the deterministic backend branch."""

import pytest
import torch

from syncai_hydranet.utils.seeding import configure_backends


@pytest.mark.parametrize("deterministic", [False, True])
@pytest.mark.parametrize("tf32", [False, True])
def test_cuda_precision_flags_are_applied_in_both_modes(deterministic, tf32):
    previous = (
        torch.are_deterministic_algorithms_enabled(),
        torch.is_deterministic_algorithms_warn_only_enabled(),
        torch.backends.cudnn.deterministic,
        torch.backends.cudnn.benchmark,
        torch.backends.cuda.matmul.allow_tf32,
        torch.backends.cudnn.allow_tf32,
    )
    try:
        # These host-side backend flags can be verified even on CPU-only CI.
        torch.backends.cuda.matmul.allow_tf32 = not tf32
        torch.backends.cudnn.allow_tf32 = not tf32
        configure_backends(
            torch.device("cuda"), deterministic=deterministic, cudnn_benchmark=True, tf32=tf32
        )
        assert torch.backends.cuda.matmul.allow_tf32 is tf32
        assert torch.backends.cudnn.allow_tf32 is tf32
        assert torch.backends.cudnn.benchmark is (not deterministic)
        if deterministic:
            assert torch.are_deterministic_algorithms_enabled()
            assert torch.backends.cudnn.deterministic
    finally:
        torch.use_deterministic_algorithms(previous[0], warn_only=previous[1])
        torch.backends.cudnn.deterministic = previous[2]
        torch.backends.cudnn.benchmark = previous[3]
        torch.backends.cuda.matmul.allow_tf32 = previous[4]
        torch.backends.cudnn.allow_tf32 = previous[5]
