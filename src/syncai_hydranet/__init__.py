"""SyncAI-Lib-HydraNet: multi-head perception network for fixed store CCTV.

A single backbone and neck (BiFPN or FPN, per config) feed four lightweight head
families that all run in one forward pass:

1. Segmentation: terrain / traversability, semantic FPN.
2. Monocular depth: defined here; the retail-security configs do not name it.
3. Object detection: FCOS, anchor-free.
4. Pose: 17 COCO keypoint heatmaps at P3, decoded inside the detection boxes.

Each config trains the subset of heads it names. The shared trunk holds most of the
parameters, so another head costs only a few percent more compute while reusing
everything already paid for.
"""

from .config import Config, load_config
from .config_schema import ConfigError, check_config
from .models.hydranet import HydraNet, build_model

# The single source of version truth: `[tool.hatch.version]` reads this file, and
# `release-please-config.json` lists it under `extra-files` so a release PR rewrites
# the line below. The trailing annotation is what release-please matches on -- remove
# it and releases will bump CHANGELOG.md and the git tag while leaving the package
# reporting the previous version, which nothing would fail on.
__version__ = "0.5.0"  # x-release-please-version

__all__ = [
    "Config",
    "ConfigError",
    "HydraNet",
    "__version__",
    "build_model",
    "check_config",
    "load_config",
]
