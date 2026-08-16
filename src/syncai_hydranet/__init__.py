"""SyncAI-Lib-HydraNet: multi-head perception network for quadruped robots.

A single backbone and BiFPN neck feed three lightweight heads that all run in one
forward pass:

1. Traversability segmentation: can the robot step here?
2. Terrain segmentation: what is the surface made of?
3. Object detection: FCOS, anchor-free.

The shared trunk holds ~84% of the parameters, so adding a fourth head costs only a
few percent more compute while reusing everything already paid for.
"""

from .config import Config, load_config
from .config_schema import ConfigError, check_config
from .models.hydranet import HydraNet, build_model

# The single source of version truth: `[tool.hatch.version]` reads this file, and
# `release-please-config.json` lists it under `extra-files` so a release PR rewrites
# the line below. The trailing annotation is what release-please matches on -- remove
# it and releases will bump CHANGELOG.md and the git tag while leaving the package
# reporting the previous version, which nothing would fail on.
__version__ = "0.2.0"  # x-release-please-version

__all__ = [
    "Config",
    "ConfigError",
    "HydraNet",
    "__version__",
    "build_model",
    "check_config",
    "load_config",
]
