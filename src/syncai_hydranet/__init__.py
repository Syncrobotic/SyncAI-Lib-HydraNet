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
from .models.hydranet import HydraNet, build_model

__version__ = "0.1.0"

__all__ = ["Config", "HydraNet", "__version__", "build_model", "load_config"]
