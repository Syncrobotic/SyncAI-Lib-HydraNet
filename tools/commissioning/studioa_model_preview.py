#!/usr/bin/env python3
"""Render a selected StudioA model against frozen camera/depth inputs."""

import argparse
from pathlib import Path

from syncai_bev3d.studioa_preview import render_view


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--camera-root", type=Path, required=True)
    parser.add_argument("--camera", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    args = parser.parse_args()
    render_view(args.run, args.camera_root, args.camera, args.out, device=args.device)


if __name__ == "__main__":
    main()
