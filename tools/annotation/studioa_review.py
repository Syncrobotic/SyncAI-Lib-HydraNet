#!/usr/bin/env python3
"""Prepare/check StudioA scene review packages without training or publishing labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from syncai_hydranet.data.studioa_contract import SOURCE_CLASSES
from syncai_hydranet.data.studioa_review import check_package, prepare_package, write_json


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest="action", required=True)
    prepare = actions.add_parser("prepare")
    prepare.add_argument(
        "--source",
        nargs=2,
        action="append",
        required=True,
        metavar=("DATASET", "SCHEME"),
        help=f"repeatable; native schemes: {', '.join(SOURCE_CLASSES)}",
    )
    prepare.add_argument("--per-camera", type=int, default=3)
    prepare.add_argument("--out", type=Path, required=True)
    check = actions.add_parser("check")
    check.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.action == "prepare":
        report = prepare_package(
            [(Path(path), scheme) for path, scheme in args.source], args.out, args.per_camera
        )
    else:
        report = check_package(args.out)
        write_json(args.out / "review_status.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 2 if args.action == "check" and report["status"] != "review_complete" else 0


if __name__ == "__main__":
    raise SystemExit(main())
