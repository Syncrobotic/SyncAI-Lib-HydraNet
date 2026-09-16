#!/usr/bin/env python3
"""Export an immutable derivative from a source-bound AI semantic review."""

import argparse
import json
from pathlib import Path

from syncai_hydranet.data.studioa_semantic_review import export_review


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = export_review(args.source, args.review, args.out)
    print(json.dumps({k: v for k, v in report.items() if k != "outputs"}, indent=2))


if __name__ == "__main__":
    main()
