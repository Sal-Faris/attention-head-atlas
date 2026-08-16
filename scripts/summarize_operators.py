"""Write a human-readable CSV of per-head operator diagnostics."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from head_atlas.diagnostics import operator_table
from head_atlas.model_io import load_operator_bundle


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    operators, _ = load_operator_bundle(args.input)
    rows = operator_table(operators)
    if not rows:
        raise RuntimeError("operator bundle is empty")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"saved {len(rows)} operator records to {args.output}")


if __name__ == "__main__":
    main()
