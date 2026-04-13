"""Hive-partitioned GPS trace pipeline (example wrapper).

This script is a thin wrapper around :func:`trucktrack.pipeline.run_pipeline`.
See that module for the full implementation and parameters.

Usage::

    uv run python examples/pipeline_hive_to_partitioned.py data/raw data/partitioned
"""

from __future__ import annotations

import sys
from pathlib import Path

from trucktrack.pipeline import run_pipeline


def main() -> None:
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <input_dir> <output_dir> [group_size]")
        sys.exit(1)
    input_dir = Path(sys.argv[1])
    output_dir = Path(sys.argv[2])
    group_size = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    run_pipeline(input_dir, output_dir, group_size=group_size, compact=True)


if __name__ == "__main__":
    main()
