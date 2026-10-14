"""Group raw GPS by truck, then split it into trips (example wrapper).

This script is a thin wrapper around :func:`trucktrack.bucket_by_truck`
and :func:`trucktrack.pipeline.run_pipeline`.  See those for the full
implementation and parameters.

Usage::

    uv run python examples/split_pipeline.py data/raw data/bucketed data/split
"""

from __future__ import annotations

import sys
from pathlib import Path

from trucktrack.buckets import DEFAULT_N_BUCKETS, bucket_by_truck
from trucktrack.pipeline import run_pipeline


def main() -> None:
    if len(sys.argv) < 4:
        print(f"Usage: {sys.argv[0]} <raw_dir> <bucketed_dir> <split_dir> [n_buckets]")
        sys.exit(1)
    raw_dir, bucketed_dir, split_dir = (Path(a) for a in sys.argv[1:4])
    n_buckets = int(sys.argv[4]) if len(sys.argv) > 4 else DEFAULT_N_BUCKETS
    bucket_by_truck(raw_dir, bucketed_dir, n_buckets=n_buckets)
    run_pipeline(bucketed_dir, split_dir)


if __name__ == "__main__":
    main()
