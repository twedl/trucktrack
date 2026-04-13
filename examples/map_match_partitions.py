"""Map-match trips from hive-partitioned output (example wrapper).

This script is a thin wrapper around
:func:`trucktrack.valhalla.pipeline.run_map_matching`.
See that module for the full implementation and parameters.

Usage::

    VALHALLA_TILE_EXTRACT=valhalla_tiles.tar \
        uv run python examples/map_match_partitions.py data/partitioned data/matched
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from trucktrack.valhalla.pipeline import run_map_matching

TILE_EXTRACT = os.environ.get("VALHALLA_TILE_EXTRACT", "valhalla_tiles.tar")
CONFIG = os.environ.get("VALHALLA_CONFIG")
MAX_WORKERS = int(os.environ.get("MAX_WORKERS", "0")) or None


def main() -> None:
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <input_dir> <output_dir>")
        sys.exit(1)
    input_dir = Path(sys.argv[1])
    output_dir = Path(sys.argv[2])
    run_map_matching(
        input_dir,
        output_dir,
        tile_extract=TILE_EXTRACT if CONFIG is None else None,
        config=CONFIG,
        max_workers=MAX_WORKERS,
    )


if __name__ == "__main__":
    main()
