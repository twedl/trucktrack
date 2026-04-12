"""Stage 1: Generate a small truck GPS database.

Creates a hive-partitioned parquet dataset at::

    output/raw/year=YYYY/chunk_id=XXX/part-0.parquet

Uses a small number of trucks and trips so the full pipeline runs
quickly.  All error types appear at least once; remaining trips use
the default error profile.

Usage::

    VALHALLA_TILE_EXTRACT=valhalla_tiles/valhalla_tiles.tar \
        uv run python examples/end_to_end/stage1_generate.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Import the shared generation logic from the sibling example.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "trace_visualizations"))
import generate_database  # noqa: E402

generate_database.OUTPUT_DIR = Path(
    os.environ.get("OUTPUT_DIR", "examples/end_to_end/output/raw")
)
generate_database.N_TRUCKS = int(os.environ.get("N_TRUCKS", "5"))
generate_database.K_TRIPS = int(os.environ.get("K_TRIPS", "5"))

if __name__ == "__main__":
    generate_database.main()
