"""End-to-end pipeline: generate, split+partition, map-match.

Stage 1 generates a small truck GPS database using the ``generate_database``
helper in ``examples/trace_visualizations/``.  Stages 2 and 3 use the
installed ``trucktrack`` package directly.

Usage::

    VALHALLA_TILE_EXTRACT=valhalla_tiles/valhalla_tiles.tar \
        uv run python examples/end_to_end.py

Environment variables::

    VALHALLA_TILE_EXTRACT  Path to Valhalla tile extract (required)
    OUTPUT_DIR             Base output directory (default: examples/end_to_end_output)
    N_TRUCKS               Number of trucks to generate (default: 5)
    K_TRIPS                Number of trips per truck (default: 5)
    MAX_WORKERS            Thread pool size for map-matching (default: 1)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Stage 1 uses the generate_database example helper, which is not part of
# the installed package (it contains Ontario-specific waypoints and error
# profiles for demo purposes).
sys.path.insert(0, str(Path(__file__).resolve().parent / "trace_visualizations"))
import generate_database  # noqa: E402
from trucktrack.pipeline import run_pipeline  # noqa: E402
from trucktrack.valhalla.pipeline import run_map_matching  # noqa: E402

OUTPUT_BASE = Path(os.environ.get("OUTPUT_DIR", "examples/end_to_end_output"))
TILE_EXTRACT = os.environ.get("VALHALLA_TILE_EXTRACT", "valhalla_tiles.tar")
MAX_WORKERS = int(os.environ.get("MAX_WORKERS", "1"))


def main() -> None:
    raw_dir = OUTPUT_BASE / "raw"
    partitioned_dir = OUTPUT_BASE / "partitioned"
    matched_dir = OUTPUT_BASE / "matched"

    # --- Stage 1: Generate ---
    print("=== Stage 1: Generating truck GPS database ===")
    generate_database.OUTPUT_DIR = raw_dir
    generate_database.N_TRUCKS = int(os.environ.get("N_TRUCKS", "5"))
    generate_database.K_TRIPS = int(os.environ.get("K_TRIPS", "5"))
    generate_database.main()

    # --- Stage 2: Split + Partition ---
    print("\n=== Stage 2: Splitting and partitioning ===")
    run_pipeline(raw_dir, partitioned_dir)

    # --- Stage 3: Map-match ---
    print("\n=== Stage 3: Map-matching ===")
    run_map_matching(
        partitioned_dir,
        matched_dir,
        tile_extract=TILE_EXTRACT,
        max_workers=MAX_WORKERS,
    )

    print("\n=== All stages complete ===")


if __name__ == "__main__":
    main()
