"""Stage 3: Map-match trips from the spatially partitioned dataset.

Reads the partitioned parquet from stage 2 and map-matches each trip
using local pyvalhalla.  Output mirrors the input hive layout at::

    output/matched/tier=.../partition_id=.../chunk.parquet

Resumable: chunks whose output file already exists are skipped.

Usage::

    VALHALLA_TILE_EXTRACT=valhalla_tiles/valhalla_tiles.tar \
        uv run python examples/end_to_end/stage3_map_match.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from map_match_partitions import run_map_matching  # noqa: E402

INPUT_DIR = Path(
    os.environ.get("INPUT_DIR", "examples/end_to_end/output/partitioned")
)
OUTPUT_DIR = Path(
    os.environ.get("OUTPUT_DIR", "examples/end_to_end/output/matched")
)


def main() -> None:
    if not INPUT_DIR.exists():
        print(f"Input directory {INPUT_DIR} does not exist. Run stage2 first.")
        return

    print(f"Stage 3: map-matching {INPUT_DIR} -> {OUTPUT_DIR}")
    run_map_matching(INPUT_DIR, OUTPUT_DIR, max_workers=1)
    print("\nStage 3 complete.")


if __name__ == "__main__":
    main()
