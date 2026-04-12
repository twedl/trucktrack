"""Stage 2: Gap-split, stop-split, traffic-filter, and spatially partition.

Reads the hive-partitioned raw database from stage 1 and produces a
spatially partitioned dataset at::

    output/partitioned/tier=.../partition_id=.../chunk.parquet

Each input chunk is processed in a thread pool (the Rust backend
releases the GIL).

Usage::

    uv run python examples/end_to_end/stage2_split_partition.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline_hive_to_partitioned import run_pipeline  # noqa: E402

INPUT_DIR = Path(
    os.environ.get("INPUT_DIR", "examples/end_to_end/output/raw")
)
OUTPUT_DIR = Path(
    os.environ.get("OUTPUT_DIR", "examples/end_to_end/output/partitioned")
)


def main() -> None:
    if not INPUT_DIR.exists():
        print(f"Input directory {INPUT_DIR} does not exist. Run stage1 first.")
        return

    print(f"Stage 2: splitting and partitioning {INPUT_DIR} -> {OUTPUT_DIR}")
    run_pipeline(INPUT_DIR, OUTPUT_DIR)
    print("\nStage 2 complete.")


if __name__ == "__main__":
    main()
