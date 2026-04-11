"""Map-match trips from hive-partitioned output.

Reads the partitioned parquet dataset produced by
``pipeline_hive_to_partitioned``, iterates over partitions, and
applies a map-matching function to each trip.

Partitions are processed in parallel using threads.

Usage::

    from examples.map_match_partitions import run_map_matching
    run_map_matching(Path("data/processed"), Path("data/matched"))
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import polars as pl


def map_match_trip(trip: pl.DataFrame) -> pl.DataFrame:
    """Map-match a single trip using Valhalla.

    TODO: call Valhalla trace_route and snap points to road network.
    """
    return trip


def _process_partition(
    partition_dir: Path,
    output_dir: Path,
) -> tuple[str, int, int]:
    """Read one partition chunk-by-chunk, map-match each trip, and write results.

    Processes one chunk at a time to keep memory free for Valhalla.
    Returns (partition_key, trips_in, rows_out).
    """
    tier = partition_dir.parent.name
    pid = partition_dir.name
    out_dir = output_dir / tier / pid
    out_dir.mkdir(parents=True, exist_ok=True)

    chunks = sorted(partition_dir.glob("*.parquet"))
    total_trips = 0
    total_rows = 0

    for chunk_path in chunks:
        df = pl.read_parquet(chunk_path)
        matched_trips = []
        for _, trip in df.group_by("id"):
            matched_trips.append(map_match_trip(trip))
            total_trips += 1

        matched = pl.concat(matched_trips)
        total_rows += len(matched)
        matched.write_parquet(out_dir / f"{chunk_path.stem}.parquet")

    return f"{tier}/{pid}", total_trips, total_rows


def _process_block(
    partition_dirs: list[Path],
    output_dir: Path,
) -> list[tuple[str, int, int]]:
    """Process a contiguous block of spatially adjacent partitions sequentially."""
    return [_process_partition(pdir, output_dir) for pdir in partition_dirs]


def _partition_sort_key(path: Path) -> tuple[str, int]:
    """Extract (tier, partition_id) for spatial sort order."""
    tier = path.parent.name.removeprefix("tier=")
    pid = int(path.name.removeprefix("partition_id="))
    return (tier, pid)


def run_map_matching(
    input_dir: Path,
    output_dir: Path,
    *,
    max_workers: int | None = None,
) -> None:
    """Map-match all trips across partitions in parallel.

    *input_dir* should be the hive-partitioned output of
    ``pipeline_hive_to_partitioned.run_pipeline``, with layout::

        input_dir/tier=.../partition_id=.../chunk.parquet

    *max_workers* defaults to ``os.cpu_count()``.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    partition_dirs = sorted(
        {
            p.parent
            for p in Path(input_dir).rglob("*.parquet")
            if p.parent.name.startswith("partition_id=")
        },
        key=_partition_sort_key,
    )

    if not partition_dirs:
        print(f"No partitions found under {input_dir}")
        return

    if max_workers is None:
        max_workers = os.cpu_count() or 1
    max_workers = min(max_workers, len(partition_dirs))

    # Split into contiguous blocks so each thread covers a geographic
    # region and Valhalla's tile cache stays warm.
    n = len(partition_dirs)
    block_size = n // max_workers
    remainder = n % max_workers
    blocks: list[list[Path]] = []
    start = 0
    for i in range(max_workers):
        end = start + block_size + (1 if i < remainder else 0)
        blocks.append(partition_dirs[start:end])
        start = end

    size_str = f"{block_size}–{block_size + 1}" if remainder else str(block_size)
    print(
        f"Found {n} partition(s), "
        f"processing with {max_workers} worker(s) "
        f"({size_str} partitions per worker)"
    )

    total_trips = 0
    total_rows = 0

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [
            pool.submit(_process_block, block, output_dir)
            for block in blocks
        ]

        for future in as_completed(futures):
            for key, trips_in, rows_out in future.result():
                total_trips += trips_in
                total_rows += rows_out
                print(f"  {key}: {trips_in:,} trips, {rows_out:,} rows")

    print("\n--- Map matching complete ---")
    print(f"  Partitions:  {len(partition_dirs):>10,}")
    print(f"  Trips:       {total_trips:>10,}")
    print(f"  Rows out:    {total_rows:>10,}")
    print(f"  Output:      {output_dir}")
