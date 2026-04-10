"""Hive-partitioned GPS trace pipeline.

Gap splitting, stop splitting, and partitioned output.

Processes input chunks in parallel using threads (the Rust backend
releases the GIL). Each chunk is written with a unique filename so
tiles shared across chunks accumulate rather than overwrite.

    1. Discover input parquet files.
    2. For each chunk (in parallel):
       a. Gap-split into discrete trip segments.
       b. Assign composite trip IDs ({id}_seg{segment_id}).
       c. Stop-split into movement and stop sub-segments.
       d. Keep only movement rows (is_stop == False).
       e. Compute spatial partition metadata.
       f. Write to hive-partitioned output with chunk-unique
          filenames.
    3. Print summary.

Usage::

    from examples.pipeline_hive_to_partitioned import run_pipeline
    run_pipeline(Path("data/raw"), Path("data/processed"))
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import timedelta
from pathlib import Path

import polars as pl
import trucktrack

# Pipeline parameters.
GAP_THRESHOLD = timedelta(minutes=30)
STOP_MAX_DIAMETER = 250.0  # meters
STOP_MIN_DURATION = timedelta(minutes=5)
MIN_SEGMENT_LENGTH = 2


def _write_chunk(
    df: pl.DataFrame,
    output_dir: Path,
    chunk_name: str,
) -> None:
    """Write one chunk into hive dirs with a chunk-unique filename."""
    for (tier, pid), group in df.group_by(["tier", "partition_id"]):
        tile_dir = output_dir / f"tier={tier}" / f"partition_id={pid}"
        tile_dir.mkdir(parents=True, exist_ok=True)
        out_path = tile_dir / f"{chunk_name}.parquet"
        group.drop(["tier", "partition_id"]).write_parquet(out_path)


def _process_chunk(
    df: pl.DataFrame,
    *,
    gap: timedelta,
    stop_max_diameter: float,
    stop_min_duration: timedelta,
    min_segment_length: int,
) -> pl.DataFrame:
    """Run gap split, stop split, filter, and partition on one chunk.

    Returns a DataFrame with tier/partition_id/hilbert_idx columns.
    """
    # Gap splitting.
    df = trucktrack.split_by_observation_gap(
        df,
        gap,
        id_col="id",
        time_col="time",
        min_length=min_segment_length,
    )

    # Rename gap segment_id before stop splitting overwrites it.
    df = df.rename({"segment_id": "gap_segment_id"})

    # Stop splitting.
    df = trucktrack.split_by_stops(
        df,
        max_diameter=stop_max_diameter,
        min_duration=stop_min_duration,
        id_col="id",
        time_col="time",
        lat_col="lat",
        lon_col="lon",
        min_length=min_segment_length,
    )

    # Composite ID: {id}_gap{gap_seg}_stop{stop_seg}
    df = df.with_columns(
        (
            pl.col("id")
            + "_gap"
            + pl.col("gap_segment_id").cast(pl.Utf8)
            + "_stop"
            + pl.col("segment_id").cast(pl.Utf8)
        ).alias("id")
    )

    # Keep only movement rows.
    df = df.filter(~pl.col("is_stop"))

    # Partition metadata.
    df = trucktrack.partition_points(df)

    return df


def _process_and_write(
    chunk_path: Path,
    input_dir: Path,
    output_dir: Path,
    gap: timedelta,
    stop_max_diameter: float,
    stop_min_duration: timedelta,
    min_segment_length: int,
) -> tuple[int, int]:
    """Read, process, and write one chunk. Returns (rows_in, rows_out)."""
    df = pl.read_parquet(chunk_path)
    n_in = len(df)

    df = _process_chunk(
        df,
        gap=gap,
        stop_max_diameter=stop_max_diameter,
        stop_min_duration=stop_min_duration,
        min_segment_length=min_segment_length,
    )

    n_out = len(df)
    _write_chunk(df, output_dir, chunk_path.stem)

    rel = chunk_path.relative_to(input_dir)
    print(f"  {rel}: {n_in:,} rows in -> {n_out:,} movement rows out")

    return n_in, n_out


def run_pipeline(
    input_dir: Path,
    output_dir: Path,
    *,
    gap: timedelta = GAP_THRESHOLD,
    stop_max_diameter: float = STOP_MAX_DIAMETER,
    stop_min_duration: timedelta = STOP_MIN_DURATION,
    min_segment_length: int = MIN_SEGMENT_LENGTH,
    max_workers: int | None = None,
) -> dict[str, int]:
    """Run the full gap+stop+partition pipeline in parallel.

    Each input parquet file is processed independently in a thread
    pool. The Rust backend releases the GIL, so threads achieve
    real parallelism. Output tiles shared across chunks accumulate
    multiple files that Polars reads back as a single dataset.

    *max_workers* defaults to ``os.cpu_count()``.

    Returns a {tier_name: partition_count} summary.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    chunks = sorted(Path(input_dir).rglob("*.parquet"))
    if not chunks:
        print(f"No parquet files found under {input_dir}")
        return {}

    if max_workers is None:
        max_workers = os.cpu_count() or 1
    max_workers = min(max_workers, len(chunks))

    print(
        f"Found {len(chunks)} input chunk(s) under {input_dir}, "
        f"processing with {max_workers} worker(s)"
    )

    total_rows_in = 0
    total_rows_out = 0

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(
                _process_and_write,
                chunk_path,
                Path(input_dir),
                output_dir,
                gap,
                stop_max_diameter,
                stop_min_duration,
                min_segment_length,
            ): chunk_path
            for chunk_path in chunks
        }

        for future in as_completed(futures):
            n_in, n_out = future.result()
            total_rows_in += n_in
            total_rows_out += n_out

    # Count distinct tiles written.
    tile_dirs: dict[str, set[str]] = {}
    for p in output_dir.rglob("*.parquet"):
        tier = p.parent.parent.name.removeprefix("tier=")
        tile_dirs.setdefault(tier, set()).add(p.parent.name)
    partition_counts = {t: len(pids) for t, pids in tile_dirs.items()}

    total_partitions = sum(partition_counts.values())
    print("\n--- Pipeline complete ---")
    print(f"  Input chunks:         {len(chunks):>10,}")
    print(f"  Workers:              {max_workers:>10,}")
    print(f"  Input rows:           {total_rows_in:>10,}")
    print(f"  Movement rows out:    {total_rows_out:>10,}")
    print(f"  Output partitions:    {total_partitions:>10,}")
    for tier, count in sorted(partition_counts.items()):
        print(f"    {tier}: {count:,} partition(s)")
    print(f"  Output directory:     {output_dir}")

    return partition_counts
