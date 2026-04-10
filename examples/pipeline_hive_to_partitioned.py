"""Hive-partitioned GPS trace pipeline.

Gap splitting, stop splitting, and partitioned output.

Reads a hive-partitioned input dataset
(e.g. data/year=2022/chunk_id=f0c/part-0.parquet), runs the full
trucktrack processing pipeline, and writes a new hive-partitioned
output:

    1. Scan all input parquet files lazily via hive partitioning.
    2. Gap-split each vehicle's trace into discrete trip segments.
    3. Assign composite trip IDs ({id}_seg{segment_id}).
    4. Stop-split each trip into movement and stop sub-segments.
    5. Keep only movement rows (is_stop == False).
    6. Compute spatial partition metadata.
    7. Write hive-partitioned parquet to the output directory.

Usage::

    from examples.pipeline_hive_to_partitioned import run_pipeline
    run_pipeline(Path("data/raw"), Path("data/processed"))
"""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import polars as pl
import trucktrack

# Pipeline parameters.
GAP_THRESHOLD = timedelta(minutes=30)
STOP_MAX_DIAMETER = 250.0  # meters
STOP_MIN_DURATION = timedelta(minutes=5)
MIN_SEGMENT_LENGTH = 2


def run_pipeline(
    input_dir: Path,
    output_dir: Path,
    *,
    gap: timedelta = GAP_THRESHOLD,
    stop_max_diameter: float = STOP_MAX_DIAMETER,
    stop_min_duration: timedelta = STOP_MIN_DURATION,
    min_segment_length: int = MIN_SEGMENT_LENGTH,
) -> dict[str, int]:
    """Run the full gap+stop+partition pipeline.

    Returns a {tier_name: partition_count} summary.
    """
    # ── 1. Scan the hive-partitioned input ──────────────────────────
    parquet_glob = str(input_dir / "**" / "*.parquet")
    print(f"Scanning input: {parquet_glob}")

    lf = pl.scan_parquet(parquet_glob, hive_partitioning=True)
    df = lf.collect()

    n_rows_raw = len(df)
    n_vehicles = df["id"].n_unique()
    print(f"Loaded {n_rows_raw:,} rows across {n_vehicles:,} vehicles.")

    # ── 2. Gap splitting ────────────────────────────────────────────
    print(f"\nStep 1/4 — Gap splitting (threshold: {gap}) ...")
    df_gap = trucktrack.split_by_observation_gap(
        df,
        gap,
        id_col="id",
        time_col="time",
        min_length=min_segment_length,
    )
    n_gap_segments = df_gap.select(pl.struct("id", "segment_id").n_unique()).item()
    print(f"  {n_gap_segments:,} trip segments after gap splitting.")

    # ── 3. Assign composite trip IDs ────────────────────────────────
    print("\nStep 2/4 — Building composite trip IDs ...")
    df_trips = df_gap.with_columns(
        (pl.col("id") + "_seg" + pl.col("segment_id").cast(pl.Utf8)).alias("trip_id")
    )

    # ── 4. Stop splitting ───────────────────────────────────────────
    print(
        f"Step 3/4 — Stop splitting "
        f"(max_diameter={stop_max_diameter} m, "
        f"min_duration={stop_min_duration}) ..."
    )
    df_split = trucktrack.split_by_stops(
        df_trips,
        max_diameter=stop_max_diameter,
        min_duration=stop_min_duration,
        id_col="trip_id",
        time_col="time",
        lat_col="lat",
        lon_col="lon",
        min_length=min_segment_length,
    )

    n_stops = (
        df_split.filter(pl.col("is_stop")).select(pl.col("trip_id").n_unique()).item()
    )
    n_moving = (
        df_split.filter(~pl.col("is_stop")).select(pl.col("trip_id").n_unique()).item()
    )
    print(f"  Stop segments: {n_stops:,}  |  Movement segments: {n_moving:,}")

    # ── 5. Keep only movement rows ──────────────────────────────────
    df_moving = df_split.filter(~pl.col("is_stop"))
    n_moving_rows = len(df_moving)
    dropped = n_rows_raw - n_moving_rows
    print(
        f"\n  Retained {n_moving_rows:,} movement rows ({dropped:,} stop rows dropped)."
    )

    # Rename trip_id back to id for the partitioner.
    df_points = df_moving.rename({"trip_id": "id"})

    # ── 6. Compute spatial partition metadata ───────────────────────
    print("Step 4/4 — Computing spatial partition metadata ...")
    df_partitioned = trucktrack.partition_points(df_points)

    # ── 7. Write hive-partitioned output ────────────────────────────
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Writing output to: {output_dir}")

    metadata = df_partitioned.select("id", "tier", "partition_id", "hilbert_idx")
    points = df_partitioned.select(
        "id", "lat", "lon", "speed", "heading", "time"
    ).rename({"time": "timestamp"})

    partition_counts = trucktrack.write_partitions(metadata, points, str(output_dir))

    # ── Summary ─────────────────────────────────────────────────────
    total_partitions = sum(partition_counts.values())
    print("\n--- Pipeline complete ---")
    print(f"  Input rows:           {n_rows_raw:>10,}")
    print(f"  Vehicles:             {n_vehicles:>10,}")
    print(f"  Trip segments:        {n_gap_segments:>10,}")
    print(f"  Movement rows out:    {n_moving_rows:>10,}")
    print(f"  Output partitions:    {total_partitions:>10,}")
    for tier, count in sorted(partition_counts.items()):
        print(f"    {tier}: {count:,} partition(s)")
    print(f"  Output directory:     {output_dir}")

    return partition_counts
