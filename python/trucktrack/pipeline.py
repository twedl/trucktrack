"""Split each truck's trace into trips, one truck bucket at a time.

Stale-ping filter, impossible-speed filter, gap splitting, stop
splitting and traffic filtering, on the buckets written by
:func:`~trucktrack.buckets.bucket_by_truck`.

Each bucket holds every point of its trucks, so a truck's trips are
split and numbered from its whole history.  Buckets are processed in
parallel on threads (the Rust filters release the GIL).

    1. Find the input buckets.
    2. For each bucket (in parallel):
       a. Drop stale GPS pings (verbatim re-emissions of earlier records).
       b. Drop points implying impossible speeds.
       c. Gap-split into discrete trip segments.
       d. Stop-split into movement and stop sub-segments.
       e. Reclassify traffic stops as movement (bearing filter).
       f. Assign composite trip IDs ({id}_gap{gap}_trip{seg}).
       g. Write to the same bucket in the output, in files of about
          ``rows_per_file`` rows, each truck in one file.
    3. Print summary.

Usage::

    from trucktrack.buckets import bucket_by_truck
    from trucktrack.pipeline import run_pipeline
    bucket_by_truck(Path("data/raw"), Path("data/bucketed"))
    run_pipeline(Path("data/bucketed"), Path("data/split"))
"""

from __future__ import annotations

import functools
import os
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import timedelta
from pathlib import Path

import polars as pl
from tqdm import tqdm

import trucktrack
from trucktrack.buckets import (
    bucket_dir,
    bucket_dirs,
    bucket_files,
    bucket_number,
    require_bucketed,
    require_empty,
    write_n_buckets,
)
from trucktrack.query import truck_id_expr

# Pipeline parameters.
GAP_THRESHOLD = timedelta(minutes=30)
STOP_MAX_DIAMETER = 250.0  # meters
STOP_MIN_DURATION = timedelta(minutes=5)
TRAFFIC_MAX_ANGLE = 30.0  # degrees
IMPOSSIBLE_SPEED_KMH = 200.0  # km/h; drop GPS points implying travel above this
MIN_SEGMENT_LENGTH = 2
# Output files are map-matching's unit of work and of resuming.
ROWS_PER_FILE = 1_000_000


def parquet_row_count(path: Path) -> int:
    """Row count from parquet footer — no row-group read."""
    return int(pl.scan_parquet(path).select(pl.len()).collect().item())


def _process_bucket(
    df: pl.DataFrame,
    *,
    gap: timedelta,
    stop_max_diameter: float,
    stop_min_duration: timedelta,
    traffic_max_angle: float,
    impossible_speed_kmh: float,
    min_segment_length: int,
) -> pl.DataFrame:
    """Run pre-process filters and segmentation on one bucket of trucks."""
    df = trucktrack.filter_stale_pings(
        df,
        id_col="id",
        time_col="time",
        lat_col="lat",
        lon_col="lon",
        speed_col="speed",
        heading_col="heading",
    )

    df = trucktrack.filter_impossible_speeds(
        df,
        max_speed_kmh=impossible_speed_kmh,
        id_col="id",
        time_col="time",
        lat_col="lat",
        lon_col="lon",
    )

    df = trucktrack.split_by_observation_gap(
        df,
        gap,
        id_col="id",
        time_col="time",
        min_length=min_segment_length,
    )

    # Rename before stop splitting overwrites segment_id.
    df = df.rename({"segment_id": "gap_segment_id"})

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

    df = trucktrack.filter_traffic_stops(
        df,
        max_angle_change=traffic_max_angle,
        id_col="id",
        lat_col="lat",
        lon_col="lon",
    )

    df = df.with_columns(
        (
            pl.col("id")
            + "_gap"
            + pl.col("gap_segment_id").cast(pl.Utf8)
            + "_trip"
            + pl.col("segment_id").cast(pl.Utf8)
        ).alias("id")
    )

    return df


def _split_bucket(
    bucket: Path,
    output_dir: Path,
    split: Callable[[pl.DataFrame], pl.DataFrame],
    rows_per_file: int,
) -> tuple[int, int, int]:
    """Split one bucket into trips and write it. Returns (rows_in, rows_out, trips)."""
    df = pl.read_parquet(bucket_files(bucket), hive_partitioning=False)
    n_in = len(df)
    df = split(df)

    if not df.is_empty():
        out_dir = bucket_dir(output_dir, bucket_number(bucket))
        out_dir.mkdir(parents=True)
        # A truck's rows all go to the file holding its first row, so no
        # trip spans two files; the splitters leave each truck contiguous.
        file_of = pl.col("_row").min().over(truck_id_expr()) // rows_per_file
        parts = (
            df.with_row_index("_row")
            .with_columns(file_of.alias("_file"))
            .drop("_row")
            .partition_by("_file", include_key=False, maintain_order=True)
        )
        for i, part in enumerate(parts):
            part.write_parquet(out_dir / f"part-{i:04d}.parquet")

    return n_in, len(df), df["id"].n_unique()


def run_pipeline(
    input_dir: Path,
    output_dir: Path,
    *,
    gap: timedelta = GAP_THRESHOLD,
    stop_max_diameter: float = STOP_MAX_DIAMETER,
    stop_min_duration: timedelta = STOP_MIN_DURATION,
    traffic_max_angle: float = TRAFFIC_MAX_ANGLE,
    impossible_speed_kmh: float = IMPOSSIBLE_SPEED_KMH,
    min_segment_length: int = MIN_SEGMENT_LENGTH,
    rows_per_file: int = ROWS_PER_FILE,
    max_workers: int | None = None,
) -> dict[str, int]:
    """Split every truck bucket under *input_dir* into trips, in parallel.

    *input_dir* is the output of
    :func:`~trucktrack.buckets.bucket_by_truck`.  Writes
    ``output_dir/bucket=K/part-NNNN.parquet`` for each bucket with any
    trip left, each file about *rows_per_file* rows with every truck in
    one file, and ``_buckets.json`` last.  *output_dir* must be empty or
    missing.  Each worker thread holds a whole bucket in memory.

    *max_workers* defaults to ``os.cpu_count()``.

    Returns counts of ``buckets``, ``rows_in``, ``rows_out`` and
    ``trips``.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    n_buckets = require_bucketed(input_dir, made_by="bucket_by_truck")
    require_empty(output_dir)

    buckets = bucket_dirs(input_dir)
    if not buckets:
        print(f"No buckets found under {input_dir}")
        return {"buckets": 0, "rows_in": 0, "rows_out": 0, "trips": 0}

    if max_workers is None:
        max_workers = os.cpu_count() or 1
    max_workers = min(max_workers, len(buckets))

    print(
        f"Found {len(buckets)} bucket(s) under {input_dir}, "
        f"processing with {max_workers} worker(s)"
    )

    rows_in = 0
    rows_out = 0
    trips = 0
    split = functools.partial(
        _process_bucket,
        gap=gap,
        stop_max_diameter=stop_max_diameter,
        stop_min_duration=stop_min_duration,
        traffic_max_angle=traffic_max_angle,
        impossible_speed_kmh=impossible_speed_kmh,
        min_segment_length=min_segment_length,
    )

    with (
        tqdm(total=len(buckets), unit="bucket", desc="Splitting") as progress,
        ThreadPoolExecutor(max_workers=max_workers) as pool,
    ):
        futures = [
            pool.submit(_split_bucket, bucket, output_dir, split, rows_per_file)
            for bucket in buckets
        ]

        for future in as_completed(futures):
            n_in, n_out, n_trips = future.result()
            rows_in += n_in
            rows_out += n_out
            trips += n_trips
            progress.update(1)
            progress.set_postfix_str(f"{n_in:,} -> {n_out:,} rows")

    # Last, so a crashed run leaves no _buckets.json to mistake for done.
    write_n_buckets(output_dir, n_buckets)

    print("\n--- Pipeline complete ---")
    print(f"  Buckets:              {len(buckets):>10,}")
    print(f"  Workers:              {max_workers:>10,}")
    print(f"  Input rows:           {rows_in:>10,}")
    print(f"  Rows out:             {rows_out:>10,}")
    print(f"  Trips:                {trips:>10,}")
    print(f"  Output directory:     {output_dir}")

    return {
        "buckets": len(buckets),
        "rows_in": rows_in,
        "rows_out": rows_out,
        "trips": trips,
    }
