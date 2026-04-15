"""Hive-partitioned GPS trace pipeline.

Stale-ping filter, gap splitting, stop splitting, traffic filtering, and
partitioned output.

Processes input chunks in parallel using threads (the Rust backend
releases the GIL). Each chunk is written with a unique filename so
tiles shared across chunks accumulate rather than overwrite.

    1. Discover input parquet files.
    2. For each chunk (in parallel):
       a. Drop stale GPS pings (verbatim re-emissions of earlier records).
       b. Gap-split into discrete trip segments.
       c. Stop-split into movement and stop sub-segments.
       d. Reclassify traffic stops as movement (bearing filter).
       e. Assign composite trip IDs ({id}_gap{gap}_trip{seg}).
       f. Compute spatial partition metadata.
       g. Write to hive-partitioned output with chunk-unique
          filenames.
    3. Print summary.

Usage::

    from trucktrack.pipeline import run_pipeline
    run_pipeline(Path("data/raw"), Path("data/processed"))
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import timedelta
from pathlib import Path

import polars as pl
from tqdm import tqdm

import trucktrack

# Pipeline parameters.
GAP_THRESHOLD = timedelta(minutes=30)
STOP_MAX_DIAMETER = 250.0  # meters
STOP_MIN_DURATION = timedelta(minutes=5)
TRAFFIC_MAX_ANGLE = 30.0  # degrees
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
    traffic_max_angle: float,
    min_segment_length: int,
) -> pl.DataFrame:
    """Run stale-ping filter, gap split, stop split, filter, and partition on one chunk.

    Returns a DataFrame with tier/partition_id/hilbert_idx columns.
    """
    df = trucktrack.filter_stale_pings(
        df,
        id_col="id",
        time_col="time",
        lat_col="lat",
        lon_col="lon",
        speed_col="speed",
        heading_col="heading",
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

    df = trucktrack.partition_points(df)

    return df


def _process_and_write(
    chunk_paths: list[Path],
    output_dir: Path,
    group_name: str,
    gap: timedelta,
    stop_max_diameter: float,
    stop_min_duration: timedelta,
    traffic_max_angle: float,
    min_segment_length: int,
) -> tuple[int, int]:
    """Read, process, and write a group of chunks. Returns (rows_in, rows_out)."""
    df = pl.read_parquet(chunk_paths)
    n_in = len(df)

    df = _process_chunk(
        df,
        gap=gap,
        stop_max_diameter=stop_max_diameter,
        stop_min_duration=stop_min_duration,
        traffic_max_angle=traffic_max_angle,
        min_segment_length=min_segment_length,
    )

    n_out = len(df)
    _write_chunk(df, output_dir, group_name)

    return n_in, n_out


def _group_chunks(
    chunks: list[Path],
    group_size: int,
) -> list[tuple[str, list[Path]]]:
    """Group chunk files for batched processing.

    Groups up to *group_size* files together with index-based names
    (``batch_0000``, ``batch_0001``, ...) to ensure unique output filenames.
    """
    groups: list[tuple[str, list[Path]]] = []
    for i in range(0, len(chunks), group_size):
        batch = chunks[i : i + group_size]
        name = f"batch_{i // group_size:04d}"
        groups.append((name, batch))
    return groups


def compact_partitions(data_dir: str | Path) -> int:
    """Merge chunk files within each partition into a single file.

    Partitions that already contain a single file are skipped.
    Returns the number of partitions compacted.

    Safe: writes to a temp file, renames to ``data.parquet``
    (atomic on the same filesystem), then deletes originals.
    """
    data_dir = Path(data_dir)

    # Single-pass: group files by partition directory.
    files_by_dir: dict[Path, list[Path]] = {}
    for p in sorted(data_dir.rglob("*.parquet")):
        files_by_dir.setdefault(p.parent, []).append(p)

    compacted = 0
    for pdir, files in sorted(files_by_dir.items()):
        if len(files) <= 1:
            continue
        df = pl.read_parquet(files)
        tmp = pdir / "_compacted.tmp"
        df.write_parquet(tmp)
        # Rename first so data.parquet exists before deleting originals.
        tmp.rename(pdir / "data.parquet")
        for f in files:
            f.unlink(missing_ok=True)
        compacted += 1
    return compacted


def run_pipeline(
    input_dir: Path,
    output_dir: Path,
    *,
    gap: timedelta = GAP_THRESHOLD,
    stop_max_diameter: float = STOP_MAX_DIAMETER,
    stop_min_duration: timedelta = STOP_MIN_DURATION,
    traffic_max_angle: float = TRAFFIC_MAX_ANGLE,
    min_segment_length: int = MIN_SEGMENT_LENGTH,
    max_workers: int | None = None,
    group_size: int = 1,
    compact: bool = False,
) -> dict[str, int]:
    """Run the full gap+stop+partition pipeline in parallel.

    Input chunk files are grouped into batches of *group_size* and
    processed together, producing fewer (larger) output files.  The
    Rust backend releases the GIL, so threads achieve real
    parallelism.

    *max_workers* defaults to ``os.cpu_count()``.

    Parameters
    ----------
    group_size
        Number of input chunks to read and process together.
        Higher values produce fewer output files but use more memory.
        With 256 chunks of ~272 MB each, ``group_size=4`` uses
        ~1 GB per worker.
    compact
        If ``True``, run :func:`compact_partitions` after processing
        to merge any remaining multi-file partitions into single files.

    Returns a {tier_name: partition_count} summary.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    chunks = sorted(input_dir.rglob("*.parquet"))
    if not chunks:
        print(f"No parquet files found under {input_dir}")
        return {}

    groups = _group_chunks(chunks, group_size)

    if max_workers is None:
        max_workers = os.cpu_count() or 1
    max_workers = min(max_workers, len(groups))

    print(
        f"Found {len(chunks)} input chunk(s) under {input_dir}, "
        f"grouped into {len(groups)} batch(es) of up to {group_size}, "
        f"processing with {max_workers} worker(s)"
    )

    total_rows_in = 0
    total_rows_out = 0

    with (
        tqdm(total=len(groups), unit="batch", desc="Processing") as progress,
        ThreadPoolExecutor(max_workers=max_workers) as pool,
    ):
        futures = {
            pool.submit(
                _process_and_write,
                group_paths,
                output_dir,
                group_name,
                gap,
                stop_max_diameter,
                stop_min_duration,
                traffic_max_angle,
                min_segment_length,
            ): group_name
            for group_name, group_paths in groups
        }

        for future in as_completed(futures):
            n_in, n_out = future.result()
            total_rows_in += n_in
            total_rows_out += n_out
            progress.update(1)
            progress.set_postfix_str(f"{n_in:,} -> {n_out:,} rows")

    if compact:
        print("\nCompacting partitions...")
        n_compacted = compact_partitions(output_dir)
        print(f"  Compacted {n_compacted} partition(s)")

    # Count distinct tiles written.
    tile_dirs: dict[str, set[str]] = {}
    for p in output_dir.rglob("*.parquet"):
        tier = p.parent.parent.name.removeprefix("tier=")
        tile_dirs.setdefault(tier, set()).add(p.parent.name)
    partition_counts = {t: len(pids) for t, pids in tile_dirs.items()}

    total_partitions = sum(partition_counts.values())
    print("\n--- Pipeline complete ---")
    print(f"  Input chunks:         {len(chunks):>10,}")
    print(f"  Groups:               {len(groups):>10,}")
    print(f"  Workers:              {max_workers:>10,}")
    print(f"  Input rows:           {total_rows_in:>10,}")
    print(f"  Rows out:             {total_rows_out:>10,}")
    print(f"  Output partitions:    {total_partitions:>10,}")
    for tier, count in sorted(partition_counts.items()):
        print(f"    {tier}: {count:,} partition(s)")
    print(f"  Output directory:     {output_dir}")

    return partition_counts
