"""Map-match trips from hive-partitioned output.

Reads the partitioned parquet dataset produced by
:func:`trucktrack.pipeline.run_pipeline`, iterates over partitions, and
applies a map-matching function to each trip.

Resumable: the output directory mirrors the input hive layout. On
restart, chunks whose output file already exists are skipped. Writes
use atomic temp-file-then-rename to avoid partial output from crashes.

Partitions are processed in parallel using threads, assigned in
contiguous spatial blocks to keep Valhalla's tile cache warm.

Usage::

    from trucktrack.valhalla.pipeline import run_map_matching
    run_map_matching(Path("data/processed"), Path("data/matched"))
"""

from __future__ import annotations

import os
import sys
import tempfile
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager, nullcontext
from datetime import datetime
from pathlib import Path
from typing import cast

import polars as pl
from tqdm import tqdm

from trucktrack.valhalla.quality import MapMatchQuality, evaluate_map_match_ways

_WAY_SCHEMA = {"id": pl.Utf8, "date": pl.Date, "way_id": pl.Int64}
_QUALITY_SCHEMA = {
    "id": pl.Utf8,
    "date": pl.Date,
    "ok": pl.Boolean,
    "error": pl.Utf8,
    "n_points": pl.Int64,
    "n_polylines": pl.Int64,
    "path_length_ratio": pl.Float64,
    "heading_reversals": pl.Int64,
}


@contextmanager
def _silence_stdout() -> Iterator[None]:
    """Redirect fd 1 to /dev/null so Valhalla's C++ log output is hidden.

    Python prints in this module go to stderr, so they survive this
    redirect. tqdm also writes to stderr by default.
    """
    sys.stdout.flush()
    devnull = os.open(os.devnull, os.O_WRONLY)
    saved = os.dup(1)
    try:
        os.dup2(devnull, 1)
        yield
    finally:
        sys.stdout.flush()
        os.dup2(saved, 1)
        os.close(devnull)
        os.close(saved)


def _null_way_result(trip_id: str, date: object) -> pl.DataFrame:
    """Single row with null way_id — used for skipped or failed trips."""
    return pl.DataFrame(
        {"id": [trip_id], "date": [date], "way_id": [None]},
        schema=_WAY_SCHEMA,
    )


def _quality_row(trip_id: str, date: object, q: MapMatchQuality) -> dict[str, object]:
    return {
        "id": trip_id,
        "date": date,
        "ok": q.ok,
        "error": q.error,
        "n_points": q.n_points,
        "n_polylines": q.n_polylines,
        "path_length_ratio": q.path_length_ratio,
        "heading_reversals": q.heading_reversals,
    }


def map_match_trip(
    trip: pl.DataFrame,
    *,
    config: str | Path | None = None,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Map-match a single trip and return (way_ids_df, quality_df).

    Uses :func:`evaluate_map_match_attributes` so that Valhalla errors
    are captured in the quality row instead of crashing the pipeline.
    """
    trip = trip.sort("time")
    trip_id = trip["id"][0]
    date = cast(datetime, trip["time"].min()).date()
    if len(trip) < 2:
        q = MapMatchQuality(trip_id=trip_id, ok=False, error="insufficient points (<2)", n_points=len(trip))
        return (
            _null_way_result(trip_id, date),
            pl.DataFrame([_quality_row(trip_id, date, q)], schema=_QUALITY_SCHEMA),
        )
    points = list(zip(trip["lat"].to_list(), trip["lon"].to_list(), strict=True))
    q = evaluate_map_match_ways(trip_id, points, config=config)
    if q.error is not None:
        print(f"  [WARN] {trip_id}: {q.error}", file=sys.stderr)
    ways = q.way_ids
    if ways:
        way_df = pl.DataFrame(
            {"id": [trip_id] * len(ways), "date": [date] * len(ways), "way_id": ways},
            schema=_WAY_SCHEMA,
        )
    else:
        way_df = _null_way_result(trip_id, date)
    quality_df = pl.DataFrame([_quality_row(trip_id, date, q)], schema=_QUALITY_SCHEMA)
    return way_df, quality_df


def _atomic_write_parquet(df: pl.DataFrame, dest: Path) -> None:
    """Write a parquet file atomically via temp-file-then-rename."""
    fd, tmp = tempfile.mkstemp(dir=dest.parent, suffix=".tmp.parquet")
    os.close(fd)
    try:
        df.write_parquet(tmp)
        os.rename(tmp, dest)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def _process_partition(
    partition_dir: Path,
    input_dir: Path,
    output_dir: Path,
    config: str | Path | None,
    progress: tqdm[None] | None = None,
) -> tuple[str, int, int, int]:
    """Read one partition chunk-by-chunk, map-match each trip, and write results.

    Processes one chunk at a time to keep memory free for Valhalla.
    Skips chunks whose output file already exists.
    Returns (partition_key, chunks_skipped, trips_matched, rows_out).
    """
    rel = partition_dir.relative_to(input_dir)
    out_dir = output_dir / rel
    quality_dir = output_dir / "_quality" / rel
    out_dir.mkdir(parents=True, exist_ok=True)
    quality_dir.mkdir(parents=True, exist_ok=True)

    chunks = sorted(partition_dir.glob("*.parquet"))
    skipped = 0
    total_trips = 0
    total_rows = 0

    for chunk_path in chunks:
        out_path = out_dir / chunk_path.name
        quality_path = quality_dir / chunk_path.name
        if out_path.exists():
            skipped += 1
            if progress is not None:
                progress.update(1)
            continue

        df = pl.read_parquet(chunk_path)
        if "is_stop" in df.columns:
            df = df.filter(~pl.col("is_stop"))
        if df.is_empty():
            if progress is not None:
                progress.update(1)
            continue
        way_dfs = []
        quality_dfs = []
        for _, trip in df.group_by("id"):
            way_df, quality_df = map_match_trip(trip, config=config)
            way_dfs.append(way_df)
            quality_dfs.append(quality_df)
            total_trips += 1

        if not way_dfs:
            if progress is not None:
                progress.update(1)
            continue

        matched = pl.concat(way_dfs)
        quality = pl.concat(quality_dfs)
        total_rows += len(matched)

        _atomic_write_parquet(matched, out_path)
        _atomic_write_parquet(quality, quality_path)

        if progress is not None:
            progress.update(1)

    return str(rel), skipped, total_trips, total_rows


def _process_block(
    partition_dirs: list[Path],
    input_dir: Path,
    output_dir: Path,
    config: str | Path | None,
    progress: tqdm[None] | None = None,
) -> list[tuple[str, int, int, int]]:
    """Process a contiguous block of spatially adjacent partitions sequentially."""
    return [
        _process_partition(pdir, input_dir, output_dir, config, progress)
        for pdir in partition_dirs
    ]


def _partition_sort_key(path: Path) -> tuple[str, int]:
    """Extract (tier, partition_id) for spatial sort order."""
    tier = path.parent.name.removeprefix("tier=")
    pid = int(path.name.removeprefix("partition_id="))
    return (tier, pid)


def run_map_matching(
    input_dir: Path,
    output_dir: Path,
    *,
    config: str | Path | None = None,
    max_workers: int | None = None,
    quiet: bool = False,
) -> None:
    """Map-match all trips across partitions in parallel.

    *config* is an optional path to ``valhalla.json``; when ``None``
    :func:`trucktrack.valhalla.find_config` discovers one in cwd
    (e.g. ``./valhalla.json``).

    *input_dir* should be the hive-partitioned output of
    :func:`trucktrack.pipeline.run_pipeline`, with layout::

        input_dir/tier=.../partition_id=.../chunk.parquet

    *output_dir* mirrors the same hive layout. Re-running skips chunks
    whose output file already exists.

    *max_workers* defaults to ``os.cpu_count()``.

    *quiet* suppresses Valhalla's C++ log output. Progress (tqdm) and
    trip-level warnings still print to stderr.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    # Single pass: collect partition dirs and count chunks.
    chunks_per_partition: dict[Path, int] = {}
    for p in input_dir.rglob("*.parquet"):
        if p.parent.name.startswith("partition_id="):
            chunks_per_partition[p.parent] = chunks_per_partition.get(p.parent, 0) + 1
    partition_dirs = sorted(chunks_per_partition, key=_partition_sort_key)

    if not partition_dirs:
        print(f"No partitions found under {input_dir}", file=sys.stderr)
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

    total_chunks = sum(chunks_per_partition.values())

    size_str = f"{block_size}\u2013{block_size + 1}" if remainder else str(block_size)
    print(
        f"Found {n} partition(s) ({total_chunks} chunks), "
        f"processing with {max_workers} worker(s) "
        f"({size_str} partitions per worker)",
        file=sys.stderr,
    )

    total_skipped = 0
    total_trips = 0
    total_rows = 0

    with (
        _silence_stdout() if quiet else nullcontext(),
        tqdm(total=total_chunks, unit="chunk", desc="Map matching") as progress,
        ThreadPoolExecutor(max_workers=max_workers) as pool,
    ):
        futures = [
            pool.submit(
                _process_block,
                block,
                input_dir,
                output_dir,
                config,
                progress,
            )
            for block in blocks
        ]

        for future in as_completed(futures):
            for _key, skipped, trips, rows in future.result():
                total_skipped += skipped
                total_trips += trips
                total_rows += rows

    print("\n--- Map matching complete ---", file=sys.stderr)
    print(f"  Partitions:  {n:>10,}", file=sys.stderr)
    print(f"  Chunks:      {total_chunks:>10,}", file=sys.stderr)
    print(f"  Skipped:     {total_skipped:>10,} chunk(s)", file=sys.stderr)
    print(f"  Matched:     {total_trips:>10,} trip(s)", file=sys.stderr)
    print(f"  Rows out:    {total_rows:>10,}", file=sys.stderr)
