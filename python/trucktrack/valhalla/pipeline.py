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
from time import perf_counter
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
_TIMING_SCHEMA = {
    "partition": pl.Utf8,
    "skipped": pl.Int64,
    "trips": pl.Int64,
    "rows_out": pl.Int64,
    "elapsed_s": pl.Float64,
}


def _quality_schema(debug: bool) -> dict[str, type[pl.DataType]]:
    if not debug:
        return _QUALITY_SCHEMA
    return {**_QUALITY_SCHEMA, "elapsed_s": pl.Float64}


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


def _quality_row(
    trip_id: str,
    date: object,
    q: MapMatchQuality,
    elapsed_s: float | None = None,
) -> dict[str, object]:
    row: dict[str, object] = {
        "id": trip_id,
        "date": date,
        "ok": q.ok,
        "error": q.error,
        "n_points": q.n_points,
        "n_polylines": q.n_polylines,
        "path_length_ratio": q.path_length_ratio,
        "heading_reversals": q.heading_reversals,
    }
    if elapsed_s is not None:
        row["elapsed_s"] = elapsed_s
    return row


def map_match_trip(
    trip: pl.DataFrame,
    *,
    config: str | Path | None = None,
    debug: bool = False,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Map-match a single trip and return (way_ids_df, quality_df).

    Uses :func:`evaluate_map_match_ways` so that Valhalla errors
    are captured in the quality row instead of crashing the pipeline.

    When *debug* is true, the quality row carries an ``elapsed_s``
    column measuring time spent in the Valhalla call for this trip.
    """
    trip = trip.sort("time")
    trip_id = trip["id"][0]
    date = cast(datetime, trip["time"].min()).date()
    schema = _quality_schema(debug)
    t0 = perf_counter() if debug else 0.0
    if len(trip) < 2:
        q = MapMatchQuality(
            trip_id=trip_id,
            ok=False,
            error="insufficient points (<2)",
            n_points=len(trip),
        )
    else:
        points = list(zip(trip["lat"].to_list(), trip["lon"].to_list(), strict=True))
        q = evaluate_map_match_ways(trip_id, points, config=config)
    elapsed = perf_counter() - t0 if debug else None
    ways = q.way_ids
    if ways:
        way_df = pl.DataFrame(
            {"id": [trip_id] * len(ways), "date": [date] * len(ways), "way_id": ways},
            schema=_WAY_SCHEMA,
        )
    else:
        way_df = _null_way_result(trip_id, date)
    quality_df = pl.DataFrame([_quality_row(trip_id, date, q, elapsed)], schema=schema)
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


def _parquet_row_count(path: Path) -> int:
    """Row count from parquet footer — no row-group read."""
    return int(pl.scan_parquet(path).select(pl.len()).collect().item())


def _process_partition(
    partition_dir: Path,
    input_dir: Path,
    output_dir: Path,
    config: str | Path | None,
    row_counts: dict[Path, int],
    progress: tqdm[None] | None = None,
    *,
    debug: bool = False,
) -> tuple[str, int, int, int, float]:
    """Read one partition chunk-by-chunk, map-match each trip, and write results.

    Processes one chunk at a time to keep memory free for Valhalla.
    Skips chunks whose output file already exists.
    Returns (partition_key, chunks_skipped, trips_matched, rows_out, elapsed_s).
    """
    t0 = perf_counter()
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
        raw_rows = row_counts[chunk_path]
        out_path = out_dir / chunk_path.name
        quality_path = quality_dir / chunk_path.name
        if out_path.exists():
            skipped += 1
            if progress is not None:
                progress.update(raw_rows)
            continue

        lf = pl.scan_parquet(chunk_path)
        if "is_stop" in lf.collect_schema():
            lf = lf.filter(~pl.col("is_stop"))
        df = lf.select(["id", "time", "lat", "lon"]).collect()
        if df.is_empty():
            if progress is not None:
                progress.update(raw_rows)
            continue
        way_dfs = []
        quality_dfs = []
        for _, trip in df.group_by("id"):
            way_df, quality_df = map_match_trip(trip, config=config, debug=debug)
            way_dfs.append(way_df)
            quality_dfs.append(quality_df)
            total_trips += 1
            if progress is not None:
                progress.update(len(trip))

        # Rows dropped by the is_stop filter still count as processed.
        filtered_out = raw_rows - len(df)
        if progress is not None and filtered_out:
            progress.update(filtered_out)

        if not way_dfs:
            continue

        matched = pl.concat(way_dfs)
        quality = pl.concat(quality_dfs)
        total_rows += len(matched)

        _atomic_write_parquet(matched, out_path)
        _atomic_write_parquet(quality, quality_path)

    return str(rel), skipped, total_trips, total_rows, perf_counter() - t0


def _process_block(
    partition_dirs: list[Path],
    input_dir: Path,
    output_dir: Path,
    config: str | Path | None,
    row_counts: dict[Path, int],
    progress: tqdm[None] | None = None,
    *,
    debug: bool = False,
) -> list[tuple[str, int, int, int, float]]:
    """Process a contiguous block of spatially adjacent partitions sequentially."""
    return [
        _process_partition(
            pdir,
            input_dir,
            output_dir,
            config,
            row_counts,
            progress,
            debug=debug,
        )
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
    debug: bool = False,
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

    *quiet* suppresses Valhalla's C++ log output. Progress (tqdm) still
    prints to stderr. Per-trip errors are recorded in the quality
    parquet rather than logged.

    *debug* adds an ``elapsed_s`` column to each quality row, so you
    can isolate pathological trips within a slow partition. Per-partition
    timing is always written to ``output_dir / "_timing.parquet"``.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    all_chunk_paths: list[Path] = []
    partition_dir_set: set[Path] = set()
    for p in input_dir.rglob("*.parquet"):
        if p.parent.name.startswith("partition_id="):
            partition_dir_set.add(p.parent)
            all_chunk_paths.append(p)
    partition_dirs = sorted(partition_dir_set, key=_partition_sort_key)
    total_chunks = len(all_chunk_paths)

    if not partition_dirs:
        print(f"No partitions found under {input_dir}", file=sys.stderr)
        return

    if max_workers is None:
        max_workers = os.cpu_count() or 1
    max_workers = min(max_workers, len(partition_dirs))

    # Parquet-footer reads are one RTT each; parallelize so networked
    # filesystems don't serialize thousands of tiny reads.
    print(
        f"Counting rows across {total_chunks:,} chunk(s)...",
        file=sys.stderr,
    )
    with ThreadPoolExecutor(max_workers=max_workers) as count_pool:
        counts = list(count_pool.map(_parquet_row_count, all_chunk_paths))
    row_counts = dict(zip(all_chunk_paths, counts, strict=True))
    total_input_rows = sum(counts)

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

    size_str = f"{block_size}\u2013{block_size + 1}" if remainder else str(block_size)
    print(
        f"Found {n} partition(s) "
        f"({total_chunks:,} chunks, {total_input_rows:,} rows), "
        f"processing with {max_workers} worker(s) "
        f"({size_str} partitions per worker)",
        file=sys.stderr,
    )

    total_skipped = 0
    total_trips = 0
    total_rows = 0

    with (
        _silence_stdout() if quiet else nullcontext(),
        tqdm(
            total=total_input_rows,
            unit="row",
            unit_scale=True,
            desc="Map matching",
        ) as progress,
        ThreadPoolExecutor(max_workers=max_workers) as pool,
    ):
        futures = [
            pool.submit(
                _process_block,
                block,
                input_dir,
                output_dir,
                config,
                row_counts,
                progress,
                debug=debug,
            )
            for block in blocks
        ]

        timing_rows: list[dict[str, object]] = []
        for future in as_completed(futures):
            for key, skipped, trips, rows, elapsed in future.result():
                total_skipped += skipped
                total_trips += trips
                total_rows += rows
                timing_rows.append(
                    {
                        "partition": key,
                        "skipped": skipped,
                        "trips": trips,
                        "rows_out": rows,
                        "elapsed_s": elapsed,
                    }
                )

    if timing_rows:
        output_dir.mkdir(parents=True, exist_ok=True)
        timing_df = pl.DataFrame(timing_rows, schema=_TIMING_SCHEMA).sort(
            "elapsed_s", descending=True
        )
        _atomic_write_parquet(timing_df, output_dir / "_timing.parquet")

    print("\n--- Map matching complete ---", file=sys.stderr)
    print(f"  Partitions:  {n:>10,}", file=sys.stderr)
    print(f"  Chunks:      {total_chunks:>10,}", file=sys.stderr)
    print(f"  Skipped:     {total_skipped:>10,} chunk(s)", file=sys.stderr)
    print(f"  Matched:     {total_trips:>10,} trip(s)", file=sys.stderr)
    print(f"  Rows out:    {total_rows:>10,}", file=sys.stderr)
    print(f"  Timing:      {output_dir / '_timing.parquet'}", file=sys.stderr)
