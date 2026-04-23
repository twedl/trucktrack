"""Map-match trips from hive-partitioned output.

Reads the partitioned parquet dataset produced by
:func:`trucktrack.pipeline.run_pipeline`, iterates over partitions, and
applies a map-matching function to each trip.

Resumable: the output directory mirrors the input hive layout. On
restart, chunks whose output file already exists are skipped. Writes
use atomic temp-file-then-rename to avoid partial output from crashes.

Partitions are processed in parallel using worker processes
(ProcessPoolExecutor), assigned in contiguous spatial blocks to keep
each worker's Valhalla tile cache warm.  Processes (not threads)
because per-trip Python work, however thin, is GIL-serialized across
workers — at 29 workers on a long run, active CPU plateaus around
14/29 with threads no matter how much Python we trim.

Usage::

    from trucktrack.valhalla.pipeline import run_map_matching
    run_map_matching(Path("data/processed"), Path("data/matched"))
"""

from __future__ import annotations

import multiprocessing as mp
import os
import sys
import tempfile
import threading
from collections.abc import Iterator
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any, cast

import polars as pl
from tqdm import tqdm

from trucktrack.pipeline import parquet_row_count, partition_sort_key
from trucktrack.valhalla._bridge import BridgeConfig
from trucktrack.valhalla.quality import (
    MapMatchQuality,
    evaluate_map_match_ways,
    evaluate_map_match_with_bridges,
)

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
    "elapsed_s": pl.Float64,
    "n_bridges": pl.Int64,
    "max_detour_ratio": pl.Float64,
    "total_bridge_m": pl.Float64,
    "any_bridge_failed": pl.Boolean,
}
_TIMING_SCHEMA = {
    "tier": pl.Utf8,
    "partition_id": pl.Int64,
    "skipped": pl.Int64,
    "trips": pl.Int64,
    "rows_out": pl.Int64,
    "elapsed_s": pl.Float64,
}

# Rows to accumulate per worker before pushing a progress update.
# Was ~8% of wall time via tqdm's RLock when we hit it per-trip from 29
# workers; batching amortizes both the mp.Queue round-trip and the
# main-thread tqdm.update().
_PROGRESS_BATCH_ROWS = 10_000

# Per-worker handle to the main-process progress queue, set in
# _worker_init.  None in the parent and in tests that call the helpers
# directly.
_PROGRESS_QUEUE: Any = None


def _report_progress(rows: int) -> None:
    """Ship a row-count increment to the main process' tqdm drainer."""
    if _PROGRESS_QUEUE is not None and rows:
        _PROGRESS_QUEUE.put(rows)


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


def _worker_init(quiet: bool, progress_queue: Any) -> None:
    """ProcessPoolExecutor initializer: silence stdout, wire progress queue.

    Stashes the shared progress queue in a module global so
    ``_report_progress`` can stream per-partition updates back to the
    main process' tqdm drainer without round-tripping through each
    future's result.  No stdout restore — the worker never writes to
    stdout for its own output.
    """
    global _PROGRESS_QUEUE
    _PROGRESS_QUEUE = progress_queue
    if quiet:
        sys.stdout.flush()
        devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull, 1)
        os.close(devnull)


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
    return {
        "id": trip_id,
        "date": date,
        "ok": q.ok,
        "error": q.error,
        "n_points": q.n_points,
        "n_polylines": q.n_polylines,
        "path_length_ratio": q.path_length_ratio,
        "heading_reversals": q.heading_reversals,
        "elapsed_s": elapsed_s,
        "n_bridges": q.n_bridges,
        "max_detour_ratio": q.max_detour_ratio,
        "total_bridge_m": q.total_bridge_m,
        "any_bridge_failed": q.any_bridge_failed,
    }


def _map_match_trip_row(
    trip: pl.DataFrame,
    *,
    config: str | Path | None = None,
    debug: bool = False,
    bridges: BridgeConfig | None = None,
) -> tuple[pl.DataFrame, dict[str, object]]:
    """Map-match one trip; return (way_ids_df, quality_row_dict).

    The quality row is left as a dict so the caller can batch many rows
    into a single ``pl.DataFrame`` construction — the per-trip list-of-
    one-dict pattern hits polars' ``_sequence_of_dict_to_pydf`` under the
    GIL and shows up as a hotspot in py-spy.
    """
    trip = trip.sort("time")
    trip_id = trip.item(0, "id")
    date = cast(datetime, trip["time"].min()).date()
    t0 = perf_counter() if debug else 0.0
    if len(trip) < 2:
        q = MapMatchQuality(
            trip_id=trip_id,
            ok=False,
            error="insufficient points (<2)",
            n_points=len(trip),
        )
    elif bridges is not None:
        q = evaluate_map_match_with_bridges(
            trip_id, trip, bridges=bridges, config=config
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
    return way_df, _quality_row(trip_id, date, q, elapsed)


def map_match_trip(
    trip: pl.DataFrame,
    *,
    config: str | Path | None = None,
    debug: bool = False,
    bridges: BridgeConfig | None = None,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Map-match a single trip and return (way_ids_df, quality_df).

    Valhalla errors are captured in the quality row instead of crashing
    the pipeline.

    When *debug* is true, the quality row's ``elapsed_s`` column holds
    the wall-clock time spent in the Valhalla call; otherwise it is null.

    When *bridges* is supplied, the trip is split at large gaps,
    matched segment-by-segment, and bridged via ``/route`` +
    ``edge_walk`` (see
    :func:`trucktrack.valhalla.quality.evaluate_map_match_with_bridges`).
    """
    way_df, quality_row = _map_match_trip_row(
        trip, config=config, debug=debug, bridges=bridges
    )
    quality_df = pl.DataFrame([quality_row], schema=_QUALITY_SCHEMA)
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
    row_counts: dict[Path, int],
    *,
    debug: bool = False,
    bridges: BridgeConfig | None = None,
) -> tuple[str, int, int, int, int, float]:
    """Read one partition chunk-by-chunk, map-match each trip, and write results.

    Processes one chunk at a time to keep memory free for Valhalla.
    Skips chunks whose output file already exists.
    Returns (tier, partition_id, chunks_skipped, trips_matched, rows_out, elapsed_s).
    """
    t0 = perf_counter()
    rel = partition_dir.relative_to(input_dir)
    tier = rel.parent.name.removeprefix("tier=")
    partition_id = int(rel.name.removeprefix("partition_id="))
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
            _report_progress(raw_rows)
            continue

        lf = pl.scan_parquet(chunk_path)
        if "is_stop" in lf.collect_schema():
            lf = lf.filter(~pl.col("is_stop"))
        df = lf.select(["id", "time", "lat", "lon"]).collect()
        if df.is_empty():
            _report_progress(raw_rows)
            continue
        way_dfs: list[pl.DataFrame] = []
        quality_rows: list[dict[str, object]] = []
        rows_pending = 0
        for _, trip in df.group_by("id"):
            way_df, quality_row = _map_match_trip_row(
                trip, config=config, debug=debug, bridges=bridges
            )
            way_dfs.append(way_df)
            quality_rows.append(quality_row)
            total_trips += 1
            rows_pending += len(trip)
            if rows_pending >= _PROGRESS_BATCH_ROWS:
                _report_progress(rows_pending)
                rows_pending = 0

        # Rows dropped by the is_stop filter still count as processed.
        filtered_out = raw_rows - len(df)
        _report_progress(rows_pending + filtered_out)

        if not way_dfs:
            continue

        matched = pl.concat(way_dfs)
        quality = pl.DataFrame(quality_rows, schema=_QUALITY_SCHEMA)
        total_rows += len(matched)

        _atomic_write_parquet(matched, out_path)
        _atomic_write_parquet(quality, quality_path)

    return tier, partition_id, skipped, total_trips, total_rows, perf_counter() - t0


def _process_chunk(
    partition_dirs: list[Path],
    input_dir: Path,
    output_dir: Path,
    config: str | Path | None,
    row_counts: dict[Path, int],
    *,
    debug: bool = False,
    bridges: BridgeConfig | None = None,
) -> list[tuple[str, int, int, int, int, float]]:
    """Process a contiguous chunk of spatially adjacent partitions sequentially.

    One chunk owns a geographically-coherent slice of Ontario, so the
    Valhalla tile LRU and the OS page cache over the mmapped tile tar
    stay hot across successive partitions.  With many chunks per
    worker, fast workers still steal from the queue after finishing
    their own chunk — we get locality within a chunk and load balance
    across chunks.
    """
    return [
        _process_partition(
            pdir,
            input_dir,
            output_dir,
            config,
            row_counts,
            debug=debug,
            bridges=bridges,
        )
        for pdir in partition_dirs
    ]


def run_map_matching(
    input_dir: Path,
    output_dir: Path,
    *,
    config: str | Path | None = None,
    max_workers: int | None = None,
    chunks_per_worker: int = 4,
    quiet: bool = False,
    debug: bool = False,
    bridges: BridgeConfig | None = None,
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

    *bridges*, when supplied, splits trips at large gaps (see
    :class:`trucktrack.valhalla.BridgeConfig`), matches each segment,
    and bridges gaps via ``/route`` + ``edge_walk``.  Populates
    ``n_bridges``, ``max_detour_ratio``, ``total_bridge_m``,
    ``any_bridge_failed`` in the quality rows.

    *chunks_per_worker* controls how many chunks of
    spatially-contiguous partitions each worker sees on average
    (default 4).  Partitions are sorted spatially and sliced into
    ``max_workers * chunks_per_worker`` chunks; workers claim whole
    chunks so the Valhalla tile LRU and OS page cache stay hot across
    a chunk's partitions, and the queue's extra chunks let fast
    workers steal from slow ones to trim the tail.  Lower values
    (1-2) bias toward locality; higher values (8+) bias toward
    balance.

    .. note::
        For best CPU utilization, set ``POLARS_MAX_THREADS=1`` in your
        environment before launching.  Each partition worker does its
        own ``pl.scan_parquet().collect()`` on a small chunk; polars'
        global rayon pool then fans that tiny work out across all
        cores, and with ``max_workers`` partition threads doing it
        simultaneously, every polars call fights every other one for
        CPU.  Serializing polars means partition-level parallelism
        runs cleanly.  A warning fires at startup if the env var isn't
        set and ``max_workers > 4``.

    .. note::
        Workers are launched with the ``spawn`` start method (required
        because polars keeps background threads in the parent that
        would otherwise deadlock across ``fork``).  Call this function
        from inside an ``if __name__ == "__main__":`` guard — without
        it, each spawned worker re-imports your script at top level
        and Python raises ``RuntimeError: An attempt has been made to
        start a new process before the current process has finished
        its bootstrapping phase``.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    all_chunk_paths: list[Path] = []
    partition_chunks: dict[Path, list[Path]] = {}
    for p in input_dir.rglob("*.parquet"):
        if p.parent.name.startswith("partition_id="):
            partition_chunks.setdefault(p.parent, []).append(p)
            all_chunk_paths.append(p)
    partition_dirs = sorted(partition_chunks, key=partition_sort_key)
    total_chunks = len(all_chunk_paths)

    if not partition_dirs:
        print(f"No partitions found under {input_dir}", file=sys.stderr)
        return

    if max_workers is None:
        max_workers = os.cpu_count() or 1
    max_workers = min(max_workers, len(partition_dirs))

    if max_workers > 4 and "POLARS_MAX_THREADS" not in os.environ:
        print(
            f"warning: POLARS_MAX_THREADS not set with max_workers={max_workers}. "
            "Polars' global rayon pool will oversubscribe CPU as each partition "
            "worker's scan_parquet/collect fans out across all cores. "
            "Set POLARS_MAX_THREADS=1 before launching for best throughput.",
            file=sys.stderr,
        )

    # Parquet-footer reads are one RTT each; parallelize so networked
    # filesystems don't serialize thousands of tiny reads.
    print(
        f"Counting rows across {total_chunks:,} chunk(s)...",
        file=sys.stderr,
    )
    with ThreadPoolExecutor(max_workers=max_workers) as count_pool:
        counts = list(count_pool.map(parquet_row_count, all_chunk_paths))
    row_counts = dict(zip(all_chunk_paths, counts, strict=True))
    total_input_rows = sum(counts)

    n = len(partition_dirs)

    # Slice partitions into spatially-contiguous chunks.  More chunks
    # than workers so the executor queue can still smooth out
    # imbalance after a worker finishes its first chunk.
    target_chunks = max(max_workers, max_workers * chunks_per_worker)
    chunk_size = max(1, (n + target_chunks - 1) // target_chunks)
    chunks: list[list[Path]] = [
        partition_dirs[i : i + chunk_size] for i in range(0, n, chunk_size)
    ]

    print(
        f"Found {n} partition(s) "
        f"({total_chunks:,} chunks, {total_input_rows:,} rows), "
        f"processing with {max_workers} worker(s) "
        f"over {len(chunks)} chunk(s) of ~{chunk_size} partition(s) each",
        file=sys.stderr,
    )

    total_skipped = 0
    total_trips = 0
    total_rows = 0

    # Longest-Processing-Time-first: submit the heaviest chunk-futures
    # earliest so the longest tasks start when the pool is full and
    # shorter tasks fill idle workers at the end.  Cuts the long tail
    # when partitions have skewed row counts (downtown vs rural).
    def _chunk_rows(pdirs: list[Path]) -> int:
        return sum(row_counts[p] for pdir in pdirs for p in partition_chunks[pdir])

    chunks_with_rows: list[tuple[list[Path], int]] = sorted(
        ((pdirs, _chunk_rows(pdirs)) for pdirs in chunks),
        key=lambda cr: cr[1],
        reverse=True,
    )

    # Shared queue — workers stream per-batch row counts; a main-process
    # drainer thread forwards them to tqdm so the bar advances inside
    # each chunk, not only when a whole chunk-future completes.
    ctx = mp.get_context("spawn")
    progress_queue: Any = ctx.Queue()

    def _drain_progress(bar: tqdm[None]) -> None:
        while (item := progress_queue.get()) is not None:
            bar.update(item)

    with (
        tqdm(
            total=total_input_rows,
            unit="row",
            unit_scale=True,
            desc="Map matching",
        ) as progress,
        ProcessPoolExecutor(
            max_workers=max_workers,
            # spawn, not the Linux fork default: the main process has
            # polars/tokio/jemalloc background threads whose mutexes get
            # inherited-locked-but-ownerless across fork, deadlocking
            # every child inside its first polars call.  Spawn starts
            # each worker from a fresh Python process with no inherited
            # thread state.  Re-imports cost a few seconds per worker at
            # startup, parallelised across the pool.
            mp_context=ctx,
            initializer=_worker_init,
            initargs=(quiet, progress_queue),
        ) as pool,
    ):
        drainer = threading.Thread(
            target=_drain_progress, args=(progress,), daemon=True
        )
        drainer.start()

        # Each future is pickled across a mp pipe — pass only the row
        # counts this worker actually needs, not the whole dataset dict.
        # With hundreds of thousands of chunks, the full dict pickles to
        # tens of MB × N workers and stalls the submit queue before any
        # worker starts.
        future_to_rows = {}
        for chunk, rows in chunks_with_rows:
            sub_counts = {
                f: row_counts[f] for pdir in chunk for f in partition_chunks[pdir]
            }
            fut = pool.submit(
                _process_chunk,
                chunk,
                input_dir,
                output_dir,
                config,
                sub_counts,
                debug=debug,
                bridges=bridges,
            )
            future_to_rows[fut] = rows

        partition_results: list[tuple[str, int, int, int, int, float]] = []
        try:
            for future in as_completed(future_to_rows):
                partition_results.extend(future.result())
        finally:
            progress_queue.put(None)
            drainer.join(timeout=5)

    for _tier, _pid, skipped, trips, rows, _elapsed in partition_results:
        total_skipped += skipped
        total_trips += trips
        total_rows += rows

    if partition_results:
        output_dir.mkdir(parents=True, exist_ok=True)
        timing_df = pl.DataFrame(
            partition_results, schema=_TIMING_SCHEMA, orient="row"
        ).sort("elapsed_s", descending=True)
        _atomic_write_parquet(timing_df, output_dir / "_timing.parquet")

    print("\n--- Map matching complete ---", file=sys.stderr)
    print(f"  Partitions:  {n:>10,}", file=sys.stderr)
    print(f"  Chunks:      {total_chunks:>10,}", file=sys.stderr)
    print(f"  Skipped:     {total_skipped:>10,} chunk(s)", file=sys.stderr)
    print(f"  Matched:     {total_trips:>10,} trip(s)", file=sys.stderr)
    print(f"  Rows out:    {total_rows:>10,}", file=sys.stderr)
    print(f"  Timing:      {output_dir / '_timing.parquet'}", file=sys.stderr)
