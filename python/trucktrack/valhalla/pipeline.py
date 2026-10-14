"""Map-match trips from the split pipeline's truck buckets.

Reads the bucketed trips produced by
:func:`trucktrack.pipeline.run_pipeline`, iterates over buckets, and
applies a map-matching function to each trip.

Resumable: the output directory mirrors the input buckets. On
restart, chunks whose requested outputs all exist are skipped. Writes
use atomic temp-file-then-rename to avoid partial output from crashes.

Opt-in edge speed observations (one row per matched GPS fix, see
:mod:`trucktrack.congestion`) come from the same ``trace_attributes``
call and land in ``_observations/`` beside ``_quality/``.

Buckets are processed in parallel using worker processes
(ProcessPoolExecutor), one bucket per task.  Processes (not threads)
because per-trip Python work, however thin, is GIL-serialized across
workers — at 29 workers on a long run, active CPU plateaus around
14/29 with threads no matter how much Python we trim.

Usage::

    from trucktrack.valhalla.pipeline import run_map_matching
    run_map_matching(Path("data/split"), Path("data/matched"))
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

from trucktrack.buckets import (
    bucket_dirs,
    bucket_files,
    bucket_number,
    read_n_buckets,
    require_bucketed,
    write_n_buckets,
)
from trucktrack.congestion import (
    FIX_COLUMNS,
    ObservationConfig,
    edge_observations,
    prepare_fixes,
)
from trucktrack.pipeline import parquet_row_count
from trucktrack.valhalla._bridge import BridgeConfig
from trucktrack.valhalla._build_id import tile_build_id
from trucktrack.valhalla.map_matching import (
    FIX_MATCH_SCHEMA,
    FixMatches,
    null_fix_matches,
)
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
    "bucket": pl.Int64,
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
    ``_report_progress`` can stream per-bucket updates back to the
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
    per_fix: bool = False,
) -> tuple[pl.DataFrame, dict[str, object], FixMatches | None]:
    """Map-match one trip; return (way_ids_df, quality_row_dict, fix_matches).

    The quality row is left as a dict so the caller can batch many rows
    into a single ``pl.DataFrame`` construction — the per-trip list-of-
    one-dict pattern hits polars' ``_sequence_of_dict_to_pydf`` under the
    GIL and shows up as a hotspot in py-spy.

    *per_fix* also returns where each row of the time-sorted trip
    landed (see :func:`~trucktrack.valhalla.parse_fix_matches`), from
    the same ``trace_attributes`` call.  The third element is ``None``
    when it wasn't requested or the trip wasn't matched.
    """
    # Stable: fix matches line up with the rows in this order.
    trip = trip.sort("time", maintain_order=True)
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
            trip_id, trip, bridges=bridges, config=config, per_fix=per_fix
        )
    else:
        points = list(zip(trip["lat"].to_list(), trip["lon"].to_list(), strict=True))
        q = evaluate_map_match_ways(trip_id, points, config=config, per_fix=per_fix)
    elapsed = perf_counter() - t0 if debug else None
    ways = q.way_ids
    if ways:
        way_df = pl.DataFrame(
            {"id": [trip_id] * len(ways), "date": [date] * len(ways), "way_id": ways},
            schema=_WAY_SCHEMA,
        )
    else:
        way_df = _null_way_result(trip_id, date)
    return way_df, _quality_row(trip_id, date, q, elapsed), q.fix_matches


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
    way_df, quality_row, _ = _map_match_trip_row(
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


def _process_bucket(
    bucket_dir: Path,
    output_dir: Path,
    config: str | Path | None,
    row_counts: dict[Path, int],
    *,
    debug: bool = False,
    bridges: BridgeConfig | None = None,
    observations: ObservationConfig | None = None,
    build_id: str | None = None,
) -> tuple[int, int, int, int, float]:
    """Read one bucket chunk-by-chunk, map-match each trip, and write results.

    Processes one chunk at a time to keep memory free for Valhalla.
    Skips chunks whose requested outputs all exist.  *build_id* tags
    observation rows (see :func:`~trucktrack.valhalla.tile_build_id`).
    Returns (bucket, chunks_skipped, trips_matched, rows_out, elapsed_s).
    """
    t0 = perf_counter()
    bucket = bucket_number(bucket_dir)
    out_dir = output_dir / bucket_dir.name
    quality_dir = output_dir / "_quality" / bucket_dir.name
    obs_dir = output_dir / "_observations" / bucket_dir.name
    out_dir.mkdir(parents=True, exist_ok=True)
    quality_dir.mkdir(parents=True, exist_ok=True)
    if observations is not None:
        obs_dir.mkdir(parents=True, exist_ok=True)

    chunks = bucket_files(bucket_dir)
    skipped = 0
    total_trips = 0
    total_rows = 0

    for chunk_path in chunks:
        raw_rows = row_counts[chunk_path]
        out_path = out_dir / chunk_path.name
        quality_path = quality_dir / chunk_path.name
        obs_path = obs_dir / chunk_path.name
        if out_path.exists() and (observations is None or obs_path.exists()):
            skipped += 1
            _report_progress(raw_rows)
            continue

        lf = pl.scan_parquet(chunk_path)
        columns = ["id", "time", "lat", "lon"]
        if observations is not None:
            # Weights and dwell flags need whole trips: before any filter.
            lf = prepare_fixes(lf, observations)
            columns += [c for c in FIX_COLUMNS if c not in columns]
        if "is_stop" in lf.collect_schema():
            lf = lf.filter(~pl.col("is_stop"))
        df = lf.select(columns).collect()
        if df.is_empty():
            _report_progress(raw_rows)
            continue
        way_dfs: list[pl.DataFrame] = []
        quality_rows: list[dict[str, object]] = []
        fix_matches = null_fix_matches(0)
        rows_pending = 0
        # In row order, so fix matches stack back onto df (sorted by
        # trip, then time, when observations are on).
        for _, trip in df.group_by("id", maintain_order=True):
            way_df, quality_row, trip_matches = _map_match_trip_row(
                trip,
                config=config,
                debug=debug,
                bridges=bridges,
                per_fix=observations is not None,
            )
            way_dfs.append(way_df)
            quality_rows.append(quality_row)
            if observations is not None:
                for col, values in (
                    trip_matches or null_fix_matches(len(trip))
                ).items():
                    fix_matches[col].extend(values)
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
        if observations is not None:
            fixes = df.select(FIX_COLUMNS).hstack(
                pl.DataFrame(fix_matches, schema=FIX_MATCH_SCHEMA)
            )
            obs = edge_observations(fixes, observations).with_columns(
                pl.lit(build_id, dtype=pl.Utf8).alias("tile_build_id")
            )
            _atomic_write_parquet(obs, obs_path)

    return bucket, skipped, total_trips, total_rows, perf_counter() - t0


def run_map_matching(
    input_dir: Path,
    output_dir: Path,
    *,
    config: str | Path | None = None,
    max_workers: int | None = None,
    quiet: bool = False,
    debug: bool = False,
    bridges: BridgeConfig | None = None,
    observations: ObservationConfig | None = None,
) -> None:
    """Map-match all trips across buckets in parallel.

    *config* is an optional path to ``valhalla.json``; when ``None``
    :func:`trucktrack.valhalla.find_config` discovers one in cwd
    (e.g. ``./valhalla.json``).

    *input_dir* should be the output of
    :func:`trucktrack.pipeline.run_pipeline`, with layout::

        input_dir/bucket=K/chunk.parquet

    *output_dir* mirrors the same buckets, as do its ``_quality/`` and
    ``_observations/``, each with a ``_buckets.json`` from the start, so
    lookups work mid-run.  Re-running skips chunks whose requested
    outputs all exist.

    *max_workers* defaults to ``os.cpu_count()``.

    *quiet* suppresses Valhalla's C++ log output. Progress (tqdm) still
    prints to stderr. Per-trip errors are recorded in the quality
    parquet rather than logged.

    *debug* adds an ``elapsed_s`` column to each quality row, so you
    can isolate pathological trips within a slow bucket. Per-bucket
    timing is always written to ``output_dir / "_timing.parquet"``.

    *bridges*, when supplied, splits trips at large gaps (see
    :class:`trucktrack.valhalla.BridgeConfig`), matches each segment,
    and bridges gaps via ``/route`` + ``edge_walk``.  Populates
    ``n_bridges``, ``max_detour_ratio``, ``total_bridge_m``,
    ``any_bridge_failed`` in the quality rows.

    *observations*, when supplied, also writes edge speed observations
    (see :mod:`trucktrack.congestion`) to ``output_dir / "_observations"``
    in the same buckets: one row per GPS fix matched to an edge,
    tagged with the tile build ID (see
    :func:`~trucktrack.valhalla.tile_build_id`).  The input needs
    ``speed`` (km/h) and ``heading`` columns.  The extra attributes ride
    on the ``trace_attributes`` call already made for way IDs.

    .. note::
        For best CPU utilization, set ``POLARS_MAX_THREADS=1`` in your
        environment before launching.  Each bucket worker does its
        own ``pl.scan_parquet().collect()`` on a chunk; polars'
        global rayon pool then fans that work out across all
        cores, and with ``max_workers`` bucket workers doing it
        simultaneously, every polars call fights every other one for
        CPU.  Serializing polars means bucket-level parallelism
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

    n_buckets = require_bucketed(input_dir, made_by="run_pipeline")
    out_n = read_n_buckets(output_dir)
    if out_n is not None and out_n != n_buckets:
        raise ValueError(
            f"{output_dir} holds {out_n} buckets, {input_dir} {n_buckets}; "
            "delete the output to redo it"
        )
    # First, so a resumed run is checked against it even after a crash.
    write_n_buckets(output_dir, n_buckets)
    write_n_buckets(output_dir / "_quality", n_buckets)
    if observations is not None:
        write_n_buckets(output_dir / "_observations", n_buckets)

    bucket_chunks = {b: bucket_files(b) for b in bucket_dirs(input_dir)}
    all_chunk_paths = [p for chunks in bucket_chunks.values() for p in chunks]
    total_chunks = len(all_chunk_paths)

    if not bucket_chunks:
        print(f"No buckets found under {input_dir}", file=sys.stderr)
        return

    if max_workers is None:
        max_workers = os.cpu_count() or 1
    max_workers = min(max_workers, len(bucket_chunks))

    build_id = tile_build_id(config) if observations is not None else None

    if max_workers > 4 and "POLARS_MAX_THREADS" not in os.environ:
        print(
            f"warning: POLARS_MAX_THREADS not set with max_workers={max_workers}. "
            "Polars' global rayon pool will oversubscribe CPU as each bucket "
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

    n = len(bucket_chunks)
    print(
        f"Found {n} bucket(s) "
        f"({total_chunks:,} chunks, {total_input_rows:,} rows), "
        f"processing with {max_workers} worker(s)",
        file=sys.stderr,
    )

    total_skipped = 0
    total_trips = 0
    total_rows = 0

    # Longest-Processing-Time-first: submit the heaviest buckets
    # earliest so the longest tasks start when the pool is full and
    # shorter tasks fill idle workers at the end.
    buckets_by_rows = sorted(
        bucket_chunks,
        key=lambda b: sum(row_counts[p] for p in bucket_chunks[b]),
        reverse=True,
    )

    # Shared queue — workers stream per-batch row counts; a main-process
    # drainer thread forwards them to tqdm so the bar advances inside
    # each bucket, not only when a whole bucket-future completes.
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
        futures = [
            pool.submit(
                _process_bucket,
                bucket,
                output_dir,
                config,
                {f: row_counts[f] for f in bucket_chunks[bucket]},
                debug=debug,
                bridges=bridges,
                observations=observations,
                build_id=build_id,
            )
            for bucket in buckets_by_rows
        ]

        bucket_results: list[tuple[int, int, int, int, float]] = []
        try:
            for future in as_completed(futures):
                bucket_results.append(future.result())
        finally:
            progress_queue.put(None)
            drainer.join(timeout=5)

    for _bucket, skipped, trips, rows, _elapsed in bucket_results:
        total_skipped += skipped
        total_trips += trips
        total_rows += rows

    timing_df = pl.DataFrame(bucket_results, schema=_TIMING_SCHEMA, orient="row").sort(
        "elapsed_s", descending=True
    )
    _atomic_write_parquet(timing_df, output_dir / "_timing.parquet")

    print("\n--- Map matching complete ---", file=sys.stderr)
    print(f"  Buckets:     {n:>10,}", file=sys.stderr)
    print(f"  Chunks:      {total_chunks:>10,}", file=sys.stderr)
    print(f"  Skipped:     {total_skipped:>10,} chunk(s)", file=sys.stderr)
    print(f"  Matched:     {total_trips:>10,} trip(s)", file=sys.stderr)
    print(f"  Rows out:    {total_rows:>10,}", file=sys.stderr)
    print(f"  Timing:      {output_dir / '_timing.parquet'}", file=sys.stderr)
    if observations is not None:
        print(
            f"  Observations: {output_dir / '_observations'} (tiles {build_id})",
            file=sys.stderr,
        )
