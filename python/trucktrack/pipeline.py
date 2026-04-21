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
import shutil
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
IMPOSSIBLE_SPEED_KMH = 200.0  # km/h; drop GPS points implying travel above this
MIN_SEGMENT_LENGTH = 2
MAX_PARTITION_BYTES = 1_000_000_000  # 1 GB
TARGET_PARTITION_ROWS = 500_000
SPLIT_THRESHOLD = 1.5


def parquet_row_count(path: Path) -> int:
    """Row count from parquet footer — no row-group read."""
    return int(pl.scan_parquet(path).select(pl.len()).collect().item())


def partition_sort_key(path: Path) -> tuple[str, int]:
    """Extract (tier, partition_id) from a ``tier=.../partition_id=...`` dir.

    Sort order is Hilbert-like because partition_id encodes the tier in
    its high bits and the tile's Hilbert key (or, post-rebalance, a
    sequential bucket assigned in Hilbert order) in the low bits.
    """
    tier = path.parent.name.removeprefix("tier=")
    pid = int(path.name.removeprefix("partition_id="))
    return (tier, pid)


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
    impossible_speed_kmh: float,
    min_segment_length: int,
) -> pl.DataFrame:
    """Run pre-process filters, segmentation, and partitioning on one chunk.

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
    impossible_speed_kmh: float,
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
        impossible_speed_kmh=impossible_speed_kmh,
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


def compact_partitions(
    data_dir: str | Path,
    *,
    max_partition_bytes: int = MAX_PARTITION_BYTES,
) -> int:
    """Merge chunk files within each partition into a single file.

    Partitions that already contain a single file are skipped.
    Partitions whose total size exceeds *max_partition_bytes*
    (default 1 GB) are skipped to avoid OOM during the sort.
    Returns the number of partitions compacted.

    Safe: writes to a temp file, renames to ``data.parquet``
    (atomic on the same filesystem), then deletes originals.

    In-place merge only — does not rebalance sizes across partitions.
    Prefer :func:`rebalance_partitions` when the goal is even work
    assignment for downstream per-partition processing.
    """
    data_dir = Path(data_dir)

    # Single-pass: group files by partition directory.
    files_by_dir: dict[Path, list[Path]] = {}
    for p in sorted(data_dir.rglob("*.parquet")):
        files_by_dir.setdefault(p.parent, []).append(p)

    compacted = 0
    skipped = 0
    for pdir, files in sorted(files_by_dir.items()):
        if len(files) <= 1:
            continue
        total_bytes = sum(f.stat().st_size for f in files)
        if total_bytes > max_partition_bytes:
            skipped += 1
            continue
        tmp = pdir / "_compacted.tmp"
        pl.scan_parquet(files).sort("hilbert_idx").sink_parquet(tmp)
        # Rename first so data.parquet exists before deleting originals.
        tmp.rename(pdir / "data.parquet")
        for f in files:
            f.unlink(missing_ok=True)
        compacted += 1
    if skipped:
        print(
            f"  Skipped {skipped} partition(s) exceeding "
            f"{max_partition_bytes / 1e9:.0f} GB"
        )
    return compacted


def _plan_buckets(
    entries: list[tuple[int, list[Path], int]],
    target_rows: int,
    split_threshold: float,
) -> list[tuple[list[Path], int | None]]:
    """Greedy-pack Hilbert-ordered partitions into ~target_rows buckets.

    Each ``entries`` item is ``(partition_id, files, rows)`` in Hilbert
    order.  Returns one plan entry per output partition:

    - ``(files, None)`` = concatenate + sort these files as-is (a
      normal or merged bucket).
    - ``(files, n_slices)`` = the files belong to a single oversized
      source partition that should be split into *n_slices* row-window
      peers after sorting.

    Oversized partitions always flush the current bucket first, then
    stand alone.  They are never packed with neighbours, so the split
    is self-contained.
    """
    plan: list[tuple[list[Path], int | None]] = []
    cur_files: list[Path] = []
    cur_rows = 0

    for _, files, rows in entries:
        if rows > target_rows * split_threshold:
            if cur_files:
                plan.append((cur_files, None))
                cur_files, cur_rows = [], 0
            n_slices = max(2, (rows + target_rows - 1) // target_rows)
            plan.append((files, n_slices))
            continue
        if cur_files and cur_rows + rows > target_rows:
            plan.append((cur_files, None))
            cur_files, cur_rows = [], 0
        cur_files.extend(files)
        cur_rows += rows

    if cur_files:
        plan.append((cur_files, None))
    return plan


def _write_bucket(
    files: list[Path],
    n_slices: int | None,
    staging: Path,
    start_idx: int,
) -> int:
    """Write one plan entry; return the number of output partitions."""
    lf = pl.scan_parquet(files).sort("hilbert_idx")
    if n_slices is None:
        new_dir = staging / f"partition_id={start_idx:06d}"
        new_dir.mkdir()
        lf.sink_parquet(new_dir / "data.parquet")
        return 1

    # Oversized split: materialize once, slice N ways.  sort + sink
    # can't stream anyway, so collect-once beats N repeated sorts.
    df = lf.collect()
    n = len(df)
    slice_rows = (n + n_slices - 1) // n_slices
    for i in range(n_slices):
        new_dir = staging / f"partition_id={start_idx + i:06d}"
        new_dir.mkdir()
        df.slice(i * slice_rows, slice_rows).write_parquet(new_dir / "data.parquet")
    return n_slices


def rebalance_partitions(
    data_dir: str | Path,
    *,
    target_rows: int = TARGET_PARTITION_ROWS,
    split_threshold: float = SPLIT_THRESHOLD,
    max_workers: int | None = None,
) -> tuple[int, int]:
    """Rewrite partitions so each holds roughly *target_rows* rows.

    For each tier under *data_dir* (``tier=local``, ``tier=regional``,
    ``tier=longhaul``):

    1. Enumerate ``partition_id=*`` dirs in Hilbert order (the
       partition_id encoding is tile-Hilbert in its low bits).
    2. Greedy-pack the sequence into buckets whose total row count is
       ≤ *target_rows*.
    3. A single partition whose rows exceed
       ``target_rows * split_threshold`` is split into row-window peers
       — one Hilbert-sorted ``data.parquet`` per slice.
    4. Atomically swap the tier's contents: old partitions move into
       ``tier=.../_rebalance.old/``, new partitions move in, then the
       ``.old`` tree is deleted.

    Downtown cells coalesce into many peers, rural cells merge into a
    super-partition.  The ``partition_id`` number after rebalance is an
    opaque sequential index per tier, zero-padded to 6 digits.

    Crash recovery: a mid-run crash leaves ``_rebalance.new`` or
    ``_rebalance.old`` in the tier dir.  ``_rebalance.new`` alone is
    safe to delete (data still lives in the old partitions).  If both
    exist, the swap was mid-way and the tier needs manual attention.

    Returns ``(n_partitions_out, n_rows_total)``.
    """
    data_dir = Path(data_dir)
    max_workers = max_workers or os.cpu_count() or 1

    total_out = 0
    total_rows = 0

    for tier_dir in sorted(data_dir.glob("tier=*")):
        staging = tier_dir / "_rebalance.new"
        retired = tier_dir / "_rebalance.old"
        if retired.exists():
            raise RuntimeError(
                f"{retired} exists — previous rebalance crashed mid-swap; "
                "inspect before retrying"
            )
        if staging.exists():
            # Incomplete previous run: old data is still in place, drop staging.
            shutil.rmtree(staging)

        parts = sorted(tier_dir.glob("partition_id=*"))
        if not parts:
            continue

        all_files = [f for pdir in parts for f in sorted(pdir.glob("*.parquet"))]
        if not all_files:
            continue

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            counts = list(pool.map(parquet_row_count, all_files))
        row_counts = dict(zip(all_files, counts, strict=True))

        entries: list[tuple[int, list[Path], int]] = []
        for pdir in parts:
            files = sorted(pdir.glob("*.parquet"))
            if not files:
                continue
            pid = int(pdir.name.removeprefix("partition_id="))
            entries.append((pid, files, sum(row_counts[f] for f in files)))
        entries.sort(key=lambda e: e[0])

        tier_rows = sum(r for _, _, r in entries)
        plan = _plan_buckets(entries, target_rows, split_threshold)

        staging.mkdir()
        new_idx = 0
        for files, n_slices in plan:
            new_idx += _write_bucket(files, n_slices, staging, new_idx)

        # Swap: move all old partition dirs into _rebalance.old (single
        # rename each — atomic within a filesystem), then move staging
        # contents into tier_dir, then drop .old.
        retired.mkdir()
        for pdir in parts:
            pdir.rename(retired / pdir.name)
        for new_dir in sorted(staging.glob("partition_id=*")):
            new_dir.rename(tier_dir / new_dir.name)
        staging.rmdir()
        shutil.rmtree(retired)

        total_out += new_idx
        total_rows += tier_rows

    return total_out, total_rows


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
    max_workers: int | None = None,
    group_size: int = 1,
    compact: bool = False,
    max_partition_bytes: int = MAX_PARTITION_BYTES,
    rebalance: bool = False,
    target_rows: int = TARGET_PARTITION_ROWS,
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
    max_partition_bytes
        Partitions whose total file size exceeds this limit are skipped
        during compaction to avoid OOM.  Default 1 GB.
    rebalance
        If ``True``, run :func:`rebalance_partitions` after processing
        to repack partitions to ~*target_rows* each (merging small
        partitions, splitting oversized ones).  Prefer this over
        *compact* when downstream stages work per-partition.
    target_rows
        Target rows per partition when *rebalance* is ``True``.

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
                impossible_speed_kmh,
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

    if rebalance:
        print(f"\nRebalancing partitions to ~{target_rows:,} rows each...")
        n_out, n_rows = rebalance_partitions(output_dir, target_rows=target_rows)
        print(f"  Wrote {n_out} partition(s), {n_rows:,} row(s)")
    elif compact:
        print("\nCompacting partitions...")
        n_compacted = compact_partitions(
            output_dir, max_partition_bytes=max_partition_bytes
        )
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
