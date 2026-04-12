"""Map-match trips from hive-partitioned output.

Reads the partitioned parquet dataset produced by
``pipeline_hive_to_partitioned``, iterates over partitions, and
applies a map-matching function to each trip.

Resumable: the output directory mirrors the input hive layout. On
restart, chunks whose output file already exists are skipped. Writes
use atomic temp-file-then-rename to avoid partial output from crashes.

Partitions are processed in parallel using threads, assigned in
contiguous spatial blocks to keep Valhalla's tile cache warm.

Usage::

    from examples.map_match_partitions import run_map_matching
    run_map_matching(Path("data/processed"), Path("data/matched"))
"""

from __future__ import annotations

import os
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import polars as pl
from tqdm import tqdm

from trucktrack.valhalla import map_match_ways

TILE_EXTRACT = os.environ.get("VALHALLA_TILE_EXTRACT", "valhalla_tiles.tar")

_WAY_SCHEMA = {"id": pl.Utf8, "date": pl.Date, "way_id": pl.Int64}


def map_match_trip(trip: pl.DataFrame) -> pl.DataFrame:
    """Map-match a single trip and return id, date, way_id rows.

    If matching fails (e.g. corrupted GPS data exceeding Valhalla's
    distance limit), returns a single row with null way_id so no
    trips are dropped.
    """
    trip_id = trip["id"][0]
    date = trip["time"].min().date()
    try:
        points = list(zip(trip["lat"].to_list(), trip["lon"].to_list(), strict=True))
        ways = map_match_ways(points, tile_extract=TILE_EXTRACT)
    except Exception as e:
        print(f"  [WARN] {trip_id[:24]}: {e}")
        return pl.DataFrame(
            {"id": [trip_id], "date": [date], "way_id": [None]},
            schema=_WAY_SCHEMA,
        )
    return pl.DataFrame(
        {"id": [trip_id] * len(ways), "date": [date] * len(ways), "way_id": ways},
        schema=_WAY_SCHEMA,
    )


def _process_partition(
    partition_dir: Path,
    input_dir: Path,
    output_dir: Path,
    progress: tqdm[None] | None = None,
) -> tuple[str, int, int, int]:
    """Read one partition chunk-by-chunk, map-match each trip, and write results.

    Processes one chunk at a time to keep memory free for Valhalla.
    Skips chunks whose output file already exists.
    Returns (partition_key, chunks_skipped, trips_matched, rows_out).
    """
    rel = partition_dir.relative_to(input_dir)
    out_dir = output_dir / rel

    chunks = sorted(partition_dir.glob("*.parquet"))
    skipped = 0
    total_trips = 0
    total_rows = 0

    for chunk_path in chunks:
        out_path = out_dir / chunk_path.name
        if out_path.exists():
            skipped += 1
            if progress is not None:
                progress.update(1)
            continue

        df = pl.read_parquet(chunk_path)
        matched_trips = []
        for _, trip in df.group_by("id"):
            matched_trips.append(map_match_trip(trip))
            total_trips += 1

        if not matched_trips:
            if progress is not None:
                progress.update(1)
            continue

        matched = pl.concat(matched_trips)
        total_rows += len(matched)

        out_dir.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=out_dir, suffix=".tmp.parquet")
        os.close(fd)
        try:
            matched.write_parquet(tmp)
            os.rename(tmp, out_path)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise

        if progress is not None:
            progress.update(1)

    return str(rel), skipped, total_trips, total_rows


def _process_block(
    partition_dirs: list[Path],
    input_dir: Path,
    output_dir: Path,
    progress: tqdm[None] | None = None,
) -> list[tuple[str, int, int, int]]:
    """Process a contiguous block of spatially adjacent partitions sequentially."""
    return [
        _process_partition(pdir, input_dir, output_dir, progress)
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
    max_workers: int | None = None,
) -> None:
    """Map-match all trips across partitions in parallel.

    *input_dir* should be the hive-partitioned output of
    ``pipeline_hive_to_partitioned.run_pipeline``, with layout::

        input_dir/tier=.../partition_id=.../chunk.parquet

    *output_dir* mirrors the same hive layout. Re-running skips chunks
    whose output file already exists.

    *max_workers* defaults to ``os.cpu_count()``.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    partition_dirs = sorted(
        {
            p.parent
            for p in input_dir.rglob("*.parquet")
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

    total_chunks = sum(
        len(list(pdir.glob("*.parquet"))) for pdir in partition_dirs
    )

    size_str = f"{block_size}–{block_size + 1}" if remainder else str(block_size)
    print(
        f"Found {n} partition(s) ({total_chunks} chunks), "
        f"processing with {max_workers} worker(s) "
        f"({size_str} partitions per worker)"
    )

    total_skipped = 0
    total_trips = 0
    total_rows = 0

    with tqdm(total=total_chunks, unit="chunk", desc="Map matching") as progress:
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = [
                pool.submit(_process_block, block, input_dir, output_dir, progress)
                for block in blocks
            ]

            for future in as_completed(futures):
                for key, skipped, trips, rows in future.result():
                    total_skipped += skipped
                    total_trips += trips
                    total_rows += rows

    print("\n--- Map matching complete ---")
    print(f"  Partitions:  {n:>10,}")
    print(f"  Chunks:      {total_chunks:>10,}")
    print(f"  Skipped:     {total_skipped:>10,} chunk(s)")
    print(f"  Matched:     {total_trips:>10,} trip(s)")
    print(f"  Rows out:    {total_rows:>10,}")
