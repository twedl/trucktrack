"""Pull one truck's or one trip's rows from any stage of the pipeline.

Every stage writes ``bucket=K/`` folders, K a stable hash of the truck
id, with a ``_buckets.json`` recording how many (see
:mod:`trucktrack.buckets`), so a lookup hashes the id and opens one
folder:

- **Bucketed points** (:func:`~trucktrack.buckets.bucket_by_truck`):
  the raw columns, ``id`` the truck id.
- **Split trips** (:func:`~trucktrack.pipeline.run_pipeline`): ``id``
  the composite trip ID ``{truck_id}_gap{N}_trip{M}``.
- **Map-matched results**
  (:func:`~trucktrack.valhalla.pipeline.run_map_matching`): ``id``,
  ``date``, ``way_id``; its ``_quality/`` and ``_observations/`` are
  bucketed the same way.

A folder without ``_buckets.json`` — the raw input, in whatever layout
it arrives — is scanned in full (:func:`~trucktrack.buckets.data_files`).

All functions return a :class:`polars.LazyFrame`, so further filters
and column selections push down into the scan.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from trucktrack.buckets import (
    bucket_dir,
    bucket_files,
    bucket_of,
    data_files,
    read_n_buckets,
)


def truck_id_from_trip(trip_id: str) -> str:
    """Extract the truck UUID from a composite trip ID.

    Trip IDs have the form ``{truck_id}_gap{N}_trip{M}``.
    """
    return trip_id.split("_gap")[0]


def truck_id_expr(col: str = "id") -> pl.Expr:
    """:func:`truck_id_from_trip` as an expression over a trip-id column."""
    return pl.col(col).str.split("_gap").list.first()


def _truck_files(data_dir: str | Path, truck_ids: set[str]) -> list[Path]:
    """Files that can hold these trucks' rows: their buckets', or all of them."""
    data_dir = Path(data_dir)
    n_buckets = read_n_buckets(data_dir)
    if n_buckets is None:
        files = data_files(data_dir)
    else:
        buckets = sorted({bucket_of(t, n_buckets) for t in truck_ids})
        files = [f for b in buckets for f in bucket_files(bucket_dir(data_dir, b))]
    if not files:
        raise FileNotFoundError(f"no files for {sorted(truck_ids)} under {data_dir}")
    return files


def scan_truck(data_dir: str | Path, truck_id: str) -> pl.LazyFrame:
    """Lazily scan every row of one truck, from any stage's *data_dir*.

    Matches ``id`` equal to *truck_id* (raw and bucketed points) or a
    trip of it (split and matched stages).
    """
    files = _truck_files(data_dir, {truck_id})
    return pl.scan_parquet(files, hive_partitioning=False).filter(
        truck_id_expr() == truck_id
    )


def scan_trip(data_dir: str | Path, trip_id: str | list[str]) -> pl.LazyFrame:
    """Lazily scan one or more trips, by composite trip ID, in one scan.

    From a split or matched stage.
    """
    trip_ids = [trip_id] if isinstance(trip_id, str) else trip_id
    files = _truck_files(data_dir, {truck_id_from_trip(t) for t in trip_ids})
    return pl.scan_parquet(files, hive_partitioning=False).filter(
        pl.col("id").is_in(trip_ids)
    )
