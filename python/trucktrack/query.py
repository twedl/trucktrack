"""Helpers for pulling individual trucks and trips from pipeline outputs.

Each stage of the pipeline uses a different layout:

- **Raw traces**: ``year=YYYY/chunk_id=XX/part-0.parquet``
  The ``chunk_id`` hive column is the last 2 hex chars of the truck UUID.

- **Partitioned trips**: ``tier=.../partition_id=.../data.parquet``
  The ``id`` column contains composite trip IDs like
  ``{truck_id}_gap{N}_trip{M}``.

- **Map-matched results**: same hive layout as partitioned trips, with
  columns ``id``, ``date``, ``way_id``.

The standalone ``scan_*`` functions scan all files with predicate
pushdown on the ``id`` column.  For repeated queries,
:class:`ChunkIndex` builds a persistent index that maps each chunk_id
to its file paths — reading only the ``id`` column once, then
saving the mapping for instant reloads.

All functions use :func:`polars.scan_parquet` for lazy evaluation so
that predicate pushdown avoids reading the full dataset.
"""

from __future__ import annotations

import functools
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import polars as pl
from tqdm import tqdm

_CHUNK_ID_LEN = 2
_INDEX_FILENAME = ".chunk_index.json"


def _chunk_id(truck_id: str) -> str:
    """Derive the chunk_id (last 2 hex chars) from a truck UUID."""
    return truck_id[-_CHUNK_ID_LEN:]


def truck_id_from_trip(trip_id: str) -> str:
    """Extract the truck UUID from a composite trip ID.

    Trip IDs have the form ``{truck_id}_gap{N}_trip{M}``.
    """
    return trip_id.split("_gap")[0]


def _scan_chunk_glob(
    data_dir: str | Path,
    glob: str,
    filter_expr: pl.Expr,
) -> pl.LazyFrame:
    """Scan parquet files matching *glob* under *data_dir* and apply a filter."""
    return pl.scan_parquet(
        Path(data_dir) / glob,
        hive_partitioning=True,
    ).filter(filter_expr)


# ---------------------------------------------------------------------------
# ChunkIndex — persistent file-path index keyed by chunk_id
# ---------------------------------------------------------------------------


class ChunkIndex:
    """Maps chunk_id to file paths for instant lookups.

    Build once with :meth:`build`, save with :meth:`save`, and reload
    in later sessions with :meth:`load`::

        idx = ChunkIndex.build("data/partitioned")
        idx.save()                       # writes .chunk_index.json
        idx = ChunkIndex.load("data/partitioned")  # instant next time
        df = idx.scan_truck(truck_id).collect()
    """

    def __init__(
        self,
        data_dir: str | Path,
        index: dict[str, list[str]],
    ) -> None:
        self._data_dir = Path(data_dir)
        self._index = index

    @classmethod
    def build(
        cls,
        data_dir: str | Path,
        *,
        max_workers: int | None = None,
        show_progress: bool = True,
    ) -> ChunkIndex:
        """Scan *data_dir* and build the chunk_id → file path mapping.

        Reads only the ``id`` column from each file to extract chunk_ids.
        Files are read in parallel with a thread pool; the Rust Parquet
        reader releases the GIL so threads achieve real parallelism.

        Parameters
        ----------
        max_workers
            Thread pool size. Defaults to ``os.cpu_count()``.
        show_progress
            Display a tqdm progress bar. Defaults to True.
        """
        data_dir = Path(data_dir)
        chunk_id_expr = (
            pl.col("id")
            .str.split("_gap")
            .list.first()
            .str.slice(-_CHUNK_ID_LEN)
            .unique()
        )

        def _extract(p: Path) -> tuple[str, list[str]]:
            rel = str(p.relative_to(data_dir))
            cids = (
                pl.read_parquet(p, columns=["id"]).select(chunk_id_expr)["id"].to_list()
            )
            return rel, cids

        files = sorted(data_dir.rglob("*.parquet"))
        if max_workers is None:
            max_workers = os.cpu_count() or 1
        max_workers = min(max_workers, len(files) or 1)

        index: dict[str, list[str]] = {}
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            iterator = pool.map(_extract, files)
            if show_progress:
                iterator = tqdm(iterator, total=len(files), unit="file", desc="Indexing")
            for rel, cids in iterator:
                for cid in cids:
                    index.setdefault(cid, []).append(rel)

        return cls(data_dir, index)

    def save(self, path: str | Path | None = None) -> Path:
        """Write the index to disk. Defaults to ``data_dir/.chunk_index.json``."""
        out = Path(path) if path is not None else self._data_dir / _INDEX_FILENAME
        out.write_text(json.dumps(self._index, sort_keys=True))
        return out

    @classmethod
    def load(cls, data_dir: str | Path, path: str | Path | None = None) -> ChunkIndex:
        """Load a previously saved index."""
        data_dir = Path(data_dir)
        src = Path(path) if path is not None else data_dir / _INDEX_FILENAME
        index = json.loads(src.read_text())
        return cls(data_dir, index)

    def _files_for_chunk(self, cid: str) -> list[Path]:
        """Resolve relative paths for a chunk_id."""
        return [self._data_dir / rel for rel in self._index.get(cid, [])]

    def scan_truck(self, truck_id: str) -> pl.LazyFrame:
        """Scan files for a truck and filter on composite trip ID prefix."""
        cid = _chunk_id(truck_id)
        files = self._files_for_chunk(cid)
        if not files:
            raise FileNotFoundError(f"No files for chunk_id={cid} in index")
        return pl.scan_parquet(files, hive_partitioning=True).filter(
            pl.col("id").str.starts_with(truck_id)
        )

    def scan_trip(self, trip_id: str) -> pl.LazyFrame:
        """Scan files for a single trip by composite trip ID."""
        cid = _chunk_id(truck_id_from_trip(trip_id))
        files = self._files_for_chunk(cid)
        if not files:
            raise FileNotFoundError(f"No files for chunk_id={cid} in index")
        return pl.scan_parquet(files, hive_partitioning=True).filter(
            pl.col("id") == trip_id
        )

    @functools.cached_property
    def chunk_ids(self) -> list[str]:
        """All chunk_ids in the index."""
        return sorted(self._index)

    def __len__(self) -> int:
        """Number of chunk_ids in the index."""
        return len(self._index)

    def __repr__(self) -> str:
        n_files = sum(len(v) for v in self._index.values())
        return (
            f"ChunkIndex({self._data_dir}, {len(self._index)} chunks, {n_files} files)"
        )


# ---------------------------------------------------------------------------
# Raw traces
# ---------------------------------------------------------------------------


def scan_raw_truck(
    data_dir: str | Path,
    truck_id: str,
) -> pl.LazyFrame:
    """Lazily scan raw traces for a single truck.

    Filters on ``chunk_id`` (hive partition) and ``id`` column to
    avoid reading the full dataset.

    Parameters
    ----------
    data_dir
        Root of the raw hive-partitioned dataset
        (``year=.../chunk_id=.../...parquet``).
    truck_id
        Full truck UUID (hex string).
    """
    cid = _chunk_id(truck_id)
    return _scan_chunk_glob(
        data_dir, f"**/chunk_id={cid}/*.parquet", pl.col("id") == truck_id
    )


# ---------------------------------------------------------------------------
# Partitioned trips (tier/partition_id layout)
# ---------------------------------------------------------------------------


def scan_partitioned_truck(
    data_dir: str | Path,
    truck_id: str,
) -> pl.LazyFrame:
    """Lazily scan partitioned trips for all trips of a truck.

    Scans all parquet files with predicate pushdown on the ``id``
    column.  For repeated queries, prefer :class:`ChunkIndex`.
    """
    return _scan_chunk_glob(
        data_dir, "**/*.parquet", pl.col("id").str.starts_with(truck_id)
    )


def scan_partitioned_trip(
    data_dir: str | Path,
    trip_id: str,
) -> pl.LazyFrame:
    """Lazily scan partitioned trips for a single trip.

    *trip_id* is the full composite ID (e.g.
    ``abc123..._gap0_trip1``).  Scans all parquet files with
    predicate pushdown.  For repeated queries, prefer
    :class:`ChunkIndex`.
    """
    return _scan_chunk_glob(data_dir, "**/*.parquet", pl.col("id") == trip_id)


# ---------------------------------------------------------------------------
# Map-matched results (same hive layout, id/date/way_id columns)
# ---------------------------------------------------------------------------


def scan_matched_truck(
    data_dir: str | Path,
    truck_id: str,
) -> pl.LazyFrame:
    """Lazily scan map-matched results for all trips of a truck.

    For repeated queries, prefer :class:`ChunkIndex`.
    """
    return _scan_chunk_glob(
        data_dir, "**/*.parquet", pl.col("id").str.starts_with(truck_id)
    )


def scan_matched_trip(
    data_dir: str | Path,
    trip_id: str,
) -> pl.LazyFrame:
    """Lazily scan map-matched results for a single trip.

    For repeated queries, prefer :class:`ChunkIndex`.
    """
    return _scan_chunk_glob(data_dir, "**/*.parquet", pl.col("id") == trip_id)
