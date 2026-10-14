"""Group each truck's points into one bucket, whatever the input layout.

Splitting a truck's trace into trips needs all of its points together,
and the raw input promises nothing about how points are spread across
files.  :func:`bucket_by_truck` streams every input file into
``bucket=K/`` folders, K a stable hash of the truck id (see
:func:`bucket_of`), so one bucket holds every point of its trucks.

Later stages keep the same buckets
(:func:`~trucktrack.pipeline.run_pipeline`,
:func:`~trucktrack.valhalla.pipeline.run_map_matching`), and a
``_buckets.json`` beside them records how many there are, so a truck's
rows at any stage sit in one folder: :func:`~trucktrack.query.scan_truck`
hashes the id and opens only that one.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import polars as pl

from trucktrack import _core

# As many as the raw chunks (~270 MB each) the pipeline was sized on.
DEFAULT_N_BUCKETS = 256

_META = "_buckets.json"


def bucket_of(truck_id: str, n_buckets: int) -> int:
    """Bucket, in ``0..n_buckets``, that holds *truck_id*'s rows."""
    return _core.bucket_of(truck_id, n_buckets)


def bucket_expr(n_buckets: int) -> pl.Expr:
    """:func:`bucket_of` over the ``id`` column, as ``bucket``."""
    return (
        pl.col("id")
        .map_batches(
            lambda ids: _core.bucket_ids(ids, n_buckets),
            return_dtype=pl.UInt32,
            is_elementwise=True,
        )
        .alias("bucket")
    )


def bucket_dir(root: str | Path, bucket: int) -> Path:
    """Folder of *bucket* under a stage's *root*."""
    return Path(root) / f"bucket={bucket}"


def bucket_number(path: Path) -> int:
    """K of a ``bucket=K`` folder."""
    return int(path.name.removeprefix("bucket="))


def bucket_dirs(root: str | Path) -> list[Path]:
    """Every bucket folder under *root*, in bucket order."""
    return sorted(Path(root).glob("bucket=*"), key=bucket_number)


def bucket_files(bucket: Path) -> list[Path]:
    """The parquet files holding one bucket's rows."""
    return sorted(bucket.glob("*.parquet"))


def data_files(root: str | Path) -> list[Path]:
    """Parquet files anywhere under *root*, in any folder layout.

    Skips paths through ``_``- or ``.``-prefixed names: side outputs
    and scratch (``_quality/``, ``_timing.parquet``, Spark's
    ``_temporary/``), not rows.
    """
    root = Path(root)
    return sorted(
        p
        for p in root.rglob("*.parquet")
        if not any(part[0] in "_." for part in p.relative_to(root).parts)
    )


def write_n_buckets(root: str | Path, n_buckets: int) -> None:
    """Record that *root* is laid out in *n_buckets* buckets."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    payload = {"version": _core.BUCKET_HASH_VERSION, "n_buckets": n_buckets}
    tmp = root / f".{_META}.tmp"
    tmp.write_text(json.dumps(payload) + "\n")
    os.replace(tmp, root / _META)


def read_n_buckets(root: str | Path) -> int | None:
    """Number of buckets *root* is laid out in, or ``None`` if it isn't."""
    path = Path(root) / _META
    if not path.exists():
        return None
    payload = json.loads(path.read_text())
    if payload.get("version") != _core.BUCKET_HASH_VERSION:
        raise ValueError(
            f"{path} was written with bucket hash version "
            f"{payload.get('version')}, this trucktrack uses "
            f"{_core.BUCKET_HASH_VERSION}; re-run bucket_by_truck"
        )
    return int(payload["n_buckets"])


def require_bucketed(root: str | Path, *, made_by: str) -> int:
    """*root*'s number of buckets; raises if it isn't bucketed."""
    n_buckets = read_n_buckets(root)
    if n_buckets is None:
        raise ValueError(
            f"{root} is not bucketed by truck (no {_META}); run {made_by} first"
        )
    return n_buckets


def require_empty(root: str | Path) -> None:
    """Raise unless *root* is missing or empty, so runs never mix."""
    root = Path(root)
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"{root} is not empty; delete it to redo this step")


def bucket_by_truck(
    input_dir: str | Path,
    output_dir: str | Path,
    *,
    n_buckets: int = DEFAULT_N_BUCKETS,
) -> int:
    """Stream every parquet file under *input_dir* into truck buckets.

    Reads :func:`data_files` in any folder layout; folder names are not
    read as columns.  Rows with a null ``id`` are dropped.  Writes
    ``output_dir/bucket=K/*.parquet`` and ``_buckets.json``, streaming,
    so the input never has to fit in memory.  Each bucket does, later:
    :func:`~trucktrack.pipeline.run_pipeline` loads one at a time, so
    pick *n_buckets* for the data's size.

    *output_dir* must be empty or missing.  Returns the number of
    buckets written (empty buckets have no folder).
    """
    output_dir = Path(output_dir)
    files = data_files(input_dir)
    if not files:
        raise FileNotFoundError(f"no parquet files under {input_dir}")
    require_empty(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (
        pl.scan_parquet(files, hive_partitioning=False)
        .filter(pl.col("id").is_not_null())
        .with_columns(bucket_expr(n_buckets))
        .sink_parquet(
            pl.PartitionBy(output_dir, key="bucket", include_key=False), mkdir=True
        )
    )
    write_n_buckets(output_dir, n_buckets)
    return len(bucket_dirs(output_dir))
