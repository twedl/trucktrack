"""Aggregate map-matched way_ids and join OSM geometry.

Counts trips per way across the matched dataset and joins against an
OSM parquet (e.g. a quackosm export) to attach geometry.

Only ``feature_id`` and ``geometry`` are read from the OSM parquet, so
unfiltered exports with a MAP-typed ``tags`` column work without help
(DuckDB prunes unused columns at the parquet reader).

The work is split across two ``COPY ... TO`` queries:

    Stage 1: matched/**/*.parquet  -> counts (way_id, trip_count)
    Stage 2: counts + OSM-by-way_id -> output with geometry, sorted

Stage 2 pushes a semi-join filter into the OSM scan on ``feature_id``
(no CAST, so DuckDB can use parquet row-group / Bloom pruning) — only
ways that actually appear in matched have their geometry decoded. Each
stage gets the full memory budget instead of contending with the other.

Spills to ``<output_parent>/.duckdb_tmp/`` when memory pressure demands.

Requires (not in pyproject.toml)::

    pip install duckdb

Usage::

    uv run python examples/way_id_counts.py \\
        data/matched \\
        /path/to/north-america-latest.osm.parquet \\
        data/way_counts.parquet

    uv run python examples/way_id_counts.py \\
        "/home/jovyan/bmp-datavol-1/atri-match" \\
        "/home/jovyan/bmp-datavol-1/north-america-latest.osm.parquet" \\
        temp/way_counts.parquet

Knobs (env vars)::

    DUCKDB_MEMORY_LIMIT   default ``64GB``
    DUCKDB_THREADS        cap thread count (default: DuckDB auto)
"""

from __future__ import annotations

import os
import sys
import time
from glob import glob
from pathlib import Path

import duckdb

MEMORY_LIMIT = os.environ.get("DUCKDB_MEMORY_LIMIT", "64GB")
THREADS = int(os.environ.get("DUCKDB_THREADS", "0"))  # 0 = auto


def matched_files(matched_dir: Path) -> list[str]:
    files = [
        f
        for f in glob(str(matched_dir / "**/*.parquet"), recursive=True)
        if "_quality" not in f
    ]
    if not files:
        raise SystemExit(f"No matched parquet files under {matched_dir}")
    return files


def _sql_literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _connect(tmp_dir: Path) -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    # COPY targets, temp_directory, and read_parquet filenames must be
    # SQL literals at bind time, hence ``_sql_literal``.
    con.execute(f"PRAGMA memory_limit = '{MEMORY_LIMIT}'")
    con.execute(f"PRAGMA temp_directory = {_sql_literal(str(tmp_dir))}")
    con.execute("PRAGMA preserve_insertion_order = false")
    if THREADS > 0:
        con.execute(f"PRAGMA threads = {THREADS}")
    return con


def _stage_counts(
    con: duckdb.DuckDBPyConnection, files: list[str], counts_path: Path
) -> None:
    matched_literal = "[" + ", ".join(_sql_literal(f) for f in files) + "]"
    con.execute(
        f"""
        COPY (
            SELECT
                way_id,
                COUNT(*) AS trip_count
            FROM read_parquet(
                {matched_literal},
                hive_partitioning = true,
                union_by_name     = true
            )
            WHERE way_id IS NOT NULL
            GROUP BY way_id
        ) TO {_sql_literal(str(counts_path))} (FORMAT PARQUET, COMPRESSION ZSTD)
        """
    )


def _stage_join(
    con: duckdb.DuckDBPyConnection,
    counts_path: Path,
    osm_parquet: Path,
    output_path: Path,
) -> None:
    con.execute(
        f"""
        COPY (
            WITH counts AS (
                SELECT * FROM read_parquet({_sql_literal(str(counts_path))})
            ),
            -- Match OSM by the original 'way/<id>' string so the parquet
            -- scan can prune row groups by feature_id min/max stats and
            -- avoid decoding geometry for ways we never touched.
            wanted AS (
                SELECT 'way/' || CAST(way_id AS VARCHAR) AS feature_id FROM counts
            ),
            osm AS (
                SELECT
                    CAST(substr(o.feature_id, 5) AS BIGINT) AS way_id,
                    o.geometry
                FROM read_parquet({_sql_literal(str(osm_parquet))}) o
                WHERE o.feature_id IN (SELECT feature_id FROM wanted)
            )
            SELECT c.way_id, c.trip_count, o.geometry
            FROM counts c
            LEFT JOIN osm o USING (way_id)
            ORDER BY c.trip_count DESC
        ) TO {_sql_literal(str(output_path))} (FORMAT PARQUET, COMPRESSION ZSTD)
        """
    )


def run(matched_dir: Path, osm_parquet: Path, output_path: Path) -> None:
    files = matched_files(matched_dir)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_dir = output_path.parent / ".duckdb_tmp"
    tmp_dir.mkdir(exist_ok=True)
    counts_path = output_path.with_suffix(".counts.parquet")

    con = _connect(tmp_dir)

    t0 = time.perf_counter()
    print(f"[1/2] aggregating counts over {len(files):,} matched files ...")
    _stage_counts(con, files, counts_path)
    n_ways = con.execute(
        f"SELECT COUNT(*) FROM read_parquet({_sql_literal(str(counts_path))})"
    ).fetchone()[0]
    print(f"      {n_ways:,} ways in {time.perf_counter() - t0:.1f}s")

    t1 = time.perf_counter()
    print("[2/2] joining OSM geometry + sorting ...")
    _stage_join(con, counts_path, osm_parquet, output_path)
    print(f"      done in {time.perf_counter() - t1:.1f}s")

    counts_path.unlink(missing_ok=True)

    total, with_geom = con.execute(
        f"SELECT COUNT(*), COUNT(geometry) "
        f"FROM read_parquet({_sql_literal(str(output_path))})"
    ).fetchone()
    size_mb = output_path.stat().st_size / 1e6
    print(f"{total:,} ways traversed, {with_geom:,} with OSM geometry")
    print(f"Wrote {output_path} ({size_mb:.1f} MB)")


def main() -> None:
    if len(sys.argv) != 4:
        print(
            f"Usage: {sys.argv[0]} <matched_dir> <osm_parquet> <output_parquet>",
            file=sys.stderr,
        )
        sys.exit(1)
    run(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]))


if __name__ == "__main__":
    main()
