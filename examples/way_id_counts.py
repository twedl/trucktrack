"""Aggregate map-matched way_ids, join OSM geometry, build a bbox index.

Counts trips per way across the matched dataset, joins against an OSM
parquet (e.g. a quackosm export) to attach geometry, and adds bbox /
centroid columns so downstream zoom queries can prune without decoding
geometry.

Only ``feature_id`` and ``geometry`` are read from the OSM parquet, so
unfiltered exports with a MAP-typed ``tags`` column work without help
(DuckDB prunes unused columns at the parquet reader).

The work is split across three cached ``COPY ... TO`` stages so re-runs
(e.g. iterating on a zoom-in plot) skip work that's already done:

    Stage 1: matched/**/*.parquet     -> <cache>/counts.parquet
    Stage 2: counts + OSM by way_id   -> <cache>/way_counts_geom.parquet
    Stage 3: + bbox / centroid + sort -> <output>

Each stage skips itself if its output file already exists. Use
``--force`` to rebuild everything or ``--force-stage N`` to rebuild
stage N and every stage after it. The cache does not auto-invalidate
when ``matched/`` changes — pass ``--force`` (or rm the cache dir)
after re-matching.

Stage 2 pushes a semi-join filter into the OSM scan via ``feature_id``
(no CAST, so DuckDB can prune row groups by min/max stats and Bloom
filters) — geometry is decoded only for ways the matched data touched.

Stage 3 uses the DuckDB ``spatial`` extension; it installs on first run
(needs network) and is cached locally afterwards.

Spills to ``<cache>/.duckdb_tmp/`` when memory pressure demands.

Requires (not in pyproject.toml)::

    pip install duckdb

Usage::

    uv run python examples/way_id_counts.py \\
        data/matched \\
        /path/to/north-america-latest.osm.parquet \\
        data/way_counts.parquet

    # Re-render with a different bbox-aware plot? Stages 1-2 stay cached:
    uv run python examples/way_id_counts.py \\
        data/matched osm.parquet temp/way_counts.parquet --force-stage 3

Knobs (env vars)::

    DUCKDB_MEMORY_LIMIT   default ``64GB``
    DUCKDB_THREADS        cap thread count (default: DuckDB auto)
"""

from __future__ import annotations

import argparse
import os
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
    joined_path: Path,
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
        ) TO {_sql_literal(str(joined_path))} (FORMAT PARQUET, COMPRESSION ZSTD)
        """
    )


def _stage_index(
    con: duckdb.DuckDBPyConnection, joined_path: Path, output_path: Path
) -> None:
    con.execute("INSTALL spatial")
    con.execute("LOAD spatial")
    con.execute(
        f"""
        COPY (
            WITH src AS (
                SELECT
                    way_id,
                    trip_count,
                    geometry,
                    ST_GeomFromWKB(geometry) AS geom
                FROM read_parquet({_sql_literal(str(joined_path))})
            )
            SELECT
                way_id,
                trip_count,
                geometry,
                ST_XMin(geom)           AS bbox_xmin,
                ST_YMin(geom)           AS bbox_ymin,
                ST_XMax(geom)           AS bbox_xmax,
                ST_YMax(geom)           AS bbox_ymax,
                ST_X(ST_Centroid(geom)) AS centroid_x,
                ST_Y(ST_Centroid(geom)) AS centroid_y
            FROM src
            ORDER BY trip_count DESC
        ) TO {_sql_literal(str(output_path))} (FORMAT PARQUET, COMPRESSION ZSTD)
        """
    )


def _should_run(stage: int, output: Path, force: bool, force_stage: int | None) -> bool:
    if force:
        return True
    if force_stage is not None and stage >= force_stage:
        return True
    return not output.exists()


def run(
    matched_dir: Path,
    osm_parquet: Path,
    output_path: Path,
    cache_dir: Path | None = None,
    force: bool = False,
    force_stage: int | None = None,
) -> None:
    if cache_dir is None:
        cache_dir = output_path.parent / ".way_counts_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    counts_path = cache_dir / "counts.parquet"
    joined_path = cache_dir / "way_counts_geom.parquet"
    tmp_dir = cache_dir / ".duckdb_tmp"
    tmp_dir.mkdir(exist_ok=True)

    con = _connect(tmp_dir)

    if _should_run(1, counts_path, force, force_stage):
        files = matched_files(matched_dir)
        t0 = time.perf_counter()
        print(f"[1/3] aggregating counts over {len(files):,} matched files ...")
        _stage_counts(con, files, counts_path)
        print(f"      wrote {counts_path} in {time.perf_counter() - t0:.1f}s")
    else:
        print(f"[1/3] cached: {counts_path}")

    if _should_run(2, joined_path, force, force_stage):
        t1 = time.perf_counter()
        print("[2/3] joining OSM geometry ...")
        _stage_join(con, counts_path, osm_parquet, joined_path)
        print(f"      wrote {joined_path} in {time.perf_counter() - t1:.1f}s")
    else:
        print(f"[2/3] cached: {joined_path}")

    if _should_run(3, output_path, force, force_stage):
        t2 = time.perf_counter()
        print("[3/3] computing bbox / centroid + sorting ...")
        _stage_index(con, joined_path, output_path)
        print(f"      wrote {output_path} in {time.perf_counter() - t2:.1f}s")
    else:
        print(f"[3/3] cached: {output_path}")

    total, with_geom = con.execute(
        f"SELECT COUNT(*), COUNT(geometry) "
        f"FROM read_parquet({_sql_literal(str(output_path))})"
    ).fetchone()
    size_mb = output_path.stat().st_size / 1e6
    print(f"{total:,} ways traversed, {with_geom:,} with OSM geometry")
    print(f"Output: {output_path} ({size_mb:.1f} MB)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Map-matched way counts joined to OSM geometry, "
        "with bbox/centroid index. Cached three-stage pipeline."
    )
    parser.add_argument("matched_dir", type=Path)
    parser.add_argument("osm_parquet", type=Path)
    parser.add_argument("output_parquet", type=Path)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Intermediate parquet location "
        "(default: <output>.parent/.way_counts_cache)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild every stage from scratch",
    )
    parser.add_argument(
        "--force-stage",
        type=int,
        choices=[1, 2, 3],
        default=None,
        help="Rebuild this stage and every stage after it",
    )
    args = parser.parse_args()
    run(
        args.matched_dir,
        args.osm_parquet,
        args.output_parquet,
        cache_dir=args.cache_dir,
        force=args.force,
        force_stage=args.force_stage,
    )


if __name__ == "__main__":
    main()
