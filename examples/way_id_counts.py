"""Aggregate map-matched way_ids and join OSM geometry.

Counts trips per way across the matched dataset and joins against an
OSM parquet (e.g. a quackosm export) to attach geometry.

Only ``feature_id`` and ``geometry`` are read from the OSM parquet, so
unfiltered exports with a MAP-typed ``tags`` column work without help
(DuckDB prunes unused columns at the parquet reader).

Uses DuckDB to stream the aggregation + join + sort through a single
``COPY ... TO`` so the result is never materialized in RAM. Spills to
``<output_parent>/.duckdb_tmp/`` when memory pressure demands.

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
"""

from __future__ import annotations

import sys
from glob import glob
from pathlib import Path

import duckdb

# Leaves headroom on a 120GB host; DuckDB spills the rest to ``temp_directory``.
MEMORY_LIMIT = "64GB"


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


def run(matched_dir: Path, osm_parquet: Path, output_path: Path) -> None:
    files = matched_files(matched_dir)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_dir = output_path.parent / ".duckdb_tmp"
    tmp_dir.mkdir(exist_ok=True)

    con = duckdb.connect()
    # DuckDB requires string literals (not ?-parameters) for COPY targets,
    # temp_directory, and ``read_parquet`` filenames at bind time, so we
    # inline quoted paths via ``_sql_literal``.
    con.execute(f"PRAGMA memory_limit = '{MEMORY_LIMIT}'")
    con.execute(f"PRAGMA temp_directory = {_sql_literal(str(tmp_dir))}")
    con.execute("PRAGMA preserve_insertion_order = false")

    matched_literal = "[" + ", ".join(_sql_literal(f) for f in files) + "]"

    con.execute(
        f"""
        COPY (
            WITH counts AS (
                SELECT
                    way_id,
                    COUNT(DISTINCT id) AS trip_count,
                    MIN(date)          AS first_date,
                    MAX(date)          AS last_date
                FROM read_parquet(
                    {matched_literal},
                    hive_partitioning = true,
                    union_by_name     = true
                )
                WHERE way_id IS NOT NULL
                GROUP BY way_id
            ),
            osm AS (
                -- quackosm writes feature_id as 'way/<id>'; strip the
                -- 4-char prefix and cast to match Valhalla's int64 way_id.
                SELECT
                    CAST(substr(feature_id, 5) AS BIGINT) AS way_id,
                    geometry
                FROM read_parquet({_sql_literal(str(osm_parquet))})
                WHERE feature_id LIKE 'way/%'
            )
            SELECT
                c.way_id,
                c.trip_count,
                c.first_date,
                c.last_date,
                o.geometry
            FROM counts c
            LEFT JOIN osm o USING (way_id)
            ORDER BY c.trip_count DESC
        ) TO {_sql_literal(str(output_path))} (FORMAT PARQUET, COMPRESSION ZSTD)
        """
    )

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
