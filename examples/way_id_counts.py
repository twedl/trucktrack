"""Aggregate map-matched way_ids and join OSM geometry + tags.

Reads the hive-partitioned map-matching output (``id``, ``date``,
``way_id`` rows produced by :mod:`trucktrack.valhalla.pipeline`),
counts how many trips traversed each way, and joins against an OSM
parquet (e.g. quackosm export with ``feature_id``, ``geometry``,
``highway``) to attach the road geometry and tags.

The output is a parquet file with one row per OSM way that was hit
by at least one matched trip.

Usage::

    uv run python examples/way_id_counts.py \\
        data/matched \\
        /path/to/ontario-latest.parquet \\
        data/way_counts.parquet

The OSM parquet is expected to have:
    - ``feature_id`` (str, like ``way/123456``)
    - ``geometry``   (WKB binary)
    - one or more tag columns (``highway``, ``name``, ...)
"""

from __future__ import annotations

import sys
from pathlib import Path

import polars as pl


def aggregate_way_counts(matched_dir: Path) -> pl.DataFrame:
    """Count distinct trips per way_id across the matched dataset.

    Drops the null way_id rows (failed/skipped trips).
    """
    return (
        pl.scan_parquet(matched_dir / "**/*.parquet", hive_partitioning=True)
        .drop_nulls("way_id")
        .group_by("way_id")
        .agg(
            pl.col("id").n_unique().alias("trip_count"),
            pl.col("date").min().alias("first_date"),
            pl.col("date").max().alias("last_date"),
        )
        .collect()
    )


def join_osm(counts: pl.DataFrame, osm_parquet: Path) -> pl.DataFrame:
    """Attach OSM geometry and tag columns to the per-way counts.

    quackosm writes ``feature_id`` as ``way/<id>``; we strip the
    ``way/`` prefix and cast to Int64 so the join key matches the
    Valhalla ``way_id``.
    """
    osm = (
        pl.scan_parquet(osm_parquet)
        .with_columns(
            pl.col("feature_id")
            .str.strip_prefix("way/")
            .cast(pl.Int64, strict=False)
            .alias("way_id")
        )
        .drop("feature_id")
        .collect()
    )
    return counts.join(osm, on="way_id", how="left").sort(
        "trip_count", descending=True
    )


def main() -> None:
    if len(sys.argv) != 4:
        print(
            f"Usage: {sys.argv[0]} <matched_dir> <osm_parquet> <output_parquet>",
            file=sys.stderr,
        )
        sys.exit(1)
    matched_dir = Path(sys.argv[1])
    osm_parquet = Path(sys.argv[2])
    output_path = Path(sys.argv[3])

    print(f"Aggregating way counts from {matched_dir} ...")
    counts = aggregate_way_counts(matched_dir)
    print(f"  {len(counts):,} distinct ways traversed")

    print(f"Joining OSM data from {osm_parquet} ...")
    enriched = join_osm(counts, osm_parquet)
    matched_geom = enriched["geometry"].is_not_null().sum()
    print(f"  {matched_geom:,} / {len(enriched):,} ways had OSM geometry")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    enriched.write_parquet(output_path)
    print(f"Wrote {output_path} ({output_path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
