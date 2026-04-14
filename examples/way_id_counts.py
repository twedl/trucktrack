"""Aggregate map-matched way_ids and join OSM geometry + tags.

Counts trips per way across the matched dataset and joins against an
OSM parquet (e.g. a quackosm export) to attach geometry and tags.

Usage::

    uv run python examples/way_id_counts.py \\
        data/matched \\
        /path/to/ontario-latest.parquet \\
        data/way_counts.parquet
"""

from __future__ import annotations

import sys
from pathlib import Path

import polars as pl


def aggregate_way_counts(matched_dir: Path) -> pl.LazyFrame:
    return (
        pl.scan_parquet(matched_dir / "**/*.parquet", hive_partitioning=True)
        .drop_nulls("way_id")
        .group_by("way_id")
        .agg(
            pl.col("id").n_unique().alias("trip_count"),
            pl.col("date").min().alias("first_date"),
            pl.col("date").max().alias("last_date"),
        )
    )


def join_osm(counts: pl.LazyFrame, osm_parquet: Path) -> pl.LazyFrame:
    # quackosm writes feature_id as "way/<id>"; strip and cast to match
    # Valhalla's int64 way_id.
    osm = pl.scan_parquet(osm_parquet).with_columns(
        pl.col("feature_id").str.strip_prefix("way/").cast(pl.Int64).alias("way_id")
    ).drop("feature_id")
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

    enriched = join_osm(aggregate_way_counts(matched_dir), osm_parquet).collect()
    matched_geom = enriched["geometry"].is_not_null().sum()
    print(
        f"{len(enriched):,} ways traversed, "
        f"{matched_geom:,} with OSM geometry"
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    enriched.write_parquet(output_path)
    print(f"Wrote {output_path} ({output_path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
