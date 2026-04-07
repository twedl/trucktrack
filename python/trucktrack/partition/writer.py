"""Hive-partitioned dataset writer (Polars).

Layout:
    <output>/
        tier=local/    partition_id=<id>/<file>.parquet
        tier=regional/ partition_id=<id>/<file>.parquet
        tier=longhaul/ partition_id=<id>/<file>.parquet

Each row is a single GPS point keyed by `id`. Within a partition file rows are
sorted by a Hilbert-curve index over the trip centroid for spatial locality.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from trucktrack.generate.models import TracePoint
from trucktrack.partition.classify import (
    assign_partitions,
    metadata_from_trace_points,
)


def _summary(metadata: pl.DataFrame) -> dict[str, int]:
    counts = (
        metadata.unique(subset=["partition_id"])
        .group_by("tier")
        .len()
        .rename({"len": "count"})
    )
    return {row["tier"]: int(row["count"]) for row in counts.iter_rows(named=True)}


def write_partitions(
    metadata: pl.DataFrame,
    points: pl.DataFrame,
    output_dir: Path,
) -> dict[str, int]:
    """Write a hive-partitioned dataset rooted at *output_dir*.

    `metadata` must contain (id, tier, partition_id, hilbert_idx).
    `points`   must contain (id, lat, lon, speed, heading, timestamp).
    Returns a {tier_name: partition_count} summary.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    merged = (
        points.join(
            metadata.select(["id", "tier", "partition_id", "hilbert_idx"]),
            on="id",
        )
        .sort(["partition_id", "hilbert_idx"])
        .drop("hilbert_idx")
    )

    # polars writes hive-style directories when partition_by is given.
    merged.write_parquet(
        output_dir,
        partition_by=["tier", "partition_id"],
    )

    return _summary(metadata)


def _trips_to_points_df(
    trips: list[tuple[list[TracePoint], str]],
) -> pl.DataFrame:
    """Flatten (points, trip_id) tuples into a one-row-per-point DataFrame."""
    ids: list[str] = []
    lats: list[float] = []
    lons: list[float] = []
    speeds: list[float] = []
    headings: list[float] = []
    timestamps: list = []
    for points, trip_id in trips:
        for pt in points:
            ids.append(trip_id)
            lats.append(pt.lat)
            lons.append(pt.lon)
            speeds.append(round(pt.speed_mph, 1))
            headings.append(round(pt.heading, 1))
            timestamps.append(pt.timestamp)
    return pl.DataFrame(
        {
            "id": ids,
            "lat": lats,
            "lon": lons,
            "speed": speeds,
            "heading": headings,
            "time": timestamps,
        }
    )


def write_trips_partitioned(
    trips: list[tuple[list[TracePoint], str]],
    output_dir: Path,
) -> dict[str, int]:
    """Write in-memory trips as a hive-partitioned parquet dataset.

    Returns a {tier_name: partition_count} summary.
    """
    metadata_rows = [
        metadata_from_trace_points(tid, pts) for pts, tid in trips if pts
    ]
    metadata = pl.DataFrame(
        {
            "id": [m.id for m in metadata_rows],
            "centroid_lat": [m.centroid_lat for m in metadata_rows],
            "centroid_lon": [m.centroid_lon for m in metadata_rows],
            "bbox_diag_km": [m.bbox_diag_km for m in metadata_rows],
        }
    )
    metadata = assign_partitions(metadata)

    points = _trips_to_points_df(trips)
    return write_partitions(metadata, points, output_dir)


def partition_existing_parquet(
    input_path: Path, output_dir: Path
) -> dict[str, int]:
    """Read a flat parquet (`id, lat, lon, ...`) and rewrite as hive-partitioned."""
    input_path = Path(input_path)
    points = pl.read_parquet(input_path)

    required = {"id", "lat", "lon"}
    missing = required - set(points.columns)
    if missing:
        raise ValueError(
            f"{input_path} is missing required columns: {sorted(missing)}"
        )

    agg = points.group_by("id").agg(
        pl.col("lat").min().alias("lat_min"),
        pl.col("lat").max().alias("lat_max"),
        pl.col("lon").min().alias("lon_min"),
        pl.col("lon").max().alias("lon_max"),
        pl.col("lat").mean().alias("centroid_lat"),
        pl.col("lon").mean().alias("centroid_lon"),
    )

    R = 6371.0
    lat1 = (pl.col("lat_min") * (3.141592653589793 / 180.0)).alias("_lat1")
    lat2 = (pl.col("lat_max") * (3.141592653589793 / 180.0)).alias("_lat2")
    agg = agg.with_columns(lat1, lat2)
    agg = agg.with_columns(
        ((pl.col("_lat2") - pl.col("_lat1"))).alias("_dlat"),
        (
            (pl.col("lon_max") - pl.col("lon_min"))
            * (3.141592653589793 / 180.0)
        ).alias("_dlon"),
    )
    agg = agg.with_columns(
        (
            (pl.col("_dlat") / 2).sin() ** 2
            + pl.col("_lat1").cos()
            * pl.col("_lat2").cos()
            * (pl.col("_dlon") / 2).sin() ** 2
        ).alias("_a")
    )
    agg = agg.with_columns(
        (R * 2 * pl.col("_a").sqrt().arcsin()).alias("bbox_diag_km")
    )

    metadata = assign_partitions(
        agg.select(["id", "centroid_lat", "centroid_lon", "bbox_diag_km"])
    )

    return write_partitions(metadata, points, output_dir)
