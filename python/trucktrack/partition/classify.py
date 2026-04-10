"""Tier classification and partition-key assignment (Polars).

Tiers:
    local    (< 100 km bbox)   → Valhalla L1 (1°×1°) tile bucket
    regional (100–800 km bbox) → Valhalla L0 (4°×4°) tile bucket
    longhaul (> 800 km bbox)   → coarse 8°×8° super-region bucket
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import polars as pl

from trucktrack import _core
from trucktrack.generate.models import TracePoint
from trucktrack.partition.tiles import (
    VALHALLA_L0_DEG,
    VALHALLA_L1_DEG,
    haversine_km,
)

EARTH_RADIUS_KM = 6371.0

# p=12 → 4096 cells per axis ≈ 1.6 km cells over the US+Canada bbox.
HILBERT_ORDER = 12


LOCAL_KM = 100.0
REGIONAL_KM = 800.0
LONGHAUL_DEG = 8.0

TIER_NAMES = ("local", "regional", "longhaul")


def classify_and_partition_key(
    centroid_lat: float,
    centroid_lon: float,
    bbox_diag_km: float,
) -> tuple[str, int]:
    """Return (tier_name, partition_id) for a single trip's centroid + bbox.

    partition_id encodes the tier in the high bits so values don't collide
    across tiers:
        bits 62–60: tier  (0=local, 1=regional, 2=longhaul)
        bits 59–0:  tile index
    """
    return _core.classify_and_partition_key(centroid_lat, centroid_lon, bbox_diag_km)


@dataclass
class TraceMetadata:
    id: str
    centroid_lat: float
    centroid_lon: float
    bbox_diag_km: float


def metadata_from_trace_points(trip_id: str, points: list[TracePoint]) -> TraceMetadata:
    """Compute centroid + bbox-diagonal metadata from in-memory TracePoints."""
    lats = [p.lat for p in points]
    lons = [p.lon for p in points]
    return TraceMetadata(
        id=trip_id,
        centroid_lat=sum(lats) / len(lats),
        centroid_lon=sum(lons) / len(lons),
        bbox_diag_km=haversine_km(min(lats), min(lons), max(lats), max(lons)),
    )


def _tile_expr(lat: pl.Expr, lon: pl.Expr, deg: float) -> pl.Expr:
    n_cols = int(360 / deg)
    col = ((lon + 180.0) / deg).floor().cast(pl.Int64).clip(0, n_cols - 1)
    row = ((lat + 90.0) / deg).floor().cast(pl.Int64).clip(0, int(180 / deg) - 1)
    return row * n_cols + col


def _compute_trip_metadata(df: pl.DataFrame) -> pl.DataFrame:
    """Aggregate per-trip centroid and bbox diagonal from a points DataFrame.

    Input must contain: id, lat, lon.
    Returns one row per trip with: id, centroid_lat, centroid_lon, bbox_diag_km.
    """
    agg = df.group_by("id").agg(
        pl.col("lat").min().alias("lat_min"),
        pl.col("lat").max().alias("lat_max"),
        pl.col("lon").min().alias("lon_min"),
        pl.col("lon").max().alias("lon_max"),
        pl.col("lat").mean().alias("centroid_lat"),
        pl.col("lon").mean().alias("centroid_lon"),
    )

    # Vectorized haversine for bbox diagonal — chained because each step
    # depends on the previous (Polars can't forward-reference within one
    # with_columns call).
    deg2rad = math.pi / 180.0
    agg = agg.with_columns(
        (pl.col("lat_min") * deg2rad).alias("_lat1"),
        (pl.col("lat_max") * deg2rad).alias("_lat2"),
    )
    agg = agg.with_columns(
        (pl.col("_lat2") - pl.col("_lat1")).alias("_dlat"),
        ((pl.col("lon_max") - pl.col("lon_min")) * deg2rad).alias("_dlon"),
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
        (EARTH_RADIUS_KM * 2 * pl.col("_a").sqrt().arcsin()).alias("bbox_diag_km")
    )

    return agg.select(["id", "centroid_lat", "centroid_lon", "bbox_diag_km"])


def partition_points(df: pl.DataFrame) -> pl.DataFrame:
    """Add tier, partition_id, and hilbert_idx columns to a points DataFrame.

    Single-call entry point for partitioning an in-memory DataFrame.
    Input must contain at least: id, lat, lon.
    Returns the original DataFrame with tier, partition_id, hilbert_idx joined in.
    """
    required = {"id", "lat", "lon"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"missing required columns: {sorted(missing)}")

    metadata = assign_partitions(_compute_trip_metadata(df))

    return df.join(
        metadata.select(["id", "tier", "partition_id", "hilbert_idx"]),
        on="id",
    )


def assign_partitions(df: pl.DataFrame) -> pl.DataFrame:
    """Add `tier`, `partition_id`, `hilbert_idx` columns to a metadata DataFrame.

    Input columns: id, centroid_lat, centroid_lon, bbox_diag_km
    """
    required = {"id", "centroid_lat", "centroid_lon", "bbox_diag_km"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"missing required columns: {sorted(missing)}")

    lat = pl.col("centroid_lat")
    lon = pl.col("centroid_lon")
    diag = pl.col("bbox_diag_km")

    l1 = _tile_expr(lat, lon, VALHALLA_L1_DEG)
    l0 = _tile_expr(lat, lon, VALHALLA_L0_DEG)
    lh = _tile_expr(lat, lon, LONGHAUL_DEG)

    tier = (
        pl.when(diag < LOCAL_KM)
        .then(0)
        .when(diag < REGIONAL_KM)
        .then(1)
        .otherwise(2)
        .cast(pl.Int64)
    )
    tile = (
        pl.when(tier == 0)
        .then(l1)
        .when(tier == 1)
        .then(l0)
        .otherwise(lh)
        .cast(pl.Int64)
    )
    partition_id = (tier * (1 << 60)) + tile
    tier_name = (
        pl.when(tier == 0)
        .then(pl.lit("local"))
        .when(tier == 1)
        .then(pl.lit("regional"))
        .otherwise(pl.lit("longhaul"))
    )

    enriched = df.with_columns(
        tier=tier_name,
        partition_id=partition_id,
    )

    lat_vals = enriched["centroid_lat"].to_list()
    lon_vals = enriched["centroid_lon"].to_list()
    hilbert = _core.hilbert_indices(lat_vals, lon_vals) if lat_vals else []

    return enriched.with_columns(
        hilbert_idx=pl.Series("hilbert_idx", hilbert, dtype=pl.Int64)
    )
