"""Tier classification and partition-key assignment (Polars).

Tiers:
    local    (< 100 km bbox)   → Valhalla L1 (1°×1°) tile bucket
    regional (100–800 km bbox) → Valhalla L0 (4°×4°) tile bucket
    longhaul (> 800 km bbox)   → coarse 8°×8° super-region bucket
"""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl
from hilbertcurve.hilbertcurve import HilbertCurve

from trucktrack.generate.models import TracePoint
from trucktrack.partition.tiles import (
    VALHALLA_L0_DEG,
    VALHALLA_L1_DEG,
    haversine_km,
    valhalla_l0_tile,
    valhalla_l1_tile,
    valhalla_tile_id,
)

# p=12 → 4096 cells per axis ≈ 1.6 km cells over the US+Canada bbox.
HILBERT_ORDER = 12
_hc = HilbertCurve(p=HILBERT_ORDER, n=2)

# Bounding box for US + Canada
LAT_MIN, LAT_MAX = 24.0, 84.0
LON_MIN, LON_MAX = -141.0, -52.0


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
    if bbox_diag_km < LOCAL_KM:
        tier = 0
        tile = valhalla_l1_tile(centroid_lat, centroid_lon)
    elif bbox_diag_km < REGIONAL_KM:
        tier = 1
        tile = valhalla_l0_tile(centroid_lat, centroid_lon)
    else:
        tier = 2
        tile = valhalla_tile_id(centroid_lat, centroid_lon, LONGHAUL_DEG)

    partition_id = (tier << 60) | tile
    return TIER_NAMES[tier], partition_id


@dataclass
class TraceMetadata:
    id: str
    centroid_lat: float
    centroid_lon: float
    bbox_diag_km: float


def metadata_from_trace_points(
    trip_id: str, points: list[TracePoint]
) -> TraceMetadata:
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

    n = (1 << HILBERT_ORDER) - 1
    lat_vals = enriched["centroid_lat"].to_list()
    lon_vals = enriched["centroid_lon"].to_list()
    coords = []
    for la, lo in zip(lat_vals, lon_vals, strict=True):
        x = max(0, min(n, int((lo - LON_MIN) / (LON_MAX - LON_MIN) * n)))
        y = max(0, min(n, int((la - LAT_MIN) / (LAT_MAX - LAT_MIN) * n)))
        coords.append([x, y])
    hilbert = _hc.distances_from_points(coords) if coords else []

    return enriched.with_columns(
        hilbert_idx=pl.Series("hilbert_idx", hilbert, dtype=pl.Int64)
    )
