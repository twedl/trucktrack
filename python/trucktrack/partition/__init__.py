"""Partition GPS traces into Valhalla-tile-aligned hive partitions.

Downstream map-matching can hit a hot tile cache instead of randomly thrashing
it. This package is a Polars-native rewrite of the original tracks.partition
module.
"""

from trucktrack.partition.classify import (
    HILBERT_ORDER,
    LOCAL_KM,
    LONGHAUL_DEG,
    REGIONAL_KM,
    TIER_NAMES,
    TraceMetadata,
    assign_partitions,
    classify_and_partition_key,
    metadata_from_trace_points,
)
from trucktrack.partition.tiles import (
    VALHALLA_L0_DEG,
    VALHALLA_L1_DEG,
    haversine_km,
    valhalla_l0_tile,
    valhalla_l1_tile,
    valhalla_tile_id,
)
from trucktrack.partition.writer import (
    partition_existing_parquet,
    write_partitions,
    write_trips_partitioned,
)

__all__ = [
    "HILBERT_ORDER",
    "LOCAL_KM",
    "LONGHAUL_DEG",
    "REGIONAL_KM",
    "TIER_NAMES",
    "TraceMetadata",
    "VALHALLA_L0_DEG",
    "VALHALLA_L1_DEG",
    "assign_partitions",
    "classify_and_partition_key",
    "haversine_km",
    "metadata_from_trace_points",
    "partition_existing_parquet",
    "valhalla_l0_tile",
    "valhalla_l1_tile",
    "valhalla_tile_id",
    "write_partitions",
    "write_trips_partitioned",
]
