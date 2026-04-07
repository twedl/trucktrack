"""trucktrack — high-performance Python package backed by Rust.

Public API is re-exported here from the compiled `_core` extension and from
the pure-Python `generate` and `partition` subpackages.
"""

from __future__ import annotations

from trucktrack._core import __version__
from trucktrack.generate import (
    RouteSegment,
    TracePoint,
    TripConfig,
    generate_trace,
    traces_to_csv,
    traces_to_parquet,
)
from trucktrack.io import (
    process_dataframe_in_rust,
    process_parquet_in_rust,
    read_dataset,
    read_parquet,
)
from trucktrack.partition import (
    assign_partitions,
    classify_and_partition_key,
    partition_existing_parquet,
    write_partitions,
    write_trips_partitioned,
)
from trucktrack.splitters import (
    split_by_observation_gap,
    split_by_observation_gap_file,
    split_by_stops,
    split_by_stops_file,
)

__all__ = [
    "__version__",
    "read_parquet",
    "read_dataset",
    "process_parquet_in_rust",
    "process_dataframe_in_rust",
    "split_by_observation_gap",
    "split_by_observation_gap_file",
    "split_by_stops",
    "split_by_stops_file",
    # generate
    "RouteSegment",
    "TracePoint",
    "TripConfig",
    "generate_trace",
    "traces_to_csv",
    "traces_to_parquet",
    # partition
    "assign_partitions",
    "classify_and_partition_key",
    "partition_existing_parquet",
    "write_partitions",
    "write_trips_partitioned",
]
