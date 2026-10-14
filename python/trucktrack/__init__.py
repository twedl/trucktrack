"""trucktrack — high-performance Python package backed by Rust.

Public API is re-exported here from the compiled `_core` extension and from
the pure-Python modules and the `generate` subpackage.
"""

from __future__ import annotations

from trucktrack._core import __version__
from trucktrack.buckets import bucket_by_truck, bucket_of
from trucktrack.generate import (
    ErrorConfig,
    RouteSegment,
    SamplingConfig,
    TracePoint,
    TripConfig,
    default_error_profile,
    generate_trace,
    random_sampling,
    traces_to_csv,
    traces_to_parquet,
)
from trucktrack.io import (
    process_dataframe_in_rust,
    process_parquet_in_rust,
    read_dataset,
    read_parquet,
)
from trucktrack.pipeline import run_pipeline
from trucktrack.query import scan_trip, scan_truck
from trucktrack.splitters import (
    filter_impossible_speeds,
    filter_impossible_speeds_file,
    filter_stale_pings,
    filter_stale_pings_file,
    filter_traffic_stops,
    filter_traffic_stops_file,
    split_by_observation_gap,
    split_by_observation_gap_file,
    split_by_stops,
    split_by_stops_file,
)

# ``trucktrack.inspect`` is intentionally not re-exported — it would shadow
# stdlib ``inspect``. Import explicitly: ``from trucktrack import inspect``.

__all__ = [
    "__version__",
    "read_parquet",
    "read_dataset",
    "process_parquet_in_rust",
    "process_dataframe_in_rust",
    "filter_impossible_speeds",
    "filter_impossible_speeds_file",
    "filter_stale_pings",
    "filter_stale_pings_file",
    "filter_traffic_stops",
    "filter_traffic_stops_file",
    "split_by_observation_gap",
    "split_by_observation_gap_file",
    "split_by_stops",
    "split_by_stops_file",
    # generate
    "ErrorConfig",
    "RouteSegment",
    "SamplingConfig",
    "TracePoint",
    "TripConfig",
    "default_error_profile",
    "generate_trace",
    "random_sampling",
    "traces_to_csv",
    "traces_to_parquet",
    # buckets
    "bucket_by_truck",
    "bucket_of",
    # pipeline
    "run_pipeline",
    # query
    "scan_truck",
    "scan_trip",
]
