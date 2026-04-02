"""trucktrack — high-performance Python package backed by Rust.

Public API is re-exported here from the compiled `_core` extension.
"""

from __future__ import annotations

from trucktrack._core import __version__
from trucktrack.io import (
    process_dataframe_in_rust,
    process_parquet_in_rust,
    read_dataset,
    read_parquet,
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
]
