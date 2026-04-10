"""Trip generation: routing, interpolation, noise, parking maneuvers."""

from trucktrack.generate.models import ErrorConfig, RouteSegment, TracePoint, TripConfig
from trucktrack.generate.trace import generate_trace, traces_to_csv, traces_to_parquet

__all__ = [
    "ErrorConfig",
    "RouteSegment",
    "TracePoint",
    "TripConfig",
    "generate_trace",
    "traces_to_csv",
    "traces_to_parquet",
]
