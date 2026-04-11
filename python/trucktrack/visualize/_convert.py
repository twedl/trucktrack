"""Convert TracePoint lists to Polars DataFrames."""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

if TYPE_CHECKING:
    from trucktrack.generate.models import TracePoint


def tracepoints_to_dataframe(points: list[TracePoint]) -> pl.DataFrame:
    """Convert a list of TracePoint to a DataFrame with standard columns."""
    return pl.DataFrame(
        {
            "lat": [p.lat for p in points],
            "lon": [p.lon for p in points],
            "speed": [p.speed_mph for p in points],
            "heading": [p.heading for p in points],
            "time": [p.timestamp for p in points],
        }
    )
