"""Trajectory splitting functions backed by Rust."""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import polars as pl

from trucktrack import _core

# ── ObservationGapSplitter ───────────────────────────────────────────────


def split_by_observation_gap(
    df: pl.DataFrame,
    gap: timedelta,
    *,
    id_col: str = "id",
    time_col: str = "time",
    min_length: int = 0,
) -> pl.DataFrame:
    """Split trajectories at temporal gaps, returning df with ``segment_id`` column."""
    gap_us = int(gap.total_seconds() * 1_000_000)
    return _core.split_by_gap_df(df, id_col, time_col, gap_us, min_length)


def split_by_observation_gap_file(
    input_path: str | Path,
    output_path: str | Path,
    gap: timedelta,
    *,
    id_col: str = "id",
    time_col: str = "time",
    min_length: int = 0,
) -> int:
    """Split at temporal gaps via parquet files entirely in Rust."""
    gap_us = int(gap.total_seconds() * 1_000_000)
    return _core.split_by_gap_file(
        str(input_path), str(output_path), id_col, time_col, gap_us, min_length
    )


# ── StopSplitter ─────────────────────────────────────────────────────────


def split_by_stops(
    df: pl.DataFrame,
    max_diameter: float,
    min_duration: timedelta,
    *,
    id_col: str = "id",
    time_col: str = "time",
    lat_col: str = "lat",
    lon_col: str = "lon",
    min_length: int = 0,
) -> pl.DataFrame:
    """Split at detected stops.

    Returns all rows with ``segment_id`` and ``is_stop`` columns.
    """
    dur_us = int(min_duration.total_seconds() * 1_000_000)
    return _core.split_by_stops_df(
        df,
        id_col,
        time_col,
        lat_col,
        lon_col,
        max_diameter,
        dur_us,
        min_length,
    )


def split_by_stops_file(
    input_path: str | Path,
    output_path: str | Path,
    max_diameter: float,
    min_duration: timedelta,
    *,
    id_col: str = "id",
    time_col: str = "time",
    lat_col: str = "lat",
    lon_col: str = "lon",
    min_length: int = 0,
) -> int:
    """Split trajectories at stops, reading/writing parquet files entirely in Rust.

    Output includes all rows with ``segment_id`` and ``is_stop`` columns.
    """
    dur_us = int(min_duration.total_seconds() * 1_000_000)
    return _core.split_by_stops_file(
        str(input_path),
        str(output_path),
        id_col,
        time_col,
        lat_col,
        lon_col,
        max_diameter,
        dur_us,
        min_length,
    )


# ── TrafficFilter ───────────────────────────────────────────────────────


def filter_traffic_stops(
    df: pl.DataFrame,
    *,
    max_angle_change: float = 30.0,
    min_distance: float = 10.0,
    id_col: str = "id",
    lat_col: str = "lat",
    lon_col: str = "lon",
) -> pl.DataFrame:
    """Remove stops where the truck stayed on the road corridor.

    Compares approach and departure bearings (computed from positions, not
    device heading) around each stop segment.  If the angular change is
    below ``max_angle_change`` degrees the stop is reclassified as movement.

    Designed to run on the output of :func:`split_by_stops`.

    Parameters
    ----------
    df
        DataFrame with ``segment_id`` and ``is_stop`` columns.
    max_angle_change
        Maximum bearing change in degrees for a stop to be considered
        traffic. Stops with a smaller change are reclassified.
    min_distance
        Minimum distance in meters between point pairs used for bearing
        computation. Filters out GPS jitter.
    id_col, lat_col, lon_col
        Column names.
    """
    return _core.filter_traffic_stops_df(
        df, id_col, lat_col, lon_col, max_angle_change, min_distance
    )


def filter_traffic_stops_file(
    input_path: str | Path,
    output_path: str | Path,
    *,
    max_angle_change: float = 30.0,
    min_distance: float = 10.0,
    id_col: str = "id",
    lat_col: str = "lat",
    lon_col: str = "lon",
) -> int:
    """File-based variant of :func:`filter_traffic_stops`."""
    return _core.filter_traffic_stops_file(
        str(input_path),
        str(output_path),
        id_col,
        lat_col,
        lon_col,
        max_angle_change,
        min_distance,
    )
