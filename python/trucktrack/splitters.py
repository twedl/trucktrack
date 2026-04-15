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


# ── StalePingFilter ─────────────────────────────────────────────────────


def filter_stale_pings(
    df: pl.DataFrame,
    *,
    window: int = 5,
    id_col: str = "id",
    time_col: str = "time",
    lat_col: str = "lat",
    lon_col: str = "lon",
    speed_col: str = "speed",
    heading_col: str = "heading",
) -> pl.DataFrame:
    """Drop stale GPS pings — verbatim re-emissions of an earlier record.

    Some devices buffer readings and occasionally re-emit an older record
    later in the stream with only the timestamp advanced, producing a
    sequence like ``T1 → T2 → T3`` where ``T3`` has the same
    ``(lat, lon, speed, heading)`` as ``T1`` but a later time.

    Per truck, each row is compared bit-exactly against the last ``window``
    distinct rows; matches are dropped.  Rows with a null in any of the four
    compared fields never match and pass through.  Rows with ``speed == 0``
    are also exempt — a stopped truck legitimately emits many repeated
    identical pings.  Output is sorted by ``(id, time)``.

    Parameters
    ----------
    window
        Lookback size in rows per truck.  Larger windows catch re-emissions
        of older records but risk dropping legitimate revisits that happen
        within the window.  Default 5.
    id_col, time_col, lat_col, lon_col, speed_col, heading_col
        Column names.
    """
    return _core.filter_stale_pings_df(
        df, id_col, time_col, lat_col, lon_col, speed_col, heading_col, window
    )


def filter_stale_pings_file(
    input_path: str | Path,
    output_path: str | Path,
    *,
    window: int = 5,
    id_col: str = "id",
    time_col: str = "time",
    lat_col: str = "lat",
    lon_col: str = "lon",
    speed_col: str = "speed",
    heading_col: str = "heading",
) -> int:
    """File-based variant of :func:`filter_stale_pings`."""
    return _core.filter_stale_pings_file(
        str(input_path),
        str(output_path),
        id_col,
        time_col,
        lat_col,
        lon_col,
        speed_col,
        heading_col,
        window,
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
