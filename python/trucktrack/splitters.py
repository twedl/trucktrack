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
    """Split at detected stops, returning movement segments with ``segment_id``."""
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
    """Split trajectories at stops, reading/writing parquet files entirely in Rust."""
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
