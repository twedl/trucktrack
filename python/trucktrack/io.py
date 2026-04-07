"""I/O utilities for reading track data into Polars DataFrames."""

from __future__ import annotations

from pathlib import Path

import polars as pl

from trucktrack import _core


def read_parquet(path: str | Path) -> pl.DataFrame:
    """Read a parquet file containing track data.

    Expected columns: id, time, speed, heading, lat, lon.
    """
    return pl.read_parquet(path)


def read_dataset(source: pl.DataFrame) -> pl.DataFrame:
    """Accept an already-loaded Polars DataFrame of track data."""
    return source


def process_parquet_in_rust(input_path: str | Path, output_path: str | Path) -> int:
    """Read *input_path*, compute derived columns in Rust, write to *output_path*.

    The entire operation happens inside the Rust extension — no Python DataFrame
    objects are created. Returns the number of rows written.
    """
    return _core.process_tracks_file(str(input_path), str(output_path))


def process_dataframe_in_rust(df: pl.DataFrame) -> pl.DataFrame:
    """Pass *df* to Rust for processing and return the result as a new DataFrame.

    The DataFrame is shared with Rust via the Arrow C Data Interface
    (``pyo3-polars``), so column buffers are not copied.
    """
    return _core.tracks_from_df(df)
