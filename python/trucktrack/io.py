"""I/O utilities for reading track data into Polars DataFrames."""

from __future__ import annotations

import io as _io
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

    Serialises to Arrow IPC bytes, hands the buffer to the Rust extension,
    then deserialises the result back into a Polars DataFrame.
    """
    buf = _io.BytesIO()
    df.write_ipc(buf)
    result_bytes = _core.tracks_from_ipc(buf.getvalue())
    return pl.read_ipc(_io.BytesIO(result_bytes))
