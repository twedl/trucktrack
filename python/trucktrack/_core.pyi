"""Type stubs for trucktrack._core (compiled Rust extension)."""

import polars as pl

__version__: str

def process_tracks_file(input_path: str, output_path: str) -> int:
    """Read parquet, compute derived columns in Rust, write to output parquet.

    Returns the number of rows written.
    """
    ...

def tracks_from_df(df: pl.DataFrame) -> pl.DataFrame:
    """Process a Polars DataFrame in Rust (zero-copy via Arrow C Data Interface)."""
    ...

def split_by_gap_df(
    df: pl.DataFrame,
    id_col: str,
    time_col: str,
    gap_us: int,
    min_length: int,
) -> pl.DataFrame:
    """Split trajectories at observation gaps (zero-copy DataFrame handoff)."""
    ...

def split_by_gap_file(
    input_path: str,
    output_path: str,
    id_col: str,
    time_col: str,
    gap_us: int,
    min_length: int,
) -> int:
    """Split trajectories at observation gaps, reading/writing parquet files."""
    ...

def split_by_stops_df(
    df: pl.DataFrame,
    id_col: str,
    time_col: str,
    lat_col: str,
    lon_col: str,
    max_diameter_m: float,
    min_duration_us: int,
    min_length: int,
) -> pl.DataFrame:
    """Split trajectories at detected stops (zero-copy DataFrame handoff).

    Returns all rows with ``segment_id`` and ``is_stop`` columns.
    """
    ...

def split_by_stops_file(
    input_path: str,
    output_path: str,
    id_col: str,
    time_col: str,
    lat_col: str,
    lon_col: str,
    max_diameter_m: float,
    min_duration_us: int,
    min_length: int,
) -> int:
    """Split trajectories at detected stops, reading/writing parquet files.

    Output includes all rows with ``segment_id`` and ``is_stop`` columns.
    """
    ...

def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Haversine distance in kilometers between two (lat, lon) points."""
    ...

def valhalla_tile_id(lat: float, lon: float, tile_deg: float) -> int:
    """Flat tile index matching Valhalla's row-major numbering.

    Raises:
        ValueError: If ``tile_deg`` is not positive.
    """
    ...

def classify_and_partition_key(
    centroid_lat: float,
    centroid_lon: float,
    bbox_diag_km: float,
) -> tuple[str, int]:
    """Return (tier_name, partition_id) for a trip's centroid + bbox diagonal."""
    ...

def hilbert_indices(lats: list[float], lons: list[float]) -> list[int]:
    """Compute Hilbert curve indices for arrays of lat/lon coordinates."""
    ...
