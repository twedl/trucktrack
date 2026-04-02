"""Type stubs for trucktrack._core (compiled Rust extension)."""

__version__: str

def process_tracks_file(input_path: str, output_path: str) -> int:
    """Read parquet, compute derived columns in Rust, write to output parquet.

    Returns the number of rows written.
    """
    ...

def tracks_from_ipc(ipc_bytes: bytes) -> bytes:
    """Accept Arrow IPC bytes, process in Rust, return Arrow IPC bytes."""
    ...

def split_by_gap_ipc(
    ipc_bytes: bytes,
    id_col: str,
    time_col: str,
    gap_us: int,
    min_length: int,
) -> bytes:
    """Split trajectories at observation gaps via Arrow IPC."""
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

def split_by_stops_ipc(
    ipc_bytes: bytes,
    id_col: str,
    time_col: str,
    lat_col: str,
    lon_col: str,
    max_diameter_m: float,
    min_duration_us: int,
    min_length: int,
) -> bytes:
    """Split trajectories at detected stops via Arrow IPC."""
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
    """Split trajectories at detected stops, reading/writing parquet files."""
    ...
