# trucktrack

High-performance trajectory splitting, generation, and partitioning, powered by Rust.

A Python package that replicates and extends
[movingpandas](https://movingpandas.org/) trajectory splitters
(ObservationGapSplitter, StopSplitter) with a Rust backend for speed.
Data flows through [Polars](https://pola.rs/) DataFrames, with the option
to process entirely in Rust (parquet in, parquet out) or share DataFrames
between Python and Rust zero-copy via `pyo3-polars`.

In addition to the Rust splitters, trucktrack ships two pure-Python
subpackages:

- **`trucktrack.generate`** — synthesize realistic truck GPS traces by
  routing through a [Valhalla](https://valhalla.github.io/valhalla/)
  instance, interpolating along the route, layering parking maneuvers,
  adding GPS noise, and optionally injecting configurable data-quality
  errors (signal dropout, multipath, traffic jams, etc.).
- **`trucktrack.partition`** — rewrite a flat parquet of trips as a
  Valhalla-tile-aligned, hive-partitioned dataset (`tier=…/partition_id=…/`)
  with rows sorted by a Hilbert-curve index for spatial locality.

## Install

```bash
pip install trucktrack
```

### From source

```bash
# Requires Python 3.11+ and Rust stable
git clone https://github.com/twedl/trucktrack.git
cd trucktrack
python3 -m venv .venv && source .venv/bin/activate
pip install "maturin>=1.7,<2.0" polars pytest
maturin develop
```

## Usage

### Python API — splitters & I/O

```python
from datetime import timedelta
import polars as pl
import trucktrack

df = pl.read_parquet("tracks.parquet")
# Expected columns: id, time, speed, heading, lat, lon

# Split at observation gaps > 2 minutes (column names are configurable)
result = trucktrack.split_by_observation_gap(
    df, timedelta(minutes=2), min_length=0
)
# Returns df with a segment_id column appended

# Split at detected stops (within 50m for at least 2 minutes)
result = trucktrack.split_by_stops(
    df,
    max_diameter=50.0,
    min_duration=timedelta(minutes=2),
    min_length=0,
)
# Returns all rows with segment_id and is_stop columns.
# Movement segments shorter than min_length are filtered;
# stop segments are always kept.

# Process entirely in Rust (parquet in, parquet out)
trucktrack.split_by_observation_gap_file(
    "input.parquet", "output.parquet", timedelta(minutes=2)
)
trucktrack.split_by_stops_file(
    "input.parquet",
    "output.parquet",
    max_diameter=50.0,
    min_duration=timedelta(minutes=2),
)

# Add speed_mps derived column (zero-copy via pyo3-polars)
result = trucktrack.process_dataframe_in_rust(df)
# …or do the whole thing as parquet → parquet inside Rust
n_rows = trucktrack.process_parquet_in_rust("input.parquet", "output.parquet")
```

### Python API — `trucktrack.generate`

```python
from datetime import datetime, UTC
from trucktrack import TripConfig, generate_trace, traces_to_parquet

config = TripConfig(
    origin=(43.6532, -79.3832),       # (lat, lon)
    destination=(45.5017, -73.5673),
    departure_time=datetime.now(UTC),
    gps_noise_meters=3.0,
    seed=42,
    origin_maneuver="alley_dock",     # parking maneuver type
    destination_maneuver="alley_dock",
    valhalla_url="http://localhost:8002",
)

# Returns list[TracePoint] (lat, lon, speed_mph, heading, timestamp).
# Requires a running Valhalla instance at the configured URL.
points = generate_trace(config)

# Write one or many trips to a single parquet (columns: id, lat, lon,
# speed, heading, time — directly consumable by the splitters above).
traces_to_parquet([(points, config.trip_id)], "trip.parquet")
```

#### Error injection

Generate traces with realistic data-quality issues for testing downstream
pipelines. Errors are configured per-trip via `ErrorConfig` and applied
after trace synthesis:

```python
from trucktrack.generate import ErrorConfig, default_error_profile

# Use the built-in profile: 19 error types at realistic probabilities
# (~1/1000 trips per type — tuned over 1000+ synthetic trips)
config = TripConfig(
    origin=(43.6532, -79.3832),
    destination=(45.5017, -73.5673),
    departure_time=datetime.now(UTC),
    valhalla_url="http://localhost:8002",
    errors=default_error_profile(),
)
points = generate_trace(config)

# Or pick specific errors:
config = TripConfig(
    ...,
    errors=[
        ErrorConfig("signal_dropout", probability=0.5, params={"gap_seconds": 30}),
        ErrorConfig("traffic_jam", probability=1.0),
    ],
)
```

Available error types:

| Category | Error type | Description |
|----------|-----------|-------------|
| GPS | `signal_dropout` | Remove points during signal loss |
| GPS | `cold_start_drift` | Initial position drift after gaps |
| GPS | `multipath` | Reflection-induced position spikes |
| GPS | `frozen_fix` | Position lock-up (repeated coordinates) |
| GPS | `timestamp_glitch` | Duplicate, jump-forward, or jump-backward timestamps |
| GPS | `coordinate_corruption` | Precision loss, lat/lon flip, or swap |
| GPS | `speed_heading_desync` | Speed/heading lag mismatch |
| GPS | `jitter_at_rest` | Brownian motion when stopped |
| Operational | `privacy_shutoff` | Entire trip segment removed |
| Operational | `relay_driving` | Multiple drivers with dwell compression |
| Operational | `yard_dwell` | Pre/post-trip parking |
| Operational | `fuel_rest_stop` | Mid-trip fuel/food break |
| Operational | `weigh_station_stop` | Commercial inspection with approach slowdown |
| Operational | `bobtail_segment` | Tractor-only faster speeds |
| Operational | `off_route_detour` | Temporary course deviation |
| Operational | `loading_dwell` | Cargo transfer dwell |
| Operational | `traffic_jam` | Highway congestion with optional full stops |
| Operational | `device_power_cycle` | GPS unit reboot with gap + drift |
| Operational | `geofence_gap` | Privacy-zone data removal |

### Python API — `trucktrack.partition`

```python
from pathlib import Path
import polars as pl
from trucktrack import (
    assign_partitions,
    partition_existing_parquet,
    write_partitions,
    write_trips_partitioned,
)

# Easiest path: rewrite a flat parquet (id, lat, lon, …) as a hive
# dataset rooted at output_dir. Returns {tier_name: partition_count}.
summary = partition_existing_parquet(
    Path("tracks.parquet"), Path("partitioned/")
)

# Or, partition in-memory generated trips end-to-end:
summary = write_trips_partitioned(
    [(points, "trip-001")], Path("partitioned/")
)

# Single-call in-memory partitioning (adds tier, partition_id, hilbert_idx):
from trucktrack.partition import partition_points
partitioned_df = partition_points(points_df)

# Lower-level: classify trips yourself, then write.
metadata = pl.DataFrame({
    "id": ["trip-001"],
    "centroid_lat": [44.5],
    "centroid_lon": [-76.5],
    "bbox_diag_km": [540.0],
})
metadata = assign_partitions(metadata)  # adds tier, partition_id, hilbert_idx
write_partitions(metadata, points_df, Path("partitioned/"))
```

Tiers are assigned by trip bounding-box diagonal:

| Tier       | Bbox diagonal | Tile bucket                  |
|------------|---------------|------------------------------|
| `local`    | < 100 km      | Valhalla L1 (1x1 deg)       |
| `regional` | 100-800 km    | Valhalla L0 (4x4 deg)       |
| `longhaul` | > 800 km      | Coarse 8x8 deg super-region |

### CLI

```bash
# Add derived columns (speed_mps), output CSV to stdout
trucktrack process tracks.parquet

# Split at observation gaps, write parquet
trucktrack split-gap tracks.parquet --gap 120 -o split.parquet

# Split at stops, output CSV
trucktrack split-stops tracks.parquet --diameter 50 --duration 120 \
    -o stops.csv --format csv

# Process / split commands support -o (file or '-' stdout), --format
# (csv/parquet), and --min-length for the splitters.
trucktrack split-gap tracks.parquet --gap 120 -o result.csv --format csv

# Synthesize a trip via Valhalla and write to parquet
trucktrack generate \
    --origin 43.6532,-79.3832 \
    --destination 45.5017,-73.5673 \
    --departure 2026-04-08T08:00:00 \
    --noise 3.0 --seed 42 \
    --valhalla-url http://localhost:8002 \
    -o trip.parquet

# Rewrite a flat parquet as a Valhalla-tile-aligned hive dataset
trucktrack partition tracks.parquet partitioned/
```

## How it works

| Path | Description |
|------|-------------|
| **Pure Rust** | `split_by_observation_gap_file()` / `split_by_stops_file()` read parquet, process, and write parquet entirely in Rust. No Python objects created. |
| **Python <-> Rust** | `split_by_observation_gap()` / `split_by_stops()` share the Polars DataFrame with Rust via `pyo3-polars` (Arrow C Data Interface, zero-copy on column buffers). The Python `polars`, Rust `polars`, and `pyo3-polars` versions must be kept in sync. |
| **Python only** | `read_parquet()` / `read_dataset()` use polars directly in Python. The `generate` and `partition` subpackages are also pure Python on top of Polars / PyArrow. |

## Project layout

```
src/
  lib.rs              # PyO3 module registration + splitter wrappers
  geo.rs              # Haversine distance (pure Rust math)
  partition.rs        # Tile classification, Hilbert indexing (Rust)
  transform.rs        # add_speed_mps, parquet helpers, error mapping
  splitters/
    gap.rs            # ObservationGapSplitter (Polars lazy expressions)
    stop.rs           # StopSplitter (Polars groupby + Rust sliding window)
python/trucktrack/
  __init__.py         # Public API re-exports
  io.py               # read_parquet, process_dataframe_in_rust, process_parquet_in_rust
  splitters.py        # split_by_observation_gap[_file], split_by_stops[_file]
  cli.py              # CLI entry point (argparse subcommands)
  _core.pyi           # Type stubs for the Rust extension
  generate/
    __init__.py       # Re-exports TripConfig, generate_trace, traces_to_*, ErrorConfig
    models.py         # TripConfig, TracePoint, RouteSegment, ErrorConfig dataclasses
    router.py         # Valhalla HTTP client
    interpolator.py   # Per-segment interpolation, bearing math
    speed_profile.py  # Per-edge speed sampling
    parking.py        # Origin/destination parking maneuvers
    noise.py          # Per-point GPS noise
    trace.py          # Orchestrator + traces_to_csv / traces_to_parquet
    random_trip.py    # Random origin/destination sampling helper
    gps_errors.py     # Physical GPS error injectors (8 types)
    operational_errors.py  # Operational pattern injectors (11 types)
  partition/
    __init__.py       # Re-exports assign_partitions, partition_points, write_*, etc.
    tiles.py          # Valhalla L0/L1 tile math, haversine
    classify.py       # Tier assignment + Hilbert-curve indexing (Polars)
    writer.py         # write_partitions, write_trips_partitioned,
                      #   partition_existing_parquet
data/
  example_tracks.parquet          # 10-row single vehicle example
  splitter_test_tracks.parquet    # 83-row, 3-vehicle test dataset
```

## Dev workflow

| Task | Command |
|------|---------|
| Build | `maturin develop` |
| Tests | `pytest tests/ -v` |
| Lint Python | `ruff check python/ tests/` |
| Format Python | `ruff format python/ tests/` |
| Lint Rust | `cargo clippy --all-targets --all-features -- -D warnings` |
| Format Rust | `cargo fmt --all` |
| Type-check | `mypy python/trucktrack` |
| Build wheel | `maturin build --release` |
