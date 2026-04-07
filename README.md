# trucktrack

High-performance trajectory splitting and analysis, powered by Rust.

A Python package that replicates and extends
[movingpandas](https://movingpandas.org/) trajectory splitters
(ObservationGapSplitter, StopSplitter) with a Rust backend for speed.
Data flows through [Polars](https://pola.rs/) DataFrames, with the option
to process entirely in Rust (parquet in, parquet out) or share DataFrames
between Python and Rust zero-copy via `pyo3-polars`.

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

### Python API

```python
from datetime import timedelta
import polars as pl
import trucktrack

df = pl.read_parquet("tracks.parquet")
# Expected columns: id, time, speed, heading, lat, lon

# Split at observation gaps > 2 minutes
result = trucktrack.split_by_observation_gap(df, timedelta(minutes=2))
# Returns df with a segment_id column appended

# Split at detected stops (within 50m for at least 2 minutes)
result = trucktrack.split_by_stops(
    df, max_diameter=50.0, min_duration=timedelta(minutes=2)
)
# Returns movement segments only, with segment_id column

# Process entirely in Rust (parquet in, parquet out)
trucktrack.split_by_observation_gap_file(
    "input.parquet", "output.parquet", timedelta(minutes=2)
)

# Add speed_mps derived column
result = trucktrack.process_dataframe_in_rust(df)
```

### CLI

```bash
# Add derived columns (speed_mps), output CSV to stdout
trucktrack process tracks.parquet

# Split at observation gaps, write parquet
trucktrack split-gap tracks.parquet --gap 120 -o split.parquet

# Split at stops, output CSV
trucktrack split-stops tracks.parquet --diameter 50 --duration 120

# All subcommands support -o (file or stdout) and --format (csv/parquet)
trucktrack split-gap tracks.parquet --gap 120 -o result.csv --format csv
```

## How it works

| Path | Description |
|------|-------------|
| **Pure Rust** | `split_by_observation_gap_file()` / `split_by_stops_file()` read parquet, process, and write parquet entirely in Rust. No Python objects created. |
| **Python <-> Rust** | `split_by_observation_gap()` / `split_by_stops()` share the Polars DataFrame with Rust via `pyo3-polars` (Arrow C Data Interface, zero-copy on column buffers). The Python `polars`, Rust `polars`, and `pyo3-polars` versions must be kept in sync. |
| **Python only** | `read_parquet()` / `read_dataset()` use polars directly in Python. |

## Project layout

```
src/
  lib.rs              # PyO3 module registration + splitter wrappers
  geo.rs              # Haversine distance (pure Rust math)
  transform.rs        # add_speed_mps, IPC/parquet helpers
  splitters/
    gap.rs            # ObservationGapSplitter (Polars lazy expressions)
    stop.rs           # StopSplitter (Polars groupby + Rust sliding window)
python/trucktrack/
  __init__.py         # Public API re-exports
  io.py               # read_parquet, process_dataframe_in_rust
  splitters.py        # split_by_observation_gap, split_by_stops
  cli.py              # CLI entry point (argparse subcommands)
  _core.pyi           # Type stubs for the Rust extension
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
