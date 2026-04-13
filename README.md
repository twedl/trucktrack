# trucktrack

High-performance trajectory splitting, generation, and partitioning, powered by Rust.

A Python package implementing logic similar to 
[movingpandas](https://movingpandas.org/) trajectory splitters
(ObservationGapSplitter, StopSplitter) with a Rust backend for speed.
Data flows through [Polars](https://pola.rs/) DataFrames, with the option
to process entirely in Rust (parquet in, parquet out) or share DataFrames
between Python and Rust zero-copy via `pyo3-polars`.

In addition to the Rust splitters, trucktrack ships pure-Python
subpackages for trace generation, spatial partitioning, map-matching,
querying, and visualization.

## Install

```bash
pip install trucktrack
```

Optional extras:

```bash
pip install trucktrack[valhalla]  # local pyvalhalla routing & map-matching
pip install trucktrack[viz]       # folium-based interactive maps
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

## Pipelines

### Split + partition

Process raw GPS traces into a spatially partitioned hive dataset:

```python
from pathlib import Path
from trucktrack import run_pipeline

run_pipeline(Path("data/raw"), Path("data/partitioned"))

# Group input chunks for fewer output files (uses more memory per worker)
run_pipeline(Path("data/raw"), Path("data/partitioned"), group_size=256)

# Compact multi-file partitions into single files after processing
run_pipeline(Path("data/raw"), Path("data/partitioned"), compact=True)
```

To compact an existing dataset without re-running the pipeline:

```python
from trucktrack import compact_partitions

compact_partitions("data/partitioned")
```

### Map-match

Map-match all trips against a local Valhalla instance:

```python
from trucktrack.valhalla.pipeline import run_map_matching

run_map_matching(
    Path("data/partitioned"),
    Path("data/matched"),
    tile_extract="valhalla_tiles.tar",
    # or: config="valhalla.json",
)
```

## Querying

Pull individual trucks or trips without scanning the full dataset.
Each function filters by `chunk_id` (last 3 hex chars of the truck UUID)
to read only the relevant files:

```python
import trucktrack as tt

# Raw traces — filters by chunk_id hive partition
df = tt.scan_raw_truck("data/raw", truck_id).collect()

# Partitioned trips — filters by chunk_id in filename
df = tt.scan_partitioned_truck("data/partitioned", truck_id).collect()
df = tt.scan_partitioned_trip("data/partitioned", trip_id).collect()

# Map-matched results
df = tt.scan_matched_truck("data/matched", truck_id).collect()
df = tt.scan_matched_trip("data/matched", trip_id).collect()
```

### ChunkIndex — persistent file-path index

For repeated queries, build an index once and reload it instantly in
later sessions:

```python
# First time — one rglob, then save to disk
idx = tt.ChunkIndex.build("data/partitioned")
idx.save()  # writes .chunk_index.json

# Later sessions — instant load, no filesystem scan
idx = tt.ChunkIndex.load("data/partitioned")
df = idx.scan_truck(truck_id).collect()
df = idx.scan_trip(trip_id).collect()
```

## Visualization

One-call helpers to query, plot, and serve an interactive map:

```python
from trucktrack.visualize import inspect_truck, inspect_trip

# All trips for a truck — opens a Flask server
inspect_truck("data/partitioned", truck_id)

# Filter to a date range
from datetime import date
inspect_truck("data/partitioned", truck_id,
              date_range=(date(2025, 1, 1), date(2025, 3, 1)))

# Single trip or multiple trips
inspect_trip("data/partitioned", trip_id)
inspect_trip("data/partitioned", [trip_id_1, trip_id_2])

# Use a ChunkIndex for fast lookups on large datasets
idx = tt.ChunkIndex.load("data/partitioned")
inspect_truck("data/partitioned", truck_id, index=idx)

# Raw traces or matched results
inspect_truck("data/raw", truck_id, stage="raw")
inspect_trip("data/matched", trip_id, stage="matched")

# Get the map object without serving (e.g. for Jupyter display)
m = inspect_trip("data/partitioned", trip_id, serve=False)

# Forward kwargs to plot_trace
inspect_trip("data/partitioned", trip_id, color_by="speed")
```

### Multi-stage overlay

Compare raw GPS, trip segments, and map-matched results on one map:

```python
from trucktrack.visualize import inspect_pipeline

# All stages for one truck
inspect_pipeline(
    truck_id,
    raw_dir="data/raw",
    partitioned_dir="data/partitioned",
    matched_dir="data/matched",
)

# Scope to specific trips (raw layer auto-filtered to matching dates)
inspect_pipeline(
    trip_id=[trip_id_1, trip_id_2],
    raw_dir="data/raw",
    partitioned_dir="data/partitioned",
    partitioned_index=idx,
)
```

For more control, use the lower-level `plot_trace`, `plot_trace_layers`,
`save_map`, and `serve_map` functions directly from `trucktrack.visualize`.

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
