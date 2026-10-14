# trucktrack

High-performance trajectory splitting, generation, and map-matching, powered by Rust.

A Python package implementing logic similar to 
[movingpandas](https://movingpandas.org/) trajectory splitters
(ObservationGapSplitter, StopSplitter) with a Rust backend for speed.
Data flows through [Polars](https://pola.rs/) DataFrames, with the option
to process entirely in Rust (parquet in, parquet out) or share DataFrames
between Python and Rust zero-copy via `pyo3-polars`.

In addition to the Rust splitters, trucktrack ships pure-Python
subpackages for trace generation, map-matching,
per-edge truck speeds as Valhalla predicted traffic (`congestion`),
querying, and visualization.  What's new in each release:
[CHANGELOG.md](https://github.com/twedl/trucktrack/blob/main/CHANGELOG.md).

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

### Group by truck, then split

Raw GPS can arrive in any file layout: parquet files anywhere under one
folder, with columns `id` (truck), `time`, `lat`, `lon`, `speed` (km/h)
and `heading`.  First gather each truck's points into one bucket, then
split each bucket's trucks into trips:

```python
from pathlib import Path
from trucktrack import bucket_by_truck, run_pipeline

bucket_by_truck(Path("data/raw"), Path("data/bucketed"))  # n_buckets=256
run_pipeline(Path("data/bucketed"), Path("data/split"))
```

Buckets are a stable hash of the truck id, and every later stage keeps
them (`bucket=K/` folders plus a `_buckets.json`).  `run_pipeline` loads
one bucket per worker, so set `n_buckets` for your data's size.

### Building Valhalla tiles

Map-matching and route generation need a local Valhalla install
(`pip install trucktrack[valhalla]`).  One-time setup (downloads Ontario
from Geofabrik, builds config + admins + tiles):

```bash
python -m trucktrack.valhalla.setup
```

Produces `valhalla_tiles/valhalla.json`, `valhalla_tiles/admin.sqlite`, and
`valhalla_tiles/valhalla_tiles.tar` — all gitignored.  `find_config()`
discovers the json automatically.  Pass `--pbf path.osm.pbf` to reuse an
existing OSM extract.  From Python, `build_tiles(pbf, tile_dir)` in
`trucktrack.valhalla.setup` does the same and returns the config path.

### Map-match

Map-match all trips against a local Valhalla instance:

```python
from trucktrack.valhalla.pipeline import run_map_matching

run_map_matching(
    Path("data/split"),
    Path("data/matched"),
    # config="valhalla.json"  # omit to auto-discover in cwd
)
```

#### Bridging large gaps

Map-match cost scales poorly when a trip has a large inter-point gap —
Meili searches candidate routes across the gap and can spend seconds
or minutes on a single trip.  Opt in to gap-splitting by passing a
`BridgeConfig`: trips are split at gaps, each sub-segment is matched
normally, and gaps are filled in with a single `/route` + `edge_walk`
call to recover way IDs:

```python
from trucktrack.valhalla import BridgeConfig, run_map_matching

run_map_matching(
    Path("data/split"),
    Path("data/matched"),
    bridges=BridgeConfig(max_dist_m=5000, time_s=240, min_dist_m=1000),
)
```

A gap triggers a split when the distance between consecutive points
exceeds `max_dist_m`, or when the time exceeds `time_s` *and* the
distance exceeds `min_dist_m` (the distance floor keeps red-light
stalls from splitting).  The `/route` bridge assumes the truck took a
shortest path through the gap — true detours are invisible.  Per-trip
quality rows pick up `n_bridges`, `max_detour_ratio`,
`total_bridge_m`, and `any_bridge_failed` so downstream can filter.
On any `/route` failure the orchestrator falls back to a single full
HMM call with `breakage_distance` pinned to the base value and sets
`any_bridge_failed=True`.

### Truck speeds as predicted traffic

Match with observations, then load each edge's hourly truck speeds into
the tiles; time-dependent truck routes then slow down where and when the
trucks did:

```python
from trucktrack.congestion import ObservationConfig

run_map_matching(
    Path("data/split"), Path("data/matched"), observations=ObservationConfig()
)
```

```bash
python -m trucktrack.valhalla.traffic data/matched
```

Roads no truck drove take their own Valhalla speed, scaled to how fast
trucks drove roads of the same class and length and shaped by their
class's week (`--measured-only` leaves them alone).

```python
from datetime import datetime
from trucktrack.valhalla import route

route(origin, destination, depart_at=datetime(2026, 3, 10, 8, 0))  # local time
```

[`trucktrack/examples/congestion.py`](https://github.com/twedl/trucktrack/blob/main/python/trucktrack/examples/congestion.py)
runs each step and prints what it produces. It is installed with the
package: copy it from the directory
`python -c "import trucktrack.examples as e; print(e.__path__[0])"`
prints, point its settings at your data, and run it.
[docs/congestion.md](https://github.com/twedl/trucktrack/blob/main/docs/congestion.md)
covers the method, how it measures up on synthetic data, and its caveats.

## Querying

Pull individual trucks or trips from any stage.  On a bucketed stage
the lookup hashes the truck id and opens only its bucket; the raw input,
in whatever layout, is scanned in full:

```python
import trucktrack as tt

df = tt.scan_truck("data/bucketed", truck_id).collect()  # or data/raw
df = tt.scan_truck("data/split", truck_id).collect()
df = tt.scan_trip("data/split", trip_id).collect()
df = tt.scan_trip("data/matched", [trip_id_1, trip_id_2]).collect()
df = tt.scan_truck("data/matched/_quality", truck_id).collect()
```

## Visualization

One-call helpers to query, plot, and serve an interactive map:

```python
from trucktrack.visualize import inspect_truck, inspect_trip

# All trips for a truck — opens a Flask server
inspect_truck("data/split", truck_id)

# Filter to a date range
from datetime import date
inspect_truck("data/split", truck_id,
              date_range=(date(2025, 1, 1), date(2025, 3, 1)))

# Single trip or multiple trips
inspect_trip("data/split", trip_id)
inspect_trip("data/split", [trip_id_1, trip_id_2])

# Any stage: raw points or matched results
inspect_truck("data/bucketed", truck_id)
inspect_trip("data/matched", trip_id)

# Get the map object without serving (e.g. for Jupyter display)
m = inspect_trip("data/split", trip_id, serve=False)

# Forward kwargs to plot_trace
inspect_trip("data/split", trip_id, color_by="speed")
```

### Multi-stage overlay

Compare raw GPS, trip segments, and map-matched results on one map:

```python
from trucktrack.visualize import inspect_pipeline

# All stages for one truck
inspect_pipeline(
    truck_id,
    raw_dir="data/bucketed",
    split_dir="data/split",
    matched_dir="data/matched",
)

# Scope to specific trips (raw layer auto-filtered to matching dates)
inspect_pipeline(
    trip_id=[trip_id_1, trip_id_2],
    raw_dir="data/bucketed",
    split_dir="data/split",
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
