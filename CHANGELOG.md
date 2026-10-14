# Changelog

## 0.2.1 (2026-10-14)

### Truck buckets replace spatial partitions

The pipeline no longer relies on how the raw input is laid out. A new
first step, `bucket_by_truck(raw_dir, bucketed_dir, n_buckets=256)`,
streams parquet files in any folder layout into `bucket=K/` folders, K
a stable hash of the truck id, so each truck's whole history is in one
bucket. Split, map-match, `_quality/` and `_observations/` all keep those
buckets, and a `_buckets.json` records how many.

This fixes trip IDs when a truck's points span several raw files (by
day, say): each file was split on its own, so trips were cut at file
boundaries and different trips got the same ID (`{id}_gap0_trip0`).

### Other changes

- The Rust splitters and filters release the GIL, so `run_pipeline`'s
  threads now run them in parallel (1.85× on 4 threads in a synthetic
  benchmark; before, threads gained nothing).

### Breaking changes

- **`run_pipeline` takes `bucket_by_truck` output**, writes
  `bucket=K/part-NNNN.parquet` (files of about `rows_per_file=1_000_000`
  rows, each truck in one file) instead of `tier=.../partition_id=.../`,
  and returns counts of buckets, rows and trips. `group_size`, `compact`,
  `rebalance` and their settings are gone, as are `compact_partitions`
  and `rebalance_partitions`. Its output folder must be empty.
- **`run_map_matching` takes `run_pipeline` output** and runs one task
  per bucket; `chunks_per_worker` and the spatially ordered work units
  are gone. `_timing.parquet` has a `bucket` column instead of `tier` and
  `partition_id`.
- **`trucktrack.partition` is gone** (tiers, Valhalla tile keys, Hilbert
  ordering), with the `trucktrack partition` command.
- **Lookups:** `scan_truck(dir, truck_id)` and `scan_trip(dir, trip_id)`
  (one ID or a list) work on any stage and replace `scan_raw_truck`, `scan_partitioned_*`,
  `scan_matched_*` and `ChunkIndex`. Raw input without `_buckets.json` is
  scanned in full, whatever its layout. `inspect_truck` and
  `inspect_trip` lose `index` and `stage`; `inspect_pipeline` takes
  `split_dir` instead of `partitioned_dir` and loses the `*_index`
  arguments; `inspect.load_truck_trace` loses `index`.

## 0.2.0 (2026-10-14)

### Truck speeds as Valhalla predicted traffic

New `trucktrack.congestion` module: measure how fast trucks drove each
road, every hour of the week, and load those speeds into the Valhalla
tiles so time-dependent truck routes slow down where and when trucks
did. See [docs/congestion.md](https://github.com/twedl/trucktrack/blob/main/docs/congestion.md)
for the method, how it measures up on synthetic data, and its caveats.

- **Observations.** `run_map_matching(..., observations=ObservationConfig())`
  writes `_observations/`: one row per GPS fix on a road edge, with the
  time it stands for and flags for fixes that don't measure traffic
  (`dwell`, `wrong_way`, `speed_mismatch`).
- **Measurement.** `edge_cells` gives each edge's space-mean speed per
  local hour of the week (or any other time bin), from its own fixes only.
- **Imputation.** `fit_model` / `historical_speeds` fill in all 168
  hours of every measured edge; `flow_speeds` gives Valhalla's
  `freeflow_speed` and `constrained_speed`; `speed_ratios` and
  `class_weeks` give roads no truck drove their road class's truck speed
  and rush hour.
- **Loading.** `trucktrack.valhalla.traffic.add_predicted_traffic`, or
  `python -m trucktrack.valhalla.traffic data/matched`, rebuilds the
  tiles with the speeds, always starting again from the tiles as first
  packed. `route(..., depart_at=datetime(...))` routes with them.
- **Example.** `trucktrack/examples/congestion.py`, installed with the
  package, runs every step, from an OSM extract and raw GPS to timed
  routes. Copy it from the directory
  `python -c "import trucktrack.examples as e; print(e.__path__[0])"`
  prints, change its settings, and run it.

Validated on synthetic data only (`data/trucks_sampling`: 200 trucks,
six trips each over four weeks).

### Other changes

- Tiles can be built from the installed package:
  `python -m trucktrack.valhalla.setup [--pbf extract.osm.pbf]`, or
  `trucktrack.valhalla.setup.build_tiles()` from Python. Each build gets
  a tile build ID, and observations carry the ID of the tiles they were
  matched on.
- `valhalla.parse_fix_matches` turns a `trace_attributes` response into
  one entry per input point.
- Synthetic traces have irregular per-truck reporting schedules
  (`SamplingConfig`), and rush-hour congestion that deepens towards
  Toronto and leans by direction.
- Supports polars 2.0 and Python 3.14.

### Breaking changes

- **Rebuild your Valhalla tiles.** pyvalhalla is now `>=3.9.1,<4`, and
  Valhalla 3.8 changed the tile format: run
  `python -m trucktrack.valhalla.setup`, then re-run map matching: new
  tiles get a new tile build ID (and edge IDs can change), and loading
  refuses observations matched on other tiles.
- **Map matching uses truck costing** (`DEFAULT_TRUCK_COSTING`
  dimensions), as routing does, instead of `auto`. Matched routes can
  differ from 0.1.x.
- **Synthetic trace speeds are km/h**: `TracePoint.speed_mph` is now
  `speed_kmh`.
- **polars `>=1.31`** (was 1.10; releases before 1.31 already failed the
  test suite).
- `scripts/setup_valhalla.py` and `scripts/add_predicted_traffic.py` are
  gone from the repo; use the `python -m` commands above.
