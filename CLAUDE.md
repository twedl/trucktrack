# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Setup

After cloning, install the pre-push hook (runs the full CI check locally):

```
ln -sf ../../scripts/pre-push .git/hooks/pre-push
```

## Build

The package is a PyO3 extension built with maturin. Rust changes are not visible to Python until you rebuild:

```
uv run maturin develop            # debug build, for iterating
uv run maturin develop --release  # required before mypy (loads the compiled .so)
```

## Test

```
uv run pytest tests/ -v
uv run pytest tests/test_splitters.py::test_name -v   # single test
```

## Lint

`scripts/pre-push` runs the canonical sequence — reproduce it exactly when checking a change:

```
cargo fmt --all -- --check
cargo clippy --all-targets --all-features -- -D warnings
uv run ruff check python/ tests/
uv run ruff format --check python/ tests/
uv run maturin develop --release && uv run mypy python/trucktrack
```

## Version check

`pyproject.toml` and `Cargo.toml` versions must match (the pre-push hook enforces this). Bump both when releasing:

```
PY_VER=$(grep -E '^version[[:space:]]*=' pyproject.toml | head -1 | sed 's/.*"\(.*\)".*/\1/')
RS_VER=$(grep -E '^version[[:space:]]*=' Cargo.toml | head -1 | sed 's/.*"\(.*\)".*/\1/')
[ "$PY_VER" = "$RS_VER" ] || echo "MISMATCH: pyproject.toml=$PY_VER Cargo.toml=$RS_VER"
```

## Architecture

trucktrack is a Rust + Python hybrid: hot-path trajectory operations live in Rust (`src/`), while orchestration, trace generation, truck bucketing, map-matching, and visualization live in pure Python (`python/trucktrack/`).

### Rust ↔ Python boundary

- `src/lib.rs` is the PyO3 module (`_core`). Every exported Rust function has two forms: `*_df` (takes/returns a `PyDataFrame` zero-copy via `pyo3-polars`) and `*_file` (reads/writes parquet directly). The file variants let the pipeline stream without materializing chunks Python-side.
- `src/splitters/{gap,stop,traffic}.rs` implement the trajectory operations (ObservationGapSplitter / StopSplitter analogues plus a bearing-based traffic-stop filter). `src/bucket.rs` is the stable truck-id hash behind every `bucket=K/` layout. `src/transform.rs` holds shared error-conversion helpers (`polars_err`, `io_err`).
- Rust releases the GIL: every wrapper goes through `transform::transform_df` / `transform_file`, which run the work inside `py.detach`, so the Python `ThreadPoolExecutor` in `pipeline.py` achieves real parallelism.
- `_core` is never imported by users — `splitters.py`, `io.py`, and `buckets.py` wrap it with Pythonic signatures (e.g. `timedelta` → microseconds). Public names are re-exported from `python/trucktrack/__init__.py`.

### Data flow — raw → bucketed → split → matched

1. **Raw input** (`data/raw/`). Parquet files in **any layout** — only the columns are a contract (`id, time, lat, lon, speed, heading`). Never rely on the provider's file layout or `chunk_id`, or on truck ids being hex UUIDs.
2. **Bucket** (`buckets.bucket_by_truck` → `data/bucketed/`). Streams every input file into `bucket=K/` (`pl.PartitionBy`), K = `bucket_of(id, n_buckets)`, so each truck's whole history is in one bucket. Splitting needs that: a truck spread over several files would be split per file, cutting trips at file edges and reusing trip IDs.
3. **Split** (`pipeline.run_pipeline` → `data/split/`). Per bucket, in parallel: stale-ping + impossible-speed filters → gap-split → stop-split → traffic-filter → compose trip IDs (`{id}_gap{n}_trip{n}`) → `bucket=K/part-NNNN.parquet`, files of about `rows_per_file` (1M) rows with each truck in one file. Files are map-matching's unit of work, memory and resume.
4. **Map-match** (`valhalla/pipeline.run_map_matching` → `data/matched/`). One process task per bucket; per-trip calls to a local Valhalla actor; `quality.py` scores each match. `_quality/` and `_observations/` mirror the buckets and files. Resumable per file.

### Bucket convention

Every stage after raw writes `bucket=K/` folders plus `_buckets.json` (`{"version", "n_buckets"}`). `bucket_by_truck` and `run_pipeline` write it last, so it also marks a finished run; `run_map_matching` writes it first, since it resumes and must check the input matches. `query.scan_truck` / `scan_trip` read it, hash the truck id with `bucket_of`, and open one folder; a dir without it (raw input) is scanned in full via `buckets.data_files`, which skips `_`/`.`-prefixed paths. Stage preconditions live in `buckets.py` (`require_bucketed`, `require_empty`, `bucket_files`, `bucket_number`) — use them rather than re-checking per stage. `bucket_of` (FNV-1a + murmur finaliser in `src/bucket.rs`) must never change — data on disk is laid out by it; the golden values in `tests/test_buckets.py` guard it, and `bucket::HASH_VERSION` (written into `_buckets.json`) must be bumped with any change. Polars' `Expr.hash` is not stable across Polars versions, so never use it for a persisted layout. `bucket_by_truck` and `run_pipeline` refuse a non-empty output dir rather than mixing runs.

### Subpackages

- `generate/` — pure-Python synthetic GPS trace generator (router, speed profile, noise, parking, operational errors). Entry point: `generate_trace`. Does not emit idle points at origin/destination, so endpoint stops aren't detected downstream.
- `valhalla/` — `_actor.py` wraps pyvalhalla; `map_matching.py` / `routing.py` are the callable surfaces; `pipeline.py` is the batch driver; `quality.py` scores matches; `_bridge.py` splits slow trips at large gaps and bridges them with `/route` + `trace_attributes(edge_walk)` (opt in via `run_map_matching(bridges=BridgeConfig(...))`). `valhalla.json` is the **sole** entry point for Valhalla configuration — do **not** read `VALHALLA_TILE_EXTRACT` (or other Valhalla env vars) directly. Use `find_config()` (re-exported from `valhalla/__init__.py`) to auto-discover `./valhalla.json` or `./valhalla_tiles/valhalla.json`, and resolve the tile directory from the loaded config. The file is not checked in; generate it with `python -m trucktrack.valhalla.setup` (`valhalla/setup.py`).
- `congestion/` — per-edge truck speeds for Valhalla predicted traffic. `fixes.py`/`observations.py` flag and weight matched fixes (written to `_observations/` by `run_map_matching(observations=...)`); `aggregate.py` measures each edge's space-mean speed per local time bin (`edge_cells`, nothing borrowed between edges); `impute.py` fills all 168 hours (`historical_speeds`, `flow_speeds`) and gives unmeasured roads their class's speed ratio and week (`speed_ratios`, `class_weeks`). `valhalla/traffic.py` loads them into the tiles (`python -m trucktrack.valhalla.traffic data/matched`), always rebuilding from the pristine `valhalla_tiles.base.tar`; `valhalla/_tiles.py` reads every edge straight from the binary tiles (Valhalla 3.x layout). See `docs/congestion.md`.
- `examples/` — scripts installed with the wheel for users to copy and adapt (`congestion.py` walks every congestion step; settings block at the top). Keep them runnable from any directory: no repo-relative imports, paths only in the settings.
- `visualize/` — folium-based. `_inspect.py` = one-call `inspect_truck`/`inspect_trip`/`inspect_pipeline` helpers; `_map.py` = lower-level `plot_trace`, `plot_trace_layers`, `save_map`, `serve_map`; `_convert.py` = DataFrame → GeoJSON.

### Gotchas when editing

- Adding a new Rust splitter/transform means: implement in `src/splitters/` (or a new module), wire `*_df` + `*_file` wrappers in `src/lib.rs` through `transform_df` / `transform_file` (so they release the GIL), register in the `_core` module, add a Python wrapper in `splitters.py`/`io.py`, re-export from `__init__.py`, then `maturin develop` before tests see it.
- The Rust splitters need a Datetime `time` (any unit, tz-aware or naive) and Float64 `lat`/`lon`/`heading`; `speed` may be any integer or float (providers send integer speeds), and the output keeps its dtype. Other types raise. Congestion reads a tz-naive `time` as UTC.
- Duplicate `(id, time)` rows make the split output vary run to run (seen on `data/trucks_sampling`: 160,453 or 160,454 rows). The Rust sorts use `SortMultipleOptions::default()`, which doesn't keep ties in order.
- **Prefer LazyFrame to eager DataFrame where possible.** Use `pl.scan_parquet` over `pl.read_parquet` so projection and predicate pushdown cut I/O and peak memory. Collect only at the boundary where an eager result is actually needed (Rust splitter call, per-row Python loop, return value).
- **Never interpolate between points with straight lines.** Trajectories belong on road geometry (via Valhalla `trace_route` legs); speed/time interpolation belongs on the route's arc-length, not the great-circle chord. Valhalla's `trace_attributes.shape` is the snapped-input polyline, *not* road geometry — do not use it for visualization or length-based quality metrics. If you need a shape between fixes, fetch it from `trace_route` or reuse `map_match_route_shape` / `map_match_full`.
