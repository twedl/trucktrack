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

trucktrack is a Rust + Python hybrid: hot-path trajectory operations live in Rust (`src/`), while orchestration, trace generation, partitioning, map-matching, and visualization live in pure Python (`python/trucktrack/`).

### Rust ↔ Python boundary

- `src/lib.rs` is the PyO3 module (`_core`). Every exported Rust function has two forms: `*_df` (takes/returns a `PyDataFrame` zero-copy via `pyo3-polars`) and `*_file` (reads/writes parquet directly). The file variants let the pipeline stream without materializing chunks Python-side.
- `src/splitters/{gap,stop,traffic}.rs` implement the trajectory operations (ObservationGapSplitter / StopSplitter analogues plus a bearing-based traffic-stop filter). `src/partition.rs` handles Valhalla-tile classification and Hilbert indexing. `src/transform.rs` holds shared error-conversion helpers (`polars_err`, `io_err`).
- Rust releases the GIL, so the Python `ThreadPoolExecutor` in `pipeline.py` achieves real parallelism.
- `_core` is never imported by users — `splitters.py`, `io.py`, and `partition/` wrap it with Pythonic signatures (e.g. `timedelta` → microseconds). Public names are re-exported from `python/trucktrack/__init__.py`.

### Data flow — raw → partitioned → matched

Three-stage funnel, each stage writing a different hive layout:

1. **Raw ingestion** (`data/raw/`). Input parquet chunks, hive-partitioned by `chunk_id` (last 2 hex chars of truck UUID).
2. **Split + partition** (`pipeline.run_pipeline` → `data/partitioned/`). Per chunk, in parallel: gap-split → stop-split → traffic-filter → compose trip IDs (`{id}_gap{n}_trip{n}`) → classify into spatial tier (local/regional/longhaul) → write to `tier=.../partition_id=.../{chunk}.parquet`. Workers write **chunk-unique filenames** into shared tile dirs so overlapping tiles accumulate rather than overwrite. `compact_partitions` is the safe merge step (temp file → atomic rename → unlink).
3. **Map-match** (`valhalla/pipeline.run_map_matching` → `data/matched/`). Per-trip calls to a local Valhalla actor; `quality.py` scores each match.

### Chunk-id convention

Many filenames and partitions key off `chunk_id` = last 2 hex chars of the truck UUID. Query helpers in `query.py` (`scan_raw_truck`, `scan_partitioned_truck`, etc.) exploit this to avoid full scans — they compute the chunk from the truck id and only open matching files. `ChunkIndex` caches the `rglob` result to `.chunk_index.json` so repeat sessions skip the filesystem walk. When compressing 3-char chunk_ids to 2, slice `[-2:]` — partitioning and querying must agree or lookups miss.

### Subpackages

- `generate/` — pure-Python synthetic GPS trace generator (router, speed profile, noise, parking, operational errors). Entry point: `generate_trace`. Does not emit idle points at origin/destination, so endpoint stops aren't detected downstream.
- `partition/` — `classify.py` (tier assignment + Hilbert), `tiles.py` (Valhalla L0/L1 tile math), `writer.py` (hive writers). Mirrors Rust `partition.rs` for DataFrames that stay in Python.
- `valhalla/` — `_actor.py` wraps pyvalhalla; `map_matching.py` / `routing.py` are the callable surfaces; `pipeline.py` is the batch driver; `quality.py` scores matches. `valhalla.json` is the canonical entry point for Valhalla configuration — do **not** read `VALHALLA_TILE_EXTRACT` (or other Valhalla env vars) directly; load `valhalla.json` and resolve the tile directory from its config.
- `visualize/` — folium-based. `_inspect.py` = one-call `inspect_truck`/`inspect_trip`/`inspect_pipeline` helpers; `_map.py` = lower-level `plot_trace`, `plot_trace_layers`, `save_map`, `serve_map`; `_convert.py` = DataFrame → GeoJSON.

### Gotchas when editing

- Adding a new Rust splitter/transform means: implement in `src/splitters/` (or a new module), wire `*_df` + `*_file` wrappers in `src/lib.rs`, register in the `_core` module, add a Python wrapper in `splitters.py`/`io.py`, re-export from `__init__.py`, then `maturin develop` before tests see it.
- The Rust splitters assume tz-naive datetime columns — passing a tz-aware column panics.
- `_write_chunk` in `pipeline.py` is a serial-I/O bottleneck on networked filesystems (k8s); batch/buffer writes rather than looping `write_parquet` per partition.
