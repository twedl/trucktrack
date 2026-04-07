# Benchmark: `trucktrack.split_by_stops` vs `movingpandas.StopSplitter`

Wall-clock time and peak Python memory for both implementations on the same
dataset and parameters, produced by
`examples/benchmark_stop_splitter_movingpandas.py`.

## Setup

- **Dataset**: `data/route.parquet` (1,884 GPS rows, 5 real trips), tiled K
  times with disjoint trip ids and forward-shifted timestamps so replicas do
  not interleave.
- **Parameters**: `max_diameter = 250 m`, `min_duration = 300 s`.
- **Method**: 1 untimed warm-up, then 10 timed passes (`time.perf_counter`)
  per implementation, then 1 additional traced pass for memory
  (`tracemalloc`). Reported `min` is the most reproducible figure.
- **Two regimes**:
  1. **Splitter-only ("algorithm")**: the movingpandas `TrajectoryCollection`
     is built and projected to UTM *before* the timed region. Head-to-head
     comparison of the two stop-detection algorithms.
  2. **End-to-end ("operational")**: starts from the polars `DataFrame`
     loaded from parquet. For movingpandas this includes
     `to_pandas()` → `GeoDataFrame` → `estimate_utm_crs()` →
     `to_crs(utm)` → `TrajectoryCollection` → `StopSplitter.split`. The real
     cost of using movingpandas in a polars pipeline.

## Headline numbers

| K | rows | trips | regime | trucktrack (min) | movingpandas (min) | speedup |
|---:|---:|---:|---|---:|---:|---:|
| 100  | 188,400   | 500   | splitter-only | 244 ms   | 3,478 ms  | **14.3×** |
| 100  | 188,400   | 500   | end-to-end    | 243 ms   | 4,832 ms  | **19.9×** |
| 1000 | 1,884,000 | 5,000 | splitter-only | 3,063 ms | 34,717 ms | **11.3×** |
| 1000 | 1,884,000 | 5,000 | end-to-end    | 3,089 ms | 48,472 ms | **15.7×** |

trucktrack is consistently **~11–14× faster** at the algorithm level and
**~16–20× faster** end-to-end. Speedup ratios are stable across K=100 → K=1000,
so the comparison is not an artifact of small-dataset noise.

## Memory (tracemalloc, peak Python allocations)

| K    | regime        | trucktrack | movingpandas |
|---:|---|---:|---:|
| 100  | splitter-only | <0.1 MiB   | 11.6 MiB     |
| 100  | end-to-end    | <0.1 MiB   | 56.8 MiB     |
| 1000 | splitter-only | <0.1 MiB   | 114.4 MiB    |
| 1000 | end-to-end    | <0.1 MiB   | 565.2 MiB    |

**Important caveat.** `tracemalloc` only sees Python allocations.
trucktrack's Rust working set is invisible to it. The result DataFrame is
also handed back to Python via the Arrow C Data Interface (`pyo3-polars`,
zero-copy on column buffers), so its column storage does not show up as a
Python allocation either. The figure reported for trucktrack is therefore
near-zero by construction and is **not** a measure of total memory use.

What is meaningful:

- The **movingpandas** column is a real cost — pandas frames, GeoPandas
  geometries, shapely `Point` objects, and the projected
  `TrajectoryCollection` all live in the Python heap. The end-to-end row
  (~57 MiB at K=100, ~565 MiB at K=1000) is what a polars-native pipeline
  has to budget for if it routes data through movingpandas.
- The trucktrack column is *not* a claim that trucktrack uses less memory
  than movingpandas at the splitter level — Rust allocations are
  structurally excluded. Use OS-level RSS if you need a fair comparison.

## Detailed timings

K = 100 (188,400 rows, 500 trips), 10 repeats:

```
Splitter-only (algorithm)
                       min      mean    median     stdev
  trucktrack      243.85ms  267.61ms  248.83ms   38.00ms
  movingpandas   3478.40ms 3522.93ms 3490.82ms   90.06ms
  speedup (mp / tt, by min): 14.3×

End-to-end (from polars DataFrame)
                       min      mean    median     stdev
  trucktrack      242.58ms  243.69ms  243.47ms    0.90ms
  movingpandas   4831.85ms 4850.24ms 4849.63ms   13.64ms
  speedup (mp / tt, by min): 19.9×
```

K = 1000 (1,884,000 rows, 5,000 trips), 10 repeats:

```
Splitter-only (algorithm)
                       min      mean    median     stdev
  trucktrack     3063.20ms 3276.38ms 3245.75ms  226.85ms
  movingpandas  34716.60ms35578.82ms34976.14ms 1125.23ms
  speedup (mp / tt, by min): 11.3×

End-to-end (from polars DataFrame)
                       min      mean    median     stdev
  trucktrack     3088.96ms 3135.75ms 3140.44ms   33.99ms
  movingpandas  48472.45ms50866.04ms51604.55ms 1865.76ms
  speedup (mp / tt, by min): 15.7×
```

## Reproducing

```bash
pip install movingpandas geopandas shapely pyproj pyarrow
python examples/benchmark_stop_splitter_movingpandas.py --repeat-dataset 100
python examples/benchmark_stop_splitter_movingpandas.py --repeat-dataset 1000
```

CLI flags: `--max-diameter` (m, default 250), `--min-duration` (s, default
300), `--repeats` (default 10), `--repeat-dataset` (default 1).

## Hardware

Numbers above were collected on a single developer machine (Apple Silicon,
macOS). Absolute timings will vary; the speedup *ratios* are the meaningful
quantity.

## Correctness

These performance numbers are only meaningful because the two implementations
agree on the segmentation result at `min_duration ≥ 5 min`. See
[`movingpandas_equivalence.md`](./movingpandas_equivalence.md) for the
parity finding.
