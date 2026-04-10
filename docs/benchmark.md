# Benchmark: `trucktrack.split_by_stops` vs `movingpandas.StopSplitter`

Wall-clock time and peak Python memory for both implementations on the same
dataset and parameters, produced by
`examples/benchmark_stop_splitter_movingpandas.py`.

## Setup

- **Dataset**: `data/route.parquet` (826 GPS rows, 5 generated trips), tiled K
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
| 100  | 82,600  | 500   | splitter-only | 9 ms     | 1,294 ms  | **145×** |
| 100  | 82,600  | 500   | end-to-end    | 9 ms     | 2,061 ms  | **235×** |
| 1000 | 826,000 | 5,000 | splitter-only | 300 ms   | 13,116 ms | **44×**  |
| 1000 | 826,000 | 5,000 | end-to-end    | 298 ms   | 20,687 ms | **70×**  |

trucktrack is consistently **~44–145× faster** at the algorithm level and
**~70–235× faster** end-to-end. The speedup ratio is higher at smaller K
because trucktrack's fixed overhead is negligible relative to movingpandas'.

## Memory (tracemalloc, peak Python allocations)

| K    | regime        | trucktrack | movingpandas |
|---:|---|---:|---:|
| 100  | splitter-only | <0.1 MiB   | 7.9 MiB      |
| 100  | end-to-end    | <0.1 MiB   | 27.3 MiB     |
| 1000 | splitter-only | <0.1 MiB   | 77.5 MiB     |
| 1000 | end-to-end    | <0.1 MiB   | 268.8 MiB    |

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
  (~27 MiB at K=100, ~269 MiB at K=1000) is what a polars-native pipeline
  has to budget for if it routes data through movingpandas.
- The trucktrack column is *not* a claim that trucktrack uses less memory
  than movingpandas at the splitter level — Rust allocations are
  structurally excluded. Use OS-level RSS if you need a fair comparison.

## Detailed timings

K = 100 (82,600 rows, 500 trips), 10 repeats:

```
Splitter-only (algorithm)
                       min      mean    median     stdev
  trucktrack        8.90ms    9.34ms    9.20ms    0.39ms
  movingpandas   1293.78ms 1297.80ms 1296.13ms    4.84ms
  speedup (mp / tt, by min): 145.3×

End-to-end (from polars DataFrame)
                       min      mean    median     stdev
  trucktrack        8.79ms    9.49ms    9.55ms    0.52ms
  movingpandas   2061.42ms 2086.58ms 2083.54ms   33.30ms
  speedup (mp / tt, by min): 234.5×
```

K = 1000 (826,000 rows, 5,000 trips), 10 repeats:

```
Splitter-only (algorithm)
                       min      mean    median     stdev
  trucktrack      299.59ms  312.40ms  311.14ms   10.63ms
  movingpandas  13116.27ms13186.34ms13160.30ms   80.75ms
  speedup (mp / tt, by min): 43.8×

End-to-end (from polars DataFrame)
                       min      mean    median     stdev
  trucktrack      297.64ms  321.53ms  320.78ms   13.55ms
  movingpandas  20686.98ms21176.75ms20869.77ms  707.27ms
  speedup (mp / tt, by min): 69.5×
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
agree on the segmentation result at `min_duration ≥ 5 min` with small
`max_diameter`. See
[`movingpandas_equivalence.md`](./movingpandas_equivalence.md) for the
parity finding.
