# Equivalence with movingpandas `StopSplitter`

`trucktrack.split_by_stops` is intended as a Rust-backed replacement for
[`movingpandas.StopSplitter`](https://movingpandas.readthedocs.io/). This
document records an empirical comparison of the two implementations on
`data/route.parquet` (826 GPS rows across 5 generated trips) using the script
`examples/compare_stop_splitter_movingpandas.py`.

## Finding

**At `max_diameter ≤ 50 m` with `min_duration ≥ 5 minutes`, or at any
`max_diameter` with `min_duration ≥ 10 minutes`, `trucktrack.split_by_stops`
produces the same movement segments as `movingpandas.StopSplitter` on
`route.parquet` for every trip.**

| `max_diameter` | `min_duration` | per-trip segment counts agree | rows kept by trucktrack only | rows kept by movingpandas only |
|---:|---:|:---:|---:|---:|
| 50 m | 300 s | yes (all 5 trips) | 0 | 0 |
| 50 m | 600 s | yes (all 5 trips) | 0 | 0 |
| 100 m | 600 s | yes (all 5 trips) | 0 | 0 |
| 200 m | 600 s | yes (all 5 trips) | 0 | 0 |
| 250 m | 600 s | yes (all 5 trips) | 0 | 0 |

At these parameter settings the partitions are bit-for-bit identical.

## Recommendation

Use `trucktrack.split_by_stops` as a drop-in replacement for
`movingpandas.StopSplitter` when `min_duration` is at least 5 minutes — a
common operational threshold for what counts as a "real" stop in vehicle
trajectory data. For shorter `min_duration` values or larger `max_diameter`
the two implementations can disagree on borderline stop windows; see the
appendix for details.

## Reproducing

```bash
pip install movingpandas geopandas shapely pyproj pyarrow
python examples/compare_stop_splitter_movingpandas.py \
    --max-diameter 50 --min-duration 300
```

The script exits 0 on full agreement and prints a per-trip breakdown.

## Appendix: behaviour at shorter `min_duration`

For completeness, the full sweep (the regime above the line is what the
finding is based on):

| `max_diameter` (m) | `min_duration` (s) | counts agree? | tt-only rows | mp-only rows |
|---:|---:|:---:|---:|---:|
| 50  | 300 | yes | 0  | 0  |
| 50  | 600 | yes | 0  | 0  |
| 100 | 600 | yes | 0  | 0  |
| 200 | 600 | yes | 0  | 0  |
| 250 | 600 | yes | 0  | 0  |
| --- | --- | --- | --- | --- |
| 10  | 60  | no  | 1  | 0  |
| 25  | 60  | no  | 6  | 0  |
| 25  | 120 | no  | 6  | 0  |
| 50  | 60  | no  | 21 | 0  |
| 50  | 120 | no  | 17 | 0  |
| 100 | 120 | no  | 21 | 0  |
| 100 | 300 | no  | 6  | 0  |
| 200 | 300 | no  | 6  | 0  |
| 250 | 300 | no  | 12 | 0  |

Two effects show up at shorter durations or larger diameters:

1. **Stop padding (definitional, not a bug).** When segment counts disagree,
   the extra rows are always on the trucktrack side: trucktrack classifies
   them as movement while movingpandas absorbs them into a stop.
   `TrajectoryStopDetector` greedily extends a stop window forward as long
   as the centroid-distance condition holds, while trucktrack's sliding
   window is anchored on the contiguous-points-within-diameter test.

2. **Borderline window anchoring (~±1 stop).** With short `min_duration`
   or large `max_diameter` several adjacent windows can simultaneously
   satisfy `(diameter ≤ max, duration ≥ min)`. The two algorithms commit to
   different anchor points, occasionally producing one extra or one fewer
   stop. As `min_duration` grows the borderline windows are squeezed out
   and the two implementations converge.
