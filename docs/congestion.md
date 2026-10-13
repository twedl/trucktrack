# Truck speeds as Valhalla predicted traffic

`trucktrack.congestion` turns map-matched truck GPS into per-edge speeds
for every hour of the week, and `trucktrack.valhalla.traffic` loads them
into the Valhalla tiles. Time-dependent truck routes then slow down where
and when the trucks did.

## Pipeline

1. **Observations** (`run_map_matching(..., observations=ObservationConfig())`):
   one row per GPS fix on a graph edge, with the time it stands for (`w`)
   and flags for fixes that don't measure traffic (`dwell`, `wrong_way`,
   `speed_mismatch`).
2. **Measurement** (`edge_cells`): each edge's space-mean speed,
   Σ speed·w / Σ w, per local hour of the week (Sunday 00:00 first, in
   `America/Toronto`, DST-aware), with its seconds, fixes and effective
   number of trucks. Only what trucks saw on that edge: an hour no truck
   drove has no row. `edge_cells(..., by=flow_window())` measures
   Valhalla's two fallback windows instead.
3. **Imputation** (`historical_speeds`, `flow_speeds`): every measured
   edge gets all 168 hours from a model of the edge,

   `speed(edge, hour) = level(edge) × pattern(hour | road class) ^ depth(edge, half of day)`

   blended with the edge's own measurement for the hour by
   `trucks / (trucks + 1)`. `level` and `depth` are the edge's own; only
   the shape of the week (when roads of a class slow down) is borrowed
   from other edges. `constrained_speed` and `freeflow_speed` are the
   harmonic means of the 168 hours over 07:00–19:00 and the rest.
4. **Loading** (`scripts/add_predicted_traffic.py data/matched`): writes
   Valhalla's traffic CSVs, runs `valhalla_add_predicted_traffic`, and
   repacks the tile extract. Each run starts again from a pristine copy
   of the extract (`valhalla_tiles.base.tar`).

Route with a departure time to use the speeds:

```python
from datetime import datetime
from trucktrack.valhalla import route

route(origin, destination, depart_at=datetime(2026, 3, 10, 8, 0))  # local time
```

Without `depart_at`, Valhalla uses each edge's `constrained_speed`.

## Measured on synthetic data

200 trucks × 6 trips over four weeks, each truck on its own reporting
schedule. Estimates come from half the trucks and are scored against the
other half's 5 s data (|error| in km/h, weighted by driving time):

| Hourly speed of an edge | |error| |
|---|---|
| hours it was measured: its own cell | 5.0 |
| hours it was measured: blended with the model | 4.5 |
| hours it wasn't: the edge's all-week mean | 14.4 |
| hours it wasn't: imputed | 6.5 |

`freeflow_speed` is off by 3.0 km/h, `constrained_speed` by 4.8. The
synthetic rush hours vary by road class, by distance from Toronto, and by
direction (into Toronto in the morning, out in the evening). Borrowing an
edge's OSM way's speed for an unmeasured hour does worse than the model
(13.9 vs 11.2 where the way has one), so ways aren't used.

## Caveats

- **Only measured edges slow down.** Edges no truck drove keep Valhalla's
  free-flow defaults, so at rush hour a time-dependent route can leave
  congested truck corridors for unmeasured roads that look fast. On the
  synthetic sample, Brampton → Toronto at Tuesday 08:00 takes 46 minutes
  with 5% of its length on loaded edges; driven on the usual route it
  takes 83. Giving unmeasured edges their class's pattern would fix the
  asymmetry but puts truck-shaped congestion on roads with no data.
- **Truck speeds apply to every costing.** Predicted, constrained and
  free-flow speeds belong to the edge, not the vehicle, so car routes on
  the same tiles get truck speeds too. They are also not capped by
  Valhalla's truck speed limit.
- **Signal delay counts twice on urban edges.** Measured speeds include
  time spent waiting at lights, and Valhalla adds its own intersection
  costs on top.
- **One time zone.** Every edge is binned in `America/Toronto`; edges west
  of 90°W (Kenora, Rainy River) run on Central time and would be an hour
  off.
- **Hourly detail.** The 168 hourly speeds are interpolated to Valhalla's
  2016 five-minute buckets and stored as 200 DCT coefficients, which
  can't hold detail much finer than an hour anyway.
- **Edges only ever seen stopped** are left at Valhalla's defaults.
- **Tile builds.** Observations carry the `tile_build_id` of the tiles
  they were matched on; edge ids mean nothing on other tiles, so loading
  refuses speeds from another build. Rebuilding the tiles with
  `scripts/setup_valhalla.py` means re-matching.
- **Validated on synthetic data only.** The settings (`ImputeConfig`) are
  the best of those tried on the synthetic sample, whose congestion is
  close to the model's form: a class shape deepened by place and leaning
  by direction (each dip capped at 85%). Real congestion that departs
  from that form leans more on each edge's own measurements.
