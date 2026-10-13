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

   `speed(edge, hour) = level(edge) × pattern(hour | road class) ^ depth(edge, time of day)`

   blended with the edge's own measurement for the hour by
   `trucks / (trucks + 1)`. `level` and `depth` are the edge's own, the
   depth separately for the morning peak (before 10:00), midday and the
   evening peak (from 15:00), since commutes lean by direction and
   midday doesn't. Only the shape of the week (when roads of a class
   slow down) is borrowed from other edges, and Monday to Friday share
   it, so no weekday hour hangs on which edges trucks happened to drive
   then. `constrained_speed` and `freeflow_speed` are the harmonic means
   of the 168 hours over 07:00–19:00 and the rest.
4. **Roads no truck drove** (`speed_ratios`, `class_weeks`): every other
   road edge a truck may drive, read straight from the tiles, takes its
   own Valhalla speed, scaled to how fast trucks drove measured roads of
   the same class and length band (pulled to the class's ratio where a
   band has little data), and shaped by its class's week, so it is as
   slow as truck roads like it and slows at rush hour too. Only classes
   with at least 40 hours of measured driving and a dip of more than 5%
   somewhere; the rest keep Valhalla's speeds.
5. **Loading** (`scripts/add_predicted_traffic.py data/matched`, or
   `--measured-only` to skip step 4): writes Valhalla's traffic CSVs,
   runs `valhalla_add_predicted_traffic`, and repacks the tile extract.
   Each run starts again from a pristine copy of the extract
   (`valhalla_tiles.base.tar`). On the sample, step 4 adds 594k edges,
   246 MB of tiles (to 579 MB) and about a minute.

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
| hours it was measured: blended with the model | 4.2 |
| hours it wasn't: the edge's all-week mean | 14.4 |
| hours it wasn't: imputed | 5.4 |

`freeflow_speed` is off by 2.1 km/h, `constrained_speed` by 4.3. The
synthetic rush hours vary by road class, by distance from Toronto, and by
direction (into Toronto in the morning, out in the evening). Borrowing an
edge's OSM way's speed for an unmeasured hour does worse than the model
(13.9 vs 11.2 where the way has one), so ways aren't used. Giving each
weekday hour its own class pattern does worse than sharing one across
Monday to Friday (5.8 vs 5.2 over all hours, 8.9 vs 7.5 at rush hours):
four weeks of trucks put too few in any one hour of the week.

For edges half the trucks never drove, Valhalla's flat speed is off by
19.9 km/h; shaped by the class's week, 10.5 (motorways 22.1 to 8.2,
secondary 17.6 to 15.0, primary 11.0 to 9.5).

On the synthetic sample, trucks drive motorways at 0.98 of Valhalla's
speed at every length, trunk roads at 0.7, and arterials at 0.3-0.55
under 200 m and 0.64-0.77 beyond.

End to end, Valhalla's time-dependent truck ETAs against the generator's
own drive of the same route (4 routes x 5 departure times):

| Loaded | mean \|error\| | bias | peak direction | worst |
|---|---|---|---|---|
| measured edges only | 11.0% | -9.0% | 32.2% | 44.9% |
| + class weeks on Valhalla's speed | 8.3% | -5.9% | 21.6% | 31.0% |
| + speed scaled by class and length | 6.9% | +5.8% | 2.9% | 24.7% |
| + weekdays share the pattern, midday depth | 6.0% | +5.4% | 1.8% | 18.5% |

## Caveats

- **Counter-peak and night trips run slow.** On the synthetic sample,
  peak-direction trips come out right (Brampton → Toronto at Tuesday
  08:00: 80 minutes against 83 driven), counter-peak ones slow (at
  17:00: 63 against 53) and night trips 3-8% slow. That route is all
  measured edges, and their raw 5 s cells alone give 58 minutes at
  17:00: about half the gap is in the measurements themselves, as at
  night, the rest is the model.
- **Unmeasured roads take their class's average.** Their speed and rush
  hour come from measured roads of the same class and length, in both
  directions. Scaling changes every unmeasured road's free-flow speed,
  not just its rush hour. Classes with too little data or no rush hour
  (here residential, service, unclassified) keep Valhalla's speeds
  entirely; `--measured-only` leaves every unmeasured road alone.
- **Truck speeds apply to every costing.** Predicted, constrained and
  free-flow speeds belong to the edge, not the vehicle, so car routes on
  the same tiles get truck speeds too. They are also not capped by
  Valhalla's truck speed limit.
- **Signal delay counts twice on urban edges.** Measured speeds include
  time spent waiting at lights, and Valhalla adds its own intersection
  costs on top. Scaling unmeasured roads by measured ratios spreads this
  to every road of a scaled class.
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
  by direction (each dip capped at 85%), the same every weekday but for
  Monday's and Friday's slightly lighter peaks. Real congestion that
  departs from that form leans more on each edge's own measurements; a
  fleet with many trucks in every hour of the week can let each weekday
  keep its own shape (`ImputeConfig.weekday_seconds`).
