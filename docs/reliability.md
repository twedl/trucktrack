# Truck travel times along a corridor

`trucktrack.valhalla.build_corridor` and `trucktrack.congestion.corridor_passes`
time every truck over a highway corridor, each section of it and each FHWA
reporting segment; `travel_time_reliability` gives the distribution of those
times by FHWA's truck travel time reliability method.
[`trucktrack/examples/corridor.py`](https://github.com/twedl/trucktrack/blob/main/python/trucktrack/examples/corridor.py)
runs it from raw GPS for the 401 between Windsor and Toronto.

## What is measured

A **pass** is one truck driving a stretch of the corridor. The truck's fixes on
the corridor's edges, in time order, make a run until a fix lands off the
highway or the truck goes unseen for more than `max_gap_s` (10 minutes). The
time it crosses a point is interpolated between its fixes either side, on the
distance along the highway. A run that crosses both ends of a stretch is a
pass: when it entered, and how long it took. Runs follow the truck, not the
trip, so a trip split by a long stop in traffic is still one pass, and that
stop counts. A stop off the highway (a service centre, an exit) ends the run.

The **corridor** is a truck route through named points, routed each way
on its own carriageway. Every motorway edge running along a route, within 300 m
of it, belongs to that direction: express and collector lanes, and the
transfers between them, are the highway.

```python
from trucktrack.congestion import corridor_passes, travel_time_reliability, tttr_index
from trucktrack.valhalla import build_corridor

corridor = build_corridor(points, before=before, after=after)  # {name: (lat, lon)}
spans = corridor.spans(urban=[(lat, lon, radius_m), ...])
edges = corridor.edges(observed_edge_ids)
passes = corridor_passes(obs, edges, spans)
reliability = travel_time_reliability(passes)
```

`spans` are the stretches timed in each direction: the whole corridor, each
section (point to point) and FHWA reporting segments, each section cut into
equal pieces of at most 1 mile in the `urban` circles and 10 miles elsewhere
(23 CFR 490.103(e)).

A truck is timed across a point only if it is seen on the highway on both sides
of it. So the routes run on past the first and last points (`before`, `after`),
and the points sit away from interchanges (a point at a junction can pull the
route onto a ramp or the other carriageway) and far enough from where trucks
join that they report a fix between. Check the routes on a map:
`corridor.py` writes them as WKT.

## FHWA's method

The Truck Travel Time Reliability measure (23 CFR 490.611):

- Each stretch's passes are averaged per **15 minutes** of entry, to the
  second: FHWA's truck travel time data set.
- Each 15 minutes falls in one of five **periods** of the week, on local time:
  weekdays 06–10 (AM peak), 10–16 (Midday) and 16–20 (PM peak), weekends
  06–20, and every night 20–06 (Overnight).
- Within a period, the averages are ranked: `p50_s` is FHWA's "normal" truck
  travel time, `p95_s` the 95th percentile (linear between ranks, as a
  spreadsheet's PERCENTILE), and `tttr` their ratio to the hundredth.
- `tttr_index` is FHWA's TTTR Index: each reporting segment's worst period,
  weighted by length (23 CFR 490.613).

FHWA defines TTTR on reporting segments only; `travel_time_reliability` gives
the same statistics for sections and the whole corridor, with the 10th and
80th percentiles, the mean, and the buffer index ((95th − mean) / mean, from
FHWA's travel time reliability guidance). FHWA's data set comes from NPMRDS,
many probes per 15 minutes; a fleet's passes are mostly one truck per 15
minutes, so its percentiles include how differently trucks drive.

## Measured on synthetic data

150 trucks shuttling for eight weeks, 70% between Windsor and Toronto and the
rest between Windsor and London, or London or Cambridge and Toronto: 1,200
drives. Each drive is generated twice, at the truck's own reporting schedule
(mostly 60 s; some 1–59 s; long gaps) and every 5 s. The generator's
congestion doesn't recur here (`DelayConfig`): each day's rush hours run deeper
or shallower (e^N(0, 0.25)), and highway incidents, 0.1 per 10 km square a day
each way, slow every truck through the square to 20–60% for 0.5–2 hours.

The 401 from 7 km past Huron Church Road to the 427 is 339 km, in sections of
168, 50, 48, 42 and 33 km and 79 reporting segments each way. From the 5 s
data, every Windsor–Toronto drive from the Ambassador Bridge makes a full pass
(66 trucks, 264 passes each way); the trucks based in east Windsor join the 401
past the first point. From the trucks' own schedules, 162–164: a pass with a gap of
more than 10 minutes is dropped.

Whole corridor, minutes, from the 5 s data:

| Direction | Period | 15-min bins | 10th | 50th | 80th | 95th | TTTR |
|---|---|---|---|---|---|---|---|
| Windsor → Toronto | AM peak | 36 | 214 | 233 | 248 | 273 | 1.17 |
| | Midday | 36 | 210 | 220 | 230 | 237 | 1.07 |
| | PM peak | 31 | 195 | 198 | 206 | 211 | 1.07 |
| | Weekend | 43 | 196 | 203 | 212 | 225 | 1.11 |
| | Overnight | 113 | 195 | 195 | 220 | 271 | 1.39 |
| Toronto → Windsor | AM peak | 26 | 206 | 211 | 215 | 221 | 1.04 |
| | Midday | 58 | 219 | 255 | 283 | 308 | 1.20 |
| | PM peak | 17 | 196 | 206 | 235 | 249 | 1.21 |
| | Weekend | 47 | 196 | 204 | 210 | 215 | 1.05 |
| | Overnight | 113 | 195 | 195 | 200 | 212 | 1.08 |

The TTTR Index over the reporting segments is 1.50 eastbound and 1.65
westbound: the 1-mile segments through the GTA swing most.

From the trucks' own schedules against the 5 s data:

- **Each pass** is timed to within a second (mean absolute error 0.1–0.2 s
  from 1-mile segments to the whole corridor). The synthetic truck holds its
  speed between route points, so interpolating between fixes is nearly exact;
  real trucks vary their speed more, and passes over short segments will be
  less exact.
- **The statistics** differ only through having fewer passes: the 95th
  percentile by 1–2% and TTTR by 0.01–0.04 on average, 0.02 for the whole
  corridor.
- **Passes needed:** subsampling the 5 s bins, a period's whole-corridor 95th
  percentile from 30 passes is typically within 3% of that from 100+, but 10%
  off one time in ten; TTTR within 0.04, 0.13 one time in ten.

## Caveats

- **Periods follow entry time.** On a 3.5-hour corridor, the congestion a truck
  meets depends on when it reaches the city, not when it enters. On the
  synthetic sample, weekday trucks leaving Windsor at 03:00–05:00 take a median
  218–243 minutes against 195 after 20:00, as they reach the GTA in the
  morning rush: FHWA's Overnight period mixes them, hence its TTTR of 1.39.
- **Trucks that stop off the highway** (rest, fuel) don't make a pass over the
  stretch with the stop. The synthetic trucks never stop, so how many real
  passes this costs is unknown.
- **The corridor is the route's.** Trucks on a parallel carriageway more than
  300 m from the route, or on a road that isn't a motorway, are off it
  (`max_offset_m`, `road_classes`). The synthetic trucks always took the
  express lanes, so trucks on the collectors are untested.
- **One time zone** for the periods (`America/Toronto` by default).
- **Synthetic incidents** strike at any hour with equal chance; real ones
  follow traffic.
