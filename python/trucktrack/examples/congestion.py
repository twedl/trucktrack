"""Truck speeds as Valhalla predicted traffic, one step at a time.

Takes raw truck GPS to time-dependent truck routes:

1. Split and partition the raw traces.
2. Map-match them, keeping one observation per fix on a road edge.
3. Measure each edge's speed in each local hour of the week it was driven.
4. Fill in every hour of every measured edge from a model of the edge.
5. Give roads no truck drove their road class's truck speed and week.
6. Load the speeds into the Valhalla tiles.
7. Route the same trips at different times of the week.

To run it on your own data, copy it out of the installed package, from
the directory this prints::

    python -c "import trucktrack.examples as e; print(e.__path__[0])"

then change the settings below and run ``python congestion.py``.
It needs the ``valhalla`` extra and tiles from
``python -m trucktrack.valhalla.setup`` (found as ``./valhalla.json``
or ``./valhalla_tiles/valhalla.json``).

Step 6 writes the speeds into those tiles.  Each load starts again from
the tiles as first packed, kept beside them as ``*.base.tar``, so a
rerun replaces the speeds instead of adding to them.  Reruns skip
partitioning and map-matching already done under ``OUTPUT``; delete it
to start over.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import polars as pl

from trucktrack import run_pipeline
from trucktrack.congestion import (
    AggregateConfig,
    ObservationConfig,
    edge_cells,
    fit_model,
    flow_speeds,
    historical_speeds,
    speed_ratios,
)
from trucktrack.valhalla import route, run_map_matching
from trucktrack.valhalla.traffic import add_predicted_traffic, graph_edges

# ── Settings: change these for your data ────────────────────────────────

# Raw GPS: parquet files anywhere under RAW_DIR, one row per fix, with
# columns id (truck), time (datetime; without a time zone it is taken as
# UTC), lat, lon, speed (km/h) and heading (degrees).  The default is the
# repo's synthetic sample: 200 trucks, six trips each over four weeks.
RAW_DIR = Path("data/trucks_sampling")
OUTPUT = Path("congestion_output")
# Speeds are binned by hour of the week on this zone's clock.
LOCAL_TZ = "America/Toronto"
# Trips to route, (lat, lon) to (lat, lon), at each departure time: local
# clock time at the origin.  10 March 2026 is a Tuesday.
BRAMPTON, TORONTO = (43.7387, -79.7271), (43.6555, -79.3800)
TRIPS = {
    "Brampton -> Toronto": (BRAMPTON, TORONTO),
    "Toronto -> Brampton": (TORONTO, BRAMPTON),
}
DEPARTURES = {
    "Tue 03:00": datetime(2026, 3, 10, 3),
    "Tue 08:00": datetime(2026, 3, 10, 8),
    "Tue 17:00": datetime(2026, 3, 10, 17),
    "Sat 08:00": datetime(2026, 3, 14, 8),
}

# ─────────────────────────────────────────────────────────────────────────

BANDS = {0: "<50 m", 1: "50-200 m", 2: "200 m-1 km", 3: ">1 km"}


def heading(text: str) -> None:
    print(f"\n=== {text} ===")


def main() -> None:
    partitioned, matched = OUTPUT / "partitioned", OUTPUT / "matched"

    heading("1. Split and partition")
    if any(partitioned.rglob("*.parquet")):
        print(f"{partitioned} already holds trips; delete it to redo this step")
    else:
        run_pipeline(RAW_DIR, partitioned)

    heading("2. Map-match, with an observation per fix")
    run_map_matching(partitioned, matched, observations=ObservationConfig())
    obs = pl.scan_parquet(matched / "_observations" / "**" / "*.parquet")
    quality = pl.scan_parquet(matched / "_quality" / "**" / "*.parquet")
    print("Fixes on a road edge, and those flagged as not measuring traffic:")
    print(
        obs.select(
            fixes=pl.len(),
            dwell=pl.col("dwell").sum(),
            wrong_way=pl.col("wrong_way").sum(),
            speed_mismatch=pl.col("speed_mismatch").sum(),
        ).collect()
    )

    heading("3. Measure: each edge's speed per local hour of the week")
    config = AggregateConfig(local_tz=LOCAL_TZ)
    cells = edge_cells(obs, quality=quality, config=config).collect()
    one_truck = cells.select((pl.col("trucks") < 1.5).mean()).item()
    print(
        f"{cells['fixes'].sum():,} usable fixes measure "
        f"{cells['edge_id'].n_unique():,} edges in {cells.height:,} edge-hours, "
        f"{one_truck:.0%} of them by one truck"
    )

    heading("4. Impute: every hour of every measured edge")
    level, depth, pattern = fit_model(cells)
    tuesday = [2 * 24 + h for h in (3, 8, 13, 17)]
    print("Each road class's speed / its night speed, on Tuesdays:")
    print(
        pattern.filter(pl.col("hour_of_week").is_in(tuesday))
        .with_columns(hour=pl.format("{}:00", pl.col("hour_of_week") % 24))
        .pivot("hour", index="road_class", values="pattern")
        .sort("road_class")
    )
    historical = historical_speeds(cells, model=(level, depth, pattern))
    print(historical["source"].value_counts().sort("source"))
    flows = flow_speeds(historical)
    print("Valhalla's speeds for requests without predicted traffic, first edges:")
    print(flows.head(3))

    heading("5. Roads no truck drove: truck speed / Valhalla speed")
    ratios = speed_ratios(level, cells, graph_edges())
    print(
        ratios.with_columns(pl.col("band").replace_strict(BANDS))
        .pivot("band", index="road_class", values="ratio")
        .sort("road_class")
    )

    heading("6. Load into the tiles")
    counts = add_predicted_traffic(historical, flows, pattern=pattern, ratios=ratios)
    print(
        f"{counts['edges']:,} measured and {counts['class_edges']:,} other edges "
        f"in {counts['files']} tiles"
    )

    heading("7. Route at different times (minutes)")
    rows = []
    for trip, (origin, destination) in TRIPS.items():
        row: dict[str, object] = {
            "trip": trip,
            "no time": route(origin, destination).total_duration_s / 60,
        }
        for label, when in DEPARTURES.items():
            timed = route(origin, destination, depart_at=when)
            row[label] = timed.total_duration_s / 60
        rows.append(row)
    print(pl.DataFrame(rows))


if __name__ == "__main__":
    with pl.Config(float_precision=2, tbl_hide_dataframe_shape=True):
        main()
