"""Truck travel times along a highway corridor, and their reliability, from raw GPS.

Takes raw truck GPS to the distribution of travel times between points
on a highway (by default the 401 from Windsor to Toronto), by FHWA's
truck travel time reliability method:

0. Build Valhalla tiles from an OpenStreetMap extract (first run only).
1. Group the raw points by truck and split them into trips.
2. Map-match them, keeping one observation per fix on a road edge.
3. Route the corridor through its points, each way, and cut it into
   sections (point to point) and FHWA reporting segments.
4. Time every truck's passes over the corridor, each section and each
   segment, from its fixes on the highway.
5. Each stretch's travel times in FHWA's five periods of the week:
   percentiles of the 15-minute averages, TTTR (95th / 50th) and the
   buffer index, and the TTTR Index of the segments.

To run it on your own data, copy it out of the installed package, from
the directory this prints::

    python -c "import trucktrack.examples as e; print(e.__path__[0])"

then change the settings below and run ``python corridor.py``.
It needs the ``valhalla`` extra (``pip install "trucktrack[valhalla]"``).

Steps 0-2 are ``congestion.py``'s: with the same ``TILES`` and ``OUTPUT``
the scripts reuse each other's tiles and matches.  Reruns skip any of
those steps whose output is already there; delete it to redo the step.
Matches belong to the tiles they were made on: delete ``OUTPUT/matched``
along with ``TILES``.

Step 3 writes ``corridor_routes.csv``: each direction's route as WKT,
to check on a map (QGIS: Layer > Add Layer > Add Delimited Text Layer)
that it stays on the highway.  Step 4 writes ``corridor_passes.parquet``
and step 5 ``corridor_reliability.csv`` to ``OUTPUT``.  The repo's
synthetic sample has no trucks west of the GTA, so on it only the
stretches east of Milton have passes.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from trucktrack import bucket_by_truck, run_pipeline
from trucktrack.buckets import read_n_buckets
from trucktrack.congestion import (
    ObservationConfig,
    corridor_passes,
    travel_time_reliability,
    tttr_index,
)
from trucktrack.valhalla import BridgeConfig, build_corridor, run_map_matching
from trucktrack.valhalla.setup import build_tiles

# ── Settings: change these for your data ────────────────────────────────

# Raw GPS: parquet files anywhere under RAW_DIR, one row per fix, with
# columns id (truck), time (datetime; without a time zone it is taken as
# UTC), lat, lon, speed (km/h) and heading (degrees).  The default is the
# repo's synthetic sample: 200 trucks, six trips each over four weeks.
RAW_DIR = Path("data/trucks_sampling")
OUTPUT = Path("congestion_output")
# Valhalla tiles, built here on the first run from OSM_EXTRACT: an
# .osm.pbf covering your trucks' roads, e.g. a Geofabrik extract.  None
# downloads Ontario (about 250 MB) to data/osm/.
TILES = Path("valhalla_tiles")
OSM_EXTRACT: Path | None = None
# FHWA's periods are on this zone's clock.
LOCAL_TZ = "America/Toronto"
# The corridor's points, (lat, lon) in driving order: each section runs
# from one to the next.  Put them on the highway, away from ramps, and
# far enough from where trucks join it that they report a fix between.
POINTS = {
    "Windsor": (42.23307, -82.98941),  # 7 km past Huron Church Road
    "London": (42.86689, -81.26432),  # east of Highway 4
    "Woodstock": (43.09117, -80.76603),  # west of the 403
    "Cambridge": (43.39392, -80.39325),  # west of Highway 8
    "Milton": (43.52421, -79.92841),  # west of the express lanes
    "Toronto": (43.6694, -79.5888),  # the 427
}
# Further along the highway before the first point and after the last,
# so trucks are seen on it on both sides of them.
BEFORE = (42.2611, -83.0418)  # Huron Church Road
AFTER = (43.69896, -79.56292)  # 4 km past the 427
# Urbanized areas, (lat, lon, radius in m): FHWA reporting segments are
# at most 1 mile long inside them and 10 miles outside.
URBAN = [
    (42.29, -82.98, 15_000),  # Windsor
    (42.98, -81.25, 15_000),  # London
    (43.43, -80.45, 15_000),  # Kitchener-Cambridge
    (43.70, -79.45, 40_000),  # Toronto, Mississauga and Milton
]
# A truck out of sight longer than this on the highway stops its pass.
MAX_GAP_S = 600.0

# ─────────────────────────────────────────────────────────────────────────


def heading(text: str) -> None:
    print(f"\n=== {text} ===")


def minutes(*columns: str) -> list[pl.Expr]:
    return [(pl.col(c) / 60).round(1) for c in columns]


def main() -> None:
    bucketed, split = OUTPUT / "bucketed", OUTPUT / "split"
    matched = OUTPUT / "matched"
    config = TILES / "valhalla.json"

    heading("0. Valhalla tiles")
    if config.exists():
        print(f"{TILES} already holds tiles; delete it to redo this step")
    else:
        build_tiles(OSM_EXTRACT, TILES)

    heading("1. Group by truck and split into trips")
    if read_n_buckets(split) is not None:
        print(f"{split} already holds trips; delete it to redo this step")
    else:
        if read_n_buckets(bucketed) is None:
            bucket_by_truck(RAW_DIR, bucketed)
        run_pipeline(bucketed, split)

    heading("2. Map-match, with an observation per fix")
    run_map_matching(
        split,
        matched,
        config=config,
        quiet=True,
        observations=ObservationConfig(),
        bridges=BridgeConfig(),
    )
    obs = pl.scan_parquet(matched / "_observations" / "**" / "*.parquet")

    heading("3. The corridor, each way")
    corridor = build_corridor(POINTS, config, before=BEFORE, after=AFTER)
    spans = corridor.spans(URBAN)
    for route in corridor.routes:
        km = [
            round(float(b - a) / 1000, 1)
            for a, b in zip(route.stops[:-1], route.stops[1:], strict=True)
        ]
        print(f"{route.direction}: sections of {km} km")
    pl.DataFrame(
        {
            "direction": [r.direction for r in corridor.routes],
            "geometry": [
                "LINESTRING ("
                + ", ".join(
                    f"{lon} {lat}" for lat, lon in zip(r.lat, r.lon, strict=True)
                )
                + ")"
                for r in corridor.routes
            ],
        }
    ).write_csv(OUTPUT / "corridor_routes.csv")
    print(spans.group_by("direction", "kind").agg(n=pl.len()).sort("direction", "kind"))

    heading("4. Passes: each truck's time over each stretch")
    seen = obs.select(pl.col("edge_id").unique()).collect()["edge_id"]
    edges = corridor.edges(seen.to_list(), config)
    passes = corridor_passes(obs, edges, spans, max_gap_s=MAX_GAP_S).collect()
    passes.write_parquet(OUTPUT / "corridor_passes.parquet")
    print(
        passes.filter(pl.col("kind") != "segment")
        .group_by("direction", "kind", "name", "begin_m")
        .agg(passes=pl.len(), trucks=pl.col("truck").n_unique())
        .sort("direction", "begin_m", "kind")
        .drop("begin_m")
    )

    heading("5. Travel time reliability, by FHWA period")
    reliability = travel_time_reliability(passes, LOCAL_TZ).collect()
    reliability.write_csv(OUTPUT / "corridor_reliability.csv")
    print(f"Wrote {OUTPUT / 'corridor_reliability.csv'}.  The whole corridor:")
    print(
        reliability.filter(pl.col("kind") == "corridor")
        .with_columns(minutes("mean_s", "p10_s", "p50_s", "p80_s", "p95_s"))
        .select(
            "direction",
            "period",
            "bins",
            "passes",
            p10_min="p10_s",
            p50_min="p50_s",
            p80_min="p80_s",
            p95_min="p95_s",
            tttr="tttr",
            buffer_index="buffer_index",
        )
    )
    print("TTTR Index over the reporting segments:")
    print(tttr_index(reliability).collect())


if __name__ == "__main__":
    with pl.Config(
        float_precision=2, tbl_hide_dataframe_shape=True, tbl_rows=40, tbl_cols=12
    ):
        main()
