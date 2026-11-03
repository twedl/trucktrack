"""Rush-hour truck speeds against free flow, road by road, from raw truck GPS.

Takes raw truck GPS to how much rush hour slows each road the trucks
drove, with the road's geometry to map it:

0. Build Valhalla tiles from an OpenStreetMap extract (first run only).
1. Group the raw points by truck and split them into trips.
2. Map-match them, keeping one observation per fix on a road edge.
3. Measure each edge's speed at night and in the weekday morning and
   evening rush hours.
4. Compare rush hour with night for each Valhalla edge and each OSM way
   in each direction, and write both with their roads.

To run it on your own data, copy it out of the installed package, from
the directory this prints::

    python -c "import trucktrack.examples as e; print(e.__path__[0])"

then change the settings below and run ``python rush_hour.py``.
It needs the ``valhalla`` extra (``pip install "trucktrack[valhalla]"``).

Steps 0-2 are ``congestion.py``'s: with the same ``TILES`` and ``OUTPUT``,
either script reuses the other's tiles and matches.  Reruns skip any of
those steps whose output is already there; delete it to redo the step.
Matches belong to the tiles they were made on: delete ``OUTPUT/matched``
along with ``TILES``.

Step 4 writes ``rush_edges.csv`` and ``rush_ways.csv`` to ``OUTPUT``, a
row per edge or directed way (``way_id``, ``forward``) with a night and
a rush cell: ``free_speed`` and ``rush_speed`` (km/h), ``ratio`` (rush /
free), ``delay_s`` (seconds rush hour adds to driving it), the effective
trucks behind each speed, and ``geometry``, the road as WKT (lon lat) in
the direction driven.  QGIS (Layer > Add Layer > Add Delimited Text
Layer), GeoPandas (``shapely.from_wkt``), R's sf and DuckDB read it.  The
two directions of a road share a line: filter or style by ``forward``.
Under 200 m a ratio rests on a fix or two and is mostly noise.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from trucktrack import bucket_by_truck, run_pipeline
from trucktrack.buckets import read_n_buckets
from trucktrack.congestion import (
    AggregateConfig,
    ObservationConfig,
    edge_cells,
    rush_speeds,
    rush_window,
    way_rush_speeds,
)
from trucktrack.valhalla import BridgeConfig, run_map_matching, with_geometry
from trucktrack.valhalla.setup import build_tiles

# ── Settings: change these for your data ────────────────────────────────

# Raw GPS: parquet files anywhere under RAW_DIR, one row per fix, with
# columns id (truck), time (datetime; without a time zone it is taken as
# UTC), lat, lon, speed (km/h) and heading (degrees).  The default is the
# repo's synthetic sample: 200 trucks, six trips each over four weeks.
RAW_DIR = Path("data/trucks_sampling")
# Your names for those columns, where they differ, as
# {"vehicle_id": "id", "gps_time": "time", "latitude": "lat"}.
COLUMNS: dict[str, str] = {}
OUTPUT = Path("congestion_output")
# Valhalla tiles, built here on the first run from OSM_EXTRACT: an
# .osm.pbf covering your trucks' roads, e.g. a Geofabrik extract.  None
# downloads Ontario (about 250 MB) to data/osm/.
TILES = Path("valhalla_tiles")
OSM_EXTRACT: Path | None = None
# The windows, as [start, end) hours on LOCAL_TZ's clock: morning and
# evening rush Monday to Friday, free flow at night on any day.
LOCAL_TZ = "America/Toronto"
AM_RUSH = (7, 9)
PM_RUSH = (16, 18)
NIGHT = (0, 5)
# Effective trucks a window needs before its speed counts.  Two cut the
# synthetic sample's coverage from 83% to 53% for a ratio error of 0.12
# instead of 0.14.
MIN_TRUCKS = 1.0

# ─────────────────────────────────────────────────────────────────────────

SHORT = 200  # m: shorter roads' ratios are mostly noise


def heading(text: str) -> None:
    print(f"\n=== {text} ===")


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
            bucket_by_truck(RAW_DIR, bucketed, columns=COLUMNS)
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
    quality = pl.scan_parquet(matched / "_quality" / "**" / "*.parquet")

    heading("3. Measure: each edge's speed at night and at rush hour")
    window = rush_window(LOCAL_TZ, am=AM_RUSH, pm=PM_RUSH, night=NIGHT)
    cells = edge_cells(
        obs, by=window, quality=quality, config=AggregateConfig(local_tz=LOCAL_TZ)
    ).collect()
    print("Edges measured in each window, and their hours of driving:")
    print(
        cells.filter(pl.col("window").is_not_null())
        .group_by("window")
        .agg(edges=pl.len(), hours=pl.col("seconds").sum() / 3600)
        .sort("window")
    )

    heading("4. Rush hour against night, by edge and by way")
    edges = with_geometry(rush_speeds(cells, MIN_TRUCKS).collect(), config)
    ways = with_geometry(way_rush_speeds(cells, MIN_TRUCKS).collect(), config)
    edges.write_csv(OUTPUT / "rush_edges.csv")
    ways.with_columns(
        pl.col("edge_ids").cast(pl.List(pl.String)).list.join(" ")
    ).write_csv(OUTPUT / "rush_ways.csv")
    print(
        f"{edges.height:,} edges and {ways.height:,} directed ways have both; "
        f"wrote {OUTPUT / 'rush_edges.csv'} and {OUTPUT / 'rush_ways.csv'}"
    )
    long = ways.filter(pl.col("length") >= SHORT)
    print(f"Ways of {SHORT} m or more, by road class:")
    print(
        long.group_by("road_class")
        .agg(
            ways=pl.len(),
            km=pl.col("length").sum() / 1000,
            median_ratio=pl.col("ratio").median(),
            am=(pl.col("rush_window") == "am").mean(),
        )
        .sort("road_class")
    )
    # One truck crawling through a rush window can top the list alone.
    print("The ten ways rush hour delays most, of those two trucks or more drove:")
    print(
        long.filter(pl.col("rush_trucks") >= 2)
        .sort("delay_s", descending=True)
        .head(10)
        .select(
            "way_id",
            "forward",
            "road_class",
            "length",
            "free_speed",
            "rush_window",
            "rush_speed",
            "ratio",
            "delay_s",
        )
    )


if __name__ == "__main__":
    with pl.Config(float_precision=2, tbl_hide_dataframe_shape=True):
        main()
