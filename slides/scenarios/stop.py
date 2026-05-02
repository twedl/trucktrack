"""Stop-detector scenario — fuel rest at a 401 service area.

A truck driving westbound on Highway 401 pulls off at Comber, parks at
a Petro Canada for 15 minutes, then re-enters 401 westbound and keeps
going.  We build this from two Valhalla truck routes joined by a hand-
injected dwell at the gas station.

Workflow:

    uv run python scenarios/stop.py preview
    uv run python scenarios/stop.py build
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timedelta
from pathlib import Path

import polars as pl
import trucktrack as tt
from trucktrack.generate import TripConfig, generate_trace
from trucktrack.generate.models import TracePoint
from trucktrack.valhalla.map_matching import map_match_route_shape

# ── parameters you can edit ────────────────────────────────────────────

# Three legs of the trip.
ORIGIN = (42.4015, -82.1850)        # Chatham, ON — east of Comber on 401
GAS_STATION = (42.239256, -82.549363)  # Petro Canada, Comber, ON
DESTINATION = (42.3020, -82.7200)   # Belle River, ON — west of Comber on 401

DEPARTURE = datetime(2026, 6, 10, 8, 0, 0)

# Dwell at the gas station: one ping per minute, for this many minutes.
DWELL_MINUTES = 15

# Stop-detector parameters (filter defaults).
MAX_DIAMETER_M = 50.0
MIN_DURATION_MIN = 2

TRIP_SEED_LEG_1 = 1
TRIP_SEED_LEG_2 = 2
VALHALLA_CONFIG = str(SLIDES.parent / "valhalla_tiles" / "valhalla.json")

# ── output paths ───────────────────────────────────────────────────────

SLIDES = Path(__file__).resolve().parent.parent
DATA_DIR = SLIDES / "filters" / "data"
TRIP_ID = "stop_demo"

# ── implementation ─────────────────────────────────────────────────────


def _trip_leg(
    origin: tuple[float, float],
    destination: tuple[float, float],
    departure_time: datetime,
    seed: int,
) -> list[TracePoint]:
    cfg = TripConfig(
        origin=origin,
        destination=destination,
        departure_time=departure_time,
        trip_id=TRIP_ID,
        seed=seed,
        gps_noise_meters=0.0,
        config=VALHALLA_CONFIG,
        errors=[],
    )
    return generate_trace(cfg)


def build_trace() -> list[TracePoint]:
    """Concatenate leg1 + dwell + leg2.

    Dwell pings are pinned to GAS_STATION coords exactly so the stop
    sits at the real Petro Canada lat/lon, not wherever the arrival
    maneuver's last ping happened to land.
    """
    leg1 = _trip_leg(ORIGIN, GAS_STATION, DEPARTURE, TRIP_SEED_LEG_1)
    end_of_leg1 = leg1[-1].timestamp
    dwell = [
        replace(
            leg1[-1],
            timestamp=end_of_leg1 + timedelta(seconds=60 * k),
            lat=GAS_STATION[0],
            lon=GAS_STATION[1],
            speed_mph=0.0,
        )
        for k in range(1, DWELL_MINUTES + 1)
    ]
    leg2_start = dwell[-1].timestamp + timedelta(seconds=60)
    leg2 = _trip_leg(GAS_STATION, DESTINATION, leg2_start, TRIP_SEED_LEG_2)
    return leg1 + dwell + leg2


def to_df(points: list[TracePoint], trip_id: str = TRIP_ID) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "id": [trip_id] * len(points),
            "time": [p.timestamp for p in points],
            "lat": [p.lat for p in points],
            "lon": [p.lon for p in points],
            "speed": [p.speed_mph for p in points],
            "heading": [p.heading for p in points],
        }
    )


def match_shape_df(df: pl.DataFrame) -> pl.DataFrame:
    ordered = df.sort("time")
    points = list(zip(ordered["lat"].to_list(), ordered["lon"].to_list(), strict=True))
    shapes = map_match_route_shape(points, costing="truck", config=VALHALLA_CONFIG)
    sids: list[int] = []
    lats: list[float] = []
    lons: list[float] = []
    for sid, shape in enumerate(shapes):
        for lat, lon in shape:
            sids.append(sid)
            lats.append(lat)
            lons.append(lon)
    return pl.DataFrame(
        {"shape_id": sids, "lat": lats, "lon": lons},
        schema={"shape_id": pl.UInt32, "lat": pl.Float64, "lon": pl.Float64},
    )


def preview() -> None:
    pts = build_trace()
    df = to_df(pts).sort("time").with_row_index("idx")
    df = df.with_columns(pl.col("time").diff().dt.total_seconds().alias("dt"))
    print(f"trace: {df.height} points "
          f"(leg1 + {DWELL_MINUTES}-min dwell + leg2)")
    print()
    with pl.Config(tbl_rows=300, tbl_cols=10, fmt_str_lengths=40):
        print(df)


def build() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    pts = build_trace()
    raw = to_df(pts).sort("time")
    seg = tt.split_by_stops(
        raw,
        max_diameter=MAX_DIAMETER_M,
        min_duration=timedelta(minutes=MIN_DURATION_MIN),
    )

    raw.write_parquet(DATA_DIR / "stop_trace.parquet")
    print(f"  stop_trace.parquet: {raw.height} rows")

    matched = match_shape_df(raw)
    matched.write_parquet(DATA_DIR / "stop_matched.parquet")
    print(f"  stop_matched.parquet: {matched.height} rows")

    n_stop = int(seg["is_stop"].sum())
    n_segs = seg["segment_id"].n_unique()
    print()
    print(f"  trip: {ORIGIN} → {GAS_STATION} → {DESTINATION}")
    print(f"  dwell at gas station: {DWELL_MINUTES} min "
          f"({DWELL_MINUTES + 1} pings — anchor + injected)")
    print(f"  filter: {n_stop} stop rows, {n_segs} segments")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["preview", "build"])
    args = parser.parse_args()
    if args.action == "preview":
        preview()
    else:
        build()


if __name__ == "__main__":
    main()
