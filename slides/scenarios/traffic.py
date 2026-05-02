"""Traffic-stop reclassification scenario — construction halt on 401.

A truck cruising westbound on Highway 401 hits a construction zone and
sits stopped in the same lane for 15 minutes, then continues westbound.

split_by_stops will flag the dwell as a stop because all 16 pings sit
inside the 50-m diameter for >2 minutes.  filter_traffic_stops then
reads the approach bearing (westbound, ~220°) and the departure
bearing (westbound, ~220°) and notes the angular change is essentially
zero — the truck never left the corridor — so it reclassifies the
"stop" as movement and merges the surrounding segments.

Workflow:

    uv run python scenarios/traffic.py preview
    uv run python scenarios/traffic.py build
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

# Same Mississauga → Cambridge 401 trip as scenarios/speed.py.  Pure
# highway cruise; no off-ramp.
ORIGIN = (43.6500, -79.5750)
DESTINATION = (43.3520, -80.3120)
DEPARTURE = datetime(2026, 6, 10, 8, 0, 0)

# Index of the cruise ping where construction halts the truck.
# Picked from `preview`; mid-401 westbound at ~62 mph.
ANCHOR_PING_INDEX = 25

# How long the truck sits in the construction backup.
DWELL_MINUTES = 15

# Stop / traffic filter parameters (filter defaults).
MAX_DIAMETER_M = 50.0
MIN_DURATION_MIN = 2
MAX_ANGLE_CHANGE = 30.0
MIN_BEARING_DISTANCE_M = 10.0

TRIP_SEED = 1
VALHALLA_CONFIG = str(SLIDES.parent / "valhalla_tiles" / "valhalla.json")

# ── output paths ───────────────────────────────────────────────────────

SLIDES = Path(__file__).resolve().parent.parent
DATA_DIR = SLIDES / "filters" / "data"
TRIP_ID = "traffic_demo"

# ── implementation ─────────────────────────────────────────────────────


def build_clean_trace() -> list[TracePoint]:
    cfg = TripConfig(
        origin=ORIGIN,
        destination=DESTINATION,
        departure_time=DEPARTURE,
        trip_id=TRIP_ID,
        seed=TRIP_SEED,
        gps_noise_meters=0.0,
        config=VALHALLA_CONFIG,
        errors=[],
    )
    return generate_trace(cfg)


def inject_dwell(
    points: list[TracePoint], anchor_idx: int, duration_minutes: int
) -> list[TracePoint]:
    """Insert duration_minutes stationary pings at points[anchor_idx]."""
    if not (0 <= anchor_idx < len(points)):
        raise ValueError(f"Anchor {anchor_idx} out of range for length {len(points)}")
    anchor = points[anchor_idx]
    dwell = [
        replace(
            anchor,
            timestamp=anchor.timestamp + timedelta(seconds=60 * k),
            speed_mph=0.0,
        )
        for k in range(1, duration_minutes + 1)
    ]
    shift = timedelta(seconds=60 * duration_minutes)
    after = [replace(p, timestamp=p.timestamp + shift) for p in points[anchor_idx + 1 :]]
    return points[: anchor_idx + 1] + dwell + after


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
    pts = build_clean_trace()
    df = to_df(pts).sort("time").with_row_index("idx")
    df = df.with_columns(pl.col("time").diff().dt.total_seconds().alias("dt"))
    print(f"clean trace: {df.height} points")
    print(f"current ANCHOR_PING_INDEX = {ANCHOR_PING_INDEX}, "
          f"DWELL_MINUTES = {DWELL_MINUTES}")
    print()
    with pl.Config(tbl_rows=300, tbl_cols=10, fmt_str_lengths=40):
        print(df)


def build() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    clean_pts = build_clean_trace()
    raw_pts = inject_dwell(clean_pts, ANCHOR_PING_INDEX, DWELL_MINUTES)
    raw = to_df(raw_pts).sort("time")

    seg = tt.split_by_stops(
        raw,
        max_diameter=MAX_DIAMETER_M,
        min_duration=timedelta(minutes=MIN_DURATION_MIN),
    )
    cleaned = tt.filter_traffic_stops(
        seg,
        max_angle_change=MAX_ANGLE_CHANGE,
        min_distance=MIN_BEARING_DISTANCE_M,
    )

    raw.write_parquet(DATA_DIR / "traffic_trace.parquet")
    print(f"  traffic_trace.parquet: {raw.height} rows")

    # BEFORE view: match the pre-dwell and post-dwell halves separately
    # so the figure can color them differently to show the trip was split
    # by the construction halt.  Each leg ends/begins at the cruise ping
    # adjacent to the dwell, so there's a natural ~1-mile gap between
    # them at the construction location.
    pre_end = ANCHOR_PING_INDEX + 1
    post_start = pre_end + DWELL_MINUTES
    pre_df = to_df(raw_pts[:pre_end]).sort("time")
    post_df = to_df(raw_pts[post_start:]).sort("time")
    pre_shape = match_shape_df(pre_df).with_columns(pl.lit(0).cast(pl.UInt32).alias("shape_id"))
    post_shape = match_shape_df(post_df).with_columns(pl.lit(1).cast(pl.UInt32).alias("shape_id"))
    matched = pl.concat([pre_shape, post_shape])
    matched.write_parquet(DATA_DIR / "traffic_matched.parquet")
    print(f"  traffic_matched.parquet: {matched.height} rows "
          f"(leg1: {pre_shape.height}, leg2: {post_shape.height})")

    # AFTER view: match the original clean trace (no dwell injected) as a
    # single continuous polyline.  This is what the route looks like once
    # filter_traffic_stops reclassifies the construction halt as movement
    # — the truck just cruises through, no gap.
    clean_df = to_df(clean_pts).sort("time")
    clean_shape = match_shape_df(clean_df).with_columns(
        pl.lit(0).cast(pl.UInt32).alias("shape_id")
    )
    clean_shape.write_parquet(DATA_DIR / "traffic_matched_after.parquet")
    print(f"  traffic_matched_after.parquet: {clean_shape.height} rows")

    n_stop_before = int(seg["is_stop"].sum())
    n_stop_after = int(cleaned["is_stop"].sum())
    n_seg_before = seg["segment_id"].n_unique()
    n_seg_after = cleaned["segment_id"].n_unique()
    anchor_row = to_df(clean_pts).row(ANCHOR_PING_INDEX, named=True)
    print()
    print(f"  construction halt at clean-trace index {ANCHOR_PING_INDEX}: "
          f"({anchor_row['lat']:.5f}, {anchor_row['lon']:.5f})")
    print(f"  injected {DWELL_MINUTES} dwell pings")
    print(f"  after split_by_stops:    {n_stop_before} stop rows, {n_seg_before} segments")
    print(f"  after filter_traffic_stops: {n_stop_after} stop rows, {n_seg_after} segments")


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
