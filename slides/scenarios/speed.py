"""Impossible-speed filter scenario — one off-route GPS spike.

Generate a clean truck trip, then mutate ONE ping's coordinates by
adding a fixed lat/lon offset.  The implied speed between the previous
ping and the spike exceeds the filter's threshold (200 km/h by default),
so the filter drops the row.

Workflow:

    uv run python scenarios/speed.py preview
        prints the clean trace.  Use it to pick SPIKE_PING_INDEX.

    uv run python scenarios/speed.py build
        writes speed_trace, speed_matched_before, speed_matched_after
        parquets into slides/filters/data/.

Edit the parameters at the top of this file to change anything.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import polars as pl
import trucktrack as tt
from trucktrack.generate import TripConfig, generate_trace
from trucktrack.generate.models import TracePoint
from trucktrack.valhalla.map_matching import map_match_route_shape

# ── parameters you can edit ────────────────────────────────────────────

# Mississauga (Dixie industrial) → Cambridge industrial — same 401 trip
# we used before.  Highway speed gives the spike enough headroom to
# trigger the 200 km/h threshold without absurd offsets.
ORIGIN = (43.6500, -79.5750)
DESTINATION = (43.3520, -80.3120)
DEPARTURE = datetime(2026, 6, 10, 8, 0, 0)

# Index of the ping whose coords get replaced with a spike.
# Pick this by running `preview`.  Mid-cruise pings on 401 work best.
SPIKE_PING_INDEX = 25

# Offset (degrees) added to the original ping's lat/lon to produce the
# spike.  Default is ~5.5 km north — visible on the map, well above the
# 200 km/h threshold for a 60-second sample interval.
SPIKE_LAT_OFFSET = 0.05
SPIKE_LON_OFFSET = 0.0

# Speed filter threshold (km/h).  Pings whose implied speed against the
# last kept anchor exceeds this are dropped.
MAX_SPEED_KMH = 200.0

# Fixed RNG seed so the clean trace is reproducible.
TRIP_SEED = 1

VALHALLA_CONFIG = str(SLIDES.parent / "valhalla_tiles" / "valhalla.json")

# ── output paths ───────────────────────────────────────────────────────

SLIDES = Path(__file__).resolve().parent.parent
DATA_DIR = SLIDES / "filters" / "data"
TRIP_ID = "speed_demo"

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


def inject_spike(
    points: list[TracePoint], idx: int, lat_offset: float, lon_offset: float
) -> list[TracePoint]:
    """Replace points[idx]'s lat/lon with original + offsets."""
    if not (0 <= idx < len(points)):
        raise ValueError(f"Spike index {idx} out of range for trace of length {len(points)}")
    p = points[idx]
    spiked = replace(p, lat=p.lat + lat_offset, lon=p.lon + lon_offset)
    return points[:idx] + [spiked] + points[idx + 1 :]


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
    try:
        shapes = map_match_route_shape(points, costing="truck", config=VALHALLA_CONFIG)
    except Exception as e:
        print(f"  matcher rejected trace: {e}")
        return pl.DataFrame(
            {"shape_id": [], "lat": [], "lon": []},
            schema={"shape_id": pl.UInt32, "lat": pl.Float64, "lon": pl.Float64},
        )
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
    print(f"current SPIKE_PING_INDEX = {SPIKE_PING_INDEX}, "
          f"offset = ({SPIKE_LAT_OFFSET:+.4f}, {SPIKE_LON_OFFSET:+.4f}) deg")
    print()
    with pl.Config(tbl_rows=200, tbl_cols=10, fmt_str_lengths=40):
        print(df)


def build() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    clean_pts = build_clean_trace()
    raw_pts = inject_spike(clean_pts, SPIKE_PING_INDEX, SPIKE_LAT_OFFSET, SPIKE_LON_OFFSET)
    raw = to_df(raw_pts).sort("time")
    cleaned = tt.filter_impossible_speeds(raw, max_speed_kmh=MAX_SPEED_KMH)

    raw.write_parquet(DATA_DIR / "speed_trace.parquet")
    print(f"  speed_trace.parquet: {raw.height} rows")

    before_shape = match_shape_df(raw)
    before_shape.write_parquet(DATA_DIR / "speed_matched_before.parquet")
    print(f"  speed_matched_before.parquet: {before_shape.height} rows")

    after_shape = match_shape_df(cleaned)
    after_shape.write_parquet(DATA_DIR / "speed_matched_after.parquet")
    print(f"  speed_matched_after.parquet: {after_shape.height} rows")

    spike_row = raw.row(SPIKE_PING_INDEX, named=True)
    orig_row = to_df(clean_pts).row(SPIKE_PING_INDEX, named=True)
    print()
    print(f"  spike row (i={SPIKE_PING_INDEX})")
    print(f"    original: ({orig_row['lat']:.5f}, {orig_row['lon']:.5f}) "
          f"spd={orig_row['speed']:.1f} hdg={orig_row['heading']:.1f}")
    print(f"    spiked:   ({spike_row['lat']:.5f}, {spike_row['lon']:.5f}) "
          f"spd={spike_row['speed']:.1f} hdg={spike_row['heading']:.1f}")
    print(f"  rows kept by filter: {cleaned.height}/{raw.height}")


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
