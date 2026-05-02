"""Stale re-emission scenario — divided arterial in Mississauga.

The trip runs along Hurontario St (a divided boulevard with a median).
We generate a clean trace, manually pick a single ping to duplicate,
and insert the copy at lag=1 — exactly the T1 → T2 → T3=T1 pattern.

When the matcher is fed the modified trace, it must visit the source
position twice in succession (once forward, once back-then-forward).
On a divided road this forces the matcher off the original carriageway
onto the opposite-direction lane and through whichever U-turn the road
network allows — producing a visible loop on the map rather than an
invisible same-edge retrace.

Workflow:

    uv run python scenarios/stale.py preview
        prints the clean trace.  Use it to pick SOURCE_PING_INDEX.

    uv run python scenarios/stale.py build
        writes stale_trace, stale_matched_before, stale_matched_after
        parquets into slides/filters/data/.

Edit the parameters at the top of this file to change anything.
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

# A short trip across Mississauga along Eglinton Ave E, a 6-lane divided
# arterial with frequent cross streets.  The two carriageways are modeled
# as separate one-way ways in OSM, so when the matcher has to revisit a
# stale-injected position it must use the opposite-direction lane and
# U-turn at a nearby intersection — visibly distinct from the original
# direction at slide-map scale.
ORIGIN = (43.6453, -79.3875)        # University Ave @ Queen, Toronto
DESTINATION = (43.6680, -79.3970)   # Avenue Rd @ Davenport (~3 km north)
DEPARTURE = datetime(2026, 6, 10, 8, 0, 0)

# Index of the ping to duplicate.  Pick this by running `preview`.
# The stale copy is inserted LAG positions later in the sequence.
SOURCE_PING_INDEX = 5
LAG = 1

# Where the stale row's timestamp lands (seconds after the displaced row).
STALE_TIME_OFFSET_S = 30.0

# Fixed RNG seed so the clean trace is reproducible — the maneuver
# generator uses random samples for dock turns.
TRIP_SEED = 1

VALHALLA_CONFIG = str(SLIDES.parent / "valhalla_tiles" / "valhalla.json")

# ── output paths ───────────────────────────────────────────────────────

SLIDES = Path(__file__).resolve().parent.parent
DATA_DIR = SLIDES / "filters" / "data"
TRIP_ID = "stale_demo"

# ── implementation ─────────────────────────────────────────────────────


def build_clean_trace() -> list[TracePoint]:
    """Generate the clean (no-error) trace along the configured route."""
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


def inject_stale(
    points: list[TracePoint], src_idx: int, lag: int, time_offset_s: float
) -> list[TracePoint]:
    """Insert a stale copy of points[src_idx] at position src_idx+lag+1.

    The copy carries the source's lat/lon/speed/heading bit-exactly and
    a timestamp `time_offset_s` after the row it displaces.
    """
    insert_at = src_idx + lag + 1
    if insert_at >= len(points):
        raise ValueError(
            f"Source index {src_idx} + lag {lag} + 1 = {insert_at} "
            f">= trace length {len(points)}"
        )
    anchor = points[src_idx + lag]
    ts = anchor.timestamp + timedelta(seconds=time_offset_s)
    stale = replace(points[src_idx], timestamp=ts)
    return points[:insert_at] + [stale] + points[insert_at:]


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
    """Print the clean trace so you can pick SOURCE_PING_INDEX."""
    pts = build_clean_trace()
    df = to_df(pts).sort("time").with_row_index("idx")
    df = df.with_columns(pl.col("time").diff().dt.total_seconds().alias("dt"))
    print(f"clean trace: {df.height} points")
    print(f"current SOURCE_PING_INDEX = {SOURCE_PING_INDEX}, LAG = {LAG}")
    print()
    with pl.Config(tbl_rows=200, tbl_cols=10, fmt_str_lengths=40):
        print(df)


def build() -> None:
    """Generate clean → injected → matched parquets."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    clean_pts = build_clean_trace()

    # Cruise pings can land at numerically identical speed/heading values,
    # which makes the stale table visually confusing — three rows in a row
    # show the same spd/hdg even though the middle one is a real (different
    # lat/lon) observation, not a stale copy.  Nudge T2 (the row between
    # source and the soon-to-be-injected stale copy) so the contrast is
    # visible: T1 source, T2 real-and-different, T3 = T1 bit-exact.
    intermediate = SOURCE_PING_INDEX + LAG
    clean_pts[intermediate] = replace(
        clean_pts[intermediate],
        speed_mph=clean_pts[intermediate].speed_mph + 4.7,
        heading=(clean_pts[intermediate].heading + 7.0) % 360,
    )

    raw_pts = inject_stale(clean_pts, SOURCE_PING_INDEX, LAG, STALE_TIME_OFFSET_S)
    raw = to_df(raw_pts).sort("time")
    cleaned = tt.filter_stale_pings(raw)

    raw.write_parquet(DATA_DIR / "stale_trace.parquet")
    print(f"  stale_trace.parquet: {raw.height} rows")

    before_shape = match_shape_df(raw)
    before_shape.write_parquet(DATA_DIR / "stale_matched_before.parquet")
    print(f"  stale_matched_before.parquet: {before_shape.height} rows")

    after_shape = match_shape_df(cleaned)
    after_shape.write_parquet(DATA_DIR / "stale_matched_after.parquet")
    print(f"  stale_matched_after.parquet: {after_shape.height} rows")

    src_row = raw.row(SOURCE_PING_INDEX, named=True)
    stale_row = raw.row(SOURCE_PING_INDEX + LAG + 1, named=True)
    print()
    print(f"  source ping (i={SOURCE_PING_INDEX}): "
          f"({src_row['lat']:.5f}, {src_row['lon']:.5f}) "
          f"spd={src_row['speed']:.1f} hdg={src_row['heading']:.1f}  "
          f"@ {src_row['time']}")
    print(f"  stale copy  (i={SOURCE_PING_INDEX + LAG + 1}): "
          f"({stale_row['lat']:.5f}, {stale_row['lon']:.5f}) "
          f"spd={stale_row['speed']:.1f} hdg={stale_row['heading']:.1f}  "
          f"@ {stale_row['time']}")
    print(f"  matcher retrace size: {before_shape.height - after_shape.height} extra points")


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
