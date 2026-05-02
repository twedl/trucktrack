"""Generate per-filter GPS traces via trucktrack.generate + local Valhalla.

One trip per filter, with only that filter's target error injected, so the
before/after contrast is unambiguous.  Runs once; parquets are committed
so the figure build doesn't need Valhalla.

Usage::

    uv run python scripts/generate_traces.py
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import polars as pl
import trucktrack as tt
from trucktrack.generate import ErrorConfig, TracePoint, TripConfig, generate_trace
from trucktrack.valhalla.map_matching import map_match_route_shape

SLIDES = Path(__file__).resolve().parent.parent
DATA_DIR = SLIDES / "filters" / "data"
VALHALLA_CONFIG = str(SLIDES.parent / "valhalla_tiles" / "valhalla.json")

# Meadowvale industrial (Mississauga) → Exeter Road industrial (London), ON:
# ~180 km, ~2 h of truck traffic along Highway 401.
ORIGIN = (43.5870, -79.7550)
DESTINATION = (42.9350, -81.2400)
DEPARTURE = datetime(2025, 6, 10, 8, 0, 0)


def _trip(
    trip_id: str,
    seed: int,
    errors: list[ErrorConfig],
    *,
    gps_noise_meters: float = 3.0,
) -> TripConfig:
    return TripConfig(
        origin=ORIGIN,
        destination=DESTINATION,
        departure_time=DEPARTURE,
        trip_id=trip_id,
        seed=seed,
        gps_noise_meters=gps_noise_meters,
        config=VALHALLA_CONFIG,
        errors=errors,
    )


def _to_df(points: list[TracePoint], trip_id: str) -> pl.DataFrame:
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


def _write(df: pl.DataFrame, name: str) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = DATA_DIR / name
    df.write_parquet(path)
    print(f"  {name}: {df.height} rows → {path.relative_to(SLIDES)}")


def _match_shape(df: pl.DataFrame) -> pl.DataFrame:
    """Map-match a trace and return its road-snapped polylines flattened.

    Returns a DataFrame with ``(shape_id, lat, lon)``.  Multiple polylines
    occur only when the matcher can't bridge a break; otherwise ``shape_id``
    is always 0 but the polyline may double back where the raw sequence had
    a backward jump (stale re-emissions, spikes).
    """
    ordered = df.sort("time")
    points = list(zip(ordered["lat"].to_list(), ordered["lon"].to_list(), strict=True))
    shapes = map_match_route_shape(points, costing="truck", config=VALHALLA_CONFIG)
    shape_ids: list[int] = []
    lats: list[float] = []
    lons: list[float] = []
    for sid, shape in enumerate(shapes):
        for lat, lon in shape:
            shape_ids.append(sid)
            lats.append(lat)
            lons.append(lon)
    return pl.DataFrame(
        {"shape_id": shape_ids, "lat": lats, "lon": lons},
        schema={"shape_id": pl.UInt32, "lat": pl.Float64, "lon": pl.Float64},
    )


# Endpoints for the gap-splitter map demo.  Both points are picked to
# snap to opposite carriageways of Highway 401 in the London area —
# OUT_END on the westbound carriageway just shy of London, BACK_START
# on the eastbound carriageway a few km closer to A.  The dock-style
# arrival / departure pings at each end get trimmed off so the gap
# really sits between two highway pings going opposite directions.
# Pinned so both natural gap endpoints land within ~600 m of each
# other near a 401 interchange east of Woodstock — both on opposite
# carriageways, both immediately east of the on/off ramps so the
# "no-splitter" matcher has to invent a U-turn through the ramps.
# The asymmetric trim (out vs back) is empirical: the back leg's first
# few pings include both maneuver and ramp-up, which cover more ground
# per ping than the out leg's slow alley-dock approach.
# Tuned so both natural gap pings land ~367 m apart on opposite
# carriageways at the Hespeler Rd / Hwy 24 interchange (exit 282) in
# Cambridge — a partial cloverleaf with loop ramps that produces a
# visibly dramatic U-turn route when Meili tries to bridge the gap
# without the splitter.
GAP_UTURN_OUT_END = (43.3600, -80.3200)
GAP_UTURN_BACK_START = (43.4300, -80.4000)
TRIM_OUT_PINGS = 20
TRIM_BACK_PINGS = 10


def _custom_trip(
    trip_id: str,
    seed: int,
    start: tuple[float, float],
    end: tuple[float, float],
    departure: datetime,
) -> TripConfig:
    return TripConfig(
        origin=start,
        destination=end,
        departure_time=departure,
        trip_id=trip_id,
        seed=seed,
        gps_noise_meters=3.0,
        config=VALHALLA_CONFIG,
        errors=[],
    )


def _make_gap_uturn_trace() -> pl.DataFrame:
    """Out-and-back trip with 10 h device shutoff between legs."""
    out_pts = generate_trace(
        _custom_trip("gap_uturn_out", 71, ORIGIN, GAP_UTURN_OUT_END, DEPARTURE)
    )
    out_pts = out_pts[:-TRIM_OUT_PINGS]

    last_ts = out_pts[-1].timestamp
    back_pts_raw = generate_trace(
        _custom_trip(
            "gap_uturn_back", 72, GAP_UTURN_BACK_START, ORIGIN,
            last_ts + timedelta(hours=10),
        )
    )
    back_pts = back_pts_raw[TRIM_BACK_PINGS:]

    return _to_df(out_pts + back_pts, "gap_uturn")


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"generating traces → {DATA_DIR.relative_to(SLIDES)}")


    # stale re-emission — GPS buffer replays older readings with a newer timestamp.
    # Noise is disabled because the generator re-applies noise *after* error
    # injection, which would jitter the "identical" copy independently and hide
    # it from the bit-exact filter.  Real devices re-emit already-noised pings.
    # seed=46 lands the single stale emission on a Mississauga arterial
    # street around 52 km/h, after the origin dock maneuver but before
    # the truck joins the 401.  Not on a highway, not in either maneuver.
    stale = _to_df(
        generate_trace(
            _trip(
                "stale_demo",
                seed=46,
                errors=[
                    ErrorConfig(
                        "stale_reemission",
                        probability=1.0,
                        params={"count": 1, "lag_min": 1, "lag_max": 1},
                    )
                ],
                gps_noise_meters=0.0,
            )
        ),
        "stale_demo",
    )
    _write(stale, "stale_trace.parquet")
    _write(_match_shape(stale), "stale_matched_before.parquet")
    _write(_match_shape(tt.filter_stale_pings(stale)), "stale_matched_after.parquet")

    # impossible speed — coordinate corruption drops wildly wrong lat/lon values.
    speed = _to_df(
        generate_trace(
            _trip(
                "speed_demo",
                seed=23,
                errors=[ErrorConfig("coordinate_corruption", probability=1.0, params={"count": 3})],
            )
        ),
        "speed_demo",
    )
    _write(speed, "speed_trace.parquet")
    # The raw trace contains wildly off-world coordinates, which the matcher
    # rejects outright.  Only after the speed filter removes them is the
    # remaining sequence matchable.
    try:
        _write(_match_shape(speed), "speed_matched_before.parquet")
    except Exception as e:
        print(f"  speed_matched_before: unmatched ({e})")
    _write(
        _match_shape(tt.filter_impossible_speeds(speed, max_speed_kmh=200.0)),
        "speed_matched_after.parquet",
    )

    # observation gap — privacy shutoff mid-drive.  Flanking pings are
    # real driving (non-zero speed, varying heading, moving coords) so
    # the row pattern can't be mistaken for a stop — the signature is
    # purely in `dt`.
    gap = _to_df(
        generate_trace(
            _trip(
                "gap_demo",
                seed=42,
                errors=[ErrorConfig("privacy_shutoff", probability=1.0)],
            )
        ),
        "gap_demo",
    )
    _write(gap, "gap_trace.parquet")

    # stop — fuel-rest dwell mid-route produces a long idle cluster.
    stop = _to_df(
        generate_trace(
            _trip(
                "stop_demo",
                seed=5,
                errors=[ErrorConfig("fuel_rest_stop", probability=1.0)],
            )
        ),
        "stop_demo",
    )
    _write(stop, "stop_trace.parquet")
    _write(_match_shape(stop), "stop_matched.parquet")

    # traffic jam — stationary window mid-route; the filter should reclassify it.
    traffic = _to_df(
        generate_trace(
            _trip(
                "traffic_demo",
                seed=99,
                errors=[ErrorConfig("traffic_jam", probability=1.0)],
            )
        ),
        "traffic_demo",
    )
    _write(traffic, "traffic_trace.parquet")
    _write(_match_shape(traffic), "traffic_matched.parquet")

    # gap-splitter map demo — out-and-back A → near-B → A with a 10 h
    # gap injected between the two legs.  The truck stops emitting on
    # the westbound 401 just shy of London, then resumes 10 h later on
    # the eastbound carriageway a few km closer to A.  Without splitting,
    # the matcher has to reconcile two pings on opposite carriageways
    # going opposite directions; with split_by_observation_gap, the
    # matcher sees two clean trips.
    gap_uturn = _make_gap_uturn_trace()
    _write(gap_uturn, "gap_uturn_trace.parquet")
    # "before" — no splitter, matcher gets the whole trace.  Force a
    # continuous shape with breakage_distance=50 km so Meili can't
    # auto-split and we see what the route looks like when the gap is
    # ignored (it has to invent a U-turn through the next interchange).
    pts = list(zip(gap_uturn.sort("time")["lat"].to_list(),
                   gap_uturn.sort("time")["lon"].to_list(), strict=True))
    cont = map_match_route_shape(
        pts, costing="truck", config=VALHALLA_CONFIG,
        trace_options={"breakage_distance": 50000},
    )
    flat = pl.DataFrame(
        {
            "shape_id": [sid for sid, s in enumerate(cont) for _ in s],
            "lat": [lat for s in cont for lat, _ in s],
            "lon": [lon for s in cont for _, lon in s],
        },
        schema={"shape_id": pl.UInt32, "lat": pl.Float64, "lon": pl.Float64},
    )
    _write(flat, "gap_uturn_matched_before.parquet")
    # "after" — split first, match each segment separately.  shape_id
    # reflects segment_id so the renderer can color the two legs.
    seg = tt.split_by_observation_gap(gap_uturn, timedelta(minutes=2)).sort("time")
    parts: list[pl.DataFrame] = []
    for sid in sorted(seg["segment_id"].unique().to_list()):
        sub = seg.filter(pl.col("segment_id") == sid).sort("time")
        sub_shape = _match_shape(sub).with_columns(
            pl.lit(int(sid)).cast(pl.UInt32).alias("shape_id")
        )
        parts.append(sub_shape)
    _write(pl.concat(parts), "gap_uturn_matched_after.parquet")

    print("done.")


if __name__ == "__main__":
    main()
