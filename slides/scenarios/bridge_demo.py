"""Bridge orchestrator scenario — Toronto → Windsor with injected gaps.

Three traces, all of the same Toronto → Windsor 401 trip:

    1. clean         — full 60-second pings, no gaps
    2. with-gaps     — same trace with three chunks of pings dropped,
                       creating distance + time gaps
    3. with-gaps     — but matched two ways:
       - no-bridge:   trace_options={"breakage_distance": 3000}, default
                      ``map_match_route_shape`` — Meili breaks the trace
                      at every gap, returns multiple disjoint shapes
       - bridge-on:   ``map_match_dataframe_with_bridges`` — splits at
                      each gap, matches sub-segments separately, and
                      bridges each gap with one ``/route`` call

Workflow:

    uv run python scenarios/bridge_demo.py
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import polars as pl
from trucktrack.generate import TripConfig, generate_trace
from trucktrack.generate.models import TracePoint
from trucktrack.valhalla._bridge import (
    BridgeConfig,
    map_match_dataframe_with_bridges,
)
from trucktrack.valhalla.map_matching import map_match_route_shape

SLIDES = Path(__file__).resolve().parent.parent
DATA = SLIDES / "valhalla" / "data"

ORIGIN = (43.6532, -79.3832)        # downtown Toronto
DESTINATION = (42.3149, -83.0364)   # Windsor
DEPARTURE = datetime(2026, 6, 10, 8, 0, 0)
TRIP_ID = "bridge_demo"
SEED = 1

VALHALLA_CONFIG = str(SLIDES.parent / "valhalla_tiles" / "valhalla.json")

# Indices into the clean trace marking the START of each gap and how
# many consecutive pings to drop.  Dropping N pings at 60 s cadence
# creates a gap of (N+1) minutes and roughly (N+1) × 1.7 km on highway.
# Gaps placed roughly at 25 % / 50 % / 75 % of the trip.
GAPS = [
    (40, 5),    # ~6 min · ~10 km
    (95, 10),   # ~11 min · ~18 km
    (155, 15),  # ~16 min · ~27 km
]

BRIDGES = BridgeConfig(max_dist_m=5000.0, time_s=240.0, min_dist_m=1000.0)


def _trip_config() -> TripConfig:
    return TripConfig(
        origin=ORIGIN,
        destination=DESTINATION,
        departure_time=DEPARTURE,
        trip_id=TRIP_ID,
        seed=SEED,
        gps_noise_meters=0.0,
        config=VALHALLA_CONFIG,
        errors=[],
    )


def _to_df(points: list[TracePoint]) -> pl.DataFrame:
    return pl.DataFrame({
        "id": [TRIP_ID] * len(points),
        "time": [p.timestamp for p in points],
        "lat": [p.lat for p in points],
        "lon": [p.lon for p in points],
        "speed": [p.speed_mph for p in points],
        "heading": [p.heading for p in points],
    })


def _drop_gaps(points: list[TracePoint], gaps: list[tuple[int, int]]
               ) -> list[TracePoint]:
    """Return a copy of ``points`` with consecutive pings dropped at
    the given (start_idx, n) ranges."""
    drop = set()
    for start, n in gaps:
        drop.update(range(start, start + n))
    return [p for i, p in enumerate(points) if i not in drop]


def _shapes_to_df(
    shapes: list[list[tuple[float, float]]],
    kinds: list[str] | None = None,
) -> pl.DataFrame:
    """Flatten list-of-shapes into one DataFrame keyed by shape_id.

    *kinds* optionally tags each shape (e.g. "segment" or "bridge")
    so renderers can color sub-segments and bridges differently.
    """
    sids: list[int] = []
    lats: list[float] = []
    lons: list[float] = []
    knds: list[str] = []
    for sid, shape in enumerate(shapes):
        kind = kinds[sid] if kinds else "segment"
        for lat, lon in shape:
            sids.append(sid)
            lats.append(lat)
            lons.append(lon)
            knds.append(kind)
    return pl.DataFrame(
        {"shape_id": sids, "kind": knds, "lat": lats, "lon": lons},
        schema={"shape_id": pl.UInt32, "kind": pl.String,
                "lat": pl.Float64, "lon": pl.Float64},
    )


def main() -> None:
    DATA.mkdir(parents=True, exist_ok=True)

    print("  generating clean trace …")
    clean_pts = generate_trace(_trip_config())
    print(f"    {len(clean_pts)} pings")

    print("  injecting gaps …")
    gappy_pts = _drop_gaps(clean_pts, GAPS)
    print(f"    {len(gappy_pts)} pings (dropped {len(clean_pts) - len(gappy_pts)})")

    clean_df = _to_df(clean_pts).sort("time")
    gappy_df = _to_df(gappy_pts).sort("time")

    clean_df.write_parquet(DATA / "bridge_clean_trace.parquet")
    gappy_df.write_parquet(DATA / "bridge_gappy_trace.parquet")

    # Match 1: clean trip
    print("  map-matching clean trace …")
    clean_pts_xy = list(zip(clean_df["lat"].to_list(),
                            clean_df["lon"].to_list(), strict=True))
    clean_shapes = map_match_route_shape(
        clean_pts_xy, costing="truck", config=VALHALLA_CONFIG,
    )
    clean_match = _shapes_to_df(clean_shapes)
    clean_match.write_parquet(DATA / "bridge_clean_matched.parquet")
    print(f"    {len(clean_shapes)} shape(s), {clean_match.height} pts")

    # Match 2: gappy trip, no bridging, breakage_distance=3000m forced
    print("  map-matching gappy trace WITHOUT bridges (breakage=3000) …")
    gappy_pts_xy = list(zip(gappy_df["lat"].to_list(),
                            gappy_df["lon"].to_list(), strict=True))
    no_bridge_shapes = map_match_route_shape(
        gappy_pts_xy, costing="truck", config=VALHALLA_CONFIG,
        trace_options={"breakage_distance": 3000},
    )
    no_bridge_match = _shapes_to_df(no_bridge_shapes)
    no_bridge_match.write_parquet(DATA / "bridge_no_bridge_matched.parquet")
    print(f"    {len(no_bridge_shapes)} shape(s), {no_bridge_match.height} pts")

    # Match 3: gappy trip, bridging on
    print("  map-matching gappy trace WITH bridges …")
    bridged = map_match_dataframe_with_bridges(
        gappy_df,
        bridges=BRIDGES,
        costing="truck",
        config=VALHALLA_CONFIG,
        collect_shapes=True,
    )
    # map_match_dataframe_with_bridges interleaves sub-segments and
    # bridges: seg, bridge, seg, bridge, ..., seg.  Even-indexed
    # shapes are matched sub-segments, odd-indexed are routed bridges.
    kinds = ["segment" if (i % 2 == 0) else "bridge"
             for i in range(len(bridged.shapes))]
    bridge_match = _shapes_to_df(bridged.shapes, kinds=kinds)
    bridge_match.write_parquet(DATA / "bridge_with_bridge_matched.parquet")
    print(f"    {len(bridged.shapes)} shape(s), {bridge_match.height} pts, "
          f"{len(bridged.fits)} bridge(s), fallback={bridged.fallback_used}")
    if bridged.fits:
        for i, fit in enumerate(bridged.fits):
            print(
                f"      bridge {i}: straight {fit.straight_m:.0f} m · "
                f"route {fit.route_m:.0f} m · "
                f"detour {fit.detour_ratio:.2f} · "
                f"gap {fit.gap_seconds:.0f} s"
            )


if __name__ == "__main__":
    main()
