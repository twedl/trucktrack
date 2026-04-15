"""Demonstrate each failure mode flagged by :func:`evaluate_map_match`.

Runs three trip segments, one per flag:

1. ``insufficient points``  — fewer than 2 input points (short-circuit).
2. ``RuntimeError``         — Valhalla raises (points far from any road
                              in the tile extract).
3. ``shape_gaps``           — Valhalla returns multiple polylines after
                              a large within-segment break.

Usage::

    VALHALLA_TILE_EXTRACT=valhalla_tiles/valhalla_tiles.tar \
        uv run python examples/trace_visualizations/quality_flag_demos.py
"""

from __future__ import annotations

import os
from datetime import UTC, datetime

from trucktrack import ErrorConfig, TripConfig, generate_trace
from trucktrack.valhalla.quality import MapMatchQuality, evaluate_map_match

TILE_EXTRACT = os.environ.get(
    "VALHALLA_TILE_EXTRACT", "valhalla_tiles/valhalla_tiles.tar"
)


def _print(label: str, q: MapMatchQuality) -> None:
    print(f"\n[{label}]")
    print(f"  trip_id      = {q.trip_id}")
    print(f"  ok           = {q.ok}")
    print(f"  error        = {q.error}")
    print(f"  n_points     = {q.n_points}")
    print(f"  n_polylines  = {q.n_polylines}")
    print(f"  shape_gaps   = {q.shape_gaps}")
    print(f"  has_issues   = {q.has_issues}")


def demo_insufficient_points() -> MapMatchQuality:
    # Single point short-circuits before the actor is called.
    return evaluate_map_match(
        trip_id="too_few_points",
        points=[(43.6426, -79.3871)],
        tile_extract=TILE_EXTRACT,
    )


def demo_valhalla_raises() -> MapMatchQuality:
    # Two points in the Atlantic, well outside the Ontario tile extract.
    # pyvalhalla raises (e.g. "No suitable edges near location").
    return evaluate_map_match(
        trip_id="off_tile",
        points=[(0.0, 0.0), (0.1, 0.1)],
        tile_extract=TILE_EXTRACT,
    )


def demo_shape_break() -> MapMatchQuality:
    # Generate a Toronto -> Ottawa trace with a 40 km geofence gap and
    # feed the full trace (no observation-gap split) to the quality
    # helper.  Override breakage_distance down to the base so the
    # matcher cleaves the trace into two polylines instead of bridging
    # the gap with a long connector.
    config = TripConfig(
        origin=(43.6426, -79.3871),
        destination=(45.4236, -75.7009),
        departure_time=datetime(2025, 6, 15, 8, 0, tzinfo=UTC),
        seed=42,
        tile_extract=TILE_EXTRACT,
        errors=[
            ErrorConfig(
                "geofence_gap",
                probability=1.0,
                params={"center": (44.2312, -76.4860), "radius_m": 20_000.0},
            ),
        ],
    )
    points = [(p.lat, p.lon) for p in generate_trace(config)]
    return evaluate_map_match(
        trip_id="break_on_gap",
        points=points,
        tile_extract=TILE_EXTRACT,
        trace_options={"breakage_distance": 3000},
    )


def main() -> None:
    _print("insufficient_points", demo_insufficient_points())
    _print("valhalla_raises", demo_valhalla_raises())
    _print("shape_break", demo_shape_break())


if __name__ == "__main__":
    main()
