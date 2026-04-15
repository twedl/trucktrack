"""Candidate reproducers for Valhalla error 444 / map_snap failure.

The 444 warning fires when `shape_match=map_snap` is set (Valhalla's default
for trace_attributes / trace_route) AND the inner Meili OfflineMatch throws
a std::exception — typically valhalla_exception_t{443} from
src/meili/map_matcher.cc when no feasible path chains the candidate edges.

Each probe below tries a different way to force that inner failure while
staying inside tile coverage (so the outer correlation step doesn't
pre-empt with a different error like 171 "No suitable edges near
location").  Running this script reports which probes actually hit 444.
"""

from __future__ import annotations

import os
from datetime import UTC, datetime

from dataclasses import dataclass

from trucktrack import TripConfig, generate_trace
from trucktrack.valhalla.map_matching import map_match_full


@dataclass
class ProbeResult:
    error: str | None


def evaluate_map_match(
    label: str,
    points: list[tuple[float, float]],
    *,
    tile_extract: str | None = None,
    trace_options: dict[str, object] | None = None,
) -> "ProbeResult":
    """Call trace_attributes (not trace_route) so we can observe the
    444 wrapper applied by trace_attributes_action.cc.
    """
    if len(points) < 2:
        return ProbeResult(error="insufficient points (<2)")
    try:
        map_match_full(points, tile_extract=tile_extract, trace_options=trace_options)
    except Exception as exc:
        return ProbeResult(error=f"{type(exc).__name__}: {exc}")
    return ProbeResult(error=None)


MapMatchQuality = ProbeResult  # keep existing probe signatures

TILE_EXTRACT = os.environ.get(
    "VALHALLA_TILE_EXTRACT", "valhalla_tiles/valhalla_tiles.tar"
)


def run(label: str, q: MapMatchQuality) -> None:
    hit = q.error and "failed to snap the shape" in q.error
    marker = "*** 444 ***" if hit else "    other  "
    print(f"{marker}  [{label}]  {q.error!r}")


def probe_lake_ontario_middle() -> MapMatchQuality:
    # Middle of Lake Ontario — tile extract covers the area, but no road
    # edges exist within search_radius of open-water points.
    return evaluate_map_match(
        "lake_middle",
        [(43.50, -78.00), (43.55, -77.80), (43.60, -77.60)],
        tile_extract=TILE_EXTRACT,
    )


def probe_teleport_across_lake() -> MapMatchQuality:
    # Each point snaps (Toronto / Rochester shorelines) but no road path
    # connects them within any reasonable route-distance factor — the
    # Viterbi stage must give up on the transition.
    return evaluate_map_match(
        "teleport_shorelines",
        [
            (43.6532, -79.3832),  # Toronto
            (43.1566, -77.6088),  # Rochester
            (43.6532, -79.3832),  # Toronto again
        ],
        tile_extract=TILE_EXTRACT,
    )


def probe_tiny_search_radius() -> MapMatchQuality:
    # A legitimate Toronto->Ottawa trace, but search_radius=1 m means
    # no edge candidates are found for any measurement.
    config = TripConfig(
        origin=(43.6426, -79.3871),
        destination=(45.4236, -75.7009),
        departure_time=datetime(2025, 6, 15, 8, 0, tzinfo=UTC),
        seed=42,
        tile_extract=TILE_EXTRACT,
        errors=[],
    )
    pts = [(p.lat, p.lon) for p in generate_trace(config)[:50]]
    return evaluate_map_match(
        "tiny_search_radius",
        pts,
        tile_extract=TILE_EXTRACT,
        trace_options={"search_radius": 1, "gps_accuracy": 1},
    )


def probe_zero_route_factor() -> MapMatchQuality:
    # Valid trace, but max_route_distance_factor near zero — no
    # transition between consecutive candidates is acceptable.
    config = TripConfig(
        origin=(43.6426, -79.3871),
        destination=(45.4236, -75.7009),
        departure_time=datetime(2025, 6, 15, 8, 0, tzinfo=UTC),
        seed=42,
        tile_extract=TILE_EXTRACT,
        errors=[],
    )
    pts = [(p.lat, p.lon) for p in generate_trace(config)[:50]]
    return evaluate_map_match(
        "zero_route_factor",
        pts,
        tile_extract=TILE_EXTRACT,
        trace_options={
            "max_route_distance_factor": 0.01,
            "max_route_time_factor": 0.01,
        },
    )


def probe_duplicate_points() -> MapMatchQuality:
    # All measurements identical — after dedup / interpolation the
    # remaining match count is below Meili's minimum.
    return evaluate_map_match(
        "duplicate_points",
        [(43.6532, -79.3832)] * 5,
        tile_extract=TILE_EXTRACT,
    )


def probe_sparse_far_apart() -> MapMatchQuality:
    # Two real Ontario road points 400+ km apart, with small
    # breakage_distance so the matcher refuses to bridge them.
    return evaluate_map_match(
        "sparse_far_apart",
        [(43.6532, -79.3832), (45.4236, -75.7009)],
        tile_extract=TILE_EXTRACT,
        trace_options={"breakage_distance": 1000},
    )


def probe_all_interpolated() -> MapMatchQuality:
    # All measurements closer together than interpolation_distance:
    # AppendMeasurements() folds them into interpolated points, leaving
    # columns_.size() < 2.  HasMinimumCandidates() returns false and
    # OfflineMatch throws 443, which the trace_route/trace_attributes
    # action wraps with 444 "map_snap algorithm failed to snap...".
    # Two points ~5 m apart, inside interpolation_distance=20.
    return evaluate_map_match(
        "all_interpolated",
        [(43.6532, -79.3832), (43.65324, -79.38324)],
        tile_extract=TILE_EXTRACT,
        trace_options={"interpolation_distance": 50},
    )


def probe_huge_interpolation_distance() -> MapMatchQuality:
    # A normal short trace with interpolation_distance set huge — every
    # point gets swallowed as interpolation, leaving <2 columns.
    config = TripConfig(
        origin=(43.6426, -79.3871),
        destination=(45.4236, -75.7009),
        departure_time=datetime(2025, 6, 15, 8, 0, tzinfo=UTC),
        seed=42,
        tile_extract=TILE_EXTRACT,
        errors=[],
    )
    pts = [(p.lat, p.lon) for p in generate_trace(config)[:20]]
    return evaluate_map_match(
        "huge_interpolation_distance",
        pts,
        tile_extract=TILE_EXTRACT,
        trace_options={"interpolation_distance": 1_000_000},
    )


def main() -> None:
    for probe in [
        probe_lake_ontario_middle,
        probe_teleport_across_lake,
        probe_tiny_search_radius,
        probe_zero_route_factor,
        probe_duplicate_points,
        probe_sparse_far_apart,
        probe_all_interpolated,
        probe_huge_interpolation_distance,
    ]:
        try:
            run(probe.__name__, probe())
        except Exception as exc:
            print(f"   ???       [{probe.__name__}]  {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()
