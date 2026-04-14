"""Evaluate the quality of a Valhalla map-match call.

Wraps :func:`map_match_route_shape` and records error and shape-break
signals for a given trip id.  Uses ``trace_route`` rather than
``trace_attributes`` because Valhalla emits each disjoint match segment
as its own polyline, giving a structural break signal instead of a
distance-threshold heuristic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from trucktrack.generate.interpolator import haversine_m
from trucktrack.valhalla.map_matching import map_match_route_shape


@dataclass
class MapMatchQuality:
    trip_id: str
    ok: bool
    error: str | None = None
    # One entry per break between consecutive returned polylines:
    # (polyline_index, jump_distance_m from prev polyline's end to this one's start).
    shape_gaps: list[tuple[int, float]] = field(default_factory=list)
    n_points: int = 0
    n_polylines: int = 0

    @property
    def has_issues(self) -> bool:
        return self.error is not None or bool(self.shape_gaps)


def _polyline_break_gaps(
    shapes: list[list[tuple[float, float]]],
) -> list[tuple[int, float]]:
    """Return the end-to-start jump distance between each consecutive pair."""
    gaps: list[tuple[int, float]] = []
    for i in range(1, len(shapes)):
        prev_end = shapes[i - 1][-1]
        cur_start = shapes[i][0]
        d = haversine_m(prev_end[0], prev_end[1], cur_start[0], cur_start[1])
        gaps.append((i, d))
    return gaps


def evaluate_map_match(
    trip_id: str,
    points: list[tuple[float, float]],
    *,
    tile_extract: str | None = None,
    costing: str = "auto",
    costing_options: dict[str, object] | None = None,
    config: str | Path | None = None,
    trace_options: dict[str, object] | None = None,
) -> MapMatchQuality:
    """Run a map-match call and return a quality report for the trip.

    Flags the trip when Valhalla raises, or when the match breaks into
    multiple polylines (each disjoint piece is emitted separately).
    """
    q = MapMatchQuality(trip_id=trip_id, ok=False, n_points=len(points))
    if len(points) < 2:
        q.error = "insufficient points (<2)"
        return q

    try:
        shapes = map_match_route_shape(
            points,
            tile_extract=tile_extract,
            costing=costing,
            costing_options=costing_options,
            config=config,
            trace_options=trace_options,
        )
    except Exception as exc:  # pyvalhalla raises RuntimeError on failure
        q.error = f"{type(exc).__name__}: {exc}"
        return q

    q.n_polylines = len(shapes)
    q.shape_gaps = _polyline_break_gaps(shapes)
    q.ok = not q.has_issues
    return q
