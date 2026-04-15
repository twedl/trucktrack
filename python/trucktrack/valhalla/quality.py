"""Evaluate the quality of a Valhalla map-match call.

Wraps :func:`map_match_route_shape` and records error and shape-break
signals for a given trip id.  Uses ``trace_route`` rather than
``trace_attributes`` because Valhalla emits each disjoint match segment
as its own polyline, giving a structural break signal instead of a
distance-threshold heuristic.

Also flags successful-but-implausible matches: when stop jitter or
multipath pushes the matcher to thread the trace through cross-streets
and U-turns, the polyline comes back as a single connected shape that
is much longer than the input and reverses direction many times.
``path_length_ratio`` and ``heading_reversals`` catch these.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from trucktrack.generate.interpolator import bearing, haversine_m
from trucktrack.valhalla.map_matching import map_match_full, map_match_route_shape

# A clean match is ~1.0–1.1; spurious detours push the ratio above 1.5.
# Legitimate U-turns are rare on truck traces, so more than a handful of
# reversals in a single trip usually means jitter-driven garbage.
_MAX_PATH_LENGTH_RATIO = 1.5
_MAX_HEADING_REVERSALS = 5
_HEADING_REVERSAL_DEG = 150.0
_MIN_SEGMENT_M = 10.0


@dataclass
class MapMatchQuality:
    trip_id: str
    ok: bool
    error: str | None = None
    # (polyline_index, jump_m from prev polyline's end to this one's start)
    shape_gaps: list[tuple[int, float]] = field(default_factory=list)
    n_points: int = 0
    n_polylines: int = 0
    # None when input straight-line length is zero (coincident points).
    path_length_ratio: float | None = None
    heading_reversals: int = 0

    @property
    def has_issues(self) -> bool:
        return (
            self.error is not None
            or bool(self.shape_gaps)
            or (
                self.path_length_ratio is not None
                and self.path_length_ratio > _MAX_PATH_LENGTH_RATIO
            )
            or self.heading_reversals > _MAX_HEADING_REVERSALS
        )


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


def _length_and_reversals(
    shape: list[tuple[float, float]],
) -> tuple[float, int]:
    """Total polyline length (m) and count of >150° bearing flips."""
    total_m = 0.0
    reversals = 0
    prev_bearing: float | None = None
    for i in range(1, len(shape)):
        lat1, lon1 = shape[i - 1]
        lat2, lon2 = shape[i]
        d = haversine_m(lat1, lon1, lat2, lon2)
        total_m += d
        if d < _MIN_SEGMENT_M:
            continue
        b = bearing(lat1, lon1, lat2, lon2)
        if prev_bearing is not None:
            delta = abs(((b - prev_bearing + 540.0) % 360.0) - 180.0)
            if delta > _HEADING_REVERSAL_DEG:
                reversals += 1
        prev_bearing = b
    return total_m, reversals


def path_quality(
    points: list[tuple[float, float]],
    shapes: list[list[tuple[float, float]]],
) -> tuple[float | None, int]:
    """Compute (path_length_ratio, total_heading_reversals) for the shapes."""
    input_len = sum(
        haversine_m(points[i - 1][0], points[i - 1][1], points[i][0], points[i][1])
        for i in range(1, len(points))
    )
    matched_len = 0.0
    reversals = 0
    for s in shapes:
        length, rev = _length_and_reversals(s)
        matched_len += length
        reversals += rev
    ratio = matched_len / input_len if input_len > 0 else None
    return ratio, reversals


def _evaluate(
    trip_id: str,
    points: list[tuple[float, float]],
    match: Callable[[], list[list[tuple[float, float]]]],
    *,
    record_breaks: bool,
) -> MapMatchQuality:
    q = MapMatchQuality(trip_id=trip_id, ok=False, n_points=len(points))
    if len(points) < 2:
        q.error = "insufficient points (<2)"
        return q

    try:
        shapes = match()
    except Exception as exc:  # pyvalhalla raises RuntimeError on failure
        q.error = f"{type(exc).__name__}: {exc}"
        return q

    if record_breaks:
        q.n_polylines = len(shapes)
        q.shape_gaps = _polyline_break_gaps(shapes)
    q.path_length_ratio, q.heading_reversals = path_quality(points, shapes)
    q.ok = not q.has_issues
    return q


def evaluate_map_match(
    trip_id: str,
    points: list[tuple[float, float]],
    *,
    costing: str = "auto",
    costing_options: dict[str, object] | None = None,
    config: str | Path | None = None,
    trace_options: dict[str, object] | None = None,
) -> MapMatchQuality:
    """Run a map-match call and return a quality report for the trip.

    Flags the trip when Valhalla raises, when the match breaks into
    multiple polylines, when the matched path is much longer than the
    input (spurious detours), or when the matched path reverses
    direction many times (stop jitter / multipath).
    """
    return _evaluate(
        trip_id,
        points,
        lambda: map_match_route_shape(
            points,
            costing=costing,
            costing_options=costing_options,
            config=config,
            trace_options=trace_options,
        ),
        record_breaks=True,
    )


def evaluate_map_match_attributes(
    trip_id: str,
    points: list[tuple[float, float]],
    *,
    costing: str = "auto",
    costing_options: dict[str, object] | None = None,
    config: str | Path | None = None,
    trace_options: dict[str, object] | None = None,
) -> MapMatchQuality:
    """Quality check via trace_attributes rather than trace_route.

    Catches the 444 "Map Match algorithm failed to find path: map_snap
    algorithm failed to snap the shape points to the correct shape."
    wrapper that trace_attributes_action.cc applies to *any* inner
    Meili exception.  trace_route re-throws the original code, so this
    function is the one to use if you want to flag trips that would
    fail the way ``trace_attributes`` reports it.

    No polyline-break detection — trace_attributes returns a single
    dense shape; breaks surface as inner-vertex jumps.  The path-length
    ratio and heading-reversal checks still apply.
    """

    def match() -> list[list[tuple[float, float]]]:
        _matched, _ways, shape = map_match_full(
            points,
            costing=costing,
            costing_options=costing_options,
            config=config,
            trace_options=trace_options,
        )
        return [shape] if shape else []

    return _evaluate(trip_id, points, match, record_breaks=False)
