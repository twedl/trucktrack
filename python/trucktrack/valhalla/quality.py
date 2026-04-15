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

import math
from dataclasses import dataclass, field
from pathlib import Path

from trucktrack.generate.interpolator import haversine_m
from trucktrack.valhalla.map_matching import map_match_full, map_match_route_shape

# Ratio thresholds for flagging implausible matches.  A clean match is
# ~1.0–1.1; spurious detours push the ratio above 1.5.  Legitimate
# U-turns are rare on truck traces, so more than a handful of heading
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
    # One entry per break between consecutive returned polylines:
    # (polyline_index, jump_distance_m from prev polyline's end to this one's start).
    shape_gaps: list[tuple[int, float]] = field(default_factory=list)
    n_points: int = 0
    n_polylines: int = 0
    # Matched-path length divided by straight-line input length.  None
    # when input length is zero (all points coincident).
    path_length_ratio: float | None = None
    # Count of >150° bearing flips between consecutive matched segments,
    # ignoring segments shorter than _MIN_SEGMENT_M.
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


def _polyline_length_m(shape: list[tuple[float, float]]) -> float:
    total = 0.0
    for i in range(1, len(shape)):
        total += haversine_m(shape[i - 1][0], shape[i - 1][1], shape[i][0], shape[i][1])
    return total


def _bearing_deg(a: tuple[float, float], b: tuple[float, float]) -> float:
    lat1, lon1 = math.radians(a[0]), math.radians(a[1])
    lat2, lon2 = math.radians(b[0]), math.radians(b[1])
    dlon = lon2 - lon1
    x = math.sin(dlon) * math.cos(lat2)
    y = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(
        dlon
    )
    return math.degrees(math.atan2(x, y))


def _heading_reversals(
    shape: list[tuple[float, float]],
    threshold_deg: float = _HEADING_REVERSAL_DEG,
    min_segment_m: float = _MIN_SEGMENT_M,
) -> int:
    """Count bearing flips > threshold between consecutive non-tiny segments."""
    count = 0
    prev_bearing: float | None = None
    for i in range(1, len(shape)):
        d = haversine_m(shape[i - 1][0], shape[i - 1][1], shape[i][0], shape[i][1])
        if d < min_segment_m:
            continue
        b = _bearing_deg(shape[i - 1], shape[i])
        if prev_bearing is not None:
            delta = abs(((b - prev_bearing + 540.0) % 360.0) - 180.0)
            if delta > threshold_deg:
                count += 1
        prev_bearing = b
    return count


def _fill_path_quality(
    q: MapMatchQuality,
    points: list[tuple[float, float]],
    shapes: list[list[tuple[float, float]]],
) -> None:
    """Populate path_length_ratio and heading_reversals from matched shapes."""
    input_len = _polyline_length_m(points)
    matched_len = sum(_polyline_length_m(s) for s in shapes)
    if input_len > 0:
        q.path_length_ratio = matched_len / input_len
    q.heading_reversals = sum(_heading_reversals(s) for s in shapes)


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

    Flags the trip when Valhalla raises, when the match breaks into
    multiple polylines, when the matched path is much longer than the
    input (spurious detours), or when the matched path reverses
    direction many times (stop jitter / multipath).
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
    _fill_path_quality(q, points, shapes)
    q.ok = not q.has_issues
    return q


def evaluate_map_match_attributes(
    trip_id: str,
    points: list[tuple[float, float]],
    *,
    tile_extract: str | None = None,
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
    q = MapMatchQuality(trip_id=trip_id, ok=False, n_points=len(points))
    if len(points) < 2:
        q.error = "insufficient points (<2)"
        return q

    try:
        _matched, _ways, shape = map_match_full(
            points,
            tile_extract=tile_extract,
            costing=costing,
            costing_options=costing_options,
            config=config,
            trace_options=trace_options,
        )
    except Exception as exc:
        q.error = f"{type(exc).__name__}: {exc}"
        return q

    _fill_path_quality(q, points, [shape] if shape else [])
    q.ok = not q.has_issues
    return q
