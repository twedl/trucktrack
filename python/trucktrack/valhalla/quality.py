"""Evaluate the quality of a Valhalla map-match call.

Wraps :func:`map_match_full` / :func:`map_match_route_shape` and records
error, warning, and shape-gap signals for a given trip id.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from trucktrack.generate.interpolator import haversine_m
from trucktrack.valhalla._actor import get_actor
from trucktrack.valhalla._parsing import decode_polyline6
from trucktrack.valhalla.map_matching import (
    _build_trace_body,
    _parse_matched_points,
    _parse_shape,
    _parse_way_ids,
)

# Valhalla trace_attributes shapes are dense (usually sub-10 m spacing).
# A jump of more than this between consecutive vertices indicates the
# matcher broke the trace into disjoint pieces.
DEFAULT_SHAPE_GAP_THRESHOLD_M = 1000.0


@dataclass
class MapMatchQuality:
    trip_id: str
    ok: bool
    error: str | None = None
    warnings: list[str] = field(default_factory=list)
    shape_gaps: list[tuple[int, float]] = field(default_factory=list)
    n_points: int = 0
    n_matched_points: int = 0
    n_shape_vertices: int = 0

    @property
    def has_issues(self) -> bool:
        return (
            self.error is not None
            or bool(self.warnings)
            or bool(self.shape_gaps)
        )


def _find_shape_gaps(
    shape: list[tuple[float, float]],
    threshold_m: float,
) -> list[tuple[int, float]]:
    """Return (index, distance_m) for each consecutive pair exceeding threshold."""
    gaps: list[tuple[int, float]] = []
    for i in range(1, len(shape)):
        d = haversine_m(shape[i - 1][0], shape[i - 1][1], shape[i][0], shape[i][1])
        if d > threshold_m:
            gaps.append((i, d))
    return gaps


def _extract_warnings(resp: dict[str, Any]) -> list[str]:
    """Pull Valhalla warning/status messages from a response dict."""
    msgs: list[str] = []
    for w in resp.get("warnings", []) or []:
        if isinstance(w, dict):
            msgs.append(str(w.get("description") or w.get("text") or w))
        else:
            msgs.append(str(w))
    # Non-zero status_code paired with an error message (shouldn't happen on
    # success path, but guards against partial responses).
    if resp.get("status_code") not in (None, 0, 200) and resp.get("error"):
        msgs.append(f"status={resp['status_code']}: {resp['error']}")
    return msgs


def evaluate_map_match(
    trip_id: str,
    points: list[tuple[float, float]],
    *,
    tile_extract: str | None = None,
    costing: str = "auto",
    costing_options: dict[str, object] | None = None,
    config: str | Path | None = None,
    trace_options: dict[str, object] | None = None,
    shape_gap_threshold_m: float = DEFAULT_SHAPE_GAP_THRESHOLD_M,
) -> MapMatchQuality:
    """Run a map-match call and return a quality report for the trip.

    Flags the trip when Valhalla raises, emits warnings, or returns a
    matched shape with large gaps between consecutive vertices.
    """
    q = MapMatchQuality(trip_id=trip_id, ok=False, n_points=len(points))
    if len(points) < 2:
        q.error = "insufficient points (<2)"
        return q

    actor = get_actor(tile_extract, config=config)
    body = _build_trace_body(
        points, costing, costing_options, trace_options=trace_options
    )
    try:
        resp = json.loads(actor.trace_attributes(json.dumps(body)))
    except Exception as exc:  # pyvalhalla raises RuntimeError on failure
        q.error = f"{type(exc).__name__}: {exc}"
        return q

    q.warnings = _extract_warnings(resp)
    shape = _parse_shape(resp)
    q.n_matched_points = len(_parse_matched_points(resp, points))
    q.n_shape_vertices = len(shape)
    q.shape_gaps = _find_shape_gaps(shape, shape_gap_threshold_m)
    q.ok = not q.has_issues
    return q
