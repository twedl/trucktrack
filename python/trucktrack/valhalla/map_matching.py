"""Map-matching via local pyvalhalla."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import polars as pl

from trucktrack.generate.interpolator import haversine_m
from trucktrack.valhalla._actor import DEFAULT_TRUCK_COSTING, get_actor
from trucktrack.valhalla._json import dumps as _json_dumps
from trucktrack.valhalla._json import loads as _json_loads
from trucktrack.valhalla._parsing import concat_leg_shapes

# Base breakage distance for typical 60-second GPS intervals.  The adaptive
# pre-scan bumps this up when it detects large spatial gaps between points.
_BASE_BREAKAGE_DISTANCE = 3000  # meters
_BREAKAGE_MULTIPLIER = 1.5  # breakage = max_gap * multiplier
# Cap so that large highway gaps (where haversine ≈ driving distance) don't
# overshoot — without a cap, a 50 km gap would request 75 km breakage,
# but 60 km is sufficient and avoids slow Meili searches.
_MAX_BREAKAGE_DISTANCE = 60_000  # meters

# Trucks match on truck costing: Meili then avoids roads trucks can't use,
# the same way the routes they follow were planned.
DEFAULT_COSTING = "truck"


@dataclass
class MatchedPoint:
    """A single GPS point snapped to the road network."""

    lat: float
    lon: float
    original_index: int
    edge_id: int | None
    distance_from_trace: float  # meters


def _adaptive_breakage_distance(
    points: list[tuple[float, float]],
    max_breakage_m: float = _MAX_BREAKAGE_DISTANCE,
) -> float:
    """Compute breakage_distance from the max gap between consecutive points.

    *max_breakage_m* caps the adaptive scaling.  When callers have
    already split a trace at gaps larger than some threshold, pass that
    threshold here so the matcher never searches further than the
    splitter would allow.
    """
    max_gap = 0.0
    for i in range(1, len(points)):
        d = haversine_m(points[i - 1][0], points[i - 1][1], points[i][0], points[i][1])
        if d > max_gap:
            max_gap = d
    scaled = max_gap * _BREAKAGE_MULTIPLIER
    return max(_BASE_BREAKAGE_DISTANCE, min(max_breakage_m, scaled))


def _build_trace_options(
    points: list[tuple[float, float]],
    trace_options: dict[str, object] | None,
    max_breakage_m: float = _MAX_BREAKAGE_DISTANCE,
) -> dict[str, object]:
    """Build trace_options with adaptive breakage_distance."""
    opts = dict(DEFAULT_TRACE_OPTIONS)
    if trace_options is not None:
        opts.update(trace_options)
    if "breakage_distance" not in (trace_options or {}):
        opts["breakage_distance"] = _adaptive_breakage_distance(
            points, max_breakage_m=max_breakage_m
        )
    return opts


def _with_costing(
    body: dict[str, object],
    costing: str,
    costing_options: dict[str, object] | None,
) -> dict[str, object]:
    """Set *costing* on a request body.

    Truck costing without options gets :data:`DEFAULT_TRUCK_COSTING`,
    the dimensions routes are planned with.
    """
    if costing_options is None and costing == "truck":
        costing_options = dict(DEFAULT_TRUCK_COSTING)
    body["costing"] = costing
    if costing_options is not None:
        body["costing_options"] = {costing: costing_options}
    return body


def _build_trace_body(
    points: list[tuple[float, float]],
    costing: str,
    costing_options: dict[str, object] | None,
    filters: dict[str, object] | None = None,
    trace_options: dict[str, object] | None = None,
    max_breakage_m: float = _MAX_BREAKAGE_DISTANCE,
) -> dict[str, object]:
    body: dict[str, object] = {
        "shape": [{"lat": lat, "lon": lon} for lat, lon in points],
        "shape_match": "map_snap",
        "trace_options": _build_trace_options(
            points, trace_options, max_breakage_m=max_breakage_m
        ),
    }
    _with_costing(body, costing, costing_options)
    if filters is not None:
        body["filters"] = filters
    return body


# Meili defaults tuned for sparse truck GPS (~60 s intervals at highway
# speed ≈ 1.7 km spacing).  breakage_distance is computed adaptively per
# call; the value here serves as the base when no large gaps are detected.
#
# gps_accuracy widens the emission Gaussian so stop-jitter and lane-drift
# don't strongly prefer the wrong candidate.  turn_penalty_factor makes
# spurious U-turns and cross-street detours structurally expensive —
# without it, Meili will happily thread jitter through a series of
# impossible maneuvers rather than stay on the main road.
DEFAULT_TRACE_OPTIONS: dict[str, object] = {
    "search_radius": 50,
    "gps_accuracy": 25,
    "breakage_distance": _BASE_BREAKAGE_DISTANCE,
    "interpolation_distance": 20,
    "max_route_distance_factor": 10,
    "max_route_time_factor": 10,
    "beta": 5,
    "turn_penalty_factor": 500,
}


def _parse_matched_points(
    resp: dict[str, Any],
    points: list[tuple[float, float]],
) -> list[MatchedPoint]:
    matched_pts: list[dict[str, Any]] = resp.get("matched_points", [])
    results: list[MatchedPoint] = []
    for i, mp in enumerate(matched_pts):
        edge_id = mp.get("edge_index")
        dist = mp.get("distance_from_trace_point", 0.0)
        lat = mp.get("lat", points[i][0])
        lon = mp.get("lon", points[i][1])
        results.append(
            MatchedPoint(
                lat=lat,
                lon=lon,
                original_index=i,
                edge_id=edge_id,
                distance_from_trace=dist,
            )
        )
    return results


def _parse_route_shape(resp: dict[str, Any]) -> list[list[tuple[float, float]]]:
    """Parse a ``trace_route`` response into one polyline per route segment.

    Returns a list of road-following polylines.  When the matcher breaks
    the trace, Valhalla returns disjoint segments as the primary ``trip``
    plus entries in ``alternates``; each becomes its own polyline so
    callers never have to bridge them with a straight chord.  Used to
    replace ``trace_attributes.shape`` (which is just the snapped input
    points and draws straight chords between sparse GPS fixes) with the
    full edge geometry Valhalla traversed.
    """
    shapes: list[list[tuple[float, float]]] = []
    primary = concat_leg_shapes(resp.get("trip", {}).get("legs", []))
    if primary:
        shapes.append(primary)
    for alt in resp.get("alternates", []):
        alt_shape = concat_leg_shapes(alt.get("trip", {}).get("legs", []))
        if alt_shape:
            shapes.append(alt_shape)
    return shapes


def _parse_way_ids(resp: dict[str, Any]) -> list[int]:
    edges: list[dict[str, Any]] = resp.get("edges", [])
    ways: list[int] = []
    for edge in edges:
        wid = edge.get("way_id")
        if wid is not None and (not ways or ways[-1] != wid):
            ways.append(wid)
    return ways


# trace_attributes keys behind the per-fix match columns (plus edge.way_id).
FIX_MATCH_ATTRIBUTES: tuple[str, ...] = (
    "edge.way_id",
    "edge.id",
    "edge.begin_heading",
    "edge.end_heading",
    "edge.forward",
    "edge.road_class",
    "edge.length",
    "matched.type",
    "matched.edge_index",
    "matched.distance_along_edge",
    "matched.distance_from_trace_point",
)

FIX_MATCH_SCHEMA: dict[str, type[pl.DataType]] = {
    "edge_id": pl.UInt64,
    "way_id": pl.Int64,
    "distance_along_edge": pl.Float64,
    "match_type": pl.Utf8,
    "distance_from_trace": pl.Float64,
    "begin_heading": pl.Float64,
    "end_heading": pl.Float64,
    "forward": pl.Boolean,
    "road_class": pl.Utf8,
    "edge_length": pl.Float64,
}

# Column name -> one value per input point, in FIX_MATCH_SCHEMA order.
FixMatches = dict[str, list[Any]]


def null_fix_matches(n: int) -> FixMatches:
    """Per-fix match columns for *n* points that weren't matched."""
    return {col: [None] * n for col in FIX_MATCH_SCHEMA}


def _edge_indices(matched: list[dict[str, Any]], n_edges: int) -> list[int | None]:
    """Each matched point's index into ``edges``, or ``None`` when it has none.

    Valhalla reports a point with no edge by omitting ``edge_index`` or
    by setting it to its invalid sentinel (2**64 - 1).  It does that to
    some ``interpolated`` points that it did place along an edge:
    fixes within ``interpolation_distance`` of the last, most often a
    truck standing in a queue.  Their ``distance_along_edge`` carries
    on from the previous point's, so such a point takes the previous
    point's edge as long as it doesn't move backwards along it.
    Unmatched points break the chain.  Without this, trucks reporting
    every few seconds lose their queue time and every queue looks
    shorter than it was.
    """
    out: list[int | None] = []
    last: tuple[int, float] | None = None  # (edge index, distance along) last placed
    for p in matched:
        i = p.get("edge_index")
        if i is not None and i >= n_edges:
            i = None
        along = p.get("distance_along_edge")
        if (
            i is None
            and p.get("type") == "interpolated"
            and last is not None
            and along is not None
            and along >= last[1]
        ):
            i = last[0]
        out.append(i)
        if p.get("type") == "unmatched":
            last = None
        elif i is not None and along is not None:
            last = (i, along)
    return out


def parse_fix_matches(resp: dict[str, Any], n_points: int) -> FixMatches:
    """Where each input point landed, from a response with FIX_MATCH_ATTRIBUTES.

    Returns :data:`FIX_MATCH_SCHEMA` columns with one value per point:
    the edge's GraphId, OSM way, begin/end headings, direction along the
    way, road class and length (m), and the point's fraction along it,
    match type and distance from the trace.  Edge columns are null for
    points with no edge (see :func:`_edge_indices`).
    """
    matched: list[dict[str, Any]] = resp.get("matched_points", [])
    edges: list[dict[str, Any]] = resp.get("edges", [])
    if len(matched) != n_points:
        raise ValueError(f"{len(matched)} matched points for {n_points} points")
    out = null_fix_matches(0)
    for p, i in zip(matched, _edge_indices(matched, len(edges)), strict=True):
        edge = edges[i] if i is not None else {}
        out["edge_id"].append(edge.get("id"))
        out["way_id"].append(edge.get("way_id"))
        out["distance_along_edge"].append(p.get("distance_along_edge"))
        out["match_type"].append(p.get("type"))
        out["distance_from_trace"].append(p.get("distance_from_trace_point"))
        out["begin_heading"].append(edge.get("begin_heading"))
        out["end_heading"].append(edge.get("end_heading"))
        out["forward"].append(edge.get("forward"))
        out["road_class"].append(edge.get("road_class"))
        length_km = edge.get("length")
        out["edge_length"].append(None if length_km is None else length_km * 1000.0)
    return out


def map_match(
    points: list[tuple[float, float]],
    *,
    costing: str = DEFAULT_COSTING,
    costing_options: dict[str, object] | None = None,
    config: str | Path | None = None,
    trace_options: dict[str, object] | None = None,
) -> list[MatchedPoint]:
    """Snap a sequence of (lat, lon) points to the road network.

    Returns one MatchedPoint per input point.

    ``trace_options`` overrides Meili defaults (see
    :data:`DEFAULT_TRACE_OPTIONS`).  Any key not supplied falls back
    to the default; ``breakage_distance`` additionally falls back to
    an adaptive per-call value when not supplied.
    """
    actor = get_actor(config=config)
    body = _build_trace_body(
        points, costing, costing_options, trace_options=trace_options
    )
    resp = _json_loads(actor.trace_attributes(_json_dumps(body)))
    return _parse_matched_points(resp, points)


def trace_attributes(
    points: list[tuple[float, float]],
    *,
    attributes: Sequence[str],
    costing: str = DEFAULT_COSTING,
    costing_options: dict[str, object] | None = None,
    config: str | Path | None = None,
    trace_options: dict[str, object] | None = None,
    max_breakage_m: float = _MAX_BREAKAGE_DISTANCE,
) -> dict[str, Any]:
    """Return the raw ``trace_attributes`` response, limited to *attributes*.

    *attributes* are Valhalla filter keys such as ``"edge.way_id"`` or
    ``"matched.edge_index"``.  ``matched_points`` has one entry per
    input point, in order.
    """
    actor = get_actor(config=config)
    body = _build_trace_body(
        points,
        costing,
        costing_options,
        filters={"attributes": list(dict.fromkeys(attributes)), "action": "include"},
        trace_options=trace_options,
        max_breakage_m=max_breakage_m,
    )
    resp: dict[str, Any] = _json_loads(actor.trace_attributes(_json_dumps(body)))
    return resp


def map_match_ways(
    points: list[tuple[float, float]],
    *,
    costing: str = DEFAULT_COSTING,
    costing_options: dict[str, object] | None = None,
    config: str | Path | None = None,
    trace_options: dict[str, object] | None = None,
    max_breakage_m: float = _MAX_BREAKAGE_DISTANCE,
) -> list[int]:
    """Return the deduplicated sequence of OSM way IDs for a matched trace.

    Consecutive duplicate way IDs are collapsed (a single OSM way may
    span multiple graph edges).
    """
    resp = trace_attributes(
        points,
        attributes=["edge.way_id"],
        costing=costing,
        costing_options=costing_options,
        config=config,
        trace_options=trace_options,
        max_breakage_m=max_breakage_m,
    )
    return _parse_way_ids(resp)


def map_match_full(
    points: list[tuple[float, float]],
    *,
    costing: str = DEFAULT_COSTING,
    costing_options: dict[str, object] | None = None,
    config: str | Path | None = None,
    trace_options: dict[str, object] | None = None,
    max_breakage_m: float = _MAX_BREAKAGE_DISTANCE,
) -> tuple[list[MatchedPoint], list[int], list[list[tuple[float, float]]]]:
    """Snap points and return matched points, OSM way IDs, and road geometry.

    Makes two Valhalla calls: ``trace_attributes`` for the snapped
    points and deduplicated way-ID sequence, and ``trace_route`` for
    the full road-following shape.  Two calls are intentional —
    ``trace_attributes.shape`` is just the snapped input polyline and
    draws straight chords across sparse GPS fixes, whereas
    ``trace_route`` returns the edge geometry Valhalla traversed
    between them.

    The shape is a list of polylines — one per matched route segment.
    When the matcher breaks the trace, disjoint segments come back
    separately so callers don't bridge them with a straight chord.
    """
    actor = get_actor(config=config)
    body = _build_trace_body(
        points,
        costing,
        costing_options,
        trace_options=trace_options,
        max_breakage_m=max_breakage_m,
    )
    attrs_resp = _json_loads(actor.trace_attributes(_json_dumps(body)))
    route_resp = _json_loads(actor.trace_route(_json_dumps(body)))
    return (
        _parse_matched_points(attrs_resp, points),
        _parse_way_ids(attrs_resp),
        _parse_route_shape(route_resp),
    )


def map_match_route_shape(
    points: list[tuple[float, float]],
    *,
    costing: str = DEFAULT_COSTING,
    costing_options: dict[str, object] | None = None,
    config: str | Path | None = None,
    trace_options: dict[str, object] | None = None,
) -> list[list[tuple[float, float]]]:
    """Return road-snapped polylines for a GPS trace.

    Uses Valhalla's ``trace_route`` endpoint, which returns the complete
    route geometry including intermediate edges between matched points.
    This avoids the gaps that ``trace_attributes`` produces when GPS
    points are sparse relative to the road network.

    When the matcher encounters breaks (unmatched points), the response
    contains multiple route segments: the primary ``trip`` plus entries
    in ``alternates``.  Each is returned as a separate polyline.
    """
    actor = get_actor(config=config)
    body = _build_trace_body(
        points, costing, costing_options, trace_options=trace_options
    )
    resp = _json_loads(actor.trace_route(_json_dumps(body)))

    shapes: list[list[tuple[float, float]]] = []

    primary = concat_leg_shapes(resp.get("trip", {}).get("legs", []))
    if primary:
        shapes.append(primary)

    for alt in resp.get("alternates", []):
        alt_shape = concat_leg_shapes(alt.get("trip", {}).get("legs", []))
        if alt_shape:
            shapes.append(alt_shape)

    return shapes


def map_match_dataframe(
    df: pl.DataFrame,
    *,
    lat_col: str = "lat",
    lon_col: str = "lon",
    costing: str = DEFAULT_COSTING,
    costing_options: dict[str, object] | None = None,
    config: str | Path | None = None,
    trace_options: dict[str, object] | None = None,
) -> pl.DataFrame:
    """Map-match a DataFrame and add matched_lat / matched_lon columns."""
    df = df.sort("time")
    points = list(zip(df[lat_col].to_list(), df[lon_col].to_list(), strict=True))
    matched = map_match(
        points,
        costing=costing,
        costing_options=costing_options,
        config=config,
        trace_options=trace_options,
    )
    return df.with_columns(
        pl.Series("matched_lat", [m.lat for m in matched]),
        pl.Series("matched_lon", [m.lon for m in matched]),
        pl.Series("distance_from_trace", [m.distance_from_trace for m in matched]),
    )


def map_match_dataframe_full(
    df: pl.DataFrame,
    *,
    lat_col: str = "lat",
    lon_col: str = "lon",
    costing: str = DEFAULT_COSTING,
    costing_options: dict[str, object] | None = None,
    config: str | Path | None = None,
    trace_options: dict[str, object] | None = None,
    max_breakage_m: float = _MAX_BREAKAGE_DISTANCE,
) -> tuple[pl.DataFrame, list[int], list[list[tuple[float, float]]]]:
    """Map-match a DataFrame and return the augmented DataFrame, way IDs, and shape.

    The shape is a list of road-snapped polylines — one per matched
    route segment.  When the matcher breaks the trace, disjoint segments
    come back separately so callers don't bridge them with a straight
    chord.  See :func:`map_match_full` for the underlying two-call
    structure.
    """
    df = df.sort("time")
    points = list(zip(df[lat_col].to_list(), df[lon_col].to_list(), strict=True))
    matched, ways, shape = map_match_full(
        points,
        costing=costing,
        costing_options=costing_options,
        config=config,
        trace_options=trace_options,
        max_breakage_m=max_breakage_m,
    )
    result = df.with_columns(
        pl.Series("matched_lat", [m.lat for m in matched]),
        pl.Series("matched_lon", [m.lon for m in matched]),
        pl.Series("distance_from_trace", [m.distance_from_trace for m in matched]),
    )
    return result, ways, shape
