"""Map-matching via local pyvalhalla."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import polars as pl

from trucktrack.generate.interpolator import haversine_m
from trucktrack.valhalla._actor import get_actor
from trucktrack.valhalla._parsing import concat_leg_shapes, decode_polyline6

# Base breakage distance for typical 60-second GPS intervals.  The adaptive
# pre-scan bumps this up when it detects large spatial gaps between points.
_BASE_BREAKAGE_DISTANCE = 3000  # meters
_BREAKAGE_MULTIPLIER = 3  # breakage = max_gap * multiplier


@dataclass
class MatchedPoint:
    """A single GPS point snapped to the road network."""

    lat: float
    lon: float
    original_index: int
    edge_id: int | None
    distance_from_trace: float  # meters


def _adaptive_breakage_distance(points: list[tuple[float, float]]) -> float:
    """Compute breakage_distance from the max gap between consecutive points."""
    max_gap = 0.0
    for i in range(1, len(points)):
        d = haversine_m(points[i - 1][0], points[i - 1][1], points[i][0], points[i][1])
        if d > max_gap:
            max_gap = d
    return max(_BASE_BREAKAGE_DISTANCE, max_gap * _BREAKAGE_MULTIPLIER)


def _build_trace_options(
    points: list[tuple[float, float]],
    trace_options: dict[str, object] | None,
) -> dict[str, object]:
    """Build trace_options with adaptive breakage_distance."""
    opts = dict(DEFAULT_TRACE_OPTIONS)
    if trace_options is not None:
        opts.update(trace_options)
    if "breakage_distance" not in (trace_options or {}):
        opts["breakage_distance"] = _adaptive_breakage_distance(points)
    return opts


def _build_trace_body(
    points: list[tuple[float, float]],
    costing: str,
    costing_options: dict[str, object] | None,
    filters: dict[str, object] | None = None,
    trace_options: dict[str, object] | None = None,
) -> dict[str, object]:
    body: dict[str, object] = {
        "shape": [{"lat": lat, "lon": lon} for lat, lon in points],
        "costing": costing,
        "shape_match": "map_snap",
        "trace_options": _build_trace_options(points, trace_options),
    }
    if costing_options is not None:
        body["costing_options"] = {costing: costing_options}
    if filters is not None:
        body["filters"] = filters
    return body


# Meili defaults tuned for sparse truck GPS (~60 s intervals at highway
# speed ≈ 1.7 km spacing).  breakage_distance is computed adaptively per
# call; the value here serves as the base when no large gaps are detected.
DEFAULT_TRACE_OPTIONS: dict[str, object] = {
    "search_radius": 50,
    "gps_accuracy": 15,
    "breakage_distance": _BASE_BREAKAGE_DISTANCE,
    "interpolation_distance": 20,
    "max_route_distance_factor": 10,
    "max_route_time_factor": 10,
    "beta": 5,
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


def _parse_shape(resp: dict[str, Any]) -> list[tuple[float, float]]:
    """Decode the matched route shape from the trace_attributes response."""
    encoded = resp.get("shape", "")
    if not encoded:
        return []
    return decode_polyline6(encoded)


def _parse_way_ids(resp: dict[str, Any]) -> list[int]:
    edges: list[dict[str, Any]] = resp.get("edges", [])
    ways: list[int] = []
    for edge in edges:
        wid = edge.get("way_id")
        if wid is not None and (not ways or ways[-1] != wid):
            ways.append(wid)
    return ways


def map_match(
    points: list[tuple[float, float]],
    tile_extract: str | None = None,
    costing: str = "auto",
    costing_options: dict[str, object] | None = None,
    config: str | Path | None = None,
    trace_options: dict[str, object] | None = None,
) -> list[MatchedPoint]:
    """Snap a sequence of (lat, lon) points to the road network.

    Returns one MatchedPoint per input point.  At least one of
    *tile_extract* or *config* must be provided.

    ``trace_options`` overrides Meili defaults (see
    :data:`DEFAULT_TRACE_OPTIONS`).  Any key not supplied falls back
    to the default; ``breakage_distance`` additionally falls back to
    an adaptive per-call value when not supplied.
    """
    actor = get_actor(tile_extract, config=config)
    body = _build_trace_body(
        points, costing, costing_options, trace_options=trace_options
    )
    resp = json.loads(actor.trace_attributes(json.dumps(body)))
    return _parse_matched_points(resp, points)


def map_match_ways(
    points: list[tuple[float, float]],
    tile_extract: str | None = None,
    costing: str = "auto",
    costing_options: dict[str, object] | None = None,
    config: str | Path | None = None,
    trace_options: dict[str, object] | None = None,
) -> list[int]:
    """Return the deduplicated sequence of OSM way IDs for a matched trace.

    Consecutive duplicate way IDs are collapsed (a single OSM way may
    span multiple graph edges).
    """
    actor = get_actor(tile_extract, config=config)
    body = _build_trace_body(
        points,
        costing,
        costing_options,
        filters={"attributes": ["edge.way_id"], "action": "include"},
        trace_options=trace_options,
    )
    resp = json.loads(actor.trace_attributes(json.dumps(body)))
    return _parse_way_ids(resp)


def map_match_full(
    points: list[tuple[float, float]],
    tile_extract: str | None = None,
    costing: str = "auto",
    costing_options: dict[str, object] | None = None,
    config: str | Path | None = None,
    trace_options: dict[str, object] | None = None,
) -> tuple[list[MatchedPoint], list[int], list[tuple[float, float]]]:
    """Snap points and return matched points, OSM way IDs, and road geometry.

    Makes a single ``trace_attributes`` call and extracts the snapped
    coordinates, deduplicated way-ID sequence, and the full matched
    route shape from the response.
    """
    actor = get_actor(tile_extract, config=config)
    body = _build_trace_body(
        points, costing, costing_options, trace_options=trace_options
    )
    resp = json.loads(actor.trace_attributes(json.dumps(body)))
    return (
        _parse_matched_points(resp, points),
        _parse_way_ids(resp),
        _parse_shape(resp),
    )


def map_match_route_shape(
    points: list[tuple[float, float]],
    tile_extract: str | None = None,
    costing: str = "auto",
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
    actor = get_actor(tile_extract, config=config)
    body = _build_trace_body(
        points, costing, costing_options, trace_options=trace_options
    )
    resp = json.loads(actor.trace_route(json.dumps(body)))

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
    tile_extract: str | None = None,
    lat_col: str = "lat",
    lon_col: str = "lon",
    costing: str = "auto",
    costing_options: dict[str, object] | None = None,
    config: str | Path | None = None,
    trace_options: dict[str, object] | None = None,
) -> pl.DataFrame:
    """Map-match a DataFrame and add matched_lat / matched_lon columns."""
    df = df.sort("time")
    points = list(zip(df[lat_col].to_list(), df[lon_col].to_list(), strict=True))
    matched = map_match(
        points,
        tile_extract,
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
    tile_extract: str | None = None,
    lat_col: str = "lat",
    lon_col: str = "lon",
    costing: str = "auto",
    costing_options: dict[str, object] | None = None,
    config: str | Path | None = None,
    trace_options: dict[str, object] | None = None,
) -> tuple[pl.DataFrame, list[int], list[tuple[float, float]]]:
    """Map-match a DataFrame and return the augmented DataFrame, way IDs, and shape.

    Single ``trace_attributes`` call — avoids the overhead of calling
    :func:`map_match_dataframe` and :func:`map_match_ways` separately.
    The shape is the full road-snapped polyline decoded from the response.
    """
    df = df.sort("time")
    points = list(zip(df[lat_col].to_list(), df[lon_col].to_list(), strict=True))
    matched, ways, shape = map_match_full(
        points,
        tile_extract,
        costing=costing,
        costing_options=costing_options,
        config=config,
        trace_options=trace_options,
    )
    result = df.with_columns(
        pl.Series("matched_lat", [m.lat for m in matched]),
        pl.Series("matched_lon", [m.lon for m in matched]),
        pl.Series("distance_from_trace", [m.distance_from_trace for m in matched]),
    )
    return result, ways, shape
