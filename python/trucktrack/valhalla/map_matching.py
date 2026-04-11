"""Map-matching via local pyvalhalla."""

from __future__ import annotations

import json
from dataclasses import dataclass

import polars as pl

from trucktrack.valhalla._actor import DEFAULT_TRUCK_COSTING, get_actor
from trucktrack.valhalla._parsing import decode_polyline6


@dataclass
class MatchedPoint:
    """A single GPS point snapped to the road network."""

    lat: float
    lon: float
    original_index: int
    edge_id: int | None
    distance_from_trace: float  # meters


def map_match(
    points: list[tuple[float, float]],
    tile_extract: str,
    costing: str = "truck",
    costing_options: dict[str, object] | None = None,
) -> list[MatchedPoint]:
    """Snap a sequence of (lat, lon) points to the road network.

    Returns one MatchedPoint per input point.
    """
    actor = get_actor(tile_extract)
    body: dict[str, object] = {
        "shape": [{"lat": lat, "lon": lon} for lat, lon in points],
        "costing": costing,
        "shape_match": "map_snap",
        "costing_options": {
            costing: costing_options or DEFAULT_TRUCK_COSTING,
        },
    }
    resp = json.loads(actor.trace_route(json.dumps(body)))

    # Extract matched points from the response.
    matched_pts = resp.get("matched_points", [])

    # Decode the snapped shape from the trip legs.
    snapped_coords: list[tuple[float, float]] = []
    for leg in resp.get("trip", {}).get("legs", []):
        shape = decode_polyline6(leg["shape"])
        if snapped_coords and shape:
            shape = shape[1:]
        snapped_coords.extend(shape)

    results: list[MatchedPoint] = []
    for i, mp in enumerate(matched_pts):
        edge_id = mp.get("edge_index")
        dist = mp.get("distance_from_trace_point", 0.0)
        # Use the snapped coordinate if available via begin_route_discontinuity
        # index, otherwise fall back to the matched_point lat/lon.
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


def map_match_dataframe(
    df: pl.DataFrame,
    tile_extract: str,
    lat_col: str = "lat",
    lon_col: str = "lon",
    costing: str = "truck",
    costing_options: dict[str, object] | None = None,
) -> pl.DataFrame:
    """Map-match a DataFrame and add matched_lat / matched_lon columns."""
    points = list(zip(df[lat_col].to_list(), df[lon_col].to_list(), strict=True))
    matched = map_match(
        points, tile_extract, costing=costing, costing_options=costing_options
    )
    return df.with_columns(
        pl.Series("matched_lat", [m.lat for m in matched]),
        pl.Series("matched_lon", [m.lon for m in matched]),
        pl.Series("distance_from_trace", [m.distance_from_trace for m in matched]),
    )
