"""Route fetching via a Valhalla instance with truck costing.

Falls back to a straight-line great-circle route if Valhalla is unreachable,
so the rest of the pipeline keeps working without an external dependency.
"""

from __future__ import annotations

import math
from typing import Any

import requests

from trucktrack.generate.models import DEFAULT_VALHALLA_URL, RouteSegment


def fetch_route(
    origin: tuple[float, float],
    destination: tuple[float, float],
    valhalla_url: str = DEFAULT_VALHALLA_URL,
) -> RouteSegment:
    """Fetch a truck route from Valhalla. Falls back to straight-line on failure."""
    body = {
        "locations": [
            {"lat": origin[0], "lon": origin[1]},
            {"lat": destination[0], "lon": destination[1]},
        ],
        "costing": "truck",
        "costing_options": {
            "truck": {
                "height": 4.11,
                "width": 2.6,
                "length": 22.0,
                "weight": 36.287,
            }
        },
        "shape_match": "map_snap",
        "units": "km",
    }

    url = valhalla_url.rstrip("/") + "/route"
    try:
        resp = requests.post(url, json=body, timeout=30)
        resp.raise_for_status()
        return _parse_valhalla_response(resp.json())
    except (requests.RequestException, KeyError, ValueError) as e:
        print(f"Valhalla error: {e}, using straight-line fallback")
        return _straight_line_fallback(origin, destination)


def _decode_polyline6(encoded: str) -> list[tuple[float, float]]:
    """Decode a Valhalla encoded polyline (precision 6) into (lat, lon) tuples."""
    coords: list[tuple[float, float]] = []
    i = 0
    lat = 0
    lon = 0
    while i < len(encoded):
        shift = 0
        result = 0
        while True:
            b = ord(encoded[i]) - 63
            i += 1
            result |= (b & 0x1F) << shift
            shift += 5
            if b < 0x20:
                break
        lat += ~(result >> 1) if (result & 1) else (result >> 1)

        shift = 0
        result = 0
        while True:
            b = ord(encoded[i]) - 63
            i += 1
            result |= (b & 0x1F) << shift
            shift += 5
            if b < 0x20:
                break
        lon += ~(result >> 1) if (result & 1) else (result >> 1)

        coords.append((lat / 1e6, lon / 1e6))
    return coords


def _parse_valhalla_response(data: dict[str, Any]) -> RouteSegment:
    trip = data["trip"]
    legs = trip["legs"]

    all_coords: list[tuple[float, float]] = []
    all_speeds: list[float] = []
    all_distances: list[float] = []

    for leg in legs:
        shape = _decode_polyline6(leg["shape"])
        if all_coords and shape:
            shape = shape[1:]
        all_coords.extend(shape)

        for maneuver in leg["maneuvers"]:
            length_m = maneuver["length"] * 1000
            time_s = maneuver["time"]
            begin = maneuver["begin_shape_index"]
            end = maneuver["end_shape_index"]
            n_segs = max(end - begin, 1)
            seg_dist = length_m / n_segs
            seg_speed = length_m / time_s if time_s > 0 else 25.0

            for _ in range(n_segs):
                all_speeds.append(seg_speed)
                all_distances.append(seg_dist)

    summary = trip["summary"]
    total_distance_m = summary["length"] * 1000
    total_duration_s = summary["time"]

    n_segments = len(all_coords) - 1
    if len(all_speeds) > n_segments:
        all_speeds = all_speeds[:n_segments]
        all_distances = all_distances[:n_segments]
    elif len(all_speeds) < n_segments:
        avg_speed = (
            total_distance_m / total_duration_s if total_duration_s > 0 else 25.0
        )
        avg_dist = total_distance_m / n_segments if n_segments > 0 else 100.0
        while len(all_speeds) < n_segments:
            all_speeds.append(avg_speed)
            all_distances.append(avg_dist)

    return RouteSegment(
        coords=all_coords,
        speeds_mps=all_speeds,
        distances_m=all_distances,
        total_distance_m=total_distance_m,
        total_duration_s=total_duration_s,
    )


def _straight_line_fallback(
    origin: tuple[float, float],
    destination: tuple[float, float],
    num_points: int = 200,
) -> RouteSegment:
    lat1, lon1 = math.radians(origin[0]), math.radians(origin[1])
    lat2, lon2 = math.radians(destination[0]), math.radians(destination[1])

    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    )
    total_dist_m = 6371000 * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    coords: list[tuple[float, float]] = []
    for i in range(num_points):
        f = i / (num_points - 1)
        lat = origin[0] + f * (destination[0] - origin[0])
        lon = origin[1] + f * (destination[1] - origin[1])
        coords.append((lat, lon))

    seg_dist = total_dist_m / (num_points - 1)
    avg_speed = 25.0
    distances_m = [seg_dist] * (num_points - 1)
    speeds_mps = [avg_speed] * (num_points - 1)

    return RouteSegment(
        coords=coords,
        speeds_mps=speeds_mps,
        distances_m=distances_m,
        total_distance_m=total_dist_m,
        total_duration_s=total_dist_m / avg_speed if avg_speed else 0.0,
    )
