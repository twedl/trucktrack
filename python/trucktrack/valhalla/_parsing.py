"""Valhalla response parsing utilities (no pyvalhalla dependency)."""

from __future__ import annotations

from typing import Any

from trucktrack.generate.models import RouteSegment


def decode_polyline6(encoded: str) -> list[tuple[float, float]]:
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


def parse_valhalla_response(data: dict[str, Any]) -> RouteSegment:
    """Parse a Valhalla route response into a RouteSegment."""
    trip = data["trip"]
    legs = trip["legs"]

    all_coords: list[tuple[float, float]] = []
    all_speeds: list[float] = []
    all_distances: list[float] = []

    for leg in legs:
        shape = decode_polyline6(leg["shape"])
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
