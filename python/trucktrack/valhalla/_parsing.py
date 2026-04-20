"""Valhalla response parsing utilities (no pyvalhalla dependency)."""

from __future__ import annotations

import math
from typing import Any

from trucktrack.generate.models import RouteSegment


def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Distance in meters between two lat/lon points."""
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(math.radians(lat1))
        * math.cos(math.radians(lat2))
        * math.sin(dlon / 2) ** 2
    )
    return 6_371_000 * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


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


def concat_leg_shapes(legs: list[dict[str, Any]]) -> list[tuple[float, float]]:
    """Decode and concatenate shape polylines from a list of route legs."""
    coords: list[tuple[float, float]] = []
    for leg in legs:
        shape = decode_polyline6(leg.get("shape", ""))
        if coords and shape:
            shape = shape[1:]
        coords.extend(shape)
    return coords


def parse_valhalla_response(data: dict[str, Any]) -> RouteSegment:
    """Parse a Valhalla route response into a RouteSegment.

    ``distances_m[i]`` is the true haversine distance between
    consecutive shape coords, not a uniform share of the maneuver
    length.  Uniform distances produced discontinuities when the
    downstream interpolator (``generate.interpolator``) interpolated
    linearly across shape coords that were unevenly spaced along a
    maneuver — short curves padded with many dense vertices and long
    straights with just a few, so "uniform per-segment" rescaling
    could teleport a sample several km inside a single 60 s step.
    """
    trip = data["trip"]
    legs = trip["legs"]

    coords = concat_leg_shapes(legs)
    n_segs = max(len(coords) - 1, 0)

    summary = trip["summary"]
    total_distance_m = summary["length"] * 1000
    total_duration_s = summary["time"]

    distances = [
        _haversine_m(coords[i - 1][0], coords[i - 1][1], coords[i][0], coords[i][1])
        for i in range(1, len(coords))
    ]

    avg_speed = total_distance_m / total_duration_s if total_duration_s > 0 else 25.0
    speeds = [avg_speed] * n_segs

    # Each maneuver carries a single length/time ratio; apply it to the
    # global segment indices it covers.  ``concat_leg_shapes`` drops the
    # duplicate first coord of each non-initial leg, so leg-local shape
    # index k maps to global coord index ``leg_start + k``.
    leg_start = 0
    for leg in legs:
        leg_shape_len = len(decode_polyline6(leg.get("shape", "")))
        for maneuver in leg["maneuvers"]:
            length_m = maneuver["length"] * 1000
            time_s = maneuver["time"]
            if time_s <= 0:
                continue
            seg_speed = length_m / time_s
            begin = maneuver["begin_shape_index"]
            end = maneuver["end_shape_index"]
            for local_seg in range(begin, end):
                global_seg = leg_start + local_seg
                if 0 <= global_seg < n_segs:
                    speeds[global_seg] = seg_speed
        leg_start += max(leg_shape_len - 1, 0)

    return RouteSegment(
        coords=coords,
        speeds_mps=speeds,
        distances_m=distances,
        total_distance_m=total_distance_m,
        total_duration_s=total_duration_s,
    )
