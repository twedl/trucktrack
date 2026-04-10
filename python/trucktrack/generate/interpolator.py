"""Walk a route polyline at fixed time intervals to produce GPS trace points."""

from __future__ import annotations

import math
import random
from datetime import datetime, timedelta

import numpy as np

from trucktrack import _core
from trucktrack.generate.models import RouteSegment, TracePoint
from trucktrack.generate.speed_profile import (
    inject_traffic_stops,
    mps_to_mph,
    smooth_speeds,
)

INTERVAL_S = 60.0


def resample_trace(points: list[TracePoint], interval_s: float = INTERVAL_S) -> list[TracePoint]:
    """Resample a list of trace points at fixed time intervals.

    Linearly interpolates lat, lon, and speed.  Heading is interpolated
    along the shortest arc to avoid wrap-around artefacts.
    """
    if len(points) < 2:
        return list(points)

    t0 = points[0].timestamp
    offsets = [(p.timestamp - t0).total_seconds() for p in points]
    total = offsets[-1]
    if total <= 0:
        return [points[0]]

    result: list[TracePoint] = []
    elapsed = 0.0
    j = 0  # index into source points

    while elapsed <= total + 0.1:
        while j < len(offsets) - 2 and offsets[j + 1] < elapsed:
            j += 1

        seg_dur = offsets[j + 1] - offsets[j]
        frac = (elapsed - offsets[j]) / seg_dur if seg_dur > 0 else 0.0
        frac = min(frac, 1.0)

        p_a, p_b = points[j], points[j + 1]

        lat = p_a.lat + frac * (p_b.lat - p_a.lat)
        lon = p_a.lon + frac * (p_b.lon - p_a.lon)
        speed = p_a.speed_mph + frac * (p_b.speed_mph - p_a.speed_mph)

        # Shortest-arc heading interpolation
        diff = (p_b.heading - p_a.heading + 540) % 360 - 180
        hdg = (p_a.heading + frac * diff) % 360

        result.append(
            TracePoint(
                lat=round(lat, 6),
                lon=round(lon, 6),
                speed_mph=round(speed, 1),
                heading=round(hdg, 1),
                timestamp=t0 + timedelta(seconds=elapsed),
            )
        )
        elapsed += interval_s

    return result


def bearing(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Compass bearing in degrees [0, 360), 0 = north, clockwise."""
    dlon = math.radians(lon2 - lon1)
    lat1_r = math.radians(lat1)
    lat2_r = math.radians(lat2)
    x = math.sin(dlon) * math.cos(lat2_r)
    y = math.cos(lat1_r) * math.sin(lat2_r) - math.sin(lat1_r) * math.cos(
        lat2_r
    ) * math.cos(dlon)
    return (math.degrees(math.atan2(x, y)) + 360) % 360


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Distance in meters between two lat/lon points."""
    return _core.haversine_km(lat1, lon1, lat2, lon2) * 1000.0


def interpolate_route(
    route: RouteSegment,
    departure_time: datetime,
    rng: random.Random,
) -> list[TracePoint]:
    """Walk the route polyline at ~60 s intervals, producing trace points."""
    coords = route.coords
    if len(coords) < 2:
        return []

    speeds = smooth_speeds(route.speeds_mps, route.distances_m)
    speeds, distances = inject_traffic_stops(speeds, route.distances_m, rng)

    orig_cum_dist = [0.0]
    for d in route.distances_m:
        orig_cum_dist.append(orig_cum_dist[-1] + d)
    total_orig_dist = orig_cum_dist[-1]

    cum_dist = [0.0]
    for d in distances:
        cum_dist.append(cum_dist[-1] + d)
    total_speed_dist = cum_dist[-1]

    seg_times: list[float] = []
    for spd, dist in zip(speeds, distances, strict=False):
        if spd < 0.01:
            seg_times.append(rng.uniform(15, 45))
        else:
            seg_times.append(dist / spd)

    points: list[TracePoint] = []
    current_time = departure_time
    elapsed = 0.0
    total_time = sum(seg_times)

    cum_time = [0.0]
    for t in seg_times:
        cum_time.append(cum_time[-1] + t)

    prev_heading: float | None = None

    while elapsed <= total_time + 0.1:
        seg_idx = int(np.searchsorted(cum_time[1:], elapsed, side="right"))
        seg_idx = min(seg_idx, len(speeds) - 1)

        seg_start_time = cum_time[seg_idx]
        seg_duration = seg_times[seg_idx]
        seg_frac = (
            min((elapsed - seg_start_time) / seg_duration, 1.0)
            if seg_duration > 0
            else 0.0
        )

        dist_along_speed = cum_dist[seg_idx] + seg_frac * distances[seg_idx]

        if total_speed_dist > 0:
            dist_along_geom = dist_along_speed * (total_orig_dist / total_speed_dist)
        else:
            dist_along_geom = 0.0

        geom_seg = int(
            np.searchsorted(orig_cum_dist[1:], dist_along_geom, side="right")
        )
        geom_seg = min(geom_seg, len(coords) - 2)

        seg_start_dist = orig_cum_dist[geom_seg]
        seg_len = orig_cum_dist[geom_seg + 1] - seg_start_dist
        if seg_len > 0:
            frac = min((dist_along_geom - seg_start_dist) / seg_len, 1.0)
        else:
            frac = 0.0

        lat = coords[geom_seg][0] + frac * (
            coords[geom_seg + 1][0] - coords[geom_seg][0]
        )
        lon = coords[geom_seg][1] + frac * (
            coords[geom_seg + 1][1] - coords[geom_seg][1]
        )

        speed_mps = speeds[seg_idx]
        speed_mph = mps_to_mph(speed_mps)

        hdg = bearing(
            coords[geom_seg][0],
            coords[geom_seg][1],
            coords[geom_seg + 1][0],
            coords[geom_seg + 1][1],
        )
        if prev_heading is not None and speed_mph < 1.0:
            hdg = prev_heading
        prev_heading = hdg

        points.append(
            TracePoint(
                lat=round(lat, 6),
                lon=round(lon, 6),
                speed_mph=round(speed_mph, 1),
                heading=round(hdg, 1),
                timestamp=current_time + timedelta(seconds=elapsed),
            )
        )

        elapsed += INTERVAL_S

    return points
