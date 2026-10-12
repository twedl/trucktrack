"""Walk a route polyline at a device's fix times to produce GPS trace points."""

from __future__ import annotations

import math
import random
import statistics
from dataclasses import replace
from datetime import datetime, timedelta

import numpy as np

from trucktrack.generate.models import (
    DEFAULT_INTERVAL_S,
    RouteSegment,
    SamplingConfig,
    TracePoint,
)
from trucktrack.generate.speed_profile import (
    inject_traffic_stops,
    mps_to_kmh,
    smooth_speeds,
)


class FixSchedule:
    """A device's fix times, continuous across the phases of a trace.

    Each phase takes the fix times that fall inside it, so a gap that
    starts in one phase ends in the next instead of restarting the
    clock at every phase boundary.
    """

    def __init__(
        self, sampling: SamplingConfig, start: datetime, rng: random.Random
    ) -> None:
        self._sampling = sampling
        self._rng = rng
        self._next = start

    def until(self, end: datetime) -> list[datetime]:
        """Take every remaining fix time at or before *end*."""
        times: list[datetime] = []
        while self._next <= end:
            times.append(self._next)
            self._next += timedelta(seconds=self._sampling.next_gap(self._rng))
        return times


def median_interval_s(points: list[TracePoint]) -> float:
    """Median positive gap between consecutive points (60 s if none).

    Error injectors use this to size dwell, dropout and window lengths
    in fixes, so they work at whatever rate the trace was sampled.
    """
    gaps = [
        dt
        for a, b in zip(points, points[1:], strict=False)
        if (dt := (b.timestamp - a.timestamp).total_seconds()) > 0
    ]
    return statistics.median(gaps) if gaps else DEFAULT_INTERVAL_S


def offset_to_latlon(
    base_lat: float,
    base_lon: float,
    dx_east: float,
    dy_north: float,
) -> tuple[float, float]:
    """Shift a lat/lon by a meter-scale offset.  Returns ``(lat, lon)``."""
    dlat = dy_north / 111_320.0
    dlon = dx_east / (111_320.0 * math.cos(math.radians(base_lat)))
    return base_lat + dlat, base_lon + dlon


def resample_trace(points: list[TracePoint], schedule: FixSchedule) -> list[TracePoint]:
    """Sample densely spaced trace points at the schedule's fix times.

    Takes every scheduled fix up to the last point's timestamp.
    Linearly interpolates lat, lon, and speed.  Each point's heading is
    the course to the next point, so a fix takes the heading of the
    segment it falls in rather than a blend of two courses.
    """
    if not points:
        return []
    if len(points) == 1:
        return [
            replace(points[0], timestamp=t) for t in schedule.until(points[0].timestamp)
        ]

    t0 = points[0].timestamp
    offsets = [(p.timestamp - t0).total_seconds() for p in points]

    result: list[TracePoint] = []
    j = 0  # index into source points

    for t in schedule.until(points[-1].timestamp):
        elapsed = (t - t0).total_seconds()
        while j < len(offsets) - 2 and offsets[j + 1] < elapsed:
            j += 1

        seg_dur = offsets[j + 1] - offsets[j]
        frac = (elapsed - offsets[j]) / seg_dur if seg_dur > 0 else 0.0
        frac = min(max(frac, 0.0), 1.0)

        p_a, p_b = points[j], points[j + 1]

        lat = p_a.lat + frac * (p_b.lat - p_a.lat)
        lon = p_a.lon + frac * (p_b.lon - p_a.lon)
        speed = p_a.speed_kmh + frac * (p_b.speed_kmh - p_a.speed_kmh)

        hdg = p_b.heading if frac >= 1.0 else p_a.heading

        result.append(
            TracePoint(
                lat=round(lat, 6),
                lon=round(lon, 6),
                speed_kmh=round(speed, 1),
                heading=round(hdg, 1),
                timestamp=t,
            )
        )

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
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(math.radians(lat1))
        * math.cos(math.radians(lat2))
        * math.sin(dlon / 2) ** 2
    )
    return 6_371_000 * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def interpolate_route(
    route: RouteSegment,
    departure_time: datetime,
    rng: random.Random,
    schedule: FixSchedule,
) -> tuple[list[TracePoint], datetime]:
    """Walk the route polyline at the schedule's fix times.

    Returns the trace points and the arrival time.  The arrival time
    comes from the speed profile, not the last fix, so a long gap at
    the end of the drive doesn't pull the arrival earlier.
    """
    coords = route.coords
    if len(coords) < 2:
        return [], departure_time

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
    total_time = sum(seg_times)
    arrival_time = departure_time + timedelta(seconds=total_time)

    cum_time = [0.0]
    for t in seg_times:
        cum_time.append(cum_time[-1] + t)

    prev_heading: float | None = None
    # Segment end times / distances, as arrays once rather than per fix.
    seg_end_times = np.asarray(cum_time[1:])
    geom_seg_ends = np.asarray(orig_cum_dist[1:])

    for fix_time in schedule.until(arrival_time):
        elapsed = (fix_time - departure_time).total_seconds()
        seg_idx = int(np.searchsorted(seg_end_times, elapsed, side="right"))
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

        geom_seg = int(np.searchsorted(geom_seg_ends, dist_along_geom, side="right"))
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
        speed_kmh = mps_to_kmh(speed_mps)

        hdg = bearing(
            coords[geom_seg][0],
            coords[geom_seg][1],
            coords[geom_seg + 1][0],
            coords[geom_seg + 1][1],
        )
        if prev_heading is not None and speed_kmh < 1.6:
            hdg = prev_heading
        prev_heading = hdg

        points.append(
            TracePoint(
                lat=round(lat, 6),
                lon=round(lon, 6),
                speed_kmh=round(speed_kmh, 1),
                heading=round(hdg, 1),
                timestamp=fix_time,
            )
        )

    return points, arrival_time
