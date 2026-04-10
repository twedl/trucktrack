"""Operational / behavioural pattern injection.

Each function takes a trace and returns a modified copy.  Signature:
``(points, rng, **params) -> list[TracePoint]``.
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable
from dataclasses import replace
from datetime import datetime, timedelta

from trucktrack.generate.interpolator import INTERVAL_S, haversine_m, offset_to_latlon
from trucktrack.generate.models import TracePoint

ErrorFn = Callable[..., list[TracePoint]]


def _shift_timestamps(
    points: list[TracePoint],
    delta: timedelta,
) -> list[TracePoint]:
    return [replace(p, timestamp=p.timestamp + delta) for p in points]


def _make_dwell_points(
    anchor: TracePoint,
    n: int,
    start_time: datetime,
    rng: random.Random,
    jitter_m: float = 2.0,
) -> list[TracePoint]:
    pts: list[TracePoint] = []
    for i in range(n):
        lat, lon = offset_to_latlon(
            anchor.lat,
            anchor.lon,
            rng.gauss(0, jitter_m),
            rng.gauss(0, jitter_m),
        )
        pts.append(
            TracePoint(
                lat=round(lat, 6),
                lon=round(lon, 6),
                speed_mph=0.0,
                heading=round(rng.uniform(0, 360), 1),
                timestamp=start_time + timedelta(seconds=i * INTERVAL_S),
            )
        )
    return pts


def privacy_shutoff(
    points: list[TracePoint],
    rng: random.Random,
    *,
    duration_s: float = 36_000.0,
) -> list[TracePoint]:
    if len(points) < 20:
        return points
    start_idx = rng.randint(int(len(points) * 0.7), int(len(points) * 0.85))
    n_remove = rng.randint(3, max(4, len(points) // 10))
    end_idx = min(start_idx + n_remove, len(points) - 2)
    actual_removed_s = (end_idx - start_idx) * INTERVAL_S
    extra = timedelta(seconds=duration_s - actual_removed_s)
    return points[:start_idx] + _shift_timestamps(points[end_idx:], extra)


def relay_driving(
    points: list[TracePoint],
    rng: random.Random,
    *,
    speed_threshold_mph: float = 2.0,
    min_stop_points: int = 3,
) -> list[TracePoint]:
    if len(points) < 10:
        return points

    segments: list[list[TracePoint]] = []
    current: list[TracePoint] = []
    stopped = 0

    for pt in points:
        if pt.speed_mph < speed_threshold_mph:
            stopped += 1
            if stopped <= min_stop_points:
                current.append(pt)
        else:
            if stopped > min_stop_points and current:
                while current and current[-1].speed_mph < speed_threshold_mph:
                    current.pop()
                if current:
                    segments.append(current)
                current = []
            stopped = 0
            current.append(pt)
    if current:
        segments.append(current)

    if not segments:
        return points

    result: list[TracePoint] = []
    t = points[0].timestamp
    for seg in segments:
        for pt in seg:
            result.append(replace(pt, timestamp=t))
            t += timedelta(seconds=INTERVAL_S)
    return result


def yard_dwell(
    points: list[TracePoint],
    rng: random.Random,
    *,
    duration_s: float = 14_400.0,
    position: str = "start",
) -> list[TracePoint]:
    if not points:
        return points
    n = max(1, int(duration_s / INTERVAL_S))
    if position == "start":
        anchor = points[0]
        t0 = anchor.timestamp - timedelta(seconds=duration_s)
        return _make_dwell_points(anchor, n, t0, rng) + points
    else:
        anchor = points[-1]
        t0 = anchor.timestamp + timedelta(seconds=INTERVAL_S)
        return points + _make_dwell_points(anchor, n, t0, rng)


def fuel_rest_stop(
    points: list[TracePoint],
    rng: random.Random,
    *,
    duration_s: float = 2_700.0,
    position_fraction: float | None = None,
) -> list[TracePoint]:
    if len(points) < 10:
        return points
    if position_fraction is None:
        position_fraction = rng.uniform(0.4, 0.6)
    idx = max(1, min(int(len(points) * position_fraction), len(points) - 2))
    anchor = points[idx]
    n_stop = max(1, int(duration_s / INTERVAL_S))
    t0 = anchor.timestamp + timedelta(seconds=INTERVAL_S)
    stop_pts = _make_dwell_points(anchor, n_stop, t0, rng, jitter_m=3.0)
    shift = timedelta(seconds=n_stop * INTERVAL_S)
    return points[: idx + 1] + stop_pts + _shift_timestamps(points[idx + 1 :], shift)


def weigh_station_stop(
    points: list[TracePoint],
    rng: random.Random,
    *,
    duration_s: float = 300.0,
    approach_points: int = 3,
) -> list[TracePoint]:
    if len(points) < 10:
        return points
    idx = rng.randint(int(len(points) * 0.3), int(len(points) * 0.6))
    result = list(points)

    for i in range(max(0, idx - approach_points), idx):
        result[i] = replace(
            result[i],
            speed_mph=round(result[i].speed_mph * 0.3, 1),
        )

    anchor = result[idx]
    n_stop = max(1, int(duration_s / INTERVAL_S))
    t0 = anchor.timestamp + timedelta(seconds=INTERVAL_S)
    stop_pts = _make_dwell_points(anchor, n_stop, t0, rng, jitter_m=1.5)
    shift = timedelta(seconds=n_stop * INTERVAL_S)
    return result[: idx + 1] + stop_pts + _shift_timestamps(result[idx + 1 :], shift)


def bobtail_segment(
    points: list[TracePoint],
    rng: random.Random,
    *,
    fraction: float = 0.3,
    speed_factor: float = 1.15,
) -> list[TracePoint]:
    if len(points) < 10:
        return points
    n = max(1, int(len(points) * fraction))
    start = rng.randint(0, len(points) - n)
    result = list(points)
    for i in range(start, start + n):
        if result[i].speed_mph > 5.0:
            result[i] = replace(
                result[i],
                speed_mph=round(result[i].speed_mph * speed_factor, 1),
            )
    return result


def off_route_detour(
    points: list[TracePoint],
    rng: random.Random,
    *,
    offset_meters: float = 500.0,
    n_points: int = 10,
) -> list[TracePoint]:
    if len(points) < n_points + 10:
        return points
    margin = len(points) // 5
    start = rng.randint(margin, len(points) - margin - n_points)
    angle = rng.uniform(0, 2 * math.pi)
    result = list(points)
    for i in range(start, start + n_points):
        progress = (i - start) / (n_points - 1)
        scale = math.sin(progress * math.pi)
        dx = offset_meters * scale * math.cos(angle)
        dy = offset_meters * scale * math.sin(angle)
        lat, lon = offset_to_latlon(result[i].lat, result[i].lon, dx, dy)
        result[i] = replace(result[i], lat=round(lat, 6), lon=round(lon, 6))
    return result


def loading_dwell(
    points: list[TracePoint],
    rng: random.Random,
    *,
    duration_s: float = 5_400.0,
    position: str = "start",
) -> list[TracePoint]:
    return yard_dwell(points, rng, duration_s=duration_s, position=position)


def device_power_cycle(
    points: list[TracePoint],
    rng: random.Random,
    *,
    gap_s: float = 180.0,
    drift_meters: float = 40.0,
    drift_points: int = 2,
) -> list[TracePoint]:
    if len(points) < 10:
        return points
    idx = rng.randint(len(points) // 4, 3 * len(points) // 4)
    n_remove = max(1, int(gap_s / INTERVAL_S))
    end_idx = min(idx + n_remove, len(points) - 1)
    actual_s = (end_idx - idx) * INTERVAL_S
    extra = timedelta(seconds=gap_s - actual_s)

    before = points[:idx]
    after_raw = points[end_idx:]

    # Shift all timestamps, then apply decaying drift to the first few
    after = _shift_timestamps(after_raw, extra)
    for j in range(min(drift_points, len(after))):
        decay = drift_meters * (1.0 - j / drift_points)
        angle = rng.uniform(0, 2 * math.pi)
        dx = decay * math.cos(angle)
        dy = decay * math.sin(angle)
        lat, lon = offset_to_latlon(after[j].lat, after[j].lon, dx, dy)
        after[j] = replace(after[j], lat=round(lat, 6), lon=round(lon, 6))

    return before + after


def traffic_jam(
    points: list[TracePoint],
    rng: random.Random,
    *,
    duration_s: float = 7_200.0,
    crawl_speed_mph: float = 3.0,
    full_stop_fraction: float = 0.4,
) -> list[TracePoint]:
    """Slow or stop a highway segment to simulate extreme traffic.

    A stretch of the trace (in the middle 30-70%) has its speeds crushed
    to *crawl_speed_mph* or zero, and stationary dwell points are inserted
    to pad the duration.  Simulates weather closures, construction,
    accidents, protests, etc.
    """
    if len(points) < 20:
        return points

    # Pick a window in the highway portion (middle 30-70%)
    start_idx = rng.randint(int(len(points) * 0.3), int(len(points) * 0.5))
    end_idx = rng.randint(int(len(points) * 0.5), int(len(points) * 0.7))
    if end_idx <= start_idx:
        end_idx = start_idx + 5

    # Slow existing points in the window to crawl or full stop
    result = list(points)
    for i in range(start_idx, min(end_idx, len(result))):
        if rng.random() < full_stop_fraction:
            result[i] = replace(result[i], speed_mph=0.0)
        else:
            result[i] = replace(
                result[i],
                speed_mph=round(rng.uniform(0.5, crawl_speed_mph), 1),
            )

    # Insert dwell points at the jam midpoint to pad the total delay
    existing_window_s = (end_idx - start_idx) * INTERVAL_S
    extra_s = max(0.0, duration_s - existing_window_s)
    n_dwell = int(extra_s / INTERVAL_S)
    if n_dwell > 0:
        mid = (start_idx + min(end_idx, len(result))) // 2
        anchor = result[mid]
        t0 = anchor.timestamp + timedelta(seconds=INTERVAL_S)
        dwell_pts = _make_dwell_points(anchor, n_dwell, t0, rng, jitter_m=1.0)
        # Override dwell headings to match travel direction (not random)
        for k in range(len(dwell_pts)):
            dwell_pts[k] = replace(dwell_pts[k], heading=anchor.heading)
        shift = timedelta(seconds=n_dwell * INTERVAL_S)
        result = (
            result[: mid + 1] + dwell_pts + _shift_timestamps(result[mid + 1 :], shift)
        )

    return result


def geofence_gap(
    points: list[TracePoint],
    rng: random.Random,
    *,
    center: tuple[float, float],
    radius_m: float = 1_000.0,
) -> list[TracePoint]:
    return [
        pt
        for pt in points
        if haversine_m(pt.lat, pt.lon, center[0], center[1]) > radius_m
    ]


OPERATIONAL_ERRORS: dict[str, ErrorFn] = {
    "privacy_shutoff": privacy_shutoff,
    "relay_driving": relay_driving,
    "yard_dwell": yard_dwell,
    "fuel_rest_stop": fuel_rest_stop,
    "weigh_station_stop": weigh_station_stop,
    "bobtail_segment": bobtail_segment,
    "off_route_detour": off_route_detour,
    "loading_dwell": loading_dwell,
    "traffic_jam": traffic_jam,
    "device_power_cycle": device_power_cycle,
    "geofence_gap": geofence_gap,
}
