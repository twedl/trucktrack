"""Operational / behavioural pattern injection.

Each function takes a trace and returns a modified copy.  Signature:
``(points, rng, **params) -> list[TracePoint]``.
"""

from __future__ import annotations

import math
import random
from datetime import timedelta

from trucktrack.generate.interpolator import INTERVAL_S, haversine_m
from trucktrack.generate.models import TracePoint


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _shift_timestamps(
    points: list[TracePoint], delta: timedelta,
) -> list[TracePoint]:
    return [
        TracePoint(
            lat=p.lat, lon=p.lon, speed_mph=p.speed_mph,
            heading=p.heading, timestamp=p.timestamp + delta,
        )
        for p in points
    ]


def _make_dwell_points(
    anchor: TracePoint,
    n: int,
    start_time,
    rng: random.Random,
    jitter_m: float = 2.0,
) -> list[TracePoint]:
    pts: list[TracePoint] = []
    for i in range(n):
        dlat = rng.gauss(0, jitter_m) / 111_320.0
        dlon = rng.gauss(0, jitter_m) / (
            111_320.0 * math.cos(math.radians(anchor.lat))
        )
        pts.append(
            TracePoint(
                lat=round(anchor.lat + dlat, 6),
                lon=round(anchor.lon + dlon, 6),
                speed_mph=0.0,
                heading=round(rng.uniform(0, 360), 1),
                timestamp=start_time + timedelta(seconds=i * INTERVAL_S),
            )
        )
    return pts


# ---------------------------------------------------------------------------
# 1. Privacy shutoff — long gap, minimal position change
# ---------------------------------------------------------------------------

def privacy_shutoff(
    points: list[TracePoint],
    rng: random.Random,
    *,
    duration_s: float = 36_000.0,
) -> list[TracePoint]:
    if len(points) < 20:
        return points
    start_idx = rng.randint(int(len(points) * 0.7), int(len(points) * 0.85))
    # Remove a small number of points but inject a large time gap
    n_remove = rng.randint(3, max(4, len(points) // 10))
    end_idx = min(start_idx + n_remove, len(points) - 2)
    actual_removed_s = (end_idx - start_idx) * INTERVAL_S
    extra = timedelta(seconds=duration_s - actual_removed_s)
    before = points[:start_idx]
    after = _shift_timestamps(points[end_idx:], extra)
    return before + after


# ---------------------------------------------------------------------------
# 2. Relay driving — remove rest stops, continuous movement
# ---------------------------------------------------------------------------

def relay_driving(
    points: list[TracePoint],
    rng: random.Random,
    *,
    speed_threshold_mph: float = 2.0,
    min_stop_points: int = 3,
) -> list[TracePoint]:
    if len(points) < 10:
        return points

    # Collect moving segments, dropping clusters of stopped points
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
                # Trim trailing stopped points
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

    # Stitch with continuous timestamps
    result: list[TracePoint] = []
    t = points[0].timestamp
    for seg in segments:
        for pt in seg:
            result.append(
                TracePoint(
                    lat=pt.lat, lon=pt.lon, speed_mph=pt.speed_mph,
                    heading=pt.heading, timestamp=t,
                )
            )
            t += timedelta(seconds=INTERVAL_S)
    return result


# ---------------------------------------------------------------------------
# 3. Yard dwell — long stationary period at start or end
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# 4. Fuel / rest stop — medium stop mid-route
# ---------------------------------------------------------------------------

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
    idx = int(len(points) * position_fraction)
    idx = max(1, min(idx, len(points) - 2))
    anchor = points[idx]
    n_stop = max(1, int(duration_s / INTERVAL_S))
    t0 = anchor.timestamp + timedelta(seconds=INTERVAL_S)
    stop_pts = _make_dwell_points(anchor, n_stop, t0, rng, jitter_m=3.0)
    shift = timedelta(seconds=n_stop * INTERVAL_S)
    return points[: idx + 1] + stop_pts + _shift_timestamps(points[idx + 1 :], shift)


# ---------------------------------------------------------------------------
# 5. Weigh station / border crossing — brief stop with slow approach
# ---------------------------------------------------------------------------

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

    # Slow approach
    for i in range(max(0, idx - approach_points), idx):
        result[i] = TracePoint(
            lat=result[i].lat, lon=result[i].lon,
            speed_mph=round(result[i].speed_mph * 0.3, 1),
            heading=result[i].heading, timestamp=result[i].timestamp,
        )

    anchor = result[idx]
    n_stop = max(1, int(duration_s / INTERVAL_S))
    t0 = anchor.timestamp + timedelta(seconds=INTERVAL_S)
    stop_pts = _make_dwell_points(anchor, n_stop, t0, rng, jitter_m=1.5)
    shift = timedelta(seconds=n_stop * INTERVAL_S)
    return result[: idx + 1] + stop_pts + _shift_timestamps(result[idx + 1 :], shift)


# ---------------------------------------------------------------------------
# 6. Bobtail / empty repositioning — higher speeds (no trailer)
# ---------------------------------------------------------------------------

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
            result[i] = TracePoint(
                lat=result[i].lat, lon=result[i].lon,
                speed_mph=round(result[i].speed_mph * speed_factor, 1),
                heading=result[i].heading, timestamp=result[i].timestamp,
            )
    return result


# ---------------------------------------------------------------------------
# 7. Off-route detour — bell-curve lateral offset
# ---------------------------------------------------------------------------

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
        dlat = dy / 111_320.0
        dlon = dx / (111_320.0 * math.cos(math.radians(result[i].lat)))
        result[i] = TracePoint(
            lat=round(result[i].lat + dlat, 6),
            lon=round(result[i].lon + dlon, 6),
            speed_mph=result[i].speed_mph,
            heading=result[i].heading,
            timestamp=result[i].timestamp,
        )
    return result


# ---------------------------------------------------------------------------
# 8. Loading / unloading dwell — extended stop at origin or destination
# ---------------------------------------------------------------------------

def loading_dwell(
    points: list[TracePoint],
    rng: random.Random,
    *,
    duration_s: float = 5_400.0,
    position: str = "start",
) -> list[TracePoint]:
    return yard_dwell(points, rng, duration_s=duration_s, position=position)


# ---------------------------------------------------------------------------
# 9. Device power cycle — small gap + cold-start drift on resume
# ---------------------------------------------------------------------------

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

    after: list[TracePoint] = []
    for j, pt in enumerate(after_raw):
        ts = pt.timestamp + extra
        if j < drift_points:
            decay = drift_meters * (1.0 - j / drift_points)
            angle = rng.uniform(0, 2 * math.pi)
            dlat = (decay * math.sin(angle)) / 111_320.0
            dlon = (decay * math.cos(angle)) / (
                111_320.0 * math.cos(math.radians(pt.lat))
            )
            after.append(
                TracePoint(
                    lat=round(pt.lat + dlat, 6),
                    lon=round(pt.lon + dlon, 6),
                    speed_mph=pt.speed_mph, heading=pt.heading,
                    timestamp=ts,
                )
            )
        else:
            after.append(
                TracePoint(
                    lat=pt.lat, lon=pt.lon, speed_mph=pt.speed_mph,
                    heading=pt.heading, timestamp=ts,
                )
            )
    return before + after


# ---------------------------------------------------------------------------
# 10. Geofence compliance gap — remove points within a zone
# ---------------------------------------------------------------------------

def geofence_gap(
    points: list[TracePoint],
    rng: random.Random,
    *,
    center: tuple[float, float],
    radius_m: float = 1_000.0,
) -> list[TracePoint]:
    return [
        pt for pt in points
        if haversine_m(pt.lat, pt.lon, center[0], center[1]) > radius_m
    ]


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

OPERATIONAL_ERRORS: dict[str, object] = {
    "privacy_shutoff": privacy_shutoff,
    "relay_driving": relay_driving,
    "yard_dwell": yard_dwell,
    "fuel_rest_stop": fuel_rest_stop,
    "weigh_station_stop": weigh_station_stop,
    "bobtail_segment": bobtail_segment,
    "off_route_detour": off_route_detour,
    "loading_dwell": loading_dwell,
    "device_power_cycle": device_power_cycle,
    "geofence_gap": geofence_gap,
}
