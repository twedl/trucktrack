"""Physical GPS error injection.

Each function takes a trace and returns a modified copy.  Signature:
``(points, rng, **params) -> list[TracePoint]``.
"""

from __future__ import annotations

import math
import random
from datetime import timedelta

from trucktrack.generate.interpolator import INTERVAL_S
from trucktrack.generate.models import TracePoint


# ---------------------------------------------------------------------------
# 1. Signal dropout — remove a contiguous stretch of points
# ---------------------------------------------------------------------------

def signal_dropout(
    points: list[TracePoint],
    rng: random.Random,
    *,
    duration_s: float = 300.0,
) -> list[TracePoint]:
    if len(points) < 10:
        return points
    n_remove = max(1, int(duration_s / INTERVAL_S))
    margin = len(points) // 5
    hi = max(margin, len(points) - margin - n_remove)
    start = rng.randint(margin, hi)
    end = min(start + n_remove, len(points))
    return points[:start] + points[end:]


# ---------------------------------------------------------------------------
# 2. Cold-start drift — large positional error after existing gaps
# ---------------------------------------------------------------------------

def cold_start_drift(
    points: list[TracePoint],
    rng: random.Random,
    *,
    n_points: int = 3,
    drift_meters: float = 50.0,
) -> list[TracePoint]:
    if len(points) < 2:
        return points
    result = list(points)
    for i in range(1, len(result)):
        gap = (result[i].timestamp - result[i - 1].timestamp).total_seconds()
        if gap > INTERVAL_S * 2:
            for j in range(i, min(i + n_points, len(result))):
                decay = drift_meters * (1.0 - (j - i) / n_points)
                angle = rng.uniform(0, 2 * math.pi)
                dlat = (decay * math.sin(angle)) / 111_320.0
                dlon = (decay * math.cos(angle)) / (
                    111_320.0 * math.cos(math.radians(result[j].lat))
                )
                result[j] = TracePoint(
                    lat=round(result[j].lat + dlat, 6),
                    lon=round(result[j].lon + dlon, 6),
                    speed_mph=result[j].speed_mph,
                    heading=result[j].heading,
                    timestamp=result[j].timestamp,
                )
    return result


# ---------------------------------------------------------------------------
# 3. Urban-canyon / multipath — large position jumps with speed spikes
# ---------------------------------------------------------------------------

def multipath(
    points: list[TracePoint],
    rng: random.Random,
    *,
    count: int = 3,
    offset_min_m: float = 50.0,
    offset_max_m: float = 200.0,
) -> list[TracePoint]:
    if len(points) < 5:
        return points
    result = list(points)
    indices = rng.sample(
        range(1, len(result) - 1), min(count, len(result) - 2)
    )
    for i in indices:
        offset = rng.uniform(offset_min_m, offset_max_m)
        angle = rng.uniform(0, 2 * math.pi)
        dlat = (offset * math.sin(angle)) / 111_320.0
        dlon = (offset * math.cos(angle)) / (
            111_320.0 * math.cos(math.radians(result[i].lat))
        )
        result[i] = TracePoint(
            lat=round(result[i].lat + dlat, 6),
            lon=round(result[i].lon + dlon, 6),
            speed_mph=round(rng.uniform(60, 120), 1),
            heading=result[i].heading,
            timestamp=result[i].timestamp,
        )
    return result


# ---------------------------------------------------------------------------
# 4. Stuck / frozen fix — same lat/lon repeated while truck moves
# ---------------------------------------------------------------------------

def frozen_fix(
    points: list[TracePoint],
    rng: random.Random,
    *,
    duration_s: float = 300.0,
) -> list[TracePoint]:
    if len(points) < 10:
        return points
    n_frozen = max(2, int(duration_s / INTERVAL_S))
    margin = len(points) // 5
    hi = max(margin, len(points) - margin - n_frozen)
    start = rng.randint(margin, hi)
    end = min(start + n_frozen, len(points))
    frozen_lat = points[start].lat
    frozen_lon = points[start].lon
    result = list(points)
    for i in range(start, end):
        result[i] = TracePoint(
            lat=frozen_lat,
            lon=frozen_lon,
            speed_mph=0.0,
            heading=result[i].heading,
            timestamp=result[i].timestamp,
        )
    return result


# ---------------------------------------------------------------------------
# 5. Timestamp glitch — jumps, duplicates, or backwards timestamps
# ---------------------------------------------------------------------------

def timestamp_glitch(
    points: list[TracePoint],
    rng: random.Random,
    *,
    count: int = 2,
) -> list[TracePoint]:
    if len(points) < 5:
        return points
    result = list(points)
    pool = range(2, len(result) - 2)
    indices = rng.sample(pool, min(count, len(pool)))
    for i in indices:
        mode = rng.choice(["duplicate", "jump_forward", "jump_backward"])
        pt = result[i]
        if mode == "duplicate":
            ts = result[i - 1].timestamp
        elif mode == "jump_forward":
            ts = pt.timestamp + timedelta(hours=rng.uniform(1, 12))
        else:
            ts = pt.timestamp - timedelta(hours=rng.uniform(1, 6))
        result[i] = TracePoint(
            lat=pt.lat, lon=pt.lon, speed_mph=pt.speed_mph,
            heading=pt.heading, timestamp=ts,
        )
    return result


# ---------------------------------------------------------------------------
# 6. Coordinate corruption — truncation, sign flip, lat/lon swap
# ---------------------------------------------------------------------------

def coordinate_corruption(
    points: list[TracePoint],
    rng: random.Random,
    *,
    count: int = 2,
) -> list[TracePoint]:
    if len(points) < 5:
        return points
    result = list(points)
    indices = rng.sample(range(len(result)), min(count, len(result)))
    for i in indices:
        pt = result[i]
        mode = rng.choice(["truncate", "flip_lat", "flip_lon", "swap"])
        if mode == "truncate":
            lat, lon = round(pt.lat, 2), round(pt.lon, 2)
        elif mode == "flip_lat":
            lat, lon = -pt.lat, pt.lon
        elif mode == "flip_lon":
            lat, lon = pt.lat, -pt.lon
        else:  # swap
            lat, lon = round(pt.lon, 6), round(pt.lat, 6)
        result[i] = TracePoint(
            lat=lat, lon=lon, speed_mph=pt.speed_mph,
            heading=pt.heading, timestamp=pt.timestamp,
        )
    return result


# ---------------------------------------------------------------------------
# 7. Speed / heading desync — zero speed while moving, lagged heading
# ---------------------------------------------------------------------------

def speed_heading_desync(
    points: list[TracePoint],
    rng: random.Random,
    *,
    duration_s: float = 300.0,
) -> list[TracePoint]:
    if len(points) < 10:
        return points
    n_affected = max(2, int(duration_s / INTERVAL_S))
    margin = len(points) // 5
    hi = max(margin, len(points) - margin - n_affected)
    start = rng.randint(margin, hi)
    end = min(start + n_affected, len(points))
    heading_lag = rng.randint(2, 5)
    result = list(points)
    for i in range(start, end):
        lagged = result[max(i - heading_lag, 0)].heading
        result[i] = TracePoint(
            lat=result[i].lat, lon=result[i].lon,
            speed_mph=0.0, heading=lagged,
            timestamp=result[i].timestamp,
        )
    return result


# ---------------------------------------------------------------------------
# 8. Jitter at rest — random walk on stationary points
# ---------------------------------------------------------------------------

def jitter_at_rest(
    points: list[TracePoint],
    rng: random.Random,
    *,
    walk_meters: float = 10.0,
) -> list[TracePoint]:
    result = list(points)
    walk_lat = 0.0
    walk_lon = 0.0
    for i in range(len(result)):
        if result[i].speed_mph < 1.0:
            step_lat = rng.gauss(0, walk_meters / 3) / 111_320.0
            step_lon = rng.gauss(0, walk_meters / 3) / (
                111_320.0 * math.cos(math.radians(result[i].lat))
            )
            walk_lat += step_lat
            walk_lon += step_lon
            result[i] = TracePoint(
                lat=round(result[i].lat + walk_lat, 6),
                lon=round(result[i].lon + walk_lon, 6),
                speed_mph=round(rng.uniform(0, 0.5), 1),
                heading=round(rng.uniform(0, 360), 1),
                timestamp=result[i].timestamp,
            )
        else:
            walk_lat = 0.0
            walk_lon = 0.0
    return result


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

GPS_ERRORS: dict[str, object] = {
    "signal_dropout": signal_dropout,
    "cold_start_drift": cold_start_drift,
    "multipath": multipath,
    "frozen_fix": frozen_fix,
    "timestamp_glitch": timestamp_glitch,
    "coordinate_corruption": coordinate_corruption,
    "speed_heading_desync": speed_heading_desync,
    "jitter_at_rest": jitter_at_rest,
}
