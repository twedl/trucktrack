"""Truck speed caps, acceleration smoothing, traffic stops, and rush hours."""

from __future__ import annotations

import math
import random
from datetime import UTC, datetime
from zoneinfo import ZoneInfo

import numpy as np

# Truck speed caps by road class (m/s)
HIGHWAY_CAP = 29.0  # ~104 km/h
ARTERIAL_CAP = 15.6  # ~56 km/h
URBAN_CAP = 11.2  # ~40 km/h
ROAD_CLASS_CAP = {"highway": HIGHWAY_CAP, "arterial": ARTERIAL_CAP, "urban": URBAN_CAP}

# Acceleration limits (m/s^2)
MAX_ACCEL = 0.5  # loaded truck, 0-60 in ~54s
MAX_DECEL = 1.5  # comfortable braking

# Traffic stop probability in urban segments
TRAFFIC_STOP_PROB = 0.12
TRAFFIC_STOP_DWELL_RANGE = (15.0, 45.0)

# Recurring congestion, in Ontario local time.  Each road class loses up
# to its depth at the height of a full-weight dip.
LOCAL_TZ = ZoneInfo("America/Toronto")
RUSH_HOUR_DEPTH = {"highway": 0.5, "arterial": 0.3, "urban": 0.15}
# (centre hour, width in hours) of the morning, midday and evening dips.
RUSH_HOUR_DIPS = ((8.0, 1.0), (13.0, 2.5), (16.5, 1.25))
# Weight of each dip by weekday, Monday first.
RUSH_HOUR_WEIGHTS = (
    (0.9, 0.35, 0.9),
    (1.0, 0.35, 1.0),
    (1.0, 0.35, 1.0),
    (1.0, 0.35, 1.0),
    (0.8, 0.45, 1.0),
    (0.0, 0.35, 0.0),
    (0.0, 0.25, 0.0),
)
# Rush hours run deepest in Toronto and ease off with distance from it:
# the depth scales from RUSH_HOUR_SCALE_PEAK downtown towards
# RUSH_HOUR_SCALE_FLOOR, by e^(-km / RUSH_HOUR_SCALE_KM).
TORONTO = (43.6532, -79.3832)  # City Hall
RUSH_HOUR_SCALE_PEAK = 1.6
RUSH_HOUR_SCALE_FLOOR = 0.4
RUSH_HOUR_SCALE_KM = 40.0


def road_class(speed_mps: float) -> str:
    """Road class of a routing-engine speed: highway, arterial, or urban."""
    if speed_mps > 22.0:
        return "highway"
    elif speed_mps > 11.0:
        return "arterial"
    else:
        return "urban"


def classify_and_cap(speed_mps: float) -> float:
    """Cap a routing-engine speed to truck limits based on road class."""
    return min(speed_mps, ROAD_CLASS_CAP[road_class(speed_mps)])


def rush_hour_scale(lat: float, lon: float) -> float:
    """How deep rush hours run at (*lat*, *lon*), as a multiple of the class depth.

    1.6 in downtown Toronto, 1.0 about 28 km out (Brampton, Pickering),
    0.7 at 60 km (Hamilton, Oshawa) and 0.5 at 100 km (Kitchener),
    approaching 0.4 beyond.  A road class's rush hour is the same shape
    everywhere but not the same depth, so a class-wide pattern misses
    edges near the core and far from it.
    """
    dlat = math.radians(lat - TORONTO[0])
    dlon = math.radians(lon - TORONTO[1]) * math.cos(math.radians(TORONTO[0]))
    km = 6371.0 * math.hypot(dlat, dlon)
    spread = RUSH_HOUR_SCALE_PEAK - RUSH_HOUR_SCALE_FLOOR
    return RUSH_HOUR_SCALE_FLOOR + spread * math.exp(-km / RUSH_HOUR_SCALE_KM)


def congestion_factor(road_class: str, when: datetime, scale: float = 1.0) -> float:
    """Recurring-congestion speed multiplier for *road_class* at *when*.

    1.0 overnight.  Weekdays dip around 08:00 and 16:30 local time with
    a shallower dip at midday; weekends have only the midday dip.  Each
    dip multiplies speed by ``1 - depth * weight * bump``, where the
    bump is a Gaussian in the local hour that peaks at 1 and the depth
    is the class's times *scale* (see :func:`rush_hour_scale`).  This
    is the ground truth that per-edge hour-of-week speeds should
    recover.  Naive *when* is taken as UTC.
    """
    if when.tzinfo is None:
        when = when.replace(tzinfo=UTC)
    local = when.astimezone(LOCAL_TZ)
    hour = local.hour + local.minute / 60 + local.second / 3600
    depth = RUSH_HOUR_DEPTH[road_class] * scale
    factor = 1.0
    for (centre, width), weight in zip(
        RUSH_HOUR_DIPS, RUSH_HOUR_WEIGHTS[local.weekday()], strict=True
    ):
        bump = math.exp(-0.5 * ((hour - centre) / width) ** 2)
        factor *= 1.0 - depth * weight * bump
    return factor


def smooth_speeds(speeds_mps: list[float], distances_m: list[float]) -> list[float]:
    """Apply acceleration/deceleration constraints to a speed sequence."""
    n = len(speeds_mps)
    if n == 0:
        return speeds_mps

    smoothed = [classify_and_cap(s) for s in speeds_mps]

    for i in range(1, n):
        dist = distances_m[i - 1] if i - 1 < len(distances_m) else distances_m[-1]
        if dist < 0.1:
            smoothed[i] = smoothed[i - 1]
            continue
        max_v = float(np.sqrt(smoothed[i - 1] ** 2 + 2 * MAX_ACCEL * dist))
        smoothed[i] = min(smoothed[i], max_v)

    for i in range(n - 2, -1, -1):
        dist = distances_m[i] if i < len(distances_m) else distances_m[-1]
        if dist < 0.1:
            smoothed[i] = smoothed[i + 1]
            continue
        max_v = float(np.sqrt(smoothed[i + 1] ** 2 + 2 * MAX_DECEL * dist))
        smoothed[i] = min(smoothed[i], max_v)

    return smoothed


def inject_traffic_stops(
    speeds_mps: list[float],
    distances_m: list[float],
    rng: random.Random,
) -> tuple[list[float], list[float], list[int]]:
    """Inject traffic-signal stops at random urban intersections.

    Returns the new speeds and distances, and for each new segment the
    index of the segment it was cut from.
    """
    new_speeds: list[float] = []
    new_distances: list[float] = []
    sources: list[int] = []

    for i, (spd, dist) in enumerate(zip(speeds_mps, distances_m, strict=False)):
        is_urban = spd < 15.0
        if is_urban and dist > 20 and rng.random() < TRAFFIC_STOP_PROB:
            decel_dist = min(dist * 0.4, 50.0)
            accel_dist = min(dist * 0.4, 50.0)
            mid_dist = dist - decel_dist - accel_dist

            new_speeds.append(spd * 0.5)
            new_distances.append(decel_dist)

            new_speeds.append(0.0)
            new_distances.append(0.1)

            new_speeds.append(spd * 0.5)
            new_distances.append(accel_dist)

            if mid_dist > 1.0:
                new_speeds.append(spd)
                new_distances.append(mid_dist)
        else:
            new_speeds.append(spd)
            new_distances.append(dist)
        sources.extend([i] * (len(new_speeds) - len(sources)))

    return new_speeds, new_distances, sources


def mps_to_kmh(speed_mps: float) -> float:
    return speed_mps * 3.6
