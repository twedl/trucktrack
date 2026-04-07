"""Truck speed caps, acceleration smoothing, and traffic-stop injection."""

from __future__ import annotations

import random

import numpy as np

# Truck speed caps by road class (m/s)
HIGHWAY_CAP = 29.0  # ~65 mph
ARTERIAL_CAP = 15.6  # ~35 mph
URBAN_CAP = 11.2  # ~25 mph

# Acceleration limits (m/s^2)
MAX_ACCEL = 0.5  # loaded truck, 0-60 in ~54s
MAX_DECEL = 1.5  # comfortable braking

# Traffic stop probability in urban segments
TRAFFIC_STOP_PROB = 0.12
TRAFFIC_STOP_DWELL_RANGE = (15.0, 45.0)


def classify_and_cap(speed_mps: float) -> float:
    """Cap a routing-engine speed to truck limits based on road class."""
    if speed_mps > 22.0:
        return min(speed_mps, HIGHWAY_CAP)
    elif speed_mps > 11.0:
        return min(speed_mps, ARTERIAL_CAP)
    else:
        return min(speed_mps, URBAN_CAP)


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
) -> tuple[list[float], list[float]]:
    """Inject traffic-signal stops at random urban intersections."""
    new_speeds: list[float] = []
    new_distances: list[float] = []

    for spd, dist in zip(speeds_mps, distances_m, strict=False):
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

    return new_speeds, new_distances


def mps_to_mph(speed_mps: float) -> float:
    return speed_mps * 2.23694
