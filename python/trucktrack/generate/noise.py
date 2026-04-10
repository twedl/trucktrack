"""Add realistic GPS noise to trace points."""

from __future__ import annotations

import math
import random

from trucktrack.generate.interpolator import offset_to_latlon
from trucktrack.generate.models import TracePoint


def apply_noise(
    points: list[TracePoint], noise_meters: float, rng: random.Random
) -> list[TracePoint]:
    """Add Gaussian GPS jitter to lat/lon, speed, and heading."""
    if noise_meters <= 0:
        return points

    sigma_m = noise_meters / 1.18

    noisy: list[TracePoint] = []
    for pt in points:
        angle = rng.uniform(0, 2 * math.pi)
        offset_m = rng.gauss(0, sigma_m)
        dx = offset_m * math.cos(angle)
        dy = offset_m * math.sin(angle)

        lat, lon = offset_to_latlon(pt.lat, pt.lon, dx, dy)

        speed_noise = rng.gauss(0, 0.5)
        noisy_speed = max(0.0, pt.speed_mph + speed_noise)
        if pt.speed_mph < 0.1:
            noisy_speed = 0.0

        if pt.speed_mph > 2.0:
            heading_noise = rng.gauss(0, 1.0)
            noisy_heading = (pt.heading + heading_noise) % 360
        else:
            noisy_heading = pt.heading

        noisy.append(
            TracePoint(
                lat=round(lat, 6),
                lon=round(lon, 6),
                speed_mph=round(noisy_speed, 1),
                heading=round(noisy_heading, 1),
                timestamp=pt.timestamp,
            )
        )

    return noisy
