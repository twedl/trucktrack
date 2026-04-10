from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from uuid import uuid4

DEFAULT_VALHALLA_URL = "http://localhost:8002"


@dataclass
class ErrorConfig:
    """A single error to inject into the trace.

    *error_type* selects the injector function (e.g. ``"signal_dropout"``).
    *probability* controls how often the error fires (0–1).
    *params* is forwarded as ``**kwargs`` to the injector.
    """

    error_type: str
    probability: float = 1.0
    params: dict[str, object] = field(default_factory=dict)


def default_error_profile() -> list[ErrorConfig]:
    """Realistic mix of errors.  Over 1000 trips every type appears at least once."""
    return [
        # --- Physical GPS errors ---
        ErrorConfig("jitter_at_rest", probability=0.25),
        ErrorConfig("signal_dropout", probability=0.08),
        ErrorConfig("cold_start_drift", probability=0.10),
        ErrorConfig("device_power_cycle", probability=0.07),
        ErrorConfig("multipath", probability=0.05, params={"count": 2}),
        ErrorConfig("speed_heading_desync", probability=0.05),
        ErrorConfig("frozen_fix", probability=0.03),
        ErrorConfig("timestamp_glitch", probability=0.01, params={"count": 1}),
        ErrorConfig("coordinate_corruption", probability=0.01, params={"count": 1}),
        # --- Operational patterns ---
        ErrorConfig("fuel_rest_stop", probability=0.20),
        ErrorConfig("loading_dwell", probability=0.15),
        ErrorConfig("weigh_station_stop", probability=0.10),
        ErrorConfig("bobtail_segment", probability=0.08),
        ErrorConfig("yard_dwell", probability=0.05),
        ErrorConfig("traffic_jam", probability=0.06),
        ErrorConfig("off_route_detour", probability=0.04),
        ErrorConfig("privacy_shutoff", probability=0.03),
        ErrorConfig(
            "geofence_gap",
            probability=0.02,
            params={
                "center": (44.25, -76.5),
                "radius_m": 2000,
            },
        ),
        ErrorConfig("relay_driving", probability=0.01),
    ]


@dataclass
class TripConfig:
    origin: tuple[float, float]  # (lat, lon)
    destination: tuple[float, float]  # (lat, lon)
    departure_time: datetime
    trip_id: str = field(default_factory=lambda: uuid4().hex[:8])
    gps_noise_meters: float = 3.0
    seed: int | None = None
    origin_maneuver: str = "alley_dock"
    destination_maneuver: str = "alley_dock"
    valhalla_url: str = DEFAULT_VALHALLA_URL
    errors: list[ErrorConfig] = field(default_factory=default_error_profile)


@dataclass
class TracePoint:
    lat: float
    lon: float
    speed_mph: float
    heading: float  # 0-360, 0=north, clockwise
    timestamp: datetime


@dataclass
class RouteSegment:
    coords: list[tuple[float, float]]  # list of (lat, lon)
    speeds_mps: list[float]  # per-segment speed in m/s
    distances_m: list[float]  # per-segment distance in m
    total_distance_m: float
    total_duration_s: float
