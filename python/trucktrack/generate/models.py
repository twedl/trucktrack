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
    errors: list[ErrorConfig] = field(default_factory=list)


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
