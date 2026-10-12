from __future__ import annotations

import random
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from uuid import uuid4

DEFAULT_VALHALLA_URL = "http://localhost:8002"
DEFAULT_INTERVAL_S = 60.0


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
        ErrorConfig("stale_reemission", probability=0.04, params={"count": 2}),
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


@dataclass(frozen=True)
class SamplingConfig:
    """When a truck's GPS device reports a fix.

    Most gaps between fixes are *interval_s*.  With probability
    *irregular_prob* a gap is instead a whole number of seconds drawn
    uniformly from [*irregular_min_s*, *irregular_max_s*].  Gaps of
    *long_gap_s* (the truck keeps moving, no fixes) occur at an average
    rate of *long_gaps_per_hour* per hour of reporting.  The default is
    a fixed 60 s schedule.
    """

    interval_s: float = DEFAULT_INTERVAL_S
    irregular_prob: float = 0.0
    irregular_min_s: int = 1
    irregular_max_s: int = 60
    long_gaps_per_hour: float = 0.0
    long_gap_s: float = 1800.0

    def next_gap(self, rng: random.Random) -> float:
        """Draw the gap in seconds before the next fix."""
        if rng.random() < self.long_gaps_per_hour * self.interval_s / 3600.0:
            return self.long_gap_s
        if rng.random() < self.irregular_prob:
            return float(rng.randint(self.irregular_min_s, self.irregular_max_s))
        return self.interval_s


def random_sampling(
    rng: random.Random,
    *,
    share_60s: float = 0.7,
    share_30min: float = 0.002,
) -> SamplingConfig:
    """Draw one truck's reporting schedule.

    The shares are of gaps between fixes, not of trucks: *share_60s* of
    all gaps come from trucks reporting every 60 s, *share_30min* from
    trucks reporting every 30 min, and the rest are spread evenly over
    trucks reporting every 1–59 s.  A truck reporting every *i* seconds
    makes 1/*i* gaps per second, so each interval is drawn with
    probability proportional to share × *i*: fast trucks are rare, but
    each contributes many gaps.

    Trucks that aren't on the 30 min schedule also report off schedule
    (1–60 s) on 5% of gaps and go 30 min without a fix about once per
    10 hours.
    """
    fast_share = (1.0 - share_60s - share_30min) / 59
    shares = {60.0: share_60s, 1800.0: share_30min} | {
        float(i): fast_share for i in range(1, 60)
    }
    interval_s = rng.choices(
        list(shares), weights=[share * i for i, share in shares.items()]
    )[0]
    if interval_s == 1800.0:
        return SamplingConfig(interval_s=1800.0)
    return SamplingConfig(
        interval_s=interval_s, irregular_prob=0.05, long_gaps_per_hour=0.1
    )


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
    config: str | Path | None = None
    errors: list[ErrorConfig] = field(default_factory=default_error_profile)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)


@dataclass
class TracePoint:
    lat: float
    lon: float
    speed_kmh: float
    heading: float  # 0-360, 0=north, clockwise
    timestamp: datetime


@dataclass
class RouteSegment:
    coords: list[tuple[float, float]]  # list of (lat, lon)
    speeds_mps: list[float]  # per-segment speed in m/s
    distances_m: list[float]  # per-segment distance in m
    total_distance_m: float
    total_duration_s: float
