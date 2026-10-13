"""Tests for trucktrack.generate — synthetic GPS trace generation."""

from __future__ import annotations

import random
from collections import Counter
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl
import pytest
import requests
import trucktrack
from trucktrack.generate import (
    RouteSegment,
    SamplingConfig,
    TracePoint,
    TripConfig,
    generate_trace,
    random_sampling,
    traces_to_csv,
    traces_to_parquet,
)
from trucktrack.generate.gps_errors import signal_dropout, stale_reemission
from trucktrack.generate.interpolator import (
    FixSchedule,
    bearing,
    haversine_m,
    interpolate_route,
    offset_to_latlon,
)
from trucktrack.generate.models import DEFAULT_VALHALLA_URL
from trucktrack.generate.operational_errors import fuel_rest_stop, relay_driving
from trucktrack.generate.parking import (
    ManeuverType,
    generate_arrival_maneuver,
    generate_departure_maneuver,
)
from trucktrack.generate.speed_profile import (
    MAX_DIP,
    TORONTO,
    congestion_factor,
    rush_hour_inbound,
    rush_hour_scale,
)


def _valhalla_reachable() -> bool:
    try:
        requests.get(DEFAULT_VALHALLA_URL, timeout=1)
    except (requests.ConnectionError, requests.Timeout):
        return False
    return True


requires_valhalla = pytest.mark.skipif(
    not _valhalla_reachable(),
    reason="Valhalla not running at " + DEFAULT_VALHALLA_URL,
)


def _config(**overrides) -> TripConfig:  # type: ignore[no-untyped-def]
    base = dict(
        origin=(43.65, -79.38),  # Toronto
        destination=(43.45, -80.49),  # Kitchener
        departure_time=datetime(2026, 1, 1, 8, 0, tzinfo=UTC),
        seed=42,
    )
    base.update(overrides)
    return TripConfig(**base)


@requires_valhalla
class TestGenerateTrace:
    def test_returns_trace_points(self) -> None:
        points = generate_trace(_config())
        assert points
        assert all(isinstance(p, TracePoint) for p in points)

    def test_seed_is_deterministic(self) -> None:
        a = generate_trace(_config())
        b = generate_trace(_config())
        assert len(a) == len(b)
        assert (a[0].lat, a[0].lon) == (b[0].lat, b[0].lon)
        assert (a[-1].lat, a[-1].lon) == (b[-1].lat, b[-1].lon)

    def test_timestamps_monotonic(self) -> None:
        points = generate_trace(_config())
        ts = [p.timestamp for p in points]
        assert ts == sorted(ts)

    def test_lat_lon_in_range(self) -> None:
        points = generate_trace(_config())
        for p in points:
            assert -90.0 <= p.lat <= 90.0
            assert -180.0 <= p.lon <= 180.0
            assert 0.0 <= p.heading < 360.0
            assert p.speed_kmh >= 0.0


@requires_valhalla
class TestTracesToParquet:
    def test_writes_file(self, tmp_path: Path) -> None:
        out = tmp_path / "trip.parquet"
        points = generate_trace(_config())
        traces_to_parquet([(points, "trip-x")], str(out))
        assert out.exists()

    def test_schema_matches_trucktrack_format(self, tmp_path: Path) -> None:
        out = tmp_path / "trip.parquet"
        points = generate_trace(_config())
        traces_to_parquet([(points, "trip-x")], str(out))
        df = pl.read_parquet(out)
        assert set(df.columns) == {
            "id",
            "lat",
            "lon",
            "speed",
            "heading",
            "time",
        }
        assert df["id"].unique().to_list() == ["trip-x"]
        assert len(df) == len(points)


@requires_valhalla
class TestTracesToCsv:
    def test_csv_has_header(self) -> None:
        points = generate_trace(_config())
        csv = traces_to_csv([(points, "trip-x")])
        first_line = csv.splitlines()[0]
        assert first_line == "id,lat,lon,speed,heading,time"

    def test_csv_row_count(self) -> None:
        points = generate_trace(_config())
        csv = traces_to_csv([(points, "trip-x")])
        # +1 for header
        assert len(csv.splitlines()) == len(points) + 1


@requires_valhalla
def test_empty_route_handled() -> None:
    # Same origin/destination — straight-line fallback still produces
    # a trace because parking maneuvers always emit points.
    cfg = _config(destination=(43.65, -79.38))
    points = generate_trace(cfg)
    assert points  # parking maneuver geometry is non-empty


# ── GPS error injectors ──────────────────────────────────────────────────


_START = datetime(2026, 1, 1, 8, 0)


def _linear_trace(n: int, interval_s: float = 60.0) -> list[TracePoint]:
    base = _START
    return [
        TracePoint(
            lat=43.0 + i * 0.001,
            lon=-79.0,
            speed_kmh=55.0,
            heading=90.0,
            timestamp=base + timedelta(seconds=interval_s * i),
        )
        for i in range(n)
    ]


class TestStaleReemissionInjector:
    def test_injects_stale_copies(self) -> None:
        points = _linear_trace(30)
        out = stale_reemission(points, random.Random(0), count=3)
        assert len(out) == 33

        src_keys = {(p.lat, p.lon, p.speed_kmh, p.heading) for p in points}
        seen: dict[tuple[float, float, float, float], int] = {}
        dup_count = 0
        for p in out:
            k = (p.lat, p.lon, p.speed_kmh, p.heading)
            if k in seen and k in src_keys:
                dup_count += 1
            seen[k] = seen.get(k, 0) + 1
        assert dup_count >= 3

    def test_filter_removes_injected_errors(self) -> None:
        """Injector + filter round-trip: trace returns to original length."""
        points = _linear_trace(30)
        corrupted = stale_reemission(points, random.Random(7), count=3)
        assert len(corrupted) == 33

        df = pl.DataFrame(
            {
                "id": ["truck_A"] * len(corrupted),
                "time": [p.timestamp for p in corrupted],
                "lat": [p.lat for p in corrupted],
                "lon": [p.lon for p in corrupted],
                "speed": [p.speed_kmh for p in corrupted],
                "heading": [p.heading for p in corrupted],
            }
        )
        cleaned = trucktrack.filter_stale_pings(df, window=8)
        assert cleaned.height == 30

    def test_noop_on_short_trace(self) -> None:
        points = _linear_trace(5)
        out = stale_reemission(points, random.Random(0), count=3)
        assert out == points


# ── Sampling schedule ────────────────────────────────────────────────────


def _straight_route(
    km: float = 60.0, mps: float = 25.0, start: tuple[float, float] = (43.0, -80.0)
) -> RouteSegment:
    """Due-east route in 500 m steps at a constant speed."""
    n = int(km * 2)
    coords = [offset_to_latlon(*start, 500.0 * i, 0.0) for i in range(n + 1)]
    distances = [
        haversine_m(*coords[i], *coords[i + 1]) for i in range(len(coords) - 1)
    ]
    total = sum(distances)
    return RouteSegment(
        coords=coords,
        speeds_mps=[mps] * len(distances),
        distances_m=distances,
        total_distance_m=total,
        total_duration_s=total / mps,
    )


_ROUTE = _straight_route()


def _gaps_s(points: list[TracePoint]) -> list[float]:
    return [
        (b.timestamp - a.timestamp).total_seconds()
        for a, b in zip(points, points[1:], strict=False)
    ]


class TestSamplingConfig:
    def test_default_is_fixed_60s(self) -> None:
        rng = random.Random(0)
        assert {SamplingConfig().next_gap(rng) for _ in range(1000)} == {60.0}

    def test_irregular_and_long_gaps(self) -> None:
        # p(long) = 36/h * 10 s / 3600 = 0.1; p(irregular) = 0.9 * 0.2 = 0.18.
        sampling = SamplingConfig(
            interval_s=10.0, irregular_prob=0.2, long_gaps_per_hour=36.0
        )
        rng = random.Random(0)
        gaps = [sampling.next_gap(rng) for _ in range(10_000)]
        assert 800 < sum(g == 1800.0 for g in gaps) < 1200
        irregular = [g for g in gaps if g not in (10.0, 1800.0)]
        assert all(1 <= g <= 60 and g == int(g) for g in irregular)
        assert 1500 < len(irregular) < 2000

    def test_random_sampling_mix_is_by_gap(self) -> None:
        rng = random.Random(0)
        configs = [random_sampling(rng) for _ in range(4000)]
        # Gaps per second of reporting, by interval.
        rate: Counter[str] = Counter()
        for c in configs:
            kind = {60.0: "60s", 1800.0: "30min"}.get(c.interval_s, "fast")
            rate[kind] += 1.0 / c.interval_s
        total = sum(rate.values())
        assert 0.65 < rate["60s"] / total < 0.75
        assert 0.001 < rate["30min"] / total < 0.004
        # By truck, the slow schedules dominate.
        trucks = Counter(c.interval_s for c in configs)
        assert trucks[60.0] / 4000 > 0.7
        assert all(1 <= i <= 59 for i in trucks if i not in (60.0, 1800.0))
        sparse = [c for c in configs if c.interval_s == 1800.0]
        assert sparse
        assert all(c.irregular_prob == c.long_gaps_per_hour == 0 for c in sparse)


class TestFixSchedule:
    def test_continuous_across_phases(self) -> None:
        sched = FixSchedule(SamplingConfig(interval_s=45.0), _START, random.Random(0))
        first = sched.until(_START + timedelta(seconds=100))
        second = sched.until(_START + timedelta(seconds=200))
        assert first == [_START + timedelta(seconds=s) for s in (0, 45, 90)]
        assert second == [_START + timedelta(seconds=s) for s in (135, 180)]


class TestInterpolateRoute:
    def test_sampling_does_not_change_the_drive(self) -> None:
        route = _ROUTE

        def drive(sampling: SamplingConfig) -> tuple[list[TracePoint], datetime]:
            sched = FixSchedule(sampling, _START, random.Random(0))
            return interpolate_route(route, _START, random.Random(1), sched)

        fine, fine_arrival = drive(SamplingConfig(interval_s=1.0))
        coarse, coarse_arrival = drive(SamplingConfig())
        sparse, sparse_arrival = drive(SamplingConfig(interval_s=1800.0))

        # Arrival comes from the speed profile, not the last fix.
        assert fine_arrival == coarse_arrival == sparse_arrival
        assert sparse[-1].timestamp < sparse_arrival - timedelta(seconds=60)

        # Every coarser fix lies on the 1 s trace.  Heading is excluded:
        # it is held from the previous fix while stopped.
        on_fine = {p.timestamp: (p.lat, p.lon, p.speed_kmh) for p in fine}
        for p in coarse + sparse:
            assert on_fine[p.timestamp] == (p.lat, p.lon, p.speed_kmh)


# ── Rush hours ───────────────────────────────────────────────────────────

# 08:00 on a Tuesday in Toronto, either side of the DST change on 8 March.
_RUSH_EST = datetime(2026, 3, 3, 13, 0, tzinfo=UTC)
_RUSH_EDT = datetime(2026, 3, 10, 12, 0, tzinfo=UTC)


class TestRushHours:
    def test_free_flow_overnight(self) -> None:
        three_am = _RUSH_EST - timedelta(hours=5)
        assert congestion_factor("highway", three_am) == pytest.approx(1.0, abs=1e-3)

    def test_peaks_in_local_time(self) -> None:
        est = congestion_factor("highway", _RUSH_EST)
        assert est < 0.55
        assert congestion_factor("highway", _RUSH_EDT) == pytest.approx(est)
        # 13:00 UTC after the change is 09:00 local, past the peak.
        assert congestion_factor("highway", _RUSH_EDT + timedelta(hours=1)) > est

    def test_deeper_on_faster_roads(self) -> None:
        factors = [
            congestion_factor(c, _RUSH_EST) for c in ("highway", "arterial", "urban")
        ]
        assert factors == sorted(factors)
        assert factors[-1] < 1.0

    def test_no_rush_hour_on_weekends(self) -> None:
        saturday = _RUSH_EST + timedelta(days=4)
        assert congestion_factor("highway", saturday) > 0.95

    def test_naive_time_is_utc(self) -> None:
        naive = _RUSH_EST.replace(tzinfo=None)
        assert congestion_factor("highway", naive) == congestion_factor(
            "highway", _RUSH_EST
        )

    def test_deepest_in_toronto(self) -> None:
        downtown = rush_hour_scale(*TORONTO)
        brampton = rush_hour_scale(43.7387, -79.7271)
        kitchener = rush_hour_scale(43.4643, -80.5204)
        assert downtown == pytest.approx(1.6)
        assert brampton == pytest.approx(1.0, abs=0.05)
        assert kitchener == pytest.approx(0.5, abs=0.05)
        factors = [congestion_factor("highway", _RUSH_EST, s) for s in (1.6, 1.0, 0.5)]
        assert factors == sorted(factors)
        assert factors[0] < 0.25

    def test_inbound_is_the_cosine_to_city_hall(self) -> None:
        hamilton = (43.2557, -79.8711)  # Toronto lies ~42 degrees east of north
        assert rush_hour_inbound(*hamilton, 42.0) == pytest.approx(1.0, abs=0.01)
        assert rush_hour_inbound(*hamilton, 222.0) == pytest.approx(-1.0, abs=0.01)
        assert rush_hour_inbound(*hamilton, 132.0) == pytest.approx(0.0, abs=0.02)
        assert rush_hour_inbound(*TORONTO, 0.0) == 0.0

    def test_inbound_leans_into_the_morning(self) -> None:
        evening = _RUSH_EST + timedelta(hours=8, minutes=30)
        in_am, out_am = (
            congestion_factor("highway", _RUSH_EST, 1.0, d) for d in (1, -1)
        )
        in_pm, out_pm = (congestion_factor("highway", evening, 1.0, d) for d in (1, -1))
        assert in_am < out_am
        assert in_pm > out_pm

    def test_no_dip_takes_more_than_its_cap(self) -> None:
        factor = congestion_factor("highway", _RUSH_EST, scale=1.6, inbound=1.0)
        assert factor >= 1.0 - MAX_DIP - 0.05  # the midday dip's tail adds a little

    def test_scale_leaves_nights_free(self) -> None:
        three_am = _RUSH_EST - timedelta(hours=5)
        assert congestion_factor("highway", three_am, 1.6) == pytest.approx(
            1.0, abs=1e-3
        )

    @staticmethod
    def _drive(
        start: datetime, route: RouteSegment = _ROUTE
    ) -> tuple[list[TracePoint], datetime]:
        sched = FixSchedule(SamplingConfig(), start, random.Random(0))
        return interpolate_route(route, start, random.Random(1), sched)

    def test_slows_the_drive_with_positions_in_step(self) -> None:
        night = _RUSH_EST - timedelta(hours=5)
        _, free_arrival = self._drive(night)
        rush, rush_arrival = self._drive(_RUSH_EST)
        # The route is ~80 km from Toronto, where rush hours run shallow.
        assert rush_arrival - _RUSH_EST > 1.25 * (free_arrival - night)
        # The truck covers the ground its reported speeds say it does.
        for a, b in zip(rush, rush[1:], strict=False):
            metres = haversine_m(a.lat, a.lon, b.lat, b.lon)
            seconds = (b.timestamp - a.timestamp).total_seconds()
            mean_mps = (a.speed_kmh + b.speed_kmh) / 2 / 3.6
            assert metres == pytest.approx(mean_mps * seconds, rel=0.02)

    def test_slows_more_near_toronto(self) -> None:
        # The same 20 km drive, ~75 km out and starting downtown, at the
        # midday dip, which doesn't lean by direction.
        midday = _RUSH_EST + timedelta(hours=5)
        far = _straight_route(km=20.0)
        near = _straight_route(km=20.0, start=TORONTO)
        _, far_arrival = self._drive(midday, far)
        _, near_arrival = self._drive(midday, near)
        assert near_arrival - midday > 1.1 * (far_arrival - midday)

    def test_inbound_slower_in_the_morning_outbound_in_the_evening(self) -> None:
        # 20 km heading east out of Toronto, and the same road back in.
        out = _straight_route(km=20.0, start=TORONTO)
        back = RouteSegment(
            coords=out.coords[::-1],
            speeds_mps=out.speeds_mps[::-1],
            distances_m=out.distances_m[::-1],
            total_distance_m=out.total_distance_m,
            total_duration_s=out.total_duration_s,
        )
        evening = _RUSH_EST + timedelta(hours=8, minutes=30)
        for start, slower, faster in ((_RUSH_EST, back, out), (evening, out, back)):
            _, slow_arrival = self._drive(start, slower)
            _, fast_arrival = self._drive(start, faster)
            assert slow_arrival - start > 1.3 * (fast_arrival - start)


class TestGenerateTraceSampling:
    @pytest.fixture(autouse=True)
    def _fake_route(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "trucktrack.generate.trace.fetch_route",
            lambda *args, **kwargs: _ROUTE,
        )

    def _trace(self, sampling: SamplingConfig) -> list[TracePoint]:
        return generate_trace(
            TripConfig(
                origin=_ROUTE.coords[0],
                destination=_ROUTE.coords[-1],
                departure_time=_START,
                seed=3,
                errors=[],
                gps_noise_meters=0.0,
                sampling=sampling,
            )
        )

    @pytest.mark.parametrize("interval_s", [1.0, 37.0, 60.0, 1800.0])
    def test_fixes_follow_the_schedule_through_every_phase(
        self, interval_s: float
    ) -> None:
        points = self._trace(SamplingConfig(interval_s=interval_s))
        assert set(_gaps_s(points)) == {interval_s}

    def test_no_jump_where_maneuvers_meet_the_route(self) -> None:
        points = self._trace(SamplingConfig(interval_s=1.0))
        implied_mps = [
            haversine_m(a.lat, a.lon, b.lat, b.lon) / g
            for a, b, g in zip(points, points[1:], _gaps_s(points), strict=False)
        ]
        # The fake route is driven at 25 m/s; nothing should go faster.
        assert max(implied_mps) < 26.0

    def test_coarse_trace_is_a_subset_of_fine_trace(self) -> None:
        fine = {
            p.timestamp: (p.lat, p.lon)
            for p in self._trace(SamplingConfig(interval_s=1.0))
        }
        coarse = self._trace(SamplingConfig())
        assert all(fine[p.timestamp] == (p.lat, p.lon) for p in coarse)


class TestInjectorsFollowSampling:
    def test_fuel_stop_dwell_uses_trace_interval(self) -> None:
        out = fuel_rest_stop(
            _linear_trace(200, interval_s=10.0), random.Random(0), duration_s=600.0
        )
        assert len(out) == 260
        assert set(_gaps_s(out)) == {10.0}

    def test_signal_dropout_removes_a_duration(self) -> None:
        out = signal_dropout(
            _linear_trace(200, interval_s=10.0), random.Random(0), duration_s=300.0
        )
        assert len(out) == 170
        assert max(_gaps_s(out)) == 310.0

    def test_signal_dropout_lasts_its_duration_with_irregular_gaps(self) -> None:
        # Mostly 1 s with a 30 s gap every fifth fix: counting fixes would
        # remove ~300 fixes, i.e. far more than 300 s.
        points = _linear_trace(1000, interval_s=1.0)
        t = _START
        for i, p in enumerate(points):
            points[i] = replace(p, timestamp=t)
            t += timedelta(seconds=30 if i % 5 == 4 else 1)
        out = signal_dropout(points, random.Random(0), duration_s=300.0)
        assert 300.0 <= max(_gaps_s(out)) <= 330.0

    def test_relay_driving_keeps_the_device_gaps(self) -> None:
        points = _linear_trace(60, interval_s=10.0)
        for i in range(20, 30):
            points[i] = replace(points[i], speed_kmh=0.0)
        points[40:] = [
            replace(p, timestamp=p.timestamp + timedelta(seconds=7))
            for p in points[40:]
        ]
        out = relay_driving(points, random.Random(0))
        # The stop is closed up, but the 17 s gap the device made survives.
        assert len(out) < len(points)
        assert sorted(set(_gaps_s(out))) == [10.0, 17.0]


class TestManeuversMeetTheRoad:
    @pytest.mark.parametrize("maneuver", list(ManeuverType))
    @pytest.mark.parametrize("arriving", [True, False])
    def test_maneuver_lines_up_with_the_road(
        self, maneuver: ManeuverType, arriving: bool
    ) -> None:
        road, heading = (43.0, -80.0), 70.0
        make = generate_arrival_maneuver if arriving else generate_departure_maneuver
        points = make(*road, heading, _START, random.Random(0), maneuver_type=maneuver)

        def turn(a: float, b: float) -> float:
            return abs((b - a + 540.0) % 360.0 - 180.0)

        moves = [
            (p, bearing(p.lat, p.lon, q.lat, q.lon))
            for p, q in zip(points, points[1:], strict=False)
            if haversine_m(p.lat, p.lon, q.lat, q.lon) > 0.5
        ]
        # Arrivals leave the road where it ends, carrying on the same way;
        # departures join it where it starts, heading the way it goes.
        road_end, road_step = (
            (points[0], moves[0]) if arriving else (points[-1], moves[-1])
        )
        assert haversine_m(road_end.lat, road_end.lon, *road) < 0.2
        assert turn(road_step[1], heading) < 1.0
        # Reported heading is GPS course (direction of motion), even reversing.
        assert all(turn(p.heading, moved) < 10.0 for p, moved in moves)
