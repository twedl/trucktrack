"""Tests for trucktrack.congestion: time weights, dwell flags, observations."""

from __future__ import annotations

import random
from datetime import UTC, datetime, timedelta

import polars as pl
import pytest
from trucktrack.congestion import (
    OBSERVATION_COLUMNS,
    ObservationConfig,
    edge_observations,
    flag_dwell,
    flag_speed_mismatch,
    time_weights,
)
from trucktrack.generate.gps_errors import speed_heading_desync
from trucktrack.generate.interpolator import FixSchedule, interpolate_route
from trucktrack.generate.models import RouteSegment, SamplingConfig, TracePoint

_T0 = datetime(2026, 1, 1, 8, 0, tzinfo=UTC)
_RUSH = datetime(2026, 3, 3, 21, 30, tzinfo=UTC)  # Tuesday 16:30 in Toronto


def _fixes(
    seconds: list[float],
    *,
    speeds: list[float | None] | None = None,
    metres: list[float] | None = None,
    trip: str = "t",
) -> pl.DataFrame:
    """Fixes at *seconds*, *metres* north of one spot (default: all on it)."""
    n = len(seconds)
    return pl.DataFrame(
        {
            "id": [trip] * n,
            "time": [_T0 + timedelta(seconds=s) for s in seconds],
            "lat": [43.65 + m / 111_195 for m in (metres or [0.0] * n)],
            "lon": [-79.38] * n,
            "speed": speeds if speeds is not None else [50.0] * n,
        },
        schema_overrides={"speed": pl.Float64},
    )


class TestTimeWeights:
    def test_half_gap_each_side(self) -> None:
        w = time_weights(_fixes([0, 60, 120, 180]))["w"].to_list()
        assert w == [30.0, 60.0, 60.0, 30.0]

    def test_each_side_capped(self) -> None:
        # Gaps 10, 60, 1800, 30: the 30-min gap gives each neighbour 30 s.
        w = time_weights(_fixes([0, 10, 70, 1870, 1900]))["w"].to_list()
        assert w == [5.0, 35.0, 60.0, 45.0, 15.0]

    def test_cap_is_configurable(self) -> None:
        config = ObservationConfig(max_half_gap_s=10.0)
        w = time_weights(_fixes([0, 60, 120]), config)["w"].to_list()
        assert w == [10.0, 20.0, 10.0]

    def test_repeated_timestamp_gets_zero_weight(self) -> None:
        df = _fixes([0, 60, 60, 120]).with_columns(tag=pl.Series(["a", "b", "c", "d"]))
        out = time_weights(df)
        assert dict(zip(out["tag"], out["w"], strict=True)) == {
            "a": 30.0,
            "b": 60.0,  # first in input order keeps the weight
            "c": 0.0,
            "d": 30.0,
        }

    def test_trips_weighted_independently(self) -> None:
        df = pl.concat([_fixes([0, 60], trip="a"), _fixes([30, 40, 50], trip="b")])
        out = time_weights(df.sample(fraction=1.0, shuffle=True, seed=1))
        assert out["id"].to_list() == ["a", "a", "b", "b", "b"]
        assert out["w"].to_list() == [30.0, 30.0, 5.0, 10.0, 5.0]

    def test_single_fix_trip_has_no_weight(self) -> None:
        assert time_weights(_fixes([0]))["w"].to_list() == [0.0]

    def test_lazyframe_stays_lazy(self) -> None:
        out = time_weights(_fixes([0, 60]).lazy())
        assert isinstance(out, pl.LazyFrame)
        assert out.collect()["w"].to_list() == [30.0, 30.0]


def _drive(
    sampling: SamplingConfig, edge_kmh: float, rng: random.Random, trip: str
) -> tuple[pl.DataFrame, float]:
    """One trip: 2 km at 80 km/h, a 500 m edge at *edge_kmh*, 2 km at 80 km/h.

    Fix times come from the generator's FixSchedule, starting at a
    random phase.  Returns the fixes (with ``on_edge``) and the time
    spent on the edge.
    """
    approach_s = 2000 / (80 / 3.6)
    edge_s = 500 / (edge_kmh / 3.6)
    total_s = 2 * approach_s + edge_s
    start = _T0 + timedelta(seconds=rng.uniform(0, sampling.interval_s))
    times = FixSchedule(sampling, start, rng).until(_T0 + timedelta(seconds=total_s))
    elapsed = [(t - _T0).total_seconds() for t in times]
    on_edge = [approach_s <= e < approach_s + edge_s for e in elapsed]
    return (
        pl.DataFrame(
            {
                "id": trip,
                "time": times,
                "speed": [edge_kmh if o else 80.0 for o in on_edge],
                "on_edge": on_edge,
            }
        ),
        edge_s,
    )


class TestEstimatorOnSyntheticSampling:
    """The weighted mean recovers space-mean speed whatever the reporting rate.

    Two fleets cross the same 500 m edge: trucks reporting every ~1 s
    crawl through at 20 km/h, trucks reporting every ~60 s cruise at
    90 km/h.  Both report irregularly as the synthetic generator does.
    The edge's space-mean speed is total metres over total seconds.
    """

    @pytest.fixture(scope="class")
    def fixes_and_truth(self) -> tuple[pl.DataFrame, float]:
        rng = random.Random(7)
        fleets = [
            (
                SamplingConfig(1.0, irregular_prob=0.05, long_gaps_per_hour=0.1),
                20.0,
                500,
            ),
            (
                SamplingConfig(60.0, irregular_prob=0.05, long_gaps_per_hour=0.1),
                90.0,
                3000,
            ),
        ]
        parts: list[pl.DataFrame] = []
        metres = seconds = 0.0
        for k, (sampling, kmh, n) in enumerate(fleets):
            for i in range(n):
                df, edge_s = _drive(sampling, kmh, rng, f"{k}-{i}")
                parts.append(df)
                metres += 500
                seconds += edge_s
        return pl.concat(parts), metres / seconds * 3.6

    def test_weighted_mean_matches_space_mean(
        self, fixes_and_truth: tuple[pl.DataFrame, float]
    ) -> None:
        fixes, truth = fixes_and_truth
        on_edge = time_weights(fixes).filter("on_edge")
        estimate = (on_edge["speed"] * on_edge["w"]).sum() / on_edge["w"].sum()
        assert estimate == pytest.approx(truth, rel=0.03)

    def test_unweighted_mean_follows_fast_reporters(
        self, fixes_and_truth: tuple[pl.DataFrame, float]
    ) -> None:
        fixes, truth = fixes_and_truth
        unweighted = fixes.filter("on_edge")["speed"].mean()
        assert isinstance(unweighted, float)
        assert unweighted < 0.5 * truth


def _dwell(df: pl.DataFrame, **kwargs: float) -> list[bool]:
    config = ObservationConfig(**kwargs)
    return flag_dwell(time_weights(df, config), config)["dwell"].to_list()


class TestFlagDwell:
    def test_signal_stop_is_queue(self) -> None:
        df = _fixes([0, 10, 20, 30, 40, 50], speeds=[40, 0, 0, 0, 0, 40])
        assert not any(_dwell(df))

    def test_long_stationary_run_is_dwell(self) -> None:
        secs = [0, 60, 120, 180, 240, 300, 360]
        df = _fixes(secs, speeds=[50, 0, 0, 0, 0.5, 0, 50])
        assert _dwell(df) == [False, True, True, True, True, True, False]

    def test_run_standing_for_threshold_is_queue(self) -> None:
        # Three zeros, 60 s each.
        df = _fixes([0, 60, 120, 180, 240], speeds=[50, 0, 0, 0, 50])
        assert not any(_dwell(df, max_queue_stop_s=180.0))
        assert sum(_dwell(df, max_queue_stop_s=179.0)) == 3

    def test_same_stop_is_dwell_at_any_reporting_rate(self) -> None:
        # A 240 s idle: 48 zeros at 5 s, or 4 zeros at 60 s.  The 60 s
        # truck's first-to-last span is only 180 s.
        for interval in (5, 60):
            stop = list(range(60, 300, interval))
            secs = [0, *stop, 300]
            speeds = [50.0, *[0.0] * len(stop), 50.0]
            dwell = _dwell(_fixes(secs, speeds=speeds))
            assert dwell == [False, *[True] * len(stop), False], interval

    def test_movement_breaks_the_run(self) -> None:
        secs = [0, 60, 120, 180, 240, 300]
        df = _fixes(secs, speeds=[0, 0, 0, 8, 0, 0])
        assert not any(_dwell(df))

    def test_zeros_down_the_road_are_separate_stops(self) -> None:
        # A 60 s truck catching a signal stop each minute, 300 m apart.
        secs = [0, 60, 120, 180, 240, 300]
        speeds = [50.0, 0, 0, 0, 0, 50]
        down_the_road = _fixes(
            secs, speeds=speeds, metres=[0, 300, 600, 900, 1200, 1500]
        )
        jitter = _fixes(secs, speeds=speeds, metres=[0, 300, 305, 298, 310, 900])
        assert not any(_dwell(down_the_road))
        assert _dwell(jitter) == [False, True, True, True, True, False]

    def test_null_speed_is_not_stationary(self) -> None:
        df = _fixes([0, 60, 120, 180, 240], speeds=[0, 0, None, 0, 0])
        assert not any(_dwell(df, max_queue_stop_s=100.0))

    def test_runs_do_not_cross_trips(self) -> None:
        df = pl.concat(
            [
                _fixes([0, 60, 120], speeds=[0, 0, 0], trip="a"),
                _fixes([200, 260, 320], speeds=[0, 0, 0], trip="b"),
            ]
        )
        assert not any(_dwell(df))


def _mismatch(df: pl.DataFrame) -> list[bool]:
    return flag_speed_mismatch(df)["speed_mismatch"].to_list()


def _route(rng: random.Random) -> RouteSegment:
    """A 20 km straight road east: urban, then highway, then urban, in 50 m steps.

    The generator's speed model drives it, with its traffic stops.
    """
    n = 400
    speeds = [11.0 if i < 100 or i >= 300 else 28.0 for i in range(n)]
    step_lon = 50 / (111_320 * 0.7236)  # cos(43.65°)
    coords = [(43.65, -79.38 + i * step_lon) for i in range(n + 1)]
    return RouteSegment(
        coords=coords,
        speeds_mps=speeds,
        distances_m=[50.0] * n,
        total_distance_m=50.0 * n,
        total_duration_s=sum(50.0 / v for v in speeds),
    )


def _trace_frame(points: list[TracePoint]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "id": "t",
            "time": [p.timestamp for p in points],
            "lat": [p.lat for p in points],
            "lon": [p.lon for p in points],
            "speed": [p.speed_kmh for p in points],
        }
    )


class TestFlagSpeedMismatch:
    def test_stuck_zero_while_cruising(self) -> None:
        # 100 km/h, a fix a minute: 1667 m apart.
        metres = [0, 1667, 3333, 5000, 6667]
        df = _fixes([0, 60, 120, 180, 240], speeds=[100, 0, 0, 0, 100], metres=metres)
        assert _mismatch(df) == [False, True, True, True, False]

    def test_dense_stuck_zero_at_city_speed(self) -> None:
        # 50 km/h, a fix every 5 s: 69 m apart.
        metres = [0, 69, 139, 208, 278]
        df = _fixes([0, 5, 10, 15, 20], speeds=[50, 0, 0, 0, 50], metres=metres)
        assert _mismatch(df) == [False, True, True, True, False]

    def test_sparse_zeros_at_city_speed_could_be_stop_and_go(self) -> None:
        metres = [0, 500, 1000, 1500]
        df = _fixes([0, 60, 120, 180], speeds=[0, 0, 0, 0], metres=metres)
        assert not any(_mismatch(df))

    def test_either_speed_explaining_the_move_passes(self) -> None:
        # Pulling away from a stop: 0 then 50 km/h, 60 m in 5 s.
        df = _fixes([0, 5], speeds=[0, 50], metres=[0, 60])
        assert not any(_mismatch(df))

    @pytest.mark.parametrize("start", [_T0, _RUSH])
    @pytest.mark.parametrize("interval_s", [1.0, 5.0, 20.0, 60.0])
    def test_generator_drives_pass(self, interval_s: float, start: datetime) -> None:
        for seed in range(5):
            rng = random.Random(seed)
            schedule = FixSchedule(
                SamplingConfig(interval_s, irregular_prob=0.05), start, rng
            )
            points, _ = interpolate_route(_route(rng), start, rng, schedule)
            assert not any(_mismatch(_trace_frame(points))), seed

    def test_generator_speed_stuck_at_zero_caught(self) -> None:
        caught = stuck = 0
        for seed in range(10):
            rng = random.Random(seed)
            schedule = FixSchedule(SamplingConfig(5.0), _T0, rng)
            clean, _ = interpolate_route(_route(rng), _T0, rng, schedule)
            faulty = speed_heading_desync(clean, rng)
            flags = _mismatch(_trace_frame(faulty))
            for before, after, flag in zip(clean, faulty, flags, strict=True):
                if before.speed_kmh > 30 and after.speed_kmh == 0:
                    stuck += 1
                    caught += flag
        assert stuck > 100
        assert caught / stuck > 0.95


def _matched(
    *,
    speed: list[float | None] | None = None,
    heading: list[float] | None = None,
    w: list[float] | None = None,
    end_heading: float = 279.0,
) -> pl.DataFrame:
    """Three matched fixes: on a straight edge, unmatched, on a curving edge."""
    return pl.DataFrame(
        {
            "id": ["t"] * 3,
            "time": [_T0 + timedelta(seconds=s) for s in (0, 60, 120)],
            "speed": speed or [55.9, 56.1, 56.1],
            "heading": heading or [119.0, 135.0, 305.0],
            "w": w or [30.0, 60.0, 30.0],
            "dwell": [False] * 3,
            "speed_mismatch": [False] * 3,
            "edge_id": [1634000558945, None, 1342686762890],
            "way_id": [370115604, None, 920928491],
            "forward": [True, None, False],
            "road_class": ["motorway", None, "primary"],
            "edge_length": [688.0, None, 101.0],
            "distance_along_edge": [0.892249, None, 0.529403],
            "match_type": ["matched", "unmatched", "matched"],
            "distance_from_trace": [1.612005, None, 32.865639],
            "begin_heading": [119.0, None, 334.0],
            "end_heading": [119.0, None, end_heading],
        },
        schema_overrides={
            "speed": pl.Float64,
            "heading": pl.Float64,
            "edge_id": pl.UInt64,
        },
    )


class TestEdgeObservations:
    def test_keeps_fixes_on_an_edge(self) -> None:
        obs = edge_observations(_matched())
        assert obs.columns == list(OBSERVATION_COLUMNS)
        assert obs["edge_id"].to_list() == [1634000558945, 1342686762890]
        assert obs["time"].to_list() == [_T0, _T0 + timedelta(seconds=120)]

    def test_fixes_without_speed_or_weight_dropped(self) -> None:
        obs = edge_observations(_matched(speed=[None, 56.1, 56.1], w=[30, 60, 0]))
        assert obs.is_empty()

    def test_heading_compared_to_bearing_along_curving_edge(self) -> None:
        # Edge turns 334 -> 279; 53% along it the bearing is ~305.
        obs = edge_observations(_matched())
        assert obs["heading_diff"].to_list() == pytest.approx([0.0, 0.1], abs=0.1)
        assert not obs["wrong_way"].any()

    def test_opposite_heading_at_speed_is_wrong_way(self) -> None:
        obs = edge_observations(
            _matched(speed=[55.9, 56.1, 10.0], heading=[299.0, 0.0, 125.0])
        )
        assert obs["heading_diff"].to_list() == pytest.approx([180.0, 180.0], abs=0.2)
        # The second fix is reversed too, but too slow to trust its course.
        assert obs["wrong_way"].to_list() == [True, False]

    def test_sharply_turning_edge_not_checked(self) -> None:
        obs = edge_observations(_matched(heading=[119.0, 0.0, 125.0], end_heading=154))
        assert obs["heading_diff"][1] is None
        assert not obs["wrong_way"][1]
