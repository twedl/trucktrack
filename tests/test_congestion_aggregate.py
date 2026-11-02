"""Tests for trucktrack.congestion.aggregate: local bins, cell speeds, pooling."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import polars as pl
import pytest
from trucktrack.congestion import (
    OBSERVATION_COLUMNS,
    AggregateConfig,
    cell_speeds,
    edge_cells,
    flow_window,
    hour_of_week,
    rush_speeds,
    rush_window,
    usable_observations,
    way_rush_speeds,
)

# Tuesday 3 March 2026, 08:00 in Toronto (EST, UTC-5).
_T0 = datetime(2026, 3, 3, 13, 0, tzinfo=UTC)

# Way 7 runs forward over edges 1, 2 and 4 (a short junction piece);
# edge 3 is its reverse direction.
_EDGES: dict[int, dict[str, Any]] = {
    1: {"way_id": 7, "forward": True, "road_class": "primary", "edge_length": 400.0},
    2: {"way_id": 7, "forward": True, "road_class": "primary", "edge_length": 250.0},
    3: {"way_id": 7, "forward": False, "road_class": "primary", "edge_length": 680.0},
    4: {"way_id": 7, "forward": True, "road_class": "primary", "edge_length": 30.0},
}


# Two trucks on edge 1: a space-mean of 50 km/h.
_PAIR = (("a", 1, 40.0, 30.0), ("b", 1, 60.0, 30.0))


def _obs(*rows: tuple[str, int, float, float]) -> pl.DataFrame:
    """Observations at _T0 from (trip, edge, speed, w) rows, as written."""
    n = len(rows)
    data: dict[str, list[Any]] = {
        "id": [r[0] for r in rows],
        "time": [_T0] * n,
        "speed": [r[2] for r in rows],
        "heading": [0.0] * n,
        "w": [r[3] for r in rows],
        "dwell": [False] * n,
        "speed_mismatch": [False] * n,
        "edge_id": [r[1] for r in rows],
        **{k: [_EDGES[r[1]][k] for r in rows] for k in _EDGES[1]},
        "distance_along_edge": [0.5] * n,
        "match_type": ["matched"] * n,
        "distance_from_trace": [1.0] * n,
        "heading_diff": [0.0] * n,
        "wrong_way": [False] * n,
    }
    return (
        pl.DataFrame(data, schema_overrides={"edge_id": pl.UInt64})
        .select(OBSERVATION_COLUMNS)
        .with_columns(tile_build_id=pl.lit("b"))
    )


class TestHourOfWeek:
    @pytest.mark.parametrize(
        ("utc", "expected"),
        [
            (datetime(2026, 3, 1, 5, 30), 0),  # Sunday 00:30 EST
            (datetime(2026, 3, 2, 13, 15), 32),  # Monday 08:15 EST
            (datetime(2026, 3, 9, 12, 15), 32),  # Monday 08:15 EDT
            (datetime(2026, 3, 8, 7, 0), 3),  # Sunday 03:00 EDT, just after the change
            (datetime(2026, 3, 8, 4, 59), 167),  # Saturday 23:59 EST
        ],
    )
    def test_local_hour_from_sunday(self, utc: datetime, expected: int) -> None:
        df = pl.DataFrame({"time": [utc.replace(tzinfo=UTC)]})
        assert df.select(hour_of_week())["hour_of_week"].item() == expected

    def test_naive_time_is_utc(self) -> None:
        df = pl.DataFrame({"time": [datetime(2026, 3, 2, 13, 15)]})
        assert df.select(hour_of_week())["hour_of_week"].item() == 32

    def test_time_zone_is_configurable(self) -> None:
        # Kenora keeps Central time: an hour behind Toronto.
        df = pl.DataFrame({"time": [datetime(2026, 3, 2, 14, 15, tzinfo=UTC)]})
        assert df.select(hour_of_week(tz="America/Winnipeg")).item() == 32


class TestUsableObservations:
    def test_flagged_rows_dropped(self) -> None:
        obs = _obs(*[("a", 1, 50.0, 30.0)] * 4).with_columns(
            dwell=pl.Series([True, False, False, False]),
            wrong_way=pl.Series([False, True, False, False]),
            speed_mismatch=pl.Series([False, False, True, False]),
        )
        assert usable_observations(obs.lazy()).collect().height == 1

    def test_speed_glitches_dropped(self) -> None:
        obs = _obs(("a", 1, 120.0, 30.0), ("a", 1, 175.0, 30.0))
        kept = usable_observations(obs.lazy()).collect()
        assert kept["speed"].to_list() == [120.0]
        config = AggregateConfig(max_speed_kmh=200.0)
        assert usable_observations(obs.lazy(), config=config).collect().height == 2

    def test_trips_failing_match_checks_dropped(self) -> None:
        obs = _obs(("a", 1, 50.0, 30.0), ("b", 1, 50.0, 30.0))
        quality = pl.DataFrame({"id": ["a", "b"], "ok": [True, False]})
        kept = usable_observations(obs.lazy(), quality.lazy()).collect()
        assert kept["id"].to_list() == ["a"]


class TestCellSpeeds:
    def _cells(self, obs: pl.DataFrame) -> pl.DataFrame:
        return cell_speeds(obs.lazy(), ["edge_id"]).collect()

    def test_space_mean_is_time_weighted(self) -> None:
        # 60 s at 20 km/h and 10 s at 90 km/h: 1.583 km in 70 s.
        cell = self._cells(_obs(("a", 1, 20.0, 60.0), ("b", 1, 90.0, 10.0)))
        assert cell["speed"].item() == pytest.approx((20 * 60 + 90 * 10) / 70)
        assert cell["seconds"].item() == 70.0
        assert cell["fixes"].item() == 2

    def test_equal_trucks_count_fully(self) -> None:
        obs = _obs(*[(t, 1, 50.0, 30.0) for t in "abc"])
        assert self._cells(obs)["trucks"].item() == pytest.approx(3.0)

    def test_one_dominant_truck_counts_about_once(self) -> None:
        obs = _obs(("a", 1, 50.0, 60.0), ("b", 1, 50.0, 2.0), ("c", 1, 50.0, 2.0))
        assert self._cells(obs)["trucks"].item() == pytest.approx(1.13, abs=0.01)

    def test_trips_of_one_truck_are_one_truck(self) -> None:
        obs = _obs(("a_gap0_trip0", 1, 50.0, 30.0), ("a_gap1_trip0", 1, 60.0, 30.0))
        assert self._cells(obs)["trucks"].item() == pytest.approx(1.0)


def _at(obs: pl.DataFrame, *local_hours: float) -> pl.DataFrame:
    """*obs* with each row moved to its hour of _T0's day, local clock."""
    return obs.with_columns(
        time=pl.Series([_T0 + timedelta(hours=h - 8) for h in local_hours])
    )


def _cells(obs: pl.DataFrame, by: pl.Expr | None = None) -> pl.DataFrame:
    return edge_cells(obs.lazy(), by=by).collect()


class TestEdgeCells:
    def test_hourly_cells_by_default(self) -> None:
        obs = _at(_obs(*_PAIR, ("c", 1, 90.0, 30.0)), 8, 8, 2)
        out = _cells(obs)
        assert out["hour_of_week"].to_list() == [2 * 24 + 2, 2 * 24 + 8]
        assert out["speed"].to_list() == pytest.approx([90.0, 50.0])
        assert out["trucks"].to_list() == pytest.approx([1.0, 2.0])
        assert out["road_class"].to_list() == ["primary"] * 2

    def test_only_cells_the_trucks_measured(self) -> None:
        # Edge 2 is seen in the day only, and its way's other edge says
        # nothing about it.
        obs = _at(_obs(*_PAIR, ("c", 2, 10.0, 30.0)), 8, 2, 8)
        out = _cells(obs, flow_window())
        assert out.select("edge_id", "flow").rows() == [
            (1, "constrained"),
            (1, "freeflow"),
            (2, "constrained"),
        ]
        assert out["speed"].to_list() == [40.0, 60.0, 10.0]

    def test_valhalla_daytime_is_seven_to_seven(self) -> None:
        # 07:00 is day; 06:59 and 19:00 are night.
        obs = _at(
            _obs(("a", 1, 40.0, 30.0), ("b", 1, 60.0, 30.0), ("c", 1, 60.0, 30.0)),
            7,
            6 + 59 / 60,
            19,
        )
        out = _cells(obs, flow_window())
        assert dict(out.select("flow", "speed").rows()) == {
            "constrained": 40.0,
            "freeflow": 60.0,
        }

    def test_edge_length_is_the_longest_seen(self) -> None:
        # A match's first edge comes with only the part driven.
        obs = _obs(*_PAIR).with_columns(edge_length=pl.Series([120.0, 400.0]))
        assert _cells(obs)["edge_length"].item() == 400.0

    def test_flagged_observations_left_out(self) -> None:
        obs = _obs(("a", 1, 40.0, 30.0), ("b", 1, 0.0, 300.0)).with_columns(
            dwell=pl.Series([False, True])
        )
        row = _cells(obs)
        assert row["speed"].item() == 40.0
        assert row["fixes"].item() == 1

    def test_custom_bin(self) -> None:
        out = _cells(_obs(*_PAIR), pl.lit(0).alias("all_day"))
        assert out["all_day"].to_list() == [0]
        assert out["speed"].item() == pytest.approx(50.0)

    def test_tile_builds_kept_apart(self) -> None:
        obs = _obs(*_PAIR).with_columns(tile_build_id=pl.Series(["x", "y"]))
        out = _cells(obs)
        assert out["tile_build_id"].to_list() == ["x", "y"]
        assert out["speed"].to_list() == [40.0, 60.0]


class TestRushWindow:
    @pytest.mark.parametrize(
        ("local_hour", "expected"),
        [
            (8, "am"),
            (17, "pm"),
            (3, "night"),
            (12, None),  # midday
            (9, None),  # the morning rush ends at 09:00
            (4 * 24 + 8, None),  # Saturday has no rush hour
            (4 * 24 + 3, "night"),  # but has a night
        ],
    )
    def test_windows(self, local_hour: int, expected: str | None) -> None:
        obs = _at(_obs(("a", 1, 50.0, 30.0)), local_hour)
        assert obs.select(rush_window()).item() == expected

    def test_hours_are_configurable(self) -> None:
        obs = _at(_obs(("a", 1, 50.0, 30.0)), 6)
        assert obs.select(rush_window(am=(6, 9))).item() == "am"


def _rush(obs: pl.DataFrame, min_trucks: float = 1.0) -> pl.DataFrame:
    return rush_speeds(_cells(obs, rush_window()), min_trucks).collect()


class TestRushSpeeds:
    def test_slower_rush_against_night(self) -> None:
        # Edge 1, 400 m: 60 km/h at night, 30 in the morning, 45 in the evening.
        obs = _at(
            _obs(("a", 1, 60.0, 30.0), ("b", 1, 30.0, 30.0), ("c", 1, 45.0, 30.0)),
            3,
            8,
            17,
        )
        row = _rush(obs).row(0, named=True)
        assert row["free_speed"] == 60.0
        assert row["rush_speed"] == 30.0
        assert row["rush_window"] == "am"
        assert row["ratio"] == pytest.approx(0.5)
        # 48 s at 30 km/h against 24 s at 60.
        assert row["delay_s"] == pytest.approx(24.0)

    def test_needs_a_night_and_a_rush(self) -> None:
        # Edge 1 is seen at night and midday, edge 2 only in the morning.
        obs = _at(
            _obs(("a", 1, 60.0, 30.0), ("b", 1, 30.0, 30.0), ("c", 2, 30.0, 30.0)),
            3,
            12,
            8,
        )
        assert _rush(obs).height == 0

    def test_thin_windows_left_out(self) -> None:
        # Two trucks at night and in the evening, one in the morning.
        obs = _at(
            _obs(
                *[(t, 1, 60.0, 30.0) for t in "ab"],
                ("c", 1, 20.0, 30.0),
                *[(t, 1, 40.0, 30.0) for t in "de"],
            ),
            3,
            3,
            8,
            17,
            17,
        )
        assert _rush(obs)["rush_window"].to_list() == ["am"]
        assert _rush(obs, min_trucks=2.0)["rush_window"].to_list() == ["pm"]
        assert _rush(obs, min_trucks=3.0).height == 0

    def test_stopped_cells_left_out(self) -> None:
        # The morning's one fix caught a truck stopped; the evening measures.
        obs = _at(
            _obs(("a", 1, 60.0, 30.0), ("b", 1, 0.0, 30.0), ("c", 1, 45.0, 30.0)),
            3,
            8,
            17,
        )
        assert _rush(obs)["rush_window"].to_list() == ["pm"]
        stopped_at_night = obs.with_columns(speed=pl.Series([0.0, 30.0, 45.0]))
        assert _rush(stopped_at_night).height == 0

    def test_tie_goes_to_the_morning(self) -> None:
        # The evening is slower only by float noise.
        obs = _at(
            _obs(
                ("a", 1, 60.0, 30.0), ("b", 1, 45.0, 30.0), ("c", 1, 45 - 1e-12, 30.0)
            ),
            3,
            8,
            17,
        )
        assert _rush(obs)["rush_window"].to_list() == ["am"]


def _ways(obs: pl.DataFrame, min_trucks: float = 1.0) -> pl.DataFrame:
    return way_rush_speeds(_cells(obs, rush_window()), min_trucks).collect()


class TestWayRushSpeeds:
    def test_travel_time_over_the_ways_edges(self) -> None:
        # Edges 1 (400 m) and 2 (250 m) of way 7 run 60 and 50 km/h at
        # night, 30 and 25 in the morning; edge 1 runs 40 in the evening.
        obs = _at(
            _obs(
                ("a", 1, 60.0, 30.0),
                ("b", 2, 50.0, 30.0),
                ("c", 1, 30.0, 30.0),
                ("d", 2, 25.0, 30.0),
                ("e", 1, 40.0, 30.0),
            ),
            3,
            3,
            8,
            8,
            17,
        )
        row = _ways(obs).row(0, named=True)
        assert (row["way_id"], row["forward"]) == (7, True)
        assert row["rush_window"] == "am"
        assert row["edge_ids"] == [1, 2]
        assert row["length"] == 650.0
        free_h = 0.4 / 60 + 0.25 / 50  # and twice that in the morning
        assert row["free_speed"] == pytest.approx(0.65 / free_h)
        assert row["ratio"] == pytest.approx(0.5)
        assert row["delay_s"] == pytest.approx(3600 * free_h)

    def test_both_speeds_from_the_same_edges(self) -> None:
        # Edge 2 has no night cell, so its slow morning doesn't count.
        obs = _at(
            _obs(("a", 1, 60.0, 30.0), ("b", 1, 45.0, 30.0), ("c", 2, 10.0, 30.0)),
            3,
            8,
            8,
        )
        row = _ways(obs).row(0, named=True)
        assert row["edge_ids"] == [1]
        assert row["ratio"] == pytest.approx(0.75)

    def test_directions_kept_apart(self) -> None:
        # Edge 3 is way 7 against its direction, slow in the evening.
        obs = _at(
            _obs(
                ("a", 1, 60.0, 30.0),
                ("b", 1, 30.0, 30.0),
                ("c", 3, 50.0, 30.0),
                ("d", 3, 45.0, 30.0),
            ),
            3,
            8,
            3,
            17,
        )
        out = _ways(obs)
        assert out.select("forward", "rush_window").rows() == [
            (False, "pm"),
            (True, "am"),
        ]

    def test_tie_goes_to_the_morning(self) -> None:
        # Edge 2's evening matches edge 1's morning but for float noise.
        obs = _at(
            _obs(
                ("a", 1, 60.0, 30.0),
                ("b", 2, 60.0, 30.0),
                ("c", 1, 30.0, 30.0),
                ("d", 2, 30 - 1e-12, 30.0),
            ),
            3,
            3,
            8,
            17,
        )
        row = _ways(obs).row(0, named=True)
        assert (row["rush_window"], row["edge_ids"]) == ("am", [1])
