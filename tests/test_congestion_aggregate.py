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
    edge_speeds,
    flow_window,
    hour_of_week,
    usable_observations,
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


def _speeds(obs: pl.DataFrame, **config: Any) -> pl.DataFrame:
    return edge_speeds(obs.lazy(), config=AggregateConfig(**config)).collect()


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


class TestEdgeSpeeds:
    def test_well_covered_edge_gives_its_own_speed(self) -> None:
        obs = _obs(*_PAIR, ("c", 2, 10.0, 30.0))
        row = _speeds(obs, min_trucks=2).filter(pl.col("edge_id") == 1)
        assert row["source"].item() == "edge"
        assert row["speed"].item() == pytest.approx(50.0)
        assert row["hour_of_week"].item() == 2 * 24 + 8

    def test_thin_edge_takes_its_ways_speed(self) -> None:
        obs = _obs(*_PAIR, ("c", 2, 10.0, 30.0))
        row = _speeds(obs, min_trucks=2).filter(pl.col("edge_id") == 2)
        assert row["source"].item() == "way"
        assert row["speed"].item() == pytest.approx(110 / 3)
        assert row["edge_speed"].item() == 10.0
        assert row["way_trucks"].item() == pytest.approx(3.0)

    def test_short_edge_does_not_take_its_ways_speed(self) -> None:
        obs = _obs(*_PAIR, ("c", 4, 10.0, 30.0))
        row = _speeds(obs, min_trucks=2).filter(pl.col("edge_id") == 4)
        assert row["speed"].item() is None
        assert row["way_speed"].item() == pytest.approx(110 / 3)
        pooled = _speeds(obs, min_trucks=2, min_pool_edge_m=0.0)
        assert pooled.filter(pl.col("edge_id") == 4)["source"].item() == "way"

    def test_way_pools_one_direction_only(self) -> None:
        obs = _obs(*_PAIR, ("c", 3, 10.0, 30.0))
        row = _speeds(obs, min_trucks=2).filter(pl.col("edge_id") == 3)
        assert row["speed"].item() is None
        assert row["source"].item() is None
        assert row["way_trucks"].item() == pytest.approx(1.0)

    def test_edge_without_a_fix_in_a_bin_takes_its_ways(self) -> None:
        # Edge 2 is seen at 08:00 only; at 09:00 only edge 1 is.
        later = _obs(*_PAIR).with_columns(time=pl.lit(_T0 + timedelta(hours=1)))
        obs = pl.concat([_obs(*_PAIR, ("c", 2, 10.0, 30.0)), later])
        rows = _speeds(obs).filter(pl.col("edge_id") == 2)
        assert rows["hour_of_week"].to_list() == [56, 57]
        assert rows["source"].to_list() == ["edge", "way"]
        assert rows["edge_fixes"].to_list() == [1, None]
        assert rows["speed"].to_list() == pytest.approx([10.0, 50.0])

    def test_default_keeps_every_edge_cell(self) -> None:
        obs = _obs(*_PAIR, ("c", 4, 10.0, 30.0))
        row = _speeds(obs).filter(pl.col("edge_id") == 4)
        assert row["source"].item() == "edge"
        assert row["speed"].item() == 10.0

    def test_flagged_observations_left_out(self) -> None:
        obs = _obs(("a", 1, 40.0, 30.0), ("b", 1, 0.0, 300.0)).with_columns(
            dwell=pl.Series([False, True])
        )
        row = _speeds(obs, min_trucks=1)
        assert row["speed"].item() == 40.0
        assert row["edge_fixes"].item() == 1

    def test_custom_bin(self) -> None:
        obs = _obs(*_PAIR)
        out = edge_speeds(obs.lazy(), by=pl.lit(0).alias("all_day")).collect()
        assert out["all_day"].to_list() == [0]
        assert out["speed"].item() == pytest.approx(50.0)

    def test_tile_builds_kept_apart(self) -> None:
        obs = _obs(*_PAIR)
        obs = obs.with_columns(tile_build_id=pl.Series(["x", "y"]))
        out = _speeds(obs, min_trucks=1)
        assert out["tile_build_id"].to_list() == ["x", "y"]
        assert out["speed"].to_list() == [40.0, 60.0]


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
