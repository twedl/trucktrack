"""Tests for trucktrack.congestion.impute: the edge model and its blend."""

from __future__ import annotations

import polars as pl
import pytest
from trucktrack.congestion import (
    ImputeConfig,
    class_weeks,
    fit_model,
    flow_speeds,
    historical_speeds,
    length_band,
    speed_ratios,
)

# The cells are exact, so no need to pull patterns towards 1.
_EXACT = ImputeConfig(pattern_seconds=0.0)

# Tuesday 08:00 and 17:00, and 03:00, as hours of the week from Sunday.
_AM, _PM, _NIGHT = 2 * 24 + 8, 2 * 24 + 17, 2 * 24 + 3
_ALL = range(168)


def _pattern(hour: int) -> float:
    """Weekday dips to 0.5 at 08:00 and 17:00; 1 otherwise."""
    weekday, h = divmod(hour, 24)
    return 0.5 if 1 <= weekday <= 5 and h in (8, 17) else 1.0


def _cells(
    edges: dict[int, tuple[float, float, float]],
    hours: dict[int, range | list[int]] | None = None,
    *,
    trucks: float = 1.0,
) -> pl.DataFrame:
    """Hourly cells of motorway edges {id: (level, am depth, pm depth)}.

    Each cell's speed is exactly level * pattern ** depth, with 600 s of
    driving behind it; *hours* limits which hours an edge was measured.
    """
    rows = []
    for edge, (level, am, pm) in edges.items():
        for h in (hours or {}).get(edge, _ALL):
            depth = am if h % 24 < 12 else pm
            rows.append((edge, h, level * _pattern(h) ** depth))
    return pl.DataFrame(
        rows, schema=["edge_id", "hour_of_week", "speed"], orient="row"
    ).with_columns(
        pl.col("edge_id").cast(pl.UInt64),
        pl.col("hour_of_week").cast(pl.Int16),
        tile_build_id=pl.lit("b"),
        way_id=pl.col("edge_id").cast(pl.Int64),
        forward=pl.lit(True),
        road_class=pl.lit("motorway"),
        edge_length=pl.lit(500.0),
        seconds=pl.lit(600.0),
        fixes=pl.lit(10, dtype=pl.UInt32),
        trucks=pl.lit(trucks),
    )


def _at(frame: pl.DataFrame, edge: int, hour: int) -> dict[str, object]:
    return frame.filter(pl.col("edge_id") == edge, pl.col("hour_of_week") == hour).row(
        0, named=True
    )


class TestFitModel:
    def test_recovers_levels_and_pattern(self) -> None:
        cells = _cells({1: (100.0, 1.0, 1.0), 2: (60.0, 1.0, 1.0)})
        level, depth, pattern = fit_model(cells, _EXACT)
        assert dict(level.select("edge_id", "level").rows()) == pytest.approx(
            {1: 100.0, 2: 60.0}, rel=1e-6
        )
        p = dict(pattern.select("hour_of_week", "pattern").rows())
        assert p[_AM] == pytest.approx(0.5, rel=1e-6)
        assert p[_NIGHT] == pytest.approx(1.0, rel=1e-6)
        assert depth["depth"].to_list() == pytest.approx([1.0] * 4, rel=1e-3)

    def test_deeper_edge_gets_deeper_depth(self) -> None:
        cells = _cells({1: (100.0, 0.5, 0.5), 2: (100.0, 1.5, 1.5)})
        _, depth, _ = fit_model(cells)
        d = depth.group_by("edge_id").agg(pl.col("depth").mean()).sort("edge_id")
        assert d["depth"][0] < 1.0 < d["depth"][1]

    def test_morning_and_evening_depths_apart(self) -> None:
        # Edge 1 slows only in the morning, edge 2 only in the evening.
        cells = _cells({1: (100.0, 1.0, 0.0), 2: (100.0, 0.0, 1.0)})
        _, depth, _ = fit_model(cells)
        d = {(e, h): v for e, h, v in depth.select("edge_id", "half", "depth").rows()}
        assert d[(1, False)] > d[(1, True)]
        assert d[(2, True)] > d[(2, False)]
        _, one_depth, _ = fit_model(cells, ImputeConfig(depth_split_hour=24))
        assert one_depth["half"].to_list() == [False, False]

    def test_stopped_edge_does_not_spoil_the_fit(self) -> None:
        cells = _cells({1: (100.0, 1.0, 1.0), 2: (0.0, 1.0, 1.0)})
        level, depth, pattern = fit_model(cells, _EXACT)
        assert level["edge_id"].to_list() == [1]
        assert not depth["depth"].is_nan().any()
        p = dict(pattern.select("hour_of_week", "pattern").rows())
        assert p[_AM] == pytest.approx(0.5, rel=1e-6)

    def test_thin_class_hours_pulled_towards_flat(self) -> None:
        cells = _cells({1: (100.0, 1.0, 1.0)})  # 600 s behind each hour
        _, _, pattern = fit_model(cells, ImputeConfig(pattern_seconds=600.0))
        p = dict(pattern.select("hour_of_week", "pattern").rows())
        assert p[_AM] == pytest.approx(0.5**0.5, rel=1e-6)
        assert pattern["seconds"].unique().to_list() == [600.0]

    def test_given_model_is_used(self) -> None:
        cells = _cells({1: (100.0, 1.0, 1.0)}, {1: [_NIGHT]})
        level, depth, pattern = fit_model(cells)
        model = (level.with_columns(level=pl.lit(50.0)), depth, pattern)
        out = historical_speeds(cells, ImputeConfig(cell_trucks=0.0), model)
        assert _at(out, 1, _NIGHT + 24)["speed"] == pytest.approx(50.0)


class TestHistoricalSpeeds:
    def test_every_hour_of_every_measured_edge_in_order(self) -> None:
        cells = _cells({2: (90.0, 1.0, 1.0), 1: (100.0, 1.0, 1.0)}, {1: [_NIGHT]})
        out = historical_speeds(cells)
        assert out["edge_id"].to_list() == [1] * 168 + [2] * 168
        assert out["hour_of_week"].to_list() == list(range(168)) * 2
        assert out["road_class"].unique().to_list() == ["motorway"]

    def test_edge_only_seen_stopped_gets_no_speeds(self) -> None:
        out = historical_speeds(_cells({1: (100.0, 1.0, 1.0), 2: (0.0, 1.0, 1.0)}))
        assert out["edge_id"].unique().to_list() == [1]

    def test_unmeasured_hours_follow_the_class(self) -> None:
        # Edge 2 is only seen at night; edge 1 teaches the pattern.
        cells = _cells(
            {1: (100.0, 1.0, 1.0), 2: (60.0, 1.0, 1.0)}, {2: [_NIGHT, _NIGHT + 24]}
        )
        row = _at(historical_speeds(cells, _EXACT), 2, _AM)
        assert row["source"] == "imputed"
        assert row["own_weight"] == 0.0
        assert row["speed"] == pytest.approx(30.0, rel=0.01)

    def test_measured_hours_blend_by_trucks(self) -> None:
        # Edge 1 ran slow one Tuesday morning: 20 km/h where the rest of
        # its class, and the model, say about 50.
        cells = _cells({e: (100.0, 1.0, 1.0) for e in range(1, 6)}, trucks=3.0)
        slow = (pl.col("edge_id") == 1) & (pl.col("hour_of_week") == _AM)
        cells = cells.with_columns(speed=pl.when(slow).then(20.0).otherwise("speed"))
        row = _at(historical_speeds(cells, ImputeConfig(cell_trucks=1.0)), 1, _AM)
        assert row["source"] == "blended"
        assert row["own_weight"] == pytest.approx(0.75)
        assert 20.0 < row["speed"] < 0.75 * 20.0 + 0.25 * 50.0 + 1.0

    def test_no_blend_keeps_measurements(self) -> None:
        cells = _cells({1: (100.0, 1.0, 1.0)}, {1: [_NIGHT]})
        out = historical_speeds(cells, ImputeConfig(cell_trucks=0.0))
        assert _at(out, 1, _NIGHT)["source"] == "measured"
        assert _at(out, 1, _AM)["source"] == "imputed"


class TestClassWeeks:
    def _pattern(self, hours: float) -> pl.DataFrame:
        cells = _cells({1: (100.0, 1.0, 1.0)}).with_columns(
            seconds=pl.lit(hours * 3600 / 168)
        )
        return fit_model(cells, _EXACT)[2]

    def test_own_speed_shaped_by_the_class(self) -> None:
        speeds = pl.DataFrame({"road_class": ["motorway"] * 2, "speed": [80.0, 100.0]})
        out = class_weeks(self._pattern(60.0), speeds)
        assert out.height == 2 * 168
        week = out.filter(pl.col("speed") == 80.0)
        assert dict(week.select("hour_of_week", "week_speed").rows())[_AM] == (
            pytest.approx(40.0)
        )
        assert week["week_speed"].max() == pytest.approx(80.0)

    def test_thinly_measured_classes_left_out(self) -> None:
        speeds = pl.DataFrame({"road_class": ["motorway"], "speed": [80.0]})
        assert class_weeks(self._pattern(10.0), speeds).is_empty()
        assert class_weeks(self._pattern(10.0), speeds, min_class_hours=5).height == 168

    def test_hours_the_class_wasnt_measured_stay_at_its_speed(self) -> None:
        pattern = self._pattern(60.0).filter(pl.col("hour_of_week") != _AM)
        speeds = pl.DataFrame({"road_class": ["motorway"], "speed": [80.0]})
        week = class_weeks(pattern, speeds)
        assert week.height == 168
        assert dict(week.select("hour_of_week", "week_speed").rows())[_AM] == 80.0

    def test_flat_classes_left_out(self) -> None:
        flat = self._pattern(60.0).with_columns(pattern=pl.lit(0.97))
        speeds = pl.DataFrame({"road_class": ["motorway"], "speed": [80.0]})
        assert class_weeks(flat, speeds).is_empty()
        assert class_weeks(flat, speeds, min_dip=0.01).height == 168

    def test_classes_without_a_pattern_left_out(self) -> None:
        speeds = pl.DataFrame({"road_class": ["residential"], "speed": [40.0]})
        assert class_weeks(self._pattern(60.0), speeds).is_empty()


class TestSpeedRatios:
    def _ratios(self, ratio_seconds: float) -> dict[tuple[str, int], float]:
        # A 30 m junction piece trucks drive at half Valhalla's 100 km/h,
        # and a 500 m block they drive at its speed, for an hour each.
        ids = pl.Series([1, 2], dtype=pl.UInt64)
        build = pl.lit("b").alias("tile_build_id")
        level = pl.DataFrame({"edge_id": ids, "level": [50.0, 100.0]}).with_columns(
            build
        )
        cells = pl.DataFrame({"edge_id": ids, "seconds": [3600.0] * 2}).with_columns(
            build
        )
        edges = pl.DataFrame(
            {
                "edge_id": ids,
                "road_class": ["primary"] * 2,
                "length": [30.0, 500.0],
                "speed": [100.0] * 2,
            }
        )
        out = speed_ratios(level, cells, edges, ratio_seconds=ratio_seconds)
        return {(c, b): r for c, b, r in out.rows()}

    def test_each_length_band_its_own_with_enough_data(self) -> None:
        assert self._ratios(0.0) == pytest.approx(
            {("primary", 0): 0.5, ("primary", 2): 1.0}
        )

    def test_thin_bands_pulled_to_their_class(self) -> None:
        # Class: 150 km in 2 h at 100 km/h = 0.75; each band has 1 h.
        assert self._ratios(3600.0) == pytest.approx(
            {("primary", 0): 0.625, ("primary", 2): 0.875}
        )

    def test_length_bands(self) -> None:
        lengths = pl.DataFrame({"length": [10.0, 50.0, 199.0, 200.0, 1000.0]})
        assert lengths.select(length_band())["band"].to_list() == [0, 1, 1, 2, 3]


class TestFlowSpeeds:
    def test_harmonic_mean_over_valhallas_windows(self) -> None:
        historical = _cells({1: (100.0, 1.0, 1.0)})
        out = flow_speeds(historical)
        # Daytime: 12 h x 7 days, ten of them at half speed.
        day = 84 / (74 / 100.0 + 10 / 50.0)
        assert out["constrained_speed"].item() == pytest.approx(day)
        assert out["freeflow_speed"].item() == pytest.approx(100.0)
