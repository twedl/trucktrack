"""Tests for trucktrack.congestion.impute: the edge model and its blend."""

from __future__ import annotations

import polars as pl
import pytest
from trucktrack.congestion import (
    ImputeConfig,
    fit_model,
    flow_speeds,
    historical_speeds,
)

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
        level, depth, pattern = fit_model(cells)
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
        level, depth, pattern = fit_model(cells)
        assert level["edge_id"].to_list() == [1]
        assert not depth["depth"].is_nan().any()
        p = dict(pattern.select("hour_of_week", "pattern").rows())
        assert p[_AM] == pytest.approx(0.5, rel=1e-6)


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
        row = _at(historical_speeds(cells), 2, _AM)
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


class TestFlowSpeeds:
    def test_harmonic_mean_over_valhallas_windows(self) -> None:
        historical = _cells({1: (100.0, 1.0, 1.0)})
        out = flow_speeds(historical)
        # Daytime: 12 h x 7 days, ten of them at half speed.
        day = 84 / (74 / 100.0 + 10 / 50.0)
        assert out["constrained_speed"].item() == pytest.approx(day)
        assert out["freeflow_speed"].item() == pytest.approx(100.0)
