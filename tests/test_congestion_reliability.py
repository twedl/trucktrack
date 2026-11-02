"""Tests for trucktrack.congestion.reliability: corridor passes and FHWA's TTTR."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import polars as pl
import pytest
from trucktrack.congestion.reliability import (
    corridor_passes,
    fhwa_period,
    travel_time_reliability,
    tttr_index,
)

# Tuesday 10 March 2026, 00:00 in Toronto (EDT, UTC-4).
_TUESDAY = datetime(2026, 3, 10, 4, tzinfo=UTC)

# Three 1 km edges, A to B, and the same road back.
_EDGES = pl.DataFrame(
    {
        "edge_id": pl.Series([1, 2, 3, 11], dtype=pl.UInt64),
        "direction": ["A to B"] * 3 + ["B to A"],
        "begin_m": [0.0, 1000.0, 2000.0, 0.0],
        "end_m": [1000.0, 2000.0, 3000.0, 1000.0],
    }
)
# The corridor from 500 m to 2500 m, in two sections.
_SPANS = pl.DataFrame(
    {
        "direction": ["A to B"] * 3,
        "kind": ["corridor", "section", "section"],
        "name": ["A to B", "A to M", "M to B"],
        "begin_m": [500.0, 500.0, 1500.0],
        "end_m": [2500.0, 1500.0, 2500.0],
    }
)


def _obs(*fixes: tuple[str, float, int, float]) -> pl.LazyFrame:
    """(trip id, seconds after _TUESDAY, edge, distance along it) per fix."""
    return pl.LazyFrame(
        {
            "id": [f[0] for f in fixes],
            "time": [_TUESDAY + timedelta(seconds=f[1]) for f in fixes],
            "edge_id": pl.Series([f[2] for f in fixes], dtype=pl.UInt64),
            "distance_along_edge": [f[3] for f in fixes],
        }
    )


def _passes(obs: pl.LazyFrame, **kwargs: float) -> dict[str, float]:
    out = corridor_passes(obs, _EDGES, _SPANS, **kwargs).collect()
    return dict(zip(out["name"], out["travel_s"], strict=True))


class TestFhwaPeriod:
    @pytest.mark.parametrize(
        ("local", "period"),
        [
            ((10, 6), "AM peak"),
            ((10, 9), "AM peak"),
            ((10, 10), "Midday"),
            ((10, 16), "PM peak"),
            ((10, 20), "Overnight"),
            ((10, 5), "Overnight"),
            ((14, 12), "Weekend"),  # Saturday
            ((14, 5), "Overnight"),
            ((15, 19), "Weekend"),  # Sunday
        ],
    )
    def test_local_clock(self, local: tuple[int, int], period: str) -> None:
        day, hour = local
        when = datetime(2026, 3, day, hour) + timedelta(hours=4)  # EDT
        frame = pl.DataFrame({"t": [when]})  # tz-naive: UTC
        assert frame.select(fhwa_period(pl.col("t")))["period"].item() == period


class TestCorridorPasses:
    def test_crossings_interpolated_on_chainage(self) -> None:
        # 100 m at 0 s, 1200 m at 60 s, 2900 m at 120 s.
        passes = _passes(
            _obs(
                ("t_gap0_trip0", 0, 1, 0.1),
                ("t_gap0_trip0", 60, 2, 0.2),
                ("t_gap0_trip0", 120, 3, 0.9),
            )
        )
        at = {
            500: 400 / 1100 * 60,
            1500: 60 + 300 / 1700 * 60,
            2500: 60 + 1300 / 1700 * 60,
        }
        assert passes["A to B"] == pytest.approx(at[2500] - at[500])
        assert passes["A to M"] == pytest.approx(at[1500] - at[500])
        assert passes["M to B"] == pytest.approx(at[2500] - at[1500])

    def test_entry_time_and_a_fix_on_the_point(self) -> None:
        obs = _obs(("t", 0, 1, 0.5), ("t", 50, 2, 0.5), ("t", 100, 3, 0.5))
        out = corridor_passes(obs, _EDGES, _SPANS).collect().sort("begin_m", "end_m")
        assert out["entry"][0] == _TUESDAY
        assert dict(zip(out["name"], out["travel_s"], strict=True)) == {
            "A to M": 50.0,
            "A to B": 100.0,
            "M to B": 50.0,
        }

    def test_one_truck_across_its_trips(self) -> None:
        # A trip split mid-corridor (a long stop in traffic) is one pass.
        obs = _obs(("t_gap0_trip0", 0, 1, 0.5), ("t_gap0_trip1", 300, 3, 0.5))
        assert _passes(obs) == {"A to B": 300.0, "A to M": 150.0, "M to B": 150.0}

    def test_off_the_corridor_or_out_of_sight(self) -> None:
        # A fix off the corridor (edge 99) between, or a long gap.
        off = _obs(("t", 0, 1, 0.5), ("t", 40, 99, 0.5), ("t", 100, 3, 0.5))
        assert _passes(off) == {}
        assert _passes(_obs(("t", 0, 1, 0.5), ("t", 700, 3, 0.5))) == {}
        assert _passes(_obs(("t", 0, 1, 0.5), ("t", 700, 3, 0.5)), max_gap_s=800) != {}

    def test_other_direction_and_other_trucks_apart(self) -> None:
        obs = _obs(
            ("t", 0, 1, 0.5),
            ("t", 10, 11, 0.5),  # the truck turns onto the other carriageway
            ("u", 0, 1, 0.0),
            ("u", 100, 3, 1.0),
        )
        out = corridor_passes(obs, _EDGES, _SPANS).collect()
        assert set(out["truck"]) == {"u"}

    def test_stopped_on_the_point_enters_on_arrival(self) -> None:
        # Stopped at 1500 m from 50 s to 200 s: M is passed at 50 s.
        obs = _obs(
            ("t", 0, 1, 0.5), ("t", 50, 2, 0.5), ("t", 200, 2, 0.5), ("t", 250, 3, 0.5)
        )
        passes = _passes(obs)
        assert passes["A to M"] == 50.0
        assert passes["M to B"] == 200.0


def _bins(
    *travel_s: float, start: datetime = _TUESDAY + timedelta(hours=10)
) -> pl.DataFrame:
    """A pass every 15 minutes from *start* (Tuesday 10:00), one per bin."""
    return pl.DataFrame(
        {
            "direction": "A to B",
            "kind": "segment",
            "name": "A to B 1/1",
            "begin_m": 0.0,
            "end_m": 1000.0,
            "truck": "t",
            "entry": [start + timedelta(minutes=15 * i) for i in range(len(travel_s))],
            "travel_s": list(travel_s),
        }
    )


class TestTravelTimeReliability:
    def test_percentiles_of_15_minute_averages(self) -> None:
        passes = pl.concat(
            [
                _bins(*map(float, range(100, 120))),
                # A second pass in the first bin: it averages 100 and 110.
                _bins(110.0).with_columns(pl.col("entry") + timedelta(minutes=5)),
            ]
        )
        row = travel_time_reliability(passes).collect().row(0, named=True)
        assert row["period"] == "Midday"
        assert (row["bins"], row["passes"]) == (20, 21)
        times = [105.0, *range(101, 120)]  # 100 and 110 average to 105
        assert row["p50_s"] == round(pl.Series(times).quantile(0.5, "linear"))
        p95 = pl.Series(times).quantile(0.95, "linear")
        assert row["p95_s"] == round(p95)
        assert row["tttr"] == round(round(p95) / row["p50_s"], 2)
        mean = sum(times) / len(times)
        assert row["buffer_index"] == pytest.approx((round(p95) - mean) / mean)

    def test_index_weights_each_segments_worst_period(self) -> None:
        reliability = pl.DataFrame(
            {
                "direction": "A to B",
                "kind": "segment",
                "name": ["s1", "s1", "s2"],
                "begin_m": [0.0, 0.0, 1000.0],
                "end_m": [1000.0, 1000.0, 4000.0],
                "period": ["AM peak", "Overnight", "AM peak"],
                "tttr": [1.5, 1.1, 1.2],
            }
        )
        out = tttr_index(reliability).collect().row(0, named=True)
        assert out["segments"] == 2
        assert out["tttr_index"] == pytest.approx((1.5 * 1000 + 1.2 * 3000) / 4000)
