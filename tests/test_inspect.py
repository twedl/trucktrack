"""Tests for ``trucktrack.inspect`` — the REPL workflow helpers.

Covers the parts that don't require Valhalla: splitter chaining, the
cached quality path, error validation, and the date-range filter in
``load_truck_trace``.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl
import pytest
from polars.testing import assert_frame_equal
from trucktrack import inspect as tti
from trucktrack.splitters import split_by_observation_gap, split_by_stops


def _trace(
    id_: str = "truck_A",
    *,
    start: datetime,
    points: list[tuple[float, float]],
    step: timedelta = timedelta(seconds=60),
) -> pl.DataFrame:
    """Build a minimal (id, time, lat, lon) DataFrame."""
    times = [start + step * i for i in range(len(points))]
    return pl.DataFrame(
        {
            "id": [id_] * len(points),
            "time": times,
            "lat": [p[0] for p in points],
            "lon": [p[1] for p in points],
        }
    )


def _moving_then_stop_then_moving() -> pl.DataFrame:
    """20 moving pts, 30 stationary pts (>5 min), 20 moving pts."""
    start = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)
    pts: list[tuple[float, float]] = []
    # Leg 1: moving east along a line.
    for i in range(20):
        pts.append((43.0, -79.0 + i * 0.001))
    # Stop: 30 points stationary at the last leg-1 position.
    stop_pt = pts[-1]
    pts.extend([stop_pt] * 30)
    # Leg 2: continue east, with a 10-minute observation gap before it.
    base_lon = stop_pt[1]
    for i in range(1, 21):
        pts.append((43.0, base_lon + i * 0.001))
    df = _trace(start=start, points=pts)
    # Insert a 10-min gap between row 49 and 50 (start of leg 2).
    new_time = df["time"].to_list()[:50] + [
        t + timedelta(minutes=10) for t in df["time"].to_list()[50:]
    ]
    return df.with_columns(pl.Series("time", new_time))


# ---------------------------------------------------------------------------
# split_trips
# ---------------------------------------------------------------------------


class TestSplitTrips:
    def test_adds_segment_id_and_is_stop(self) -> None:
        # Disable the traffic filter so the stationary cluster stays
        # classified as a stop — the fixture moves in a straight line on
        # either side of it, so approach/departure bearings match.
        df = _moving_then_stop_then_moving()
        out = tti.split_trips(
            df,
            gap=timedelta(minutes=5),
            stop_max_diameter=30.0,
            stop_min_duration=timedelta(minutes=2),
            traffic_max_angle_change=None,
        )
        assert "segment_id" in out.columns
        assert "is_stop" in out.columns
        assert out.filter(pl.col("is_stop")).height > 0
        # Gap should produce more than one non-stop segment.
        moving = out.filter(~pl.col("is_stop"))
        assert moving["segment_id"].n_unique() >= 2

    def test_skip_traffic_filter_equals_split_by_stops(self) -> None:
        df = _moving_then_stop_then_moving()
        gap = timedelta(minutes=5)
        gapped = split_by_observation_gap(df, gap, min_length=3)
        baseline = split_by_stops(gapped, 30.0, timedelta(minutes=2), min_length=3)
        out = tti.split_trips(
            df,
            gap=gap,
            stop_max_diameter=30.0,
            stop_min_duration=timedelta(minutes=2),
            traffic_max_angle_change=None,
        )
        assert_frame_equal(out, baseline)


# ---------------------------------------------------------------------------
# evaluate_quality
# ---------------------------------------------------------------------------


def _straight_split_df() -> pl.DataFrame:
    """A single non-stop segment with 10 collinear points."""
    start = datetime(2026, 1, 1, tzinfo=UTC)
    pts = [(43.0, -79.0 + i * 0.001) for i in range(10)]
    df = _trace(start=start, points=pts)
    return df.with_columns(
        pl.lit(1, dtype=pl.Int64).alias("segment_id"),
        pl.lit(False).alias("is_stop"),
    )


class TestEvaluateQualityCached:
    def test_cached_path_ratio_and_flag(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A shape twice the length of the input should flag has_issues=True."""

        # Ensure we never call Valhalla on the cached path.
        def _boom(*_a: object, **_k: object) -> object:
            raise AssertionError("evaluate_map_match must not be called on cached path")

        monkeypatch.setattr("trucktrack.inspect.evaluate_map_match", _boom)

        split = _straight_split_df()
        pts = list(zip(split["lat"].to_list(), split["lon"].to_list(), strict=True))
        # Build a shape that retraces the input — total length = 2× input.
        shape = pts + list(reversed(pts))
        tm = tti.TripMatch(
            segment_id=1,
            matched_df=split,
            way_ids=[],
            shape=shape,
        )
        out = tti.evaluate_quality(split, trips={1: tm})

        assert out.height == 1
        row = out.row(0, named=True)
        assert row["segment_id"] == 1
        assert row["n_points"] == len(pts)
        assert row["path_length_ratio"] is not None
        assert row["path_length_ratio"] == pytest.approx(2.0, rel=1e-3)
        # Straight retrace: one ~180° flip → reversals == 1.
        assert row["heading_reversals"] >= 1
        assert row["has_issues"] is True
        assert row["ok"] is False
        assert row["error"] is None

    def test_requires_trips_or_tile_extract(self) -> None:
        split = _straight_split_df()
        with pytest.raises(ValueError, match="trips=.* or tile_extract="):
            tti.evaluate_quality(split)


# ---------------------------------------------------------------------------
# load_truck_trace time filter (raw hive layout fixture)
# ---------------------------------------------------------------------------


class TestLoadTruckTrace:
    def test_time_filter_tz_aware(self, tmp_path: Path) -> None:
        truck_id = "deadbeef"  # chunk_id = "ef"
        chunk_dir = tmp_path / "year=2026" / f"chunk_id={truck_id[-2:]}"
        chunk_dir.mkdir(parents=True)
        times = [
            datetime(2026, 1, 1, 10, 0, tzinfo=UTC),
            datetime(2026, 1, 1, 10, 30, tzinfo=UTC),
            datetime(2026, 1, 1, 12, 0, tzinfo=UTC),
        ]
        df = pl.DataFrame(
            {
                "id": [truck_id] * 3,
                "time": times,
                "lat": [43.0, 43.1, 43.2],
                "lon": [-79.0, -79.1, -79.2],
            }
        )
        df.write_parquet(chunk_dir / "part-0.parquet")

        # Request [10:00, 11:00) — naive datetimes should be coerced to UTC.
        out = tti.load_truck_trace(
            truck_id,
            datetime(2026, 1, 1, 10, 0),
            datetime(2026, 1, 1, 11, 0),
            data_dir=tmp_path,
        )
        assert out.height == 2
        assert out["time"].to_list() == times[:2]

    def test_empty_result_raises(self, tmp_path: Path) -> None:
        truck_id = "cafecafe"
        chunk_dir = tmp_path / "year=2026" / f"chunk_id={truck_id[-2:]}"
        chunk_dir.mkdir(parents=True)
        pl.DataFrame(
            {
                "id": [truck_id],
                "time": [datetime(2026, 1, 1, tzinfo=UTC)],
                "lat": [43.0],
                "lon": [-79.0],
            }
        ).write_parquet(chunk_dir / "part-0.parquet")

        with pytest.raises(ValueError, match="no rows"):
            tti.load_truck_trace(
                truck_id,
                datetime(2027, 1, 1),
                datetime(2027, 1, 2),
                data_dir=tmp_path,
            )

    def test_requires_data_dir_or_index(self) -> None:
        with pytest.raises(ValueError, match="data_dir or index"):
            tti.load_truck_trace(
                "abcd",
                datetime(2026, 1, 1),
                datetime(2026, 1, 2),
            )
