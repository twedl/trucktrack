"""Tests for the visualize submodule."""

from __future__ import annotations

import tempfile
from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl
import pytest

folium = pytest.importorskip("folium")

from trucktrack.generate.models import TracePoint  # noqa: E402
from trucktrack.visualize import plot_trace, save_map  # noqa: E402


def _make_tracepoints(n: int = 10) -> list[TracePoint]:
    base = datetime(2025, 6, 15, 8, 0, tzinfo=UTC)
    return [
        TracePoint(
            lat=43.65 + i * 0.01,
            lon=-79.38 + i * 0.01,
            speed_mph=55.0 + i,
            heading=90.0,
            timestamp=base + timedelta(minutes=i),
        )
        for i in range(n)
    ]


def _make_raw_df(n: int = 10) -> pl.DataFrame:
    base = datetime(2025, 6, 15, 8, 0, tzinfo=UTC)
    return pl.DataFrame(
        {
            "lat": [43.65 + i * 0.01 for i in range(n)],
            "lon": [-79.38 + i * 0.01 for i in range(n)],
            "speed": [55.0 + i for i in range(n)],
            "heading": [90.0] * n,
            "time": [base + timedelta(minutes=i) for i in range(n)],
        }
    )


def _make_gap_split_df() -> pl.DataFrame:
    df = _make_raw_df(10)
    seg_ids = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
    return df.with_columns(pl.Series("segment_id", seg_ids, dtype=pl.UInt32))


def _make_stop_split_df() -> pl.DataFrame:
    df = _make_gap_split_df()
    is_stop = [False, False, False, True, True, True, False, False, False, False]
    seg_ids = [0, 0, 0, 1, 1, 1, 2, 2, 2, 2]
    return df.with_columns(
        pl.Series("segment_id", seg_ids, dtype=pl.UInt32),
        pl.Series("is_stop", is_stop),
    )


def _make_matched_df() -> pl.DataFrame:
    df = _make_raw_df(10)
    return df.with_columns(
        pl.Series("matched_lat", [43.651 + i * 0.01 for i in range(10)]),
        pl.Series("matched_lon", [-79.379 + i * 0.01 for i in range(10)]),
        pl.Series("distance_from_trace", [1.5 + i * 0.3 for i in range(10)]),
    )


class TestPlotTraceRaw:
    def test_from_tracepoints(self) -> None:
        points = _make_tracepoints()
        m = plot_trace(points)
        assert isinstance(m, folium.Map)

    def test_from_dataframe(self) -> None:
        df = _make_raw_df()
        m = plot_trace(df)
        assert isinstance(m, folium.Map)

    def test_color_by_speed(self) -> None:
        df = _make_raw_df()
        m = plot_trace(df, color_by="speed")
        html = m._repr_html_()
        assert "speed" in html


class TestPlotTraceGapSplit:
    def test_segments_rendered(self) -> None:
        df = _make_gap_split_df()
        m = plot_trace(df)
        assert isinstance(m, folium.Map)


class TestPlotTraceStopSplit:
    def test_stops_and_movement(self) -> None:
        df = _make_stop_split_df()
        m = plot_trace(df)
        assert isinstance(m, folium.Map)
        html = m._repr_html_()
        assert "Stop" in html


class TestPlotTraceMatched:
    def test_matched_trace(self) -> None:
        df = _make_matched_df()
        m = plot_trace(df)
        assert isinstance(m, folium.Map)

    def test_color_by_distance(self) -> None:
        df = _make_matched_df()
        m = plot_trace(df, color_by="distance_from_trace")
        html = m._repr_html_()
        assert "distance_from_trace" in html


class TestSaveMap:
    def test_save_html(self) -> None:
        df = _make_raw_df()
        m = plot_trace(df)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "test.html"
            save_map(m, path)
            content = path.read_text()
            assert "<html>" in content.lower() or "leaflet" in content.lower()
