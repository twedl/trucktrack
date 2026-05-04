"""Tests for the visualize submodule."""

from __future__ import annotations

import tempfile
from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl
import pytest

folium = pytest.importorskip("folium")

from trucktrack.generate.models import TracePoint  # noqa: E402
from trucktrack.visualize import plot_trace, plot_trace_layers, save_map  # noqa: E402


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

    def test_stop_renders_interior_polyline_and_connectors(self) -> None:
        # The fixture has segments 0 (move) → 1 (stop, 3 pts) → 2 (move),
        # so the stop should produce an interior polyline plus dashed
        # connectors on both sides.
        html = plot_trace(_make_stop_split_df())._repr_html_()
        # dashArray "4 4" comes from the connector polylines (HTML-escaped).
        assert "dashArray&quot;: &quot;4 4" in html
        assert "circle_marker" in html.lower()

    def test_stops_default_to_segment_palette(self) -> None:
        # Stop sits at segment_id=1 → second palette entry (#ff2050, bright
        # red).  Default red should not appear.
        html = plot_trace(_make_stop_split_df())._repr_html_()
        assert "ff2050" in html.lower()
        assert "&quot;red&quot;" not in html.lower()

    def test_stop_color_override_applies_to_all_stops(self) -> None:
        html = plot_trace(_make_stop_split_df(), stop_color="red")._repr_html_()
        assert "&quot;red&quot;" in html.lower()


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

    def test_matched_falls_back_to_matched_color_without_segment_id(self) -> None:
        # Default fallback color "#1e90ff" should appear on the matched
        # polyline since the fixture has no segment_id column.
        html = plot_trace(_make_matched_df())._repr_html_()
        assert "1e90ff" in html.lower()

    def test_matched_uses_palette_per_segment_id(self) -> None:
        df = _make_matched_df().with_columns(
            pl.Series("segment_id", [0] * 5 + [2] * 5, dtype=pl.UInt32),
        )
        html = plot_trace(df)._repr_html_()
        # Segment 0 → palette[0] (#00ffff), segment 2 → palette[2] (#c020ff).
        # Default matched_color (#1e90ff) should be absent.
        assert "00ffff" in html.lower()
        assert "c020ff" in html.lower()
        assert "1e90ff" not in html.lower()

    def test_matched_skips_stop_segments(self) -> None:
        df = _make_matched_df().with_columns(
            pl.Series("segment_id", [0] * 3 + [1] * 4 + [2] * 3, dtype=pl.UInt32),
            pl.Series("is_stop", [False] * 3 + [True] * 4 + [False] * 3),
        )
        html = plot_trace(df)._repr_html_()
        # Movement segments 0 and 2 contribute palette polylines; the stop
        # segment 1 (palette[1] = #ff2050) must not appear in the matched
        # layer — but it *will* show in the stop markers, so this just
        # confirms _add_matched_segments doesn't draw it as a polyline.
        # We can't easily disambiguate the two contexts from raw HTML, so
        # this test is mostly a smoke check that the code path runs.
        assert isinstance(plot_trace(df), folium.Map)
        assert "00ffff" in html.lower()  # segment 0
        assert "c020ff" in html.lower()  # segment 2


class TestPlotTraceLayers:
    def test_all_layers(self) -> None:
        raw = _make_raw_df()
        segments = _make_gap_split_df()
        matched = _make_matched_df()
        m = plot_trace_layers(raw=raw, segments=segments, matched=matched)
        assert isinstance(m, folium.Map)
        html = m._repr_html_()
        assert "Raw trace" in html
        assert "Segments" in html
        assert "Map-matched" in html

    def test_raw_only(self) -> None:
        m = plot_trace_layers(raw=_make_raw_df())
        assert isinstance(m, folium.Map)

    def test_segments_with_stops(self) -> None:
        m = plot_trace_layers(segments=_make_stop_split_df())
        assert isinstance(m, folium.Map)
        html = m._repr_html_()
        assert "Stop" in html

    def test_from_tracepoints(self) -> None:
        m = plot_trace_layers(raw=_make_tracepoints())
        assert isinstance(m, folium.Map)

    def test_empty_returns_map(self) -> None:
        m = plot_trace_layers()
        assert isinstance(m, folium.Map)

    def test_matched_shape_groups_share_color_across_subshapes(self) -> None:
        # Two trips, the first has two sub-shapes (e.g. main + bridge).
        # All shapes in trip 0 get palette[0] (#00ffff); trip 1 gets
        # palette[1] (#ff2050).  Default matched_color (#1e90ff) should
        # not appear.  A matched DataFrame is supplied for bounds.
        trip0 = [
            [(43.65, -79.38), (43.66, -79.37)],
            [(43.66, -79.37), (43.67, -79.36)],
        ]
        trip1 = [[(43.70, -79.40), (43.71, -79.39)]]
        html = plot_trace_layers(
            matched=_make_matched_df(),
            matched_shape=[trip0, trip1],
        )._repr_html_()
        assert "00ffff" in html.lower()
        assert "ff2050" in html.lower()
        assert "1e90ff" not in html.lower()


class TestSatelliteBasemap:
    def test_plot_trace_registered_by_default(self) -> None:
        html = plot_trace(_make_raw_df())._repr_html_()
        assert "Satellite (Esri)" in html
        assert "World_Imagery" in html

    def test_plot_trace_layers_registered_by_default(self) -> None:
        html = plot_trace_layers(raw=_make_raw_df())._repr_html_()
        assert "Satellite (Esri)" in html
        assert "World_Imagery" in html

    def test_can_be_disabled(self) -> None:
        html = plot_trace(_make_raw_df(), satellite=False)._repr_html_()
        assert "World_Imagery" not in html
        html = plot_trace_layers(raw=_make_raw_df(), satellite=False)._repr_html_()
        assert "World_Imagery" not in html


class TestSaveMap:
    def test_save_html(self) -> None:
        df = _make_raw_df()
        m = plot_trace(df)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "test.html"
            save_map(m, path)
            content = path.read_text()
            assert "<html>" in content.lower() or "leaflet" in content.lower()
