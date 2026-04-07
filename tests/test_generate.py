"""Tests for trucktrack.generate — synthetic GPS trace generation."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import polars as pl
from trucktrack.generate import (
    TracePoint,
    TripConfig,
    generate_trace,
    traces_to_csv,
    traces_to_parquet,
)

# Use an unreachable Valhalla URL so the router falls back to straight-line
# routing — keeps the test self-contained, no external service required.
UNREACHABLE_VALHALLA = "http://127.0.0.1:1"


def _config(**overrides) -> TripConfig:  # type: ignore[no-untyped-def]
    base = dict(
        origin=(43.65, -79.38),  # Toronto
        destination=(43.45, -80.49),  # Kitchener
        departure_time=datetime(2026, 1, 1, 8, 0, tzinfo=UTC),
        seed=42,
        valhalla_url=UNREACHABLE_VALHALLA,
    )
    base.update(overrides)
    return TripConfig(**base)


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
            assert p.speed_mph >= 0.0


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


def test_empty_route_handled() -> None:
    # Same origin/destination — straight-line fallback still produces
    # a trace because parking maneuvers always emit points.
    cfg = _config(destination=(43.65, -79.38))
    points = generate_trace(cfg)
    assert points  # parking maneuver geometry is non-empty
