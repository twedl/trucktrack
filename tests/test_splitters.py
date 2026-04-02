"""Tests for trucktrack trajectory splitters."""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import polars as pl
import pytest

import trucktrack

DATA_DIR = Path(__file__).parent.parent / "data"
SPLITTER_PARQUET = DATA_DIR / "splitter_test_tracks.parquet"


@pytest.fixture()
def tracks() -> pl.DataFrame:
    return pl.read_parquet(SPLITTER_PARQUET)


@pytest.fixture()
def truck_a(tracks: pl.DataFrame) -> pl.DataFrame:
    return tracks.filter(pl.col("id") == "truck_A")


@pytest.fixture()
def truck_b(tracks: pl.DataFrame) -> pl.DataFrame:
    return tracks.filter(pl.col("id") == "truck_B")


@pytest.fixture()
def truck_c(tracks: pl.DataFrame) -> pl.DataFrame:
    return tracks.filter(pl.col("id") == "truck_C")


# ── ObservationGapSplitter ───────────────────────────────────────────────


class TestObservationGapSplitter:
    def test_has_segment_id(self, tracks: pl.DataFrame) -> None:
        result = trucktrack.split_by_observation_gap(tracks, timedelta(minutes=2))
        assert "segment_id" in result.columns

    def test_truck_a_splits_at_10min_gap(self, truck_a: pl.DataFrame) -> None:
        """truck_A has a 10-min gap; a 2-min threshold should produce 2 segments."""
        result = trucktrack.split_by_observation_gap(truck_a, timedelta(minutes=2))
        n_segs = result["segment_id"].n_unique()
        assert n_segs == 2, f"Expected 2 segments, got {n_segs}"

    def test_truck_a_no_split_with_large_gap(self, truck_a: pl.DataFrame) -> None:
        """With a 1-hour threshold, truck_A should stay as one segment."""
        result = trucktrack.split_by_observation_gap(truck_a, timedelta(hours=1))
        assert result["segment_id"].n_unique() == 1

    def test_truck_a_row_count_preserved(self, truck_a: pl.DataFrame) -> None:
        result = trucktrack.split_by_observation_gap(truck_a, timedelta(minutes=2))
        assert len(result) == len(truck_a)

    def test_multiple_vehicles_independent(self, tracks: pl.DataFrame) -> None:
        """truck_A has 1 gap, truck_C has 1 gap, truck_B has none (with 2-min threshold)."""
        result = trucktrack.split_by_observation_gap(tracks, timedelta(minutes=2))
        a_segs = result.filter(pl.col("id") == "truck_A")["segment_id"].n_unique()
        b_segs = result.filter(pl.col("id") == "truck_B")["segment_id"].n_unique()
        c_segs = result.filter(pl.col("id") == "truck_C")["segment_id"].n_unique()
        assert a_segs == 2
        assert b_segs == 1
        assert c_segs == 2

    def test_tiny_gap_splits_every_row(self, truck_a: pl.DataFrame) -> None:
        """A 1-second gap threshold should split at every 30s interval."""
        result = trucktrack.split_by_observation_gap(truck_a, timedelta(seconds=1))
        # Every row except the first becomes its own segment boundary
        assert result["segment_id"].n_unique() == len(truck_a)

    def test_min_length_filters_short_segments(self, truck_a: pl.DataFrame) -> None:
        """With min_length=20, segments < 20 rows should be dropped."""
        result = trucktrack.split_by_observation_gap(
            truck_a, timedelta(minutes=2), min_length=20
        )
        # truck_A has 15 rows per segment, so both get dropped
        assert len(result) == 0

    def test_file_io_path(self, tmp_path: Path) -> None:
        out = tmp_path / "gap_out.parquet"
        n = trucktrack.split_by_observation_gap_file(
            SPLITTER_PARQUET, out, timedelta(minutes=2)
        )
        assert out.exists()
        assert n == 83  # all rows preserved
        result = pl.read_parquet(out)
        assert "segment_id" in result.columns


# ── StopSplitter ─────────────────────────────────────────────────────────


class TestStopSplitter:
    def test_has_segment_id(self, tracks: pl.DataFrame) -> None:
        result = trucktrack.split_by_stops(
            tracks, max_diameter=50.0, min_duration=timedelta(minutes=2)
        )
        assert "segment_id" in result.columns

    def test_truck_b_stop_detected(self, truck_b: pl.DataFrame) -> None:
        """truck_B has a stop (9 rows over ~5 min within ~25m). Should produce 2 movement segments."""
        result = trucktrack.split_by_stops(
            truck_b, max_diameter=50.0, min_duration=timedelta(minutes=2)
        )
        n_segs = result["segment_id"].n_unique()
        assert n_segs == 2, f"Expected 2 movement segments, got {n_segs}"

    def test_truck_b_stop_rows_excluded(self, truck_b: pl.DataFrame) -> None:
        """Stop rows should be excluded from output."""
        result = trucktrack.split_by_stops(
            truck_b, max_diameter=50.0, min_duration=timedelta(minutes=2)
        )
        assert len(result) < len(truck_b)

    def test_no_stops_single_segment(self, truck_a: pl.DataFrame) -> None:
        """truck_A has no stops, should return 1 segment with all rows."""
        result = trucktrack.split_by_stops(
            truck_a, max_diameter=50.0, min_duration=timedelta(minutes=2)
        )
        assert result["segment_id"].n_unique() == 1
        assert len(result) == len(truck_a)

    def test_large_diameter_no_stops(self, truck_b: pl.DataFrame) -> None:
        """With a very large diameter, nothing is a stop (duration won't be met in small area)."""
        result = trucktrack.split_by_stops(
            truck_b, max_diameter=100_000.0, min_duration=timedelta(hours=24)
        )
        # Duration threshold is 24h, no window that long, so no stops detected
        assert len(result) == len(truck_b)

    def test_truck_c_has_stop(self, truck_c: pl.DataFrame) -> None:
        """truck_C has a stop (6 rows, ~3 min within ~15m)."""
        result = trucktrack.split_by_stops(
            truck_c, max_diameter=50.0, min_duration=timedelta(minutes=2)
        )
        assert result["segment_id"].n_unique() == 2
        assert len(result) < len(truck_c)

    def test_min_length_filters(self, truck_b: pl.DataFrame) -> None:
        """With min_length=15, short movement segments after splitting should be dropped."""
        result = trucktrack.split_by_stops(
            truck_b,
            max_diameter=50.0,
            min_duration=timedelta(minutes=2),
            min_length=15,
        )
        # Both movement segments in truck_B are ~10-11 rows, so all get filtered
        assert len(result) == 0

    def test_file_io_path(self, tmp_path: Path) -> None:
        out = tmp_path / "stop_out.parquet"
        n = trucktrack.split_by_stops_file(
            SPLITTER_PARQUET,
            out,
            max_diameter=50.0,
            min_duration=timedelta(minutes=2),
        )
        assert out.exists()
        result = pl.read_parquet(out)
        assert "segment_id" in result.columns
        assert n < 83  # some rows removed as stops
