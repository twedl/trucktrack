"""Tests for trucktrack trajectory splitters."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
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
        """A/C have 1 gap each, B has none (2-min threshold)."""
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

    def test_tz_aware_datetimes(self, truck_a: pl.DataFrame) -> None:
        """tz-aware datetime columns should not panic the Rust splitter."""
        tz_df = truck_a.with_columns(pl.col("time").dt.replace_time_zone("UTC"))
        result = trucktrack.split_by_observation_gap(tz_df, timedelta(minutes=2))
        naive_result = trucktrack.split_by_observation_gap(
            truck_a, timedelta(minutes=2)
        )
        assert result["segment_id"].n_unique() == naive_result["segment_id"].n_unique()

    @pytest.mark.parametrize("unit", ["ms", "us", "ns"])
    def test_split_independent_of_time_unit(self, unit: str) -> None:
        """Splitting must be invariant to the column's Datetime TimeUnit.

        Regression: previously the Rust side cast the time column to Int64
        directly, yielding integers in the column's native TimeUnit and
        silently miscomparing against the microsecond gap threshold for
        Datetime[ms] inputs.
        """
        base = datetime(2025, 6, 15, 8, 0)
        df = pl.DataFrame(
            {
                "id": ["A", "A"],
                "lat": [43.65, 43.66],
                "lon": [-79.38, -79.37],
                "time": [base, base + timedelta(hours=3, minutes=30)],
            }
        ).with_columns(pl.col("time").cast(pl.Datetime(unit)))
        result = trucktrack.split_by_observation_gap(df, timedelta(hours=1))
        assert result["segment_id"].to_list() == [0, 1], (
            f"3.5h gap not detected with Datetime[{unit}]"
        )

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
    def test_has_segment_id_and_is_stop(self, tracks: pl.DataFrame) -> None:
        result = trucktrack.split_by_stops(
            tracks, max_diameter=50.0, min_duration=timedelta(minutes=2)
        )
        assert "segment_id" in result.columns
        assert "is_stop" in result.columns

    def test_truck_b_stop_detected(self, truck_b: pl.DataFrame) -> None:
        """truck_B stop produces 3 segments (movement, stop, movement)."""
        result = trucktrack.split_by_stops(
            truck_b, max_diameter=50.0, min_duration=timedelta(minutes=2)
        )
        n_segs = result["segment_id"].n_unique()
        assert n_segs == 3, f"Expected 3 segments, got {n_segs}"

    def test_truck_b_all_rows_preserved(self, truck_b: pl.DataFrame) -> None:
        """All rows should be preserved, with stop rows marked via is_stop."""
        result = trucktrack.split_by_stops(
            truck_b, max_diameter=50.0, min_duration=timedelta(minutes=2)
        )
        assert len(result) == len(truck_b)
        assert result["is_stop"].sum() > 0

    def test_no_stops_single_segment(self, truck_a: pl.DataFrame) -> None:
        """truck_A has no stops, should return 1 segment with all rows."""
        result = trucktrack.split_by_stops(
            truck_a, max_diameter=50.0, min_duration=timedelta(minutes=2)
        )
        assert result["segment_id"].n_unique() == 1
        assert len(result) == len(truck_a)
        assert result["is_stop"].sum() == 0

    def test_large_diameter_no_stops(self, truck_b: pl.DataFrame) -> None:
        """Large diameter + long duration threshold = no stops."""
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
        assert result["segment_id"].n_unique() == 3
        assert len(result) == len(truck_c)

    def test_min_length_filters(self, truck_b: pl.DataFrame) -> None:
        """min_length=15 drops short movement segments but keeps stop segments."""
        result = trucktrack.split_by_stops(
            truck_b,
            max_diameter=50.0,
            min_duration=timedelta(minutes=2),
            min_length=15,
        )
        # Both movement segments in truck_B are ~10-11 rows, so they get filtered.
        # Stop segments survive regardless of min_length.
        assert len(result) > 0
        assert result["is_stop"].all()

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
        assert "is_stop" in result.columns
        assert n == 83  # all rows preserved


# ── TrafficFilter ───────────────────────────────────────────────────────


def _make_trajectory(
    lats: list[float],
    lons: list[float],
    is_stop: list[bool],
) -> pl.DataFrame:
    """Helper: build a minimal DataFrame with segment_id derived from is_stop."""
    seg_ids: list[int] = [0]
    for i in range(1, len(is_stop)):
        seg_ids.append(seg_ids[-1] + (1 if is_stop[i] != is_stop[i - 1] else 0))
    return pl.DataFrame(
        {
            "id": ["t"] * len(lats),
            "lat": lats,
            "lon": lons,
            "is_stop": is_stop,
            "segment_id": pl.Series("segment_id", seg_ids, dtype=pl.UInt32),
        }
    )


class TestFilterTrafficStops:
    def test_collinear_stop_reclassified(self) -> None:
        """A stop along a straight northbound corridor should be filtered."""
        lats = [43.0, 43.001, 43.002, 43.0025, 43.0025, 43.003, 43.004]
        lons = [-79.0] * 7
        is_stop = [False, False, False, True, True, False, False]
        df = _make_trajectory(lats, lons, is_stop)

        result = trucktrack.filter_traffic_stops(
            df, max_angle_change=30.0, min_distance=5.0
        )

        assert result["is_stop"].sum() == 0

    def test_real_stop_preserved(self) -> None:
        """A stop with a bearing change (north in, east out) should survive."""
        lats = [43.0, 43.001, 43.002, 43.0025, 43.0025, 43.0025, 43.0025]
        lons = [-79.0, -79.0, -79.0, -79.0, -79.0, -78.999, -78.998]
        is_stop = [False, False, False, True, True, False, False]
        df = _make_trajectory(lats, lons, is_stop)

        result = trucktrack.filter_traffic_stops(
            df, max_angle_change=30.0, min_distance=5.0
        )

        assert result["is_stop"].sum() == 2

    def test_no_stops_unchanged(self) -> None:
        """A trajectory with no stops should pass through unchanged."""
        lats = [43.0, 43.001, 43.002, 43.003]
        lons = [-79.0] * 4
        is_stop = [False] * 4
        df = _make_trajectory(lats, lons, is_stop)

        result = trucktrack.filter_traffic_stops(df)

        assert len(result) == 4
        assert result["is_stop"].sum() == 0

    def test_segment_ids_reassigned(self) -> None:
        """After reclassifying a traffic stop, segment_ids should be sequential."""
        lats = [43.0, 43.001, 43.002, 43.0025, 43.0025, 43.003, 43.004]
        lons = [-79.0] * 7
        is_stop = [False, False, False, True, True, False, False]
        df = _make_trajectory(lats, lons, is_stop)

        result = trucktrack.filter_traffic_stops(
            df, max_angle_change=30.0, min_distance=5.0
        )

        # All movement now, single segment
        assert result["segment_id"].n_unique() == 1


# ── StalePingFilter ──────────────────────────────────────────────────────


def _make_raw(
    rows: list[tuple[str, int, float, float, float, float]],
) -> pl.DataFrame:
    """Build a raw-schema DataFrame: (id, time, lat, lon, speed, heading).

    ``time`` is given as seconds-since-epoch for readability.
    """
    from datetime import datetime

    return pl.DataFrame(
        {
            "id": [r[0] for r in rows],
            "time": [
                datetime.fromtimestamp(r[1], tz=UTC).replace(tzinfo=None) for r in rows
            ],
            "lat": [r[2] for r in rows],
            "lon": [r[3] for r in rows],
            "speed": [r[4] for r in rows],
            "heading": [r[5] for r in rows],
        }
    )


class TestStalePingFilter:
    def test_drops_verbatim_reemission(self) -> None:
        """T1 → T2 → T3 where T3 has identical fields to T1."""
        df = _make_raw(
            [
                ("a", 0, 43.0, -79.0, 55.0, 90.0),
                ("a", 60, 43.001, -79.0, 56.0, 91.0),
                ("a", 63, 43.0, -79.0, 55.0, 90.0),  # stale copy of row 0
            ]
        )
        out = trucktrack.filter_stale_pings(df)
        assert out.height == 2
        assert sorted(out["lat"].to_list()) == [43.0, 43.001]

    def test_clean_data_passes_through(self) -> None:
        df = _make_raw(
            [
                ("a", 0, 43.0, -79.0, 55.0, 90.0),
                ("a", 60, 43.001, -79.0, 56.0, 91.0),
                ("a", 120, 43.002, -79.0, 57.0, 92.0),
            ]
        )
        out = trucktrack.filter_stale_pings(df)
        assert out.height == 3

    def test_trucks_are_independent(self) -> None:
        """Truck b's fresh ping must not match truck a's buffer."""
        df = _make_raw(
            [
                ("a", 0, 43.0, -79.0, 55.0, 90.0),
                ("b", 0, 43.0, -79.0, 55.0, 90.0),  # same coords, different truck
                ("a", 60, 43.0, -79.0, 55.0, 90.0),  # stale re-emission for a
            ]
        )
        out = trucktrack.filter_stale_pings(df)
        assert out.height == 2
        assert set(out["id"].to_list()) == {"a", "b"}

    def test_outside_window_kept(self) -> None:
        """A repeat outside the lookback window is not dropped."""
        df = _make_raw(
            [
                ("a", 0, 43.0, -79.0, 55.0, 90.0),
                ("a", 1, 43.1, -79.0, 50.0, 90.0),
                ("a", 2, 43.2, -79.0, 50.0, 90.0),
                ("a", 3, 43.3, -79.0, 50.0, 90.0),
                ("a", 4, 43.4, -79.0, 50.0, 90.0),
                # matches row 0 but window=3 has evicted it
                ("a", 5, 43.0, -79.0, 55.0, 90.0),
            ]
        )
        out = trucktrack.filter_stale_pings(df, window=3)
        assert out.height == 6

    def test_partial_match_kept(self) -> None:
        """Same lat/lon but different speed must not be dropped."""
        df = _make_raw(
            [
                ("a", 0, 43.0, -79.0, 55.0, 90.0),
                ("a", 60, 43.001, -79.0, 0.0, 0.0),
                ("a", 120, 43.0, -79.0, 56.0, 90.0),  # lat/lon match, speed differs
            ]
        )
        out = trucktrack.filter_stale_pings(df)
        assert out.height == 3

    def test_stopped_truck_repeats_kept(self) -> None:
        """A stopped truck (speed=0) emitting repeated identical pings is legitimate."""
        df = _make_raw(
            [
                ("a", 0, 43.0, -79.0, 0.0, 0.0),
                ("a", 60, 43.0, -79.0, 0.0, 0.0),
                ("a", 120, 43.0, -79.0, 0.0, 0.0),
                ("a", 180, 43.0, -79.0, 0.0, 0.0),
            ]
        )
        out = trucktrack.filter_stale_pings(df)
        assert out.height == 4

    def test_zero_speed_does_not_poison_buffer(self) -> None:
        """A later moving ping that coincidentally matches a zero-speed row must pass.

        Without the speed=0 exemption, a stationary (43.0, -79.0, 0, 0) would sit in
        the buffer and spuriously flag a later moving ping at the same coords.
        """
        df = _make_raw(
            [
                ("a", 0, 43.0, -79.0, 0.0, 0.0),  # stopped
                ("a", 60, 43.1, -79.0, 55.0, 90.0),  # moving away
                ("a", 120, 43.0, -79.0, 55.0, 90.0),  # returns, now moving — keep
            ]
        )
        out = trucktrack.filter_stale_pings(df)
        assert out.height == 3


# ── ImpossibleSpeedFilter ────────────────────────────────────────────────


class TestFilterImpossibleSpeeds:
    def test_keeps_plausible_trajectory(self) -> None:
        """~110 m / 60 s ≈ 1.85 m/s — well under threshold."""
        df = _make_raw(
            [
                ("a", 0, 43.0, -79.0, 10.0, 0.0),
                ("a", 60, 43.001, -79.0, 10.0, 0.0),
                ("a", 120, 43.002, -79.0, 10.0, 0.0),
            ]
        )
        out = trucktrack.filter_impossible_speeds(df)
        assert out.height == 3

    def test_drops_single_spike(self) -> None:
        """A 1° jump (~111 km) in 1 s is clearly impossible; row is dropped."""
        df = _make_raw(
            [
                ("a", 0, 43.0, -79.0, 10.0, 0.0),
                ("a", 1, 44.0, -79.0, 10.0, 0.0),  # spike
                ("a", 2, 43.00001, -79.0, 10.0, 0.0),
            ]
        )
        out = trucktrack.filter_impossible_speeds(df)
        assert out.height == 2
        assert sorted(out["lat"].to_list()) == pytest.approx([43.0, 43.00001])

    def test_first_point_always_kept(self) -> None:
        """Single point per truck has no prior anchor and survives."""
        df = _make_raw([("a", 0, 43.0, -79.0, 10.0, 0.0)])
        out = trucktrack.filter_impossible_speeds(df)
        assert out.height == 1

    def test_consecutive_spikes_collapse_against_anchor(self) -> None:
        """Two adjacent spikes both drop; the anchor is row 0, not the last spike."""
        df = _make_raw(
            [
                ("a", 0, 43.0, -79.0, 10.0, 0.0),
                ("a", 1, 44.0, -79.0, 10.0, 0.0),  # spike 1
                ("a", 2, 45.0, -79.0, 10.0, 0.0),  # spike 2
                ("a", 3, 43.00002, -79.0, 10.0, 0.0),  # within tolerance of row 0
            ]
        )
        out = trucktrack.filter_impossible_speeds(df)
        assert out.height == 2
        assert sorted(out["lat"].to_list()) == pytest.approx([43.0, 43.00002])

    def test_trucks_are_independent(self) -> None:
        """One truck's spike must not cause another truck's point to drop."""
        df = _make_raw(
            [
                ("a", 0, 43.0, -79.0, 10.0, 0.0),
                ("a", 1, 44.0, -79.0, 10.0, 0.0),  # a's spike
                ("b", 0, 50.0, -100.0, 10.0, 0.0),
                ("b", 60, 50.001, -100.0, 10.0, 0.0),
            ]
        )
        out = trucktrack.filter_impossible_speeds(df)
        # a: 1 kept (row 0), b: 2 kept.
        assert out.height == 3
        counts = {tid: 0 for tid in ("a", "b")}
        for tid in out["id"].to_list():
            counts[tid] += 1
        assert counts == {"a": 1, "b": 2}

    def test_null_coords_pass_through(self) -> None:
        """Null lat keeps the row and does not advance the anchor."""
        df = pl.DataFrame(
            {
                "id": ["a", "a", "a"],
                "time": [
                    datetime(2026, 1, 1, 0, 0, 0),
                    datetime(2026, 1, 1, 0, 1, 0),
                    datetime(2026, 1, 1, 0, 2, 0),
                ],
                "lat": [43.0, None, 43.002],
                "lon": [-79.0, -79.0, -79.0],
            }
        )
        out = trucktrack.filter_impossible_speeds(df)
        assert out.height == 3

    def test_duplicate_timestamp_kept(self) -> None:
        """dt == 0 between consecutive valid points — keep without dividing by zero."""
        df = _make_raw(
            [
                ("a", 0, 43.0, -79.0, 10.0, 0.0),
                ("a", 0, 43.001, -79.0, 10.0, 0.0),
            ]
        )
        out = trucktrack.filter_impossible_speeds(df)
        assert out.height == 2

    def test_threshold_is_configurable(self) -> None:
        """A 100 m / 60 s fix (~6 km/h) drops when the threshold is set absurdly low."""
        df = _make_raw(
            [
                ("a", 0, 43.0, -79.0, 10.0, 0.0),
                ("a", 60, 43.001, -79.0, 10.0, 0.0),
            ]
        )
        out = trucktrack.filter_impossible_speeds(df, max_speed_kmh=1.0)
        # ~6.7 km/h implied, threshold 1 km/h → second row dropped.
        assert out.height == 1

    def test_file_io_path(self, tmp_path: Path) -> None:
        """`_file` variant round-trips parquet and drops the spike row."""
        df = _make_raw(
            [
                ("a", 0, 43.0, -79.0, 10.0, 0.0),
                ("a", 1, 44.0, -79.0, 10.0, 0.0),  # spike
                ("a", 2, 43.00001, -79.0, 10.0, 0.0),
            ]
        )
        inp = tmp_path / "speed_in.parquet"
        out = tmp_path / "speed_out.parquet"
        df.write_parquet(inp)
        n = trucktrack.filter_impossible_speeds_file(inp, out)
        assert out.exists()
        assert n == 2
        result = pl.read_parquet(out)
        assert result.height == 2
