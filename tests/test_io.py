"""Tests for trucktrack.io — parquet reading and dataset handling."""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest
import trucktrack

DATA_DIR = Path(__file__).parent.parent / "data"
EXAMPLE_PARQUET = DATA_DIR / "example_tracks.parquet"

EXPECTED_COLUMNS = {"id", "time", "speed", "heading", "lat", "lon"}


class TestReadParquet:
    def test_returns_dataframe(self) -> None:
        df = trucktrack.read_parquet(EXAMPLE_PARQUET)
        assert isinstance(df, pl.DataFrame)

    def test_row_count(self) -> None:
        df = trucktrack.read_parquet(EXAMPLE_PARQUET)
        assert len(df) == 10

    def test_expected_columns(self) -> None:
        df = trucktrack.read_parquet(EXAMPLE_PARQUET)
        assert set(df.columns) == EXPECTED_COLUMNS

    def test_id_column(self) -> None:
        df = trucktrack.read_parquet(EXAMPLE_PARQUET)
        assert df["id"].to_list() == list(range(1, 11))

    def test_speed_column_is_float(self) -> None:
        df = trucktrack.read_parquet(EXAMPLE_PARQUET)
        assert df["speed"].dtype == pl.Float64

    def test_heading_within_range(self) -> None:
        df = trucktrack.read_parquet(EXAMPLE_PARQUET)
        assert df["heading"].min() >= 0.0
        assert df["heading"].max() < 360.0

    def test_lat_lon_within_range(self) -> None:
        df = trucktrack.read_parquet(EXAMPLE_PARQUET)
        assert df["lat"].min() >= -90.0
        assert df["lat"].max() <= 90.0
        assert df["lon"].min() >= -180.0
        assert df["lon"].max() <= 180.0

    def test_accepts_path_object(self) -> None:
        df = trucktrack.read_parquet(Path(EXAMPLE_PARQUET))
        assert len(df) == 10

    def test_accepts_string_path(self) -> None:
        df = trucktrack.read_parquet(str(EXAMPLE_PARQUET))
        assert len(df) == 10

    def test_missing_file_raises(self) -> None:
        with pytest.raises((FileNotFoundError, OSError)):
            trucktrack.read_parquet("nonexistent.parquet")


class TestReadDataset:
    def test_passthrough(self) -> None:
        df = pl.DataFrame(
            {
                "id": [1, 2],
                "time": ["t1", "t2"],
                "speed": [1.0, 2.0],
                "heading": [0.0, 90.0],
                "lat": [0.0, 1.0],
                "lon": [0.0, 1.0],
            }
        )
        result = trucktrack.read_dataset(df)
        assert result.shape == df.shape
        assert result.columns == df.columns


class TestProcessTracksFile:
    def test_writes_output_file(self, tmp_path: Path) -> None:
        out = tmp_path / "out.parquet"
        trucktrack.process_parquet_in_rust(EXAMPLE_PARQUET, out)
        assert out.exists()

    def test_returns_row_count(self, tmp_path: Path) -> None:
        out = tmp_path / "out.parquet"
        n = trucktrack.process_parquet_in_rust(EXAMPLE_PARQUET, out)
        assert n == 10

    def test_output_has_speed_mps(self, tmp_path: Path) -> None:
        out = tmp_path / "out.parquet"
        trucktrack.process_parquet_in_rust(EXAMPLE_PARQUET, out)
        result = pl.read_parquet(out)
        assert "speed_mps" in result.columns

    def test_speed_mps_values(self, tmp_path: Path) -> None:
        out = tmp_path / "out.parquet"
        trucktrack.process_parquet_in_rust(EXAMPLE_PARQUET, out)
        result = pl.read_parquet(out)
        original = pl.read_parquet(EXAMPLE_PARQUET)
        expected = original["speed"] * (1000.0 / 3600.0)
        assert result["speed_mps"].to_list() == pytest.approx(expected.to_list())

    def test_missing_input_raises(self, tmp_path: Path) -> None:
        with pytest.raises((OSError, RuntimeError)):
            trucktrack.process_parquet_in_rust(
                "nonexistent.parquet", tmp_path / "out.parquet"
            )


class TestProcessDataframeInRust:
    def test_returns_dataframe(self) -> None:
        df = pl.read_parquet(EXAMPLE_PARQUET)
        result = trucktrack.process_dataframe_in_rust(df)
        assert isinstance(result, pl.DataFrame)

    def test_adds_speed_mps(self) -> None:
        df = pl.read_parquet(EXAMPLE_PARQUET)
        result = trucktrack.process_dataframe_in_rust(df)
        assert "speed_mps" in result.columns

    def test_preserves_original_columns(self) -> None:
        df = pl.read_parquet(EXAMPLE_PARQUET)
        result = trucktrack.process_dataframe_in_rust(df)
        for col in df.columns:
            assert col in result.columns

    def test_row_count_unchanged(self) -> None:
        df = pl.read_parquet(EXAMPLE_PARQUET)
        result = trucktrack.process_dataframe_in_rust(df)
        assert len(result) == len(df)

    def test_speed_mps_values(self) -> None:
        df = pl.read_parquet(EXAMPLE_PARQUET)
        result = trucktrack.process_dataframe_in_rust(df)
        expected = (df["speed"] * (1000.0 / 3600.0)).to_list()
        assert result["speed_mps"].to_list() == pytest.approx(expected)
