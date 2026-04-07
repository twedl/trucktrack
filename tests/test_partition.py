"""Tests for trucktrack.partition — hive-partitioned dataset writer."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl
import pytest
from trucktrack.generate import TracePoint
from trucktrack.partition import (
    LOCAL_KM,
    REGIONAL_KM,
    assign_partitions,
    classify_and_partition_key,
    partition_existing_parquet,
    write_trips_partitioned,
)


def _trip(
    trip_id: str,
    start_lat: float,
    start_lon: float,
    end_lat: float,
    end_lon: float,
    n: int = 10,
) -> tuple[list[TracePoint], str]:
    base = datetime(2026, 1, 1, tzinfo=UTC)
    points: list[TracePoint] = []
    for i in range(n):
        f = i / max(n - 1, 1)
        points.append(
            TracePoint(
                lat=start_lat + f * (end_lat - start_lat),
                lon=start_lon + f * (end_lon - start_lon),
                speed_mph=55.0,
                heading=90.0,
                timestamp=base + timedelta(minutes=i),
            )
        )
    return points, trip_id


class TestClassifyAndPartitionKey:
    def test_local_tier(self) -> None:
        tier, pid = classify_and_partition_key(43.65, -79.38, bbox_diag_km=10.0)
        assert tier == "local"
        assert pid >> 60 == 0

    def test_regional_tier(self) -> None:
        tier, pid = classify_and_partition_key(43.65, -79.38, bbox_diag_km=300.0)
        assert tier == "regional"
        assert pid >> 60 == 1

    def test_longhaul_tier(self) -> None:
        tier, pid = classify_and_partition_key(40.0, -100.0, bbox_diag_km=2000.0)
        assert tier == "longhaul"
        assert pid >> 60 == 2

    def test_boundary_local_to_regional(self) -> None:
        tier, _ = classify_and_partition_key(43.0, -79.0, bbox_diag_km=LOCAL_KM)
        assert tier == "regional"

    def test_boundary_regional_to_longhaul(self) -> None:
        tier, _ = classify_and_partition_key(43.0, -79.0, bbox_diag_km=REGIONAL_KM)
        assert tier == "longhaul"


class TestAssignPartitions:
    def test_adds_expected_columns(self) -> None:
        df = pl.DataFrame(
            {
                "id": ["a", "b"],
                "centroid_lat": [43.65, 40.0],
                "centroid_lon": [-79.38, -100.0],
                "bbox_diag_km": [10.0, 2000.0],
            }
        )
        out = assign_partitions(df)
        assert {"tier", "partition_id", "hilbert_idx"}.issubset(out.columns)
        tiers = out["tier"].to_list()
        assert tiers == ["local", "longhaul"]

    def test_missing_columns_raises(self) -> None:
        with pytest.raises(ValueError):
            assign_partitions(pl.DataFrame({"id": ["a"]}))


class TestWriteTripsPartitioned:
    def test_creates_hive_layout(self, tmp_path: Path) -> None:
        trips = [
            _trip("local-1", 43.65, -79.38, 43.66, -79.37),  # local
            _trip("local-2", 43.65, -79.38, 43.67, -79.36),  # local, same tile
            _trip("longhaul-1", 40.0, -100.0, 50.0, -90.0),  # longhaul
        ]
        summary = write_trips_partitioned(trips, tmp_path)
        assert summary.get("local", 0) >= 1
        assert summary.get("longhaul", 0) >= 1

        # Hive layout: tier=*/partition_id=*/*.parquet
        tier_dirs = sorted(p.name for p in tmp_path.iterdir() if p.is_dir())
        assert any(d.startswith("tier=") for d in tier_dirs)

        files = list(tmp_path.rglob("*.parquet"))
        assert files
        for f in files:
            assert "tier=" in str(f)
            assert "partition_id=" in str(f)

    def test_inner_columns_are_point_schema(self, tmp_path: Path) -> None:
        trips = [_trip("t1", 43.65, -79.38, 43.66, -79.37)]
        write_trips_partitioned(trips, tmp_path)
        files = list(tmp_path.rglob("*.parquet"))
        assert files
        df = pl.read_parquet(files[0])
        # tier and partition_id come from directory names, not the inner file
        assert {"id", "lat", "lon", "speed", "heading", "time"}.issubset(df.columns)


class TestPartitionExistingParquet:
    def test_round_trip_from_flat_parquet(self, tmp_path: Path) -> None:
        flat = tmp_path / "flat.parquet"
        df = pl.DataFrame(
            {
                "id": ["a"] * 5 + ["b"] * 5,
                "lat": [43.65 + i * 0.001 for i in range(5)]
                + [40.0 + i * 0.5 for i in range(5)],
                "lon": [-79.38 + i * 0.001 for i in range(5)]
                + [-100.0 + i * 0.5 for i in range(5)],
                "speed": [55.0] * 10,
                "heading": [90.0] * 10,
                "time": [datetime(2026, 1, 1, tzinfo=UTC)] * 10,
            }
        )
        df.write_parquet(flat)

        out_dir = tmp_path / "parts"
        summary = partition_existing_parquet(flat, out_dir)
        assert summary
        assert sum(summary.values()) >= 1
        assert list(out_dir.rglob("*.parquet"))

    def test_missing_columns_raises(self, tmp_path: Path) -> None:
        flat = tmp_path / "bad.parquet"
        pl.DataFrame({"id": ["a"], "x": [1.0]}).write_parquet(flat)
        with pytest.raises(ValueError):
            partition_existing_parquet(flat, tmp_path / "parts")
