"""Tests for ``trucktrack.query`` — truck and trip lookups at any stage."""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest
from trucktrack.buckets import bucket_dir, bucket_of, write_n_buckets
from trucktrack.query import scan_trip, scan_truck, truck_id_from_trip

N = 8


def _write(path: Path, ids: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame({"id": ids}).write_parquet(path)


def _bucketed(root: Path, trip_ids: list[str]) -> Path:
    """Trips written to their trucks' buckets, as run_pipeline would."""
    by_bucket: dict[int, list[str]] = {}
    for trip_id in trip_ids:
        bucket = bucket_of(truck_id_from_trip(trip_id), N)
        by_bucket.setdefault(bucket, []).append(trip_id)
    for bucket, ids in by_bucket.items():
        _write(bucket_dir(root, bucket) / "data.parquet", ids)
    write_n_buckets(root, N)
    return root


class TestScanTruck:
    def test_bucketed_opens_only_the_trucks_bucket(self, tmp_path: Path) -> None:
        _bucketed(tmp_path, ["truck-a_gap0_trip0", "truck-b_gap0_trip0"])
        # A file in another bucket isn't read, even if it names the truck.
        other = (bucket_of("truck-a", N) + 1) % N
        _write(bucket_dir(tmp_path, other) / "x.parquet", ["truck-a_gap9_trip9"])

        out = scan_truck(tmp_path, "truck-a").collect()
        assert out["id"].to_list() == ["truck-a_gap0_trip0"]

    def test_unbucketed_scans_any_layout_but_side_outputs(self, tmp_path: Path) -> None:
        _write(tmp_path / "year=2026/chunk_id=01/part-0.parquet", ["truck-a", "b"])
        _write(tmp_path / "elsewhere.parquet", ["truck-a"])
        # A side output with other columns would break the scan.
        (tmp_path / "_quality").mkdir()
        pl.DataFrame({"ok": [True]}).write_parquet(tmp_path / "_quality" / "q.parquet")

        out = scan_truck(tmp_path, "truck-a").collect()
        assert out.columns == ["id"]  # folder names aren't columns
        assert out["id"].to_list() == ["truck-a", "truck-a"]

    def test_whole_truck_ids_not_prefixes(self, tmp_path: Path) -> None:
        _write(tmp_path / "f.parquet", ["truck-a_gap0_trip0", "truck-ab_gap0_trip0"])
        assert scan_truck(tmp_path, "truck-a").collect()["id"].to_list() == [
            "truck-a_gap0_trip0"
        ]

    def test_missing_bucket_raises(self, tmp_path: Path) -> None:
        write_n_buckets(tmp_path, N)
        with pytest.raises(FileNotFoundError, match="truck-a"):
            scan_truck(tmp_path, "truck-a")


class TestScanTrip:
    def test_exact_trip(self, tmp_path: Path) -> None:
        _bucketed(tmp_path, ["truck-a_gap0_trip0", "truck-a_gap0_trip1"])
        out = scan_trip(tmp_path, "truck-a_gap0_trip1").collect()
        assert out["id"].to_list() == ["truck-a_gap0_trip1"]

    def test_trips_of_several_trucks(self, tmp_path: Path) -> None:
        wanted = ["truck-a_gap0_trip1", "truck-b_gap2_trip0"]
        _bucketed(tmp_path, ["truck-a_gap0_trip0", *wanted])
        out = scan_trip(tmp_path, wanted).collect()
        assert sorted(out["id"].to_list()) == wanted
