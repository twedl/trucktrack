"""Tests for ``trucktrack.buckets`` and the bucket-wise split pipeline."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl
import pytest
from trucktrack.buckets import (
    bucket_by_truck,
    bucket_dirs,
    bucket_expr,
    bucket_files,
    bucket_of,
    data_files,
    read_n_buckets,
    write_n_buckets,
)
from trucktrack.pipeline import run_pipeline


def _points(truck_id: str, start: datetime, n: int, lat0: float) -> pl.DataFrame:
    """*n* fixes a minute apart, driving north at ~33 km/h."""
    return pl.DataFrame(
        {
            "id": [truck_id] * n,
            "time": [start + timedelta(minutes=i) for i in range(n)],
            "lat": [lat0 + 0.005 * i for i in range(n)],
            "lon": [-79.0] * n,
            "speed": [33.0] * n,
            "heading": [0.0] * n,
        }
    )


def _raw_by_day(root: Path, trucks: list[str]) -> Path:
    """Raw input split by day, not by truck: each truck is in both files."""
    raw = root / "raw"
    for day in (1, 2):
        out = raw / f"day={day}"
        out.mkdir(parents=True)
        start = datetime(2026, 1, day, 8, tzinfo=UTC)
        pl.concat(
            _points(t, start, 20, lat0=43.0 + day + i) for i, t in enumerate(trucks)
        ).write_parquet(out / "part.parquet")
    return raw


class TestBucketOf:
    def test_known_buckets_never_change(self) -> None:
        # Data on disk is laid out by these; update only with a migration.
        assert bucket_of("", 256) == 38
        assert bucket_of("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaa01", 256) == 44
        assert bucket_of("truck-42", 1000) == 166

    def test_expr_matches_scalar_and_keeps_nulls(self) -> None:
        df = pl.DataFrame({"id": ["a", None, "truck-42"]})
        assert df.select(bucket_expr(16))["bucket"].to_list() == [
            bucket_of("a", 16),
            None,
            bucket_of("truck-42", 16),
        ]

    def test_zero_buckets_rejected(self) -> None:
        with pytest.raises(ValueError, match="n_buckets"):
            bucket_of("a", 0)


class TestBucketMeta:
    def test_roundtrip(self, tmp_path: Path) -> None:
        assert read_n_buckets(tmp_path) is None
        write_n_buckets(tmp_path, 12)
        assert read_n_buckets(tmp_path) == 12

    def test_rejects_other_hash_version(self, tmp_path: Path) -> None:
        (tmp_path / "_buckets.json").write_text(
            json.dumps({"version": 0, "n_buckets": 12})
        )
        with pytest.raises(ValueError, match="hash version"):
            read_n_buckets(tmp_path)


def test_data_files_skip_side_outputs_and_scratch(tmp_path: Path) -> None:
    for rel in ["a.parquet", "x=1/b.parquet", "_quality/c.parquet", ".tmp/d.parquet"]:
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).touch()
    names = [p.relative_to(tmp_path).as_posix() for p in data_files(tmp_path)]
    assert names == ["a.parquet", "x=1/b.parquet"]


class TestBucketByTruck:
    def test_each_truck_lands_in_its_one_bucket(self, tmp_path: Path) -> None:
        trucks = ["truck-a", "truck-b", "truck-c"]
        raw = _raw_by_day(tmp_path, trucks)
        out = tmp_path / "bucketed"

        n_written = bucket_by_truck(raw, out, n_buckets=8)

        df = pl.read_parquet(
            out / "**" / "*.parquet", hive_partitioning=False, include_file_paths="f"
        )
        # Folder names (day=, bucket=) never become columns.
        assert df.columns == ["id", "time", "lat", "lon", "speed", "heading", "f"]
        assert len(df) == 2 * 20 * len(trucks)
        buckets = df.group_by("id").agg(
            pl.col("f").str.extract(r"bucket=(\d+)").cast(pl.Int64).unique()
        )
        for truck, found in buckets.iter_rows():
            assert found == [bucket_of(truck, 8)]
        assert n_written == len({bucket_of(t, 8) for t in trucks})
        assert read_n_buckets(out) == 8

    def test_drops_null_ids(self, tmp_path: Path) -> None:
        raw = tmp_path / "raw"
        raw.mkdir()
        pl.DataFrame({"id": ["truck-a", None], "x": [1, 2]}).write_parquet(
            raw / "part.parquet"
        )
        bucket_by_truck(raw, tmp_path / "out", n_buckets=4)
        df = pl.read_parquet(tmp_path / "out" / "**" / "*.parquet")
        assert df["id"].to_list() == ["truck-a"]

    def test_refuses_non_empty_output(self, tmp_path: Path) -> None:
        raw = _raw_by_day(tmp_path, ["truck-a"])
        out = tmp_path / "out"
        out.mkdir()
        (out / "stale.parquet").touch()
        with pytest.raises(FileExistsError, match="not empty"):
            bucket_by_truck(raw, out)

    def test_no_input_files(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="no parquet"):
            bucket_by_truck(tmp_path, tmp_path / "out")


class TestRunPipeline:
    def test_truck_split_across_raw_files_gets_distinct_trips(
        self, tmp_path: Path
    ) -> None:
        # One truck, a day apart in two raw files: two trips, numbered
        # from its whole history, not trip 0 of each file.
        raw = _raw_by_day(tmp_path, ["truck-a"])
        bucket_by_truck(raw, tmp_path / "bucketed", n_buckets=4)

        summary = run_pipeline(tmp_path / "bucketed", tmp_path / "split")

        df = pl.read_parquet(tmp_path / "split" / "**" / "*.parquet")
        trips = df.group_by("id").agg(pl.col("time").min()).sort("time")
        assert trips["id"].to_list() == ["truck-a_gap0_trip0", "truck-a_gap1_trip0"]
        assert trips["time"].dt.day().to_list() == [1, 2]
        assert summary == {"buckets": 1, "rows_in": 40, "rows_out": 40, "trips": 2}

    def test_output_keeps_the_buckets(self, tmp_path: Path) -> None:
        raw = _raw_by_day(tmp_path, ["truck-a", "truck-b", "truck-c"])
        bucket_by_truck(raw, tmp_path / "bucketed", n_buckets=8)
        run_pipeline(tmp_path / "bucketed", tmp_path / "split")

        def names(root: Path) -> list[str]:
            return [p.name for p in bucket_dirs(root)]

        assert names(tmp_path / "split") == names(tmp_path / "bucketed")
        assert read_n_buckets(tmp_path / "split") == 8

    def test_files_hold_whole_trucks(self, tmp_path: Path) -> None:
        trucks = [f"truck-{i}" for i in range(6)]
        raw = _raw_by_day(tmp_path, trucks)
        bucket_by_truck(raw, tmp_path / "bucketed", n_buckets=1)

        run_pipeline(tmp_path / "bucketed", tmp_path / "split", rows_per_file=100)

        files = bucket_files(bucket_dirs(tmp_path / "split")[0])
        # Trucks of 40 rows start at rows 0, 40, ..., 200; each goes to
        # the 100-row slot its first row falls in.
        assert [pl.read_parquet(f).height for f in files] == [120, 80, 40]
        owners = pl.concat(
            pl.read_parquet(f).select(
                pl.col("id").str.split("_gap").list.first().unique(), file=pl.lit(i)
            )
            for i, f in enumerate(files)
        )
        assert owners["id"].is_unique().all()

    def test_requires_bucketed_input(self, tmp_path: Path) -> None:
        raw = _raw_by_day(tmp_path, ["truck-a"])
        with pytest.raises(ValueError, match="bucket_by_truck"):
            run_pipeline(raw, tmp_path / "split")

    def test_refuses_non_empty_output(self, tmp_path: Path) -> None:
        raw = _raw_by_day(tmp_path, ["truck-a"])
        bucket_by_truck(raw, tmp_path / "bucketed", n_buckets=4)
        (tmp_path / "split").mkdir()
        (tmp_path / "split" / "stale.parquet").touch()
        with pytest.raises(FileExistsError, match="not empty"):
            run_pipeline(tmp_path / "bucketed", tmp_path / "split")
