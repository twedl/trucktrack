"""Tests for ``ChunkIndex`` — chunk_id resolution and save/load versioning."""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl
import pytest
from trucktrack.query import ChunkIndex


def _write_trips(path: Path, trip_ids: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame({"id": trip_ids}).write_parquet(path)


def test_chunkindex_keys_use_three_chars(tmp_path: Path) -> None:
    # Two trucks sharing the last 2 hex chars but differing at the 3rd:
    # under a 2-char index they'd land in one bucket; 3-char splits them.
    truck_a = "aaaaaaa0abc"  # last 3 = "abc", last 2 = "bc"
    truck_b = "bbbbbbb1xbc"  # last 3 = "xbc", last 2 = "bc"
    _write_trips(
        tmp_path / "tier=local/partition_id=0/f.parquet",
        [f"{truck_a}_gap0_trip0"],
    )
    _write_trips(
        tmp_path / "tier=local/partition_id=1/f.parquet",
        [f"{truck_b}_gap0_trip0"],
    )

    idx = ChunkIndex.build(tmp_path, show_progress=False)

    assert set(idx.chunk_ids) == {"abc", "xbc"}
    # scan_truck for truck_a opens only its file, not truck_b's.
    df_a = idx.scan_truck(truck_a).collect()
    assert df_a["id"].to_list() == [f"{truck_a}_gap0_trip0"]


def test_chunkindex_save_load_roundtrip(tmp_path: Path) -> None:
    truck = "0123456789abcdef"
    _write_trips(
        tmp_path / "tier=local/partition_id=0/f.parquet",
        [f"{truck}_gap0_trip0"],
    )

    idx = ChunkIndex.build(tmp_path, show_progress=False)
    idx.save()

    reloaded = ChunkIndex.load(tmp_path)
    assert reloaded.chunk_ids == idx.chunk_ids


def test_chunkindex_load_rejects_wrong_resolution(tmp_path: Path) -> None:
    (tmp_path / ".chunk_index.json").write_text(
        json.dumps({"version": 1, "chunk_id_len": 2, "index": {"ef": ["a.parquet"]}})
    )
    with pytest.raises(ValueError, match="chunk_id_len=2"):
        ChunkIndex.load(tmp_path)


def test_chunkindex_load_rejects_legacy_format(tmp_path: Path) -> None:
    # Pre-versioning format was the bare index dict.
    (tmp_path / ".chunk_index.json").write_text(json.dumps({"ef": ["a.parquet"]}))
    with pytest.raises(ValueError, match="legacy"):
        ChunkIndex.load(tmp_path)
