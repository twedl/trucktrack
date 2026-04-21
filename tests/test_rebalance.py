"""Tests for ``rebalance_partitions`` — row-budget repack of tier dirs."""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest
from trucktrack.pipeline import _plan_buckets, rebalance_partitions


def _write_partition(
    tier_dir: Path,
    partition_id: int,
    rows: int,
    *,
    start_hilbert: int = 0,
    n_chunks: int = 1,
) -> None:
    """Write a synthetic partition with *rows* rows split across *n_chunks* files.

    hilbert_idx runs contiguously from *start_hilbert*, so after sort the
    concatenated partition is in sorted order.
    """
    pdir = tier_dir / f"partition_id={partition_id}"
    pdir.mkdir(parents=True)
    rows_per_chunk = max(1, rows // n_chunks)
    written = 0
    for c in range(n_chunks):
        take = rows - written if c == n_chunks - 1 else rows_per_chunk
        df = pl.DataFrame(
            {
                "id": [f"t{partition_id}_r{i}" for i in range(written, written + take)],
                "hilbert_idx": list(
                    range(start_hilbert + written, start_hilbert + written + take)
                ),
            }
        )
        df.write_parquet(pdir / f"chunk_{c}.parquet")
        written += take


def test_plan_buckets_merges_small_partitions() -> None:
    entries = [(i, [Path(f"p{i}")], 100) for i in range(10)]
    plan = _plan_buckets(entries, target_rows=500, split_threshold=1.5)
    # Two buckets of ~500 rows each (5 * 100 = 500).
    assert len(plan) == 2
    assert all(n_slices is None for _, n_slices in plan)
    assert sum(len(files) for files, _ in plan) == 10


def test_plan_buckets_splits_oversized_partition() -> None:
    # One 2500-row partition among small ones; target=500, threshold=1.5
    # so >750 rows triggers split.  2500/500 = 5 slices.
    entries = [
        (0, [Path("small0")], 200),
        (1, [Path("big")], 2500),
        (2, [Path("small1")], 200),
    ]
    plan = _plan_buckets(entries, target_rows=500, split_threshold=1.5)
    # Expect: bucket(small0) | split(big, 5) | bucket(small1)
    assert len(plan) == 3
    files0, n0 = plan[0]
    files1, n1 = plan[1]
    files2, n2 = plan[2]
    assert n0 is None and files0 == [Path("small0")]
    assert n1 == 5 and files1 == [Path("big")]
    assert n2 is None and files2 == [Path("small1")]


def test_plan_buckets_flushes_before_oversized() -> None:
    # Non-empty bucket must never be merged into an oversized partition:
    # the oversized one stands alone so its row-window split is self-contained.
    entries = [
        (0, [Path("small")], 200),
        (1, [Path("big")], 2000),
    ]
    plan = _plan_buckets(entries, target_rows=500, split_threshold=1.5)
    assert plan[0] == ([Path("small")], None)
    files, n = plan[1]
    assert files == [Path("big")]
    assert n == 4


def test_rebalance_merges_tiny_partitions(tmp_path: Path) -> None:
    tier_dir = tmp_path / "tier=local"
    for pid in range(10):
        _write_partition(tier_dir, pid, rows=100, start_hilbert=pid * 100)

    n_out, n_rows = rebalance_partitions(tmp_path, target_rows=500)

    assert n_rows == 1000
    assert n_out == 2
    remaining = sorted(tier_dir.glob("partition_id=*"))
    assert len(remaining) == 2
    # Each output has the expected row count and is hilbert-sorted.
    for d in remaining:
        df = pl.read_parquet(d / "data.parquet")
        assert len(df) == 500
        assert df["hilbert_idx"].is_sorted()


def test_rebalance_splits_oversized(tmp_path: Path) -> None:
    tier_dir = tmp_path / "tier=longhaul"
    _write_partition(tier_dir, 0, rows=100)  # small
    _write_partition(tier_dir, 1, rows=2000, start_hilbert=10_000, n_chunks=3)
    _write_partition(tier_dir, 2, rows=100, start_hilbert=20_000)

    n_out, n_rows = rebalance_partitions(tmp_path, target_rows=500)

    assert n_rows == 2200
    # bucket(100) + split(2000/500 = 4) + bucket(100)  = 6
    assert n_out == 6
    remaining = sorted(tier_dir.glob("partition_id=*"))
    assert len(remaining) == 6
    # Slices of the oversized partition together preserve sort order.
    slice_rows = [
        pl.read_parquet(d / "data.parquet")["hilbert_idx"].to_list() for d in remaining
    ]
    flat = [r for rs in slice_rows for r in rs]
    assert flat == sorted(flat)


def test_rebalance_per_tier_isolation(tmp_path: Path) -> None:
    # local stays as-is (already balanced); longhaul gets merged.
    _write_partition(tmp_path / "tier=local", 0, rows=500)
    for pid in range(5):
        _write_partition(
            tmp_path / "tier=longhaul", pid, rows=50, start_hilbert=pid * 50
        )

    n_out, _ = rebalance_partitions(tmp_path, target_rows=500)

    assert n_out == 2  # 1 local + 1 longhaul
    assert len(list((tmp_path / "tier=local").glob("partition_id=*"))) == 1
    assert len(list((tmp_path / "tier=longhaul").glob("partition_id=*"))) == 1


def test_rebalance_cleans_stale_staging(tmp_path: Path) -> None:
    tier_dir = tmp_path / "tier=local"
    _write_partition(tier_dir, 0, rows=100)
    stale = tier_dir / "_rebalance.new"
    stale.mkdir()
    (stale / "garbage.txt").write_text("left over")

    rebalance_partitions(tmp_path, target_rows=500)

    assert not stale.exists()


def test_rebalance_aborts_on_mid_swap_state(tmp_path: Path) -> None:
    tier_dir = tmp_path / "tier=local"
    _write_partition(tier_dir, 0, rows=100)
    (tier_dir / "_rebalance.old").mkdir()

    with pytest.raises(RuntimeError, match="crashed mid-swap"):
        rebalance_partitions(tmp_path, target_rows=500)
