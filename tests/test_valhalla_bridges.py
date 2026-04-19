"""Tests for gap-split + bridge orchestration that don't need pyvalhalla.

Exercises :func:`trucktrack.valhalla._bridge._find_gap_indices` and the
pure-Python stitch helpers.  The orchestrator itself needs a live
Valhalla actor and is covered alongside the other integration tests.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import polars as pl
import pytest
from trucktrack.valhalla._bridge import (
    BridgeConfig,
    _extend_dedup,
    _find_gap_indices,
    _split_at,
)

_CFG = BridgeConfig(max_dist_m=5000, time_s=240, min_dist_m=1000)


def _trip(
    rows: list[tuple[datetime, float, float]],
) -> pl.DataFrame:
    """Build a (time, lat, lon) DataFrame from literal rows."""
    return pl.DataFrame(
        {
            "time": [r[0] for r in rows],
            "lat": [r[1] for r in rows],
            "lon": [r[2] for r in rows],
        }
    )


_T0 = datetime(2026, 4, 19, 12, 0, 0)


class TestFindGapIndices:
    def test_no_gaps_returns_empty(self) -> None:
        # 60s apart, ~17m spacing — dense trace, nothing triggers.
        rows = [
            (_T0 + timedelta(seconds=60 * i), 43.65, -79.38 + i * 0.0002)
            for i in range(5)
        ]
        df = _trip(rows)
        assert _find_gap_indices(df, bridges=_CFG) == []

    def test_distance_only_triggers(self) -> None:
        # Points 0..1 are 20s apart but ~15 km apart — distance rule fires.
        rows = [
            (_T0, 43.65, -79.38),
            (_T0 + timedelta(seconds=20), 43.75, -79.50),
            (_T0 + timedelta(seconds=40), 43.76, -79.51),
        ]
        df = _trip(rows)
        assert _find_gap_indices(df, bridges=_CFG) == [0]

    def test_time_alone_does_not_trigger_without_distance(self) -> None:
        # Big time gap (10 min) but nearly coincident points — red-light case.
        # Distance between (43.65, -79.38) and (43.6501, -79.3801) is ~14 m,
        # way below min_dist_m=1000, so the time rule doesn't fire.
        rows = [
            (_T0, 43.65, -79.38),
            (_T0 + timedelta(minutes=10), 43.6501, -79.3801),
            (_T0 + timedelta(minutes=11), 43.6502, -79.3802),
        ]
        df = _trip(rows)
        assert _find_gap_indices(df, bridges=_CFG) == []

    def test_time_plus_min_distance_triggers(self) -> None:
        # 6 min gap, ~1.2 km apart — short-distance stall in dense urban.
        rows = [
            (_T0, 43.65, -79.38),
            (_T0 + timedelta(minutes=6), 43.66, -79.39),
            (_T0 + timedelta(minutes=7), 43.661, -79.391),
        ]
        df = _trip(rows)
        assert _find_gap_indices(df, bridges=_CFG) == [0]

    def test_multiple_gaps(self) -> None:
        rows = [
            (_T0, 43.65, -79.38),
            (_T0 + timedelta(seconds=30), 43.75, -79.50),
            (_T0 + timedelta(seconds=60), 43.76, -79.51),
            (_T0 + timedelta(minutes=10), 43.80, -79.55),
        ]
        df = _trip(rows)
        assert _find_gap_indices(df, bridges=_CFG) == [0, 2]

    def test_single_point_returns_empty(self) -> None:
        df = _trip([(_T0, 43.65, -79.38)])
        assert _find_gap_indices(df, bridges=_CFG) == []

    def test_empty_df_returns_empty(self) -> None:
        df = _trip([])
        assert _find_gap_indices(df, bridges=_CFG) == []


class TestSplitAt:
    def test_no_gap_returns_single_segment(self) -> None:
        df = _trip([(_T0 + timedelta(seconds=i), 43.65, -79.38) for i in range(3)])
        [seg] = _split_at(df, [])
        assert len(seg) == 3

    def test_splits_at_gap_index(self) -> None:
        df = _trip([(_T0 + timedelta(seconds=i), 43.65, -79.38) for i in range(5)])
        s0, s1 = _split_at(df, [1])
        assert len(s0) == 2
        assert len(s1) == 3


class TestExtendDedup:
    def test_dedups_at_seam(self) -> None:
        acc = [1, 2, 3]
        _extend_dedup(acc, [3, 4, 5])
        assert acc == [1, 2, 3, 4, 5]

    def test_no_dedup_when_different(self) -> None:
        acc = [1, 2, 3]
        _extend_dedup(acc, [4, 5])
        assert acc == [1, 2, 3, 4, 5]

    def test_empty_new_is_noop(self) -> None:
        acc = [1, 2, 3]
        _extend_dedup(acc, [])
        assert acc == [1, 2, 3]

    def test_empty_acc_accepts_new(self) -> None:
        acc: list[int] = []
        _extend_dedup(acc, [1, 2])
        assert acc == [1, 2]


@pytest.mark.parametrize(
    "gap_idxs,expected_lengths",
    [
        ([], [5]),
        ([0], [1, 4]),
        ([1, 3], [2, 2, 1]),
        ([4], [5, 0]),
    ],
)
def test_split_at_preserves_row_count(
    gap_idxs: list[int], expected_lengths: list[int]
) -> None:
    df = _trip([(_T0 + timedelta(seconds=i), 43.65, -79.38) for i in range(5)])
    segments = _split_at(df, gap_idxs)
    assert [len(s) for s in segments] == expected_lengths
    assert sum(len(s) for s in segments) == len(df)
