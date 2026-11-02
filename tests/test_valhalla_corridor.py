"""Tests for trucktrack.valhalla.corridor: spans and edges along a corridor."""

from __future__ import annotations

import math

import numpy as np
import polars as pl
import pytest

pytest.importorskip("valhalla")

from trucktrack.valhalla.corridor import (  # noqa: E402
    RURAL_SEGMENT_M,
    URBAN_SEGMENT_M,
    Corridor,
    Route,
    _haversine_m,
    locate_edges,
)

_LAT = 43.0
_M_PER_DEG_LON = math.radians(1) * 6_371_000 * math.cos(math.radians(_LAT))
_M_PER_DEG_LAT = math.radians(1) * 6_371_000


def _at(east_m: float, north_m: float = 0.0) -> tuple[float, float]:
    """(lat, lon) *east_m* east and *north_m* north of (43, -80)."""
    return _LAT + north_m / _M_PER_DEG_LAT, -80.0 + east_m / _M_PER_DEG_LON


def _route(names: tuple[str, ...], xs: list[float], stops_x: list[float]) -> Route:
    """A straight route along the points at *xs* metres east, stopping at *stops_x*."""
    lat, lon = map(np.array, zip(*(_at(x) for x in xs), strict=True))
    chainage = np.concatenate([[0.0], np.cumsum(_haversine_m(lat, lon))])
    stops = np.interp(stops_x, xs, chainage) if xs[0] < xs[-1] else None
    if stops is None:  # heading west: chainage grows as x falls
        stops = np.interp([-x for x in stops_x], [-x for x in xs], chainage)
    return Route(names, lat, lon, chainage, stops)


# 0-50 km east and back, with stops at 1 km (A), 21 km (B) and 49 km (C).
_XS = [float(x) for x in range(0, 50_001, 500)]
_CORRIDOR = Corridor(
    (
        _route(("A", "B", "C"), _XS, [1000, 21_000, 49_000]),
        _route(("C", "B", "A"), _XS[::-1], [49_000, 21_000, 1000]),
    )
)


class TestSpans:
    def test_corridor_sections_and_fhwa_segments(self) -> None:
        # Urban within 5 km of x = 1 km: A to 6 km.
        spans = _CORRIDOR.spans(urban=[(*_at(1000), 5000.0)])
        east = spans.filter(pl.col("direction") == "A to C")
        corridor = east.filter(pl.col("kind") == "corridor").row(0, named=True)
        assert corridor["name"] == "A to C"
        assert corridor["end_m"] - corridor["begin_m"] == pytest.approx(
            48_000, rel=1e-3
        )
        sections = east.filter(pl.col("kind") == "section")
        assert sections["name"].to_list() == ["A to B", "B to C"]
        segments = east.filter(pl.col("kind") == "segment").sort("begin_m")
        lengths = (segments["end_m"] - segments["begin_m"]).to_numpy()
        # A-B: 5 km urban in 4 pieces, then 15 km rural in 1; B-C: 28 km in 2.
        assert len(segments) == 4 + 1 + 2
        assert (lengths[:4] <= URBAN_SEGMENT_M).all()
        assert (lengths[4:] <= RURAL_SEGMENT_M).all()
        # Contiguous, from A to C.
        np.testing.assert_allclose(segments["begin_m"][1:], segments["end_m"][:-1])
        assert segments["name"][0] == "A to B 1/5"
        assert segments["name"][-1] == "B to C 2/2"

    def test_each_direction_on_its_own_route(self) -> None:
        spans = _CORRIDOR.spans()
        west = spans.filter(
            pl.col("direction") == "C to A", pl.col("kind") == "section"
        )
        assert west.sort("begin_m")["name"].to_list() == ["C to B", "B to A"]
        first = west.sort("begin_m").row(0, named=True)
        # C is 1 km from the westbound route's start.
        assert first["begin_m"] == pytest.approx(1000, rel=1e-3)
        assert first["end_m"] - first["begin_m"] == pytest.approx(28_000, rel=1e-3)


def _shapes(*edges: tuple[int, str, list[tuple[float, float]]]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "edge_id": pl.Series([e[0] for e in edges], dtype=pl.UInt64),
            "road_class": [e[1] for e in edges],
            "shape": pl.Series(
                [[list(_at(*p)) for p in e[2]] for e in edges],
                dtype=pl.List(pl.Array(pl.Float64, 2)),
            ),
        }
    )


class TestLocateEdges:
    def test_carriageways_and_lanes_alongside(self) -> None:
        shapes = _shapes(
            (1, "motorway", [(10_000, -15), (10_400, -15)]),  # eastbound
            (2, "motorway", [(10_400, 15), (10_000, 15)]),  # westbound
            (3, "motorway", [(20_000, -250), (21_000, -250)]),  # collector lanes
        )
        out = locate_edges(_CORRIDOR, shapes).sort("edge_id")
        assert out["direction"].to_list() == ["A to C", "C to A", "A to C"]
        east, west, collector = out.rows(named=True)
        assert east["begin_m"] == pytest.approx(10_000, abs=2)
        assert east["end_m"] == pytest.approx(10_400, abs=2)
        # The westbound route starts at x = 50 km.
        assert west["begin_m"] == pytest.approx(39_600, abs=2)
        assert west["end_m"] == pytest.approx(40_000, abs=2)
        assert collector["end_m"] - collector["begin_m"] == pytest.approx(1000, abs=2)

    def test_roads_that_dont_run_along_it(self) -> None:
        shapes = _shapes(
            (4, "motorway", [(30_000, 0), (30_300, 200), (30_500, 600)]),  # exit ramp
            (5, "motorway", [(31_000, 40), (31_000, -40)]),  # highway crossing
            (6, "trunk", [(32_000, 100), (33_000, 100)]),  # road beside it
            (7, "motorway", [(34_000, 2000), (35_000, 2000)]),  # another highway
        )
        assert locate_edges(_CORRIDOR, shapes).height == 0
        beside = locate_edges(_CORRIDOR, shapes, road_classes=("motorway", "trunk"))
        assert beside["edge_id"].to_list() == [6]
