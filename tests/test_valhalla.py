"""Tests for the trucktrack.valhalla submodule."""

from __future__ import annotations

import os

import pytest

# Skip the entire module if pyvalhalla is not installed.
_pyvalhalla_available = False
try:
    import valhalla as _valhalla  # noqa: F401

    _pyvalhalla_available = True
except ImportError:
    pass

requires_pyvalhalla = pytest.mark.skipif(
    not _pyvalhalla_available,
    reason="pyvalhalla not installed",
)

_TILE_EXTRACT = os.environ.get("VALHALLA_TILE_EXTRACT", "")

requires_tiles = pytest.mark.skipif(
    not _TILE_EXTRACT,
    reason="VALHALLA_TILE_EXTRACT env var not set",
)


@requires_pyvalhalla
class TestActorCache:
    def test_get_actor_returns_same_instance(self) -> None:
        if not _TILE_EXTRACT:
            pytest.skip("VALHALLA_TILE_EXTRACT not set")
        from trucktrack.valhalla._actor import get_actor

        a1 = get_actor(_TILE_EXTRACT)
        a2 = get_actor(_TILE_EXTRACT)
        assert a1 is a2


@requires_pyvalhalla
@requires_tiles
class TestRoute:
    def test_route_returns_route_segment(self) -> None:
        from trucktrack.generate.models import RouteSegment
        from trucktrack.valhalla import route

        result = route(
            origin=(43.65, -79.38),
            destination=(43.70, -79.40),
            tile_extract=_TILE_EXTRACT,
        )
        assert isinstance(result, RouteSegment)
        assert len(result.coords) >= 2
        assert result.total_distance_m > 0
        assert result.total_duration_s > 0
        assert len(result.speeds_mps) == len(result.coords) - 1


@requires_pyvalhalla
@requires_tiles
class TestMapMatch:
    def test_map_match_returns_matched_points(self) -> None:
        from trucktrack.valhalla import MatchedPoint, map_match

        points = [(43.65, -79.38), (43.651, -79.381), (43.652, -79.382)]
        result = map_match(points, tile_extract=_TILE_EXTRACT)
        assert len(result) == len(points)
        for mp in result:
            assert isinstance(mp, MatchedPoint)

    def test_map_match_dataframe(self) -> None:
        import polars as pl
        from trucktrack.valhalla import map_match_dataframe

        df = pl.DataFrame(
            {
                "lat": [43.65, 43.651, 43.652],
                "lon": [-79.38, -79.381, -79.382],
            }
        )
        result = map_match_dataframe(df, tile_extract=_TILE_EXTRACT)
        assert "matched_lat" in result.columns
        assert "matched_lon" in result.columns
        assert "distance_from_trace" in result.columns
        assert len(result) == len(df)
