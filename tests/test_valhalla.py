"""Tests for the trucktrack.valhalla submodule."""

from __future__ import annotations

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

def _valhalla_works() -> bool:
    """True iff a ``valhalla.json`` is discoverable AND Actor construction
    succeeds with it.  Catches both missing configs and stale ones that
    point at unavailable or incompatible tile extracts.
    """
    if not _pyvalhalla_available:
        return False
    try:
        from trucktrack.valhalla._actor import get_actor

        get_actor()
    except Exception:
        return False
    return True


_valhalla_ok = _valhalla_works()

requires_tiles = pytest.mark.skipif(
    not _valhalla_ok,
    reason="no working valhalla.json discoverable",
)


@requires_pyvalhalla
@requires_tiles
class TestActorCache:
    def test_get_actor_returns_same_instance(self) -> None:
        from trucktrack.valhalla._actor import get_actor

        a1 = get_actor()
        a2 = get_actor()
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
        result = map_match(points)
        assert len(result) == len(points)
        for mp in result:
            assert isinstance(mp, MatchedPoint)

    def test_map_match_ways_returns_way_ids(self) -> None:
        from trucktrack.valhalla import map_match_ways

        points = [(43.65, -79.38), (43.651, -79.381), (43.652, -79.382)]
        result = map_match_ways(points)
        assert isinstance(result, list)
        assert len(result) >= 1
        assert all(isinstance(w, int) for w in result)
        # Consecutive way IDs should be deduplicated
        for i in range(1, len(result)):
            assert result[i] != result[i - 1]

    def test_map_match_dataframe(self) -> None:
        import polars as pl
        from trucktrack.valhalla import map_match_dataframe

        df = pl.DataFrame(
            {
                "lat": [43.65, 43.651, 43.652],
                "lon": [-79.38, -79.381, -79.382],
            }
        )
        result = map_match_dataframe(df)
        assert "matched_lat" in result.columns
        assert "matched_lon" in result.columns
        assert "distance_from_trace" in result.columns
        assert len(result) == len(df)
