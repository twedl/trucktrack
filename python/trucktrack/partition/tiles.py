"""Valhalla-compatible tile indexing and bbox-diagonal helpers.

The tile functions replicate Valhalla's GraphId tile logic for L0 and L1.
Source: valhalla/baldr/graphid.h
"""

from __future__ import annotations

from trucktrack import _core

VALHALLA_L1_DEG = 1.0  # 1° × 1° tiles
VALHALLA_L0_DEG = 4.0  # 4° × 4° tiles


def valhalla_tile_id(lat: float, lon: float, tile_deg: float) -> int:
    """Flat tile index matching Valhalla's row-major numbering from (-90, -180)."""
    return _core.valhalla_tile_id(lat, lon, tile_deg)


def valhalla_l1_tile(lat: float, lon: float) -> int:
    return _core.valhalla_tile_id(lat, lon, VALHALLA_L1_DEG)


def valhalla_l0_tile(lat: float, lon: float) -> int:
    return _core.valhalla_tile_id(lat, lon, VALHALLA_L0_DEG)


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    return _core.haversine_km(lat1, lon1, lat2, lon2)
