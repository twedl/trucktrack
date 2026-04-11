"""Singleton pyvalhalla Actor wrapper."""

from __future__ import annotations

import json
from typing import Any

DEFAULT_TRUCK_COSTING: dict[str, float] = {
    "height": 4.11,
    "width": 2.6,
    "length": 22.0,
    "weight": 36.287,
}

_actors: dict[str, Any] = {}


def get_actor(tile_extract: str) -> Any:
    """Return a cached Valhalla Actor for the given tile extract path."""
    try:
        import valhalla
    except ImportError as exc:
        raise ImportError(
            "pyvalhalla is required for local Valhalla routing. "
            "Install it with: pip install trucktrack[valhalla]"
        ) from exc

    if tile_extract not in _actors:
        config = valhalla.get_config(tile_extract=tile_extract)
        _actors[tile_extract] = valhalla.Actor(json.dumps(config))
    return _actors[tile_extract]
