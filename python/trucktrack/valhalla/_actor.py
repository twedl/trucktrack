"""Singleton pyvalhalla Actor wrapper."""

from __future__ import annotations

from pathlib import Path
from typing import Any

DEFAULT_TRUCK_COSTING: dict[str, float] = {
    "height": 4.11,
    "width": 2.6,
    "length": 22.0,
    "weight": 36.287,
}

_actors: dict[str, Any] = {}

CONFIG_FILENAME = "valhalla.json"


def _find_config(tile_extract: str) -> Path | None:
    """Look for an existing valhalla.json next to or inside *tile_extract*."""
    p = Path(tile_extract)
    if p.is_dir():
        candidate = p / CONFIG_FILENAME
        if candidate.is_file():
            return candidate
    # Sibling of a .tar / directory (e.g. valhalla_tiles/valhalla.json for
    # valhalla_tiles.tar sitting in the same parent).
    candidate = p.parent / CONFIG_FILENAME
    if candidate.is_file():
        return candidate
    return None


def get_actor(tile_extract: str) -> Any:
    """Return a cached Valhalla Actor for the given tile extract path."""
    try:
        import valhalla
    except ImportError as exc:
        raise ImportError(
            "pyvalhalla is required for local Valhalla routing. "
            "Install it with: pip install trucktrack[valhalla]"
        ) from exc

    key = str(Path(tile_extract).resolve())
    if key not in _actors:
        config_path = _find_config(tile_extract)
        if config_path is not None:
            _actors[key] = valhalla.Actor(config_path)
        else:
            config = valhalla.get_config(tile_extract=tile_extract)
            _actors[key] = valhalla.Actor(config)
    return _actors[key]
