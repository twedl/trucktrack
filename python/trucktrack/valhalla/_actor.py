"""Thread-local pyvalhalla Actor wrapper."""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

DEFAULT_TRUCK_COSTING: dict[str, float] = {
    "height": 4.11,
    "width": 2.6,
    "length": 22.0,
    "weight": 36.287,
}

_local = threading.local()

CONFIG_FILENAME = "valhalla.json"


def _find_config(tile_extract: str) -> Path | None:
    """Look for an existing valhalla.json next to, inside, or in the cwd.

    Search order:

    1. Inside *tile_extract* (when it is a directory).
    2. Sibling of *tile_extract* (e.g. ``valhalla_tiles/valhalla.json``
       next to ``valhalla_tiles.tar``).
    3. Current working directory.
    """
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
    candidate = Path.cwd() / CONFIG_FILENAME
    if candidate.is_file():
        return candidate
    return None


def get_actor(
    tile_extract: str | None = None,
    config: str | Path | None = None,
) -> Any:
    """Return a cached Valhalla Actor for the given tile extract or config.

    At least one of *tile_extract* or *config* must be provided.

    Parameters
    ----------
    tile_extract
        Path to the Valhalla tile extract (``.tar`` file or directory).
        Optional when *config* is provided (the config already contains
        the tile path).
    config
        Explicit path to a ``valhalla.json`` config file.  When provided
        this takes priority over automatic discovery.
    """
    try:
        import valhalla
    except ImportError as exc:
        raise ImportError(
            "pyvalhalla is required for local Valhalla routing. "
            "Install it with: pip install trucktrack[valhalla]"
        ) from exc

    actors: dict[str, Any] = getattr(_local, "actors", None) or {}
    _local.actors = actors

    if config is not None:
        key = str(Path(config).resolve())
        if key not in actors:
            actors[key] = valhalla.Actor(Path(config))
    elif tile_extract is not None:
        key = str(Path(tile_extract).resolve())
        if key not in actors:
            found = _find_config(tile_extract)
            if found is not None:
                actors[key] = valhalla.Actor(found)
            else:
                actors[key] = valhalla.Actor(
                    valhalla.get_config(tile_extract=tile_extract)
                )
    else:
        raise ValueError("At least one of tile_extract or config must be provided.")
    return actors[key]
