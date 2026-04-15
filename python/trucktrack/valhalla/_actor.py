"""Thread-local pyvalhalla Actor wrapper."""

from __future__ import annotations

import json
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


def _config_points_to_existing_tiles(config_path: Path) -> bool:
    """True when the config's ``mjolnir.tile_extract``/``tile_dir`` resolves locally.

    Configs generated in containers often embed absolute paths like
    ``/custom_files/valhalla_tiles.tar`` that don't exist on the host.
    Skipping those lets us fall back to a freshly built config.
    """
    try:
        data = json.loads(config_path.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    mjolnir = data.get("mjolnir", {})
    for key in ("tile_extract", "tile_dir"):
        value = mjolnir.get(key)
        if value and Path(value).exists():
            return True
    return False


def _find_config(tile_extract: str) -> Path | None:
    """Look for a *valid* valhalla.json next to, inside, or in the cwd.

    Search order:

    1. Inside *tile_extract* (when it is a directory).
    2. Sibling of *tile_extract* (e.g. ``valhalla_tiles/valhalla.json``
       next to ``valhalla_tiles.tar``).
    3. Current working directory.

    A candidate is only accepted when its embedded tile path exists
    locally — stale container-style configs (e.g. pointing at
    ``/custom_files/...``) are skipped so ``get_actor`` can fall back
    to :func:`valhalla.get_config`.
    """
    p = Path(tile_extract)
    candidates: list[Path] = []
    if p.is_dir():
        candidates.append(p / CONFIG_FILENAME)
    candidates.append(p.parent / CONFIG_FILENAME)
    candidates.append(Path.cwd() / CONFIG_FILENAME)
    for candidate in candidates:
        if candidate.is_file() and _config_points_to_existing_tiles(candidate):
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

    if not hasattr(_local, "actors"):
        _local.actors = {}
    actors: dict[str, Any] = _local.actors

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
