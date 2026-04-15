"""Thread-local pyvalhalla Actor wrapper."""

from __future__ import annotations

import functools
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

_SEARCH_PATHS: tuple[Path, ...] = (
    Path(CONFIG_FILENAME),
    Path("valhalla_tiles") / CONFIG_FILENAME,
)


def _looks_like_valhalla_config(path: Path) -> bool:
    """Cheap sanity check: JSON-parses and contains a ``mjolnir`` key."""
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    return isinstance(data, dict) and "mjolnir" in data


@functools.lru_cache(maxsize=1)
def find_config() -> Path | None:
    """Discover ``valhalla.json`` in conventional locations relative to cwd.

    Search order:

    1. ``./valhalla.json``
    2. ``./valhalla_tiles/valhalla.json``

    Cached for the process lifetime — assumes cwd doesn't change.  Call
    ``find_config.cache_clear()`` if you chdir and want to rediscover.
    """
    for candidate in _SEARCH_PATHS:
        if _looks_like_valhalla_config(candidate):
            return candidate
    return None


def get_actor(config: str | Path | None = None) -> Any:
    """Return a cached Valhalla Actor for the given config.

    When *config* is ``None``, discovers ``valhalla.json`` via
    :func:`find_config`.  Raises :class:`FileNotFoundError` when no
    config is found.
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

    if config is None:
        config = find_config()
        if config is None:
            raise FileNotFoundError(
                "No valhalla.json found. Create one at ./valhalla.json "
                "pointing at your tile extract, or pass config=..."
            )
    config_path = Path(config).resolve()
    key = str(config_path)
    if key not in actors:
        actors[key] = valhalla.Actor(config_path)
    return actors[key]
