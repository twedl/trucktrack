"""Thread-local pyvalhalla Actor wrapper.

The per-thread cache (via :data:`_local`) is **load-bearing for
correctness**, not just defensive.  ``pyvalhalla.Actor`` is not safe
to share across threads: concurrent ``trace_attributes`` calls against
a single Actor crash with SIGTRAP on pyvalhalla 3.6.x (measured
2026-04-16 on macOS, 4 threads × 50 calls reproduced reliably).  See
``tests/test_valhalla_concurrency.py`` for the regression guard.

Actor construction is cheap in this configuration — about 10 ms per
Actor on our hardware, because ``mjolnir.tile_extract`` mmaps the tile
archive at the process level and subsequent Actors reuse that mapping.
In a representative pipeline run, per-thread init was ~1% of aggregate
worker time, well below any threshold that would justify a shared
cache.  The numbers and plan are archived at
``/Users/tweedle/.claude/plans/crystalline-floating-micali.md``.

If a future pyvalhalla version gains concurrent-safe
``trace_attributes`` / ``trace_route``, the regression test above will
start failing — that's the signal to revisit this module and swap
:data:`_local` for a module-level cache.
"""

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

    Actors are cached **per thread** via :data:`_local` because
    ``pyvalhalla.Actor`` is not safe to share across threads — see the
    module docstring for details and the measured evidence.  Do not
    replace this with a module-level cache without first re-running
    ``tests/test_valhalla_concurrency.py`` against the target
    pyvalhalla version.

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
