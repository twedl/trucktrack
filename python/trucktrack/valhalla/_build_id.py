"""Identify the tile build that produced a set of edge GraphIds.

GraphIds are fixed when tiles are built.  Adding predicted traffic
rewrites the tile archive but keeps them, so the ID can't be a hash of
the archive as it is now.  It is a hash of the archive as first packed,
stored in a ``tile_build_id`` file beside it.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path

from trucktrack.valhalla._actor import find_config

BUILD_ID_FILENAME = "tile_build_id"
_HASH_CHUNK = 8 * 1024 * 1024


def write_tile_build_id(tile_extract: str | Path) -> str:
    """Hash *tile_extract* and store the result beside it; return the ID.

    Call this right after packing freshly built tiles, before predicted
    traffic is added.  Overwrites any earlier ID.
    """
    extract = Path(tile_extract)
    digest = hashlib.sha256()
    with extract.open("rb") as f:
        while chunk := f.read(_HASH_CHUNK):
            digest.update(chunk)
    build_id = digest.hexdigest()[:16]
    fd, tmp = tempfile.mkstemp(dir=extract.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(build_id + "\n")
        os.replace(tmp, extract.parent / BUILD_ID_FILENAME)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
    return build_id


def tile_build_id(config: str | Path | None = None) -> str:
    """Return the build ID of the tiles that *config* points at.

    *config* is a ``valhalla.json``; ``None`` discovers one with
    :func:`~trucktrack.valhalla.find_config`.  The tile archive is the
    config's ``mjolnir.tile_extract``.  If its ``tile_build_id`` file is
    missing (tiles built before IDs existed), the archive is hashed now
    and the ID written, so later calls agree.
    """
    if config is None:
        config = find_config()
        if config is None:
            raise FileNotFoundError("No valhalla.json found; pass config=...")
    mjolnir = json.loads(Path(config).read_text())["mjolnir"]
    extract = Path(mjolnir["tile_extract"])
    id_path = extract.parent / BUILD_ID_FILENAME
    if id_path.exists():
        return id_path.read_text().strip()
    return write_tile_build_id(extract)
