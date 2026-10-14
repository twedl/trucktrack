"""Identify the tile build that produced a set of edge GraphIds.

GraphIds are fixed when tiles are built.  Adding predicted traffic
rewrites the tile archive but keeps them, so the ID can't be a hash of
the archive as it is now.  It is a hash of the archive as first packed,
stored in a ``tile_build_id`` file beside it.  Adding traffic also
needs the archive as first packed, to start again from: a copy of it
is kept beside it too (see :func:`pristine_extract`).
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path

from trucktrack.valhalla._actor import resolve_config

BUILD_ID_FILENAME = "tile_build_id"


def _extract_path(config: str | Path | None) -> Path:
    mjolnir = json.loads(resolve_config(config).read_text())["mjolnir"]
    return Path(mjolnir["tile_extract"])


def base_extract_path(tile_extract: str | Path) -> Path:
    """Where the pristine copy of *tile_extract* is kept (``*.base.tar``)."""
    return Path(tile_extract).with_suffix(".base.tar")


def extract_digest(tile_extract: str | Path) -> str:
    """The build ID *tile_extract* would get if it were freshly packed."""
    with Path(tile_extract).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()[:16]


def write_tile_build_id(tile_extract: str | Path) -> str:
    """Hash *tile_extract* and store the result beside it; return the ID.

    Call this right after packing freshly built tiles, before predicted
    traffic is added.  Overwrites any earlier ID, and drops any pristine
    copy of the earlier tiles.
    """
    extract = Path(tile_extract)
    base_extract_path(extract).unlink(missing_ok=True)
    build_id = extract_digest(extract)
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
    extract = _extract_path(config)
    id_path = extract.parent / BUILD_ID_FILENAME
    if id_path.exists():
        return id_path.read_text().strip()
    return write_tile_build_id(extract)


def pristine_extract(config: str | Path | None = None) -> Path:
    """The tile extract *config* points at, as first packed, before any traffic.

    Kept beside the extract (:func:`base_extract_path`), copied from it
    the first time it is asked for.  That is only safe while the extract
    is still the archive its build ID was taken from, so it is checked.
    """
    extract = _extract_path(config)
    base = base_extract_path(extract)
    if base.exists():
        return base
    build_id = tile_build_id(config)
    if extract_digest(extract) != build_id:
        raise RuntimeError(
            f"{extract} isn't the archive its tile build ID {build_id} was "
            f"taken from (predicted traffic already added?) and there is no "
            f"{base.name} to start again from.  Rebuild the tiles with "
            "python -m trucktrack.valhalla.setup."
        )
    fd, tmp = tempfile.mkstemp(dir=extract.parent, suffix=".tar.tmp")
    os.close(fd)
    try:
        shutil.copyfile(extract, tmp)
        os.replace(tmp, base)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
    return base
