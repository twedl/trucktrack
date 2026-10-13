"""pyvalhalla's command-line tools, and packing tile directories into an extract."""

from __future__ import annotations

import os
import tarfile
import tempfile
from pathlib import Path

import valhalla


def valhalla_bin(name: str) -> Path:
    """Path to a pyvalhalla-bundled binary (``valhalla_build_tiles``, ...)."""
    path = Path(valhalla.__file__).parent / "bin" / name
    if not path.is_file():
        raise FileNotFoundError(
            f"{name} not found at {path}. Install pyvalhalla: uv sync --extra valhalla"
        )
    return path


def pack_tiles(tile_dir: Path, tar_path: Path) -> list[Path]:
    """Pack the tile hierarchy's level directories in *tile_dir* into *tar_path*.

    The levels go in as top-level directories (``0/``, ``1/``, ``2/``),
    the layout Valhalla's ``tile_extract`` reads.  *tar_path* is
    replaced in one step, so readers never see a half-written extract.
    Returns the level directories packed.
    """
    levels = sorted(p for p in tile_dir.iterdir() if p.is_dir() and p.name.isdigit())
    fd, tmp = tempfile.mkstemp(dir=tar_path.parent, suffix=".tar.tmp")
    os.close(fd)
    try:
        with tarfile.open(tmp, "w") as tar:
            for level in levels:
                tar.add(level, arcname=level.name)
        os.replace(tmp, tar_path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
    return levels
