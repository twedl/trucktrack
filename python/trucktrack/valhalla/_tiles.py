"""pyvalhalla's command-line tools, and reading and packing tile extracts."""

from __future__ import annotations

import os
import tarfile
import tempfile
from pathlib import Path

import numpy as np
import polars as pl
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


# Valhalla 3.x tile layout: a 272-byte header, then 32-byte nodes, 8-byte
# node transitions and 48-byte (six-word) directed edges.  Bit positions
# are those of GraphTileHeader and DirectedEdge, packed from the low bit.
_HEADER_BYTES = 272
_NODE_BYTES = 32
_TRANSITION_BYTES = 8
_EDGE_WORDS = 6
# Valhalla's RoadClass, by its 3-bit value; names as trace_attributes gives them.
ROAD_CLASSES = (
    "motorway",
    "trunk",
    "primary",
    "secondary",
    "tertiary",
    "unclassified",
    "residential",
    "service_other",
)
_TRUCK_ACCESS = 8
# Use values below this are roads (road, ramp, turn channel, ..., service
# road); paths, ferries, construction and transit come after.
_ROAD_USES = 12


def _bits(words: np.ndarray, low: int, width: int) -> np.ndarray:
    return (words >> np.uint64(low)) & np.uint64((1 << width) - 1)


def read_edges(tile_extract: Path) -> pl.DataFrame:
    """Every road edge a truck may drive, from the tiles in *tile_extract*.

    Returns ``edge_id`` (the GraphId value, as observations carry it),
    ``road_class``, ``length`` (m) and ``speed``: the km/h Valhalla's
    truck costing takes without traffic, the edge's speed or its truck
    speed where one is tagged and lower.  Shortcuts, edges trucks may
    not drive in their direction, and non-road uses (ferries, paths)
    are left out.
    """
    frames = []
    with tarfile.open(tile_extract) as tar:
        for member in tar:
            if not member.name.endswith(".gph"):
                continue
            f = tar.extractfile(member)
            if f is None:
                continue
            tile = f.read()
            header = np.frombuffer(tile[:_HEADER_BYTES], dtype=np.uint64)
            tile_id = header[0] & np.uint64((1 << 46) - 1)
            nodes = int(_bits(header[5], 0, 21))
            count = int(_bits(header[5], 21, 21))
            transitions = int(np.frombuffer(tile[48:52], dtype=np.uint32)[0]) & (
                (1 << 22) - 1
            )
            start = (
                _HEADER_BYTES + nodes * _NODE_BYTES + transitions * _TRANSITION_BYTES
            )
            edges = np.frombuffer(
                tile[start : start + count * _EDGE_WORDS * 8], dtype=np.uint64
            ).reshape(count, _EDGE_WORDS)
            speed, truck_speed = _bits(edges[:, 2], 0, 8), _bits(edges[:, 2], 24, 8)
            keep = (
                ((_bits(edges[:, 3], 0, 12) & np.uint64(_TRUCK_ACCESS)) > 0)
                & (_bits(edges[:, 5], 60, 1) == 0)  # not a shortcut
                & (_bits(edges[:, 2], 40, 6) < _ROAD_USES)
                & (speed > 0)
            )
            index = np.arange(count, dtype=np.uint64)[keep]
            frames.append(
                pl.DataFrame(
                    {
                        "edge_id": tile_id | (index << np.uint64(25)),
                        "road_class": _bits(edges[keep, 2], 54, 3).astype(np.int64),
                        "length": _bits(edges[keep, 4], 32, 24).astype(np.float64),
                        "speed": np.where(
                            (truck_speed > 0) & (truck_speed < speed),
                            truck_speed,
                            speed,
                        )[keep].astype(np.float64),
                    }
                )
            )
    classes = pl.Enum(ROAD_CLASSES)
    return pl.concat(frames).with_columns(
        pl.col("road_class").replace_strict(dict(enumerate(ROAD_CLASSES))).cast(classes)
    )
