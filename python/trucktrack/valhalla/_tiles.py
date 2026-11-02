"""pyvalhalla's command-line tools, and reading and packing tile extracts."""

from __future__ import annotations

import os
import struct
import tarfile
import tempfile
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import polars as pl


def valhalla_bin(name: str) -> Path:
    """Path to a pyvalhalla-bundled binary (``valhalla_build_tiles``, ...)."""
    # Here, not at the top: reading tiles doesn't need pyvalhalla, and
    # trucktrack.valhalla must import without it.
    import valhalla

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
# Checked against pyvalhalla's GraphTileHeader (valhalla.baldr) on tiles
# built by 3.6.3 and 3.9.1: recheck after a Valhalla upgrade.
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
# The header's uint32 at this byte is where the edge info section starts,
# from the tile's start (after the complex restrictions' offsets, before
# the text list's).  An edge info record is three uint32 words (way id;
# more way id bits; name count and shape size), a 4-byte entry per name,
# then the shape, as zigzag varint deltas of lat and lon in 1e-6 degrees.
_EDGEINFO_AT = 104
_EDGEINFO_BYTES = 12
_NAME_BYTES = 4
# GraphId value bits below an edge's index: its level and tile.
_TILE_BITS = 25


def _bits(words: np.ndarray, low: int, width: int) -> np.ndarray:
    return (words >> np.uint64(low)) & np.uint64((1 << width) - 1)


def _directed_edges(tile: bytes) -> tuple[int, np.ndarray]:
    """A tile's GraphId value (level and tile) and its directed edges' words."""
    header = np.frombuffer(tile[:_HEADER_BYTES], dtype=np.uint64)
    tile_id = int(header[0]) & ((1 << 46) - 1)
    nodes = int(_bits(header[5], 0, 21))
    count = int(_bits(header[5], 21, 21))
    transitions = int(np.frombuffer(tile[48:52], dtype=np.uint32)[0]) & ((1 << 22) - 1)
    start = _HEADER_BYTES + nodes * _NODE_BYTES + transitions * _TRANSITION_BYTES
    edges = np.frombuffer(
        tile[start : start + count * _EDGE_WORDS * 8], dtype=np.uint64
    ).reshape(count, _EDGE_WORDS)
    return tile_id, edges


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
            tile_id, edges = _directed_edges(f.read())
            count = len(edges)
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
                        "edge_id": np.uint64(tile_id) | (index << np.uint64(25)),
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


def _varints(buf: bytes) -> list[int]:
    """The zigzag varints in *buf*."""
    out, value, shift = [], 0, 0
    for byte in buf:
        value |= (byte & 0x7F) << shift
        shift += 7
        if not byte & 0x80:
            out.append(~(value >> 1) if value & 1 else value >> 1)
            value, shift = 0, 0
    return out


def read_shapes(tile_extract: Path, edge_ids: Iterable[int]) -> pl.DataFrame:
    """The OSM way and shape of each of *edge_ids*, from the tiles in *tile_extract*.

    Returns ``edge_id``, ``way_id``, ``road_class`` and ``shape``: the
    edge's (lat, lon) points from its begin node to its end node.  Ids
    not in the extract are left out.  Decodes in Python, so it suits thousands of edges,
    not the whole graph.
    """
    wanted: dict[int, set[int]] = {}
    for edge_id in edge_ids:
        tile_key = edge_id & ((1 << _TILE_BITS) - 1)
        wanted.setdefault(tile_key, set()).add(edge_id >> _TILE_BITS)
    ids: list[int] = []
    ways: list[int] = []
    classes: list[int] = []
    shapes: list[np.ndarray] = []
    with tarfile.open(tile_extract) as tar:
        for member in tar:
            if not member.name.endswith(".gph"):
                continue
            f = tar.extractfile(member)
            if f is None:
                continue
            header = f.read(_HEADER_BYTES)
            tile_key = int(np.frombuffer(header[:8], dtype=np.uint64)[0]) & (
                (1 << _TILE_BITS) - 1
            )
            if tile_key not in wanted:
                continue
            tile = header + f.read()
            tile_id, edges = _directed_edges(tile)
            (section,) = struct.unpack_from("<I", tile, _EDGEINFO_AT)
            for index in sorted(i for i in wanted[tile_key] if i < len(edges)):
                word0, word1 = int(edges[index, 0]), int(edges[index, 1])
                at = section + (word1 & ((1 << 25) - 1))
                way, more, sizes = struct.unpack_from("<3I", tile, at)
                start = at + _EDGEINFO_BYTES + (sizes & 0xF) * _NAME_BYTES
                end = start + ((sizes >> 4) & 0xFFFF)
                way |= (more >> 24) << 32 | ((sizes >> 20) & 0xFF) << 40
                for k in range((sizes >> 28) & 0x3):
                    way |= tile[end + k] << (48 + 8 * k)
                deltas = np.array(_varints(tile[start:end]), dtype=np.int64)
                points = (
                    np.column_stack([np.cumsum(deltas[0::2]), np.cumsum(deltas[1::2])])
                    / 1e6
                )
                # Stored along the OSM way; an edge against it runs backwards.
                forward = (word0 >> 61) & 1
                ids.append(tile_id | (index << _TILE_BITS))
                ways.append(way)
                classes.append(int(_bits(edges[index, 2], 54, 3)))
                shapes.append(points if forward else points[::-1])
    return pl.DataFrame(
        {
            "edge_id": pl.Series(ids, dtype=pl.UInt64),
            "way_id": pl.Series(ways, dtype=pl.Int64),
            "road_class": pl.Series(
                [ROAD_CLASSES[c] for c in classes], dtype=pl.Enum(ROAD_CLASSES)
            ),
            "shape": pl.Series(
                [p.tolist() for p in shapes], dtype=pl.List(pl.Array(pl.Float64, 2))
            ),
        }
    )
