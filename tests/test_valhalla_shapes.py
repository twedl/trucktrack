"""Tests for trucktrack.valhalla.shapes: edge geometry from the tiles."""

from __future__ import annotations

import io
import json
import struct
import tarfile
from pathlib import Path

import numpy as np
import polars as pl
import pytest

pytest.importorskip("valhalla")

from trucktrack.valhalla import (  # noqa: E402
    edge_shapes,
    tile_build_id,
    with_geometry,
)
from valhalla.baldr import GraphId  # noqa: E402

_TILE = (770804, 2)
_A, _B = GraphId(*_TILE, 0).value, GraphId(*_TILE, 1).value


def _varints(values: list[int]) -> bytes:
    out = bytearray()
    for v in values:
        z = v << 1 if v >= 0 else (~v << 1) | 1
        while z >= 0x80:
            out.append(z & 0x7F | 0x80)
            z >>= 7
        out.append(z)
    return bytes(out)


def _edgeinfo(way_id: int, points: list[tuple[float, float]], names: int = 0) -> bytes:
    """An edge info record: *way_id*, *names* name entries, *points* along the way."""
    ints = [round(c * 1e6) for p in points for c in p]
    shape = _varints([b - a for a, b in zip([0, 0, *ints[:-2]], ints, strict=True)])
    high = way_id >> 32
    words = struct.pack(
        "<3I",
        way_id & 0xFFFFFFFF,
        (high & 0xFF) << 24,
        names | len(shape) << 4 | (high >> 8 & 0xFF) << 20 | (high > 0xFFFF) << 28,
    )
    return (
        words + bytes(4 * names) + shape + bytes([high >> 16 & 0xFF] * (high > 0xFFFF))
    )


def _extract(tmp_path: Path, edges: list[tuple[bool, bytes]]) -> Path:
    """valhalla.json for one level-2 tile of (forward, edge info) edges."""
    infos = [info for _, info in edges]
    offsets = np.cumsum([0, *map(len, infos)])
    words = np.zeros((len(edges), 6), dtype=np.uint64)
    for i, (forward, _) in enumerate(edges):
        words[i, 0] = int(forward) << 61
        words[i, 1] = offsets[i]
    header = np.zeros(34, dtype=np.uint64)
    header[0] = _TILE[1] | _TILE[0] << 3
    header[5] = 1 | len(edges) << 21
    section = header.nbytes + 32 + words.nbytes
    header.view(np.uint32)[26] = section  # byte 104
    tile = header.tobytes() + bytes(32) + words.tobytes() + b"".join(infos)
    tar_path = tmp_path / "tiles.tar"
    with tarfile.open(tar_path, "w") as tar:
        info = tarfile.TarInfo("2/000/770/804.gph")
        info.size = len(tile)
        tar.addfile(info, io.BytesIO(tile))
    config = tmp_path / "valhalla.json"
    config.write_text(json.dumps({"mjolnir": {"tile_extract": str(tar_path)}}))
    return config


@pytest.fixture
def config(tmp_path: Path) -> Path:
    # Edge A runs along way 123; edge B against way 2**40 + 5, which has a name.
    return _extract(
        tmp_path,
        [
            (True, _edgeinfo(123, [(43.1, -79.2), (43.2, -79.3)])),
            (
                False,
                _edgeinfo(2**40 + 5, [(43.0, -79.0), (43.5, -79.5), (44.0, -80.0)], 1),
            ),
        ],
    )


class TestEdgeShapes:
    def test_way_and_points_in_driving_order(self, config: Path) -> None:
        missing = GraphId(1, 2, 0).value
        out = edge_shapes([_B, missing, _A], config)
        assert out["edge_id"].to_list() == [_A, _B]
        assert out["way_id"].to_list() == [123, 2**40 + 5]
        a, b = out["shape"].to_list()
        np.testing.assert_allclose(a, [[43.1, -79.2], [43.2, -79.3]])
        np.testing.assert_allclose(b, [[44.0, -80.0], [43.5, -79.5], [43.0, -79.0]])


class TestWithGeometry:
    def test_edge_rows_get_a_linestring(self, config: Path) -> None:
        rows = pl.DataFrame({"edge_id": pl.Series([_A, 99], dtype=pl.UInt64)})
        assert with_geometry(rows, config)["geometry"].to_list() == [
            "LINESTRING (-79.2 43.1, -79.3 43.2)",
            None,
        ]

    def test_way_rows_get_their_edges(self, config: Path) -> None:
        rows = pl.DataFrame(
            {"way_id": [7], "edge_ids": [pl.Series([_A, _B], dtype=pl.UInt64)]}
        )
        assert with_geometry(rows, config)["geometry"].item() == (
            "MULTILINESTRING ((-79.2 43.1, -79.3 43.2), "
            "(-80.0 44.0, -79.5 43.5, -79.0 43.0))"
        )

    def test_rows_from_other_tiles_refused(self, config: Path) -> None:
        rows = pl.DataFrame(
            {
                "tile_build_id": [tile_build_id(config)],
                "edge_id": pl.Series([_A], dtype=pl.UInt64),
            }
        )
        assert with_geometry(rows, config)["geometry"].item() is not None
        rebuilt = rows.with_columns(tile_build_id=pl.lit("older"))
        with pytest.raises(ValueError, match="tile builds"):
            with_geometry(rebuilt, config)
