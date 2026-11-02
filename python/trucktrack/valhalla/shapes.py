"""Road geometry of Valhalla edges, from the tiles, to map results by edge or way."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import polars as pl

from trucktrack.valhalla._build_id import _extract_path, tile_build_id
from trucktrack.valhalla._tiles import read_shapes


def edge_shapes(
    edge_ids: Iterable[int], config: str | Path | None = None
) -> pl.DataFrame:
    """The OSM way and road geometry of each of *edge_ids*, from *config*'s tiles.

    Returns ``edge_id``, ``way_id``, ``road_class`` and ``shape``: the
    edge's (lat, lon) points in the direction it runs.  Edge ids belong
    to one tile build (see :func:`~trucktrack.valhalla.tile_build_id`);
    ids not in these tiles are left out.  Loading traffic leaves the
    shapes as they were, so the extract is read as it is.
    """
    return read_shapes(_extract_path(config), edge_ids)


def _wkt_points(shape: pl.Expr) -> pl.Expr:
    """``(lon lat, lon lat, ...)``: a list of (lat, lon) points as WKT."""
    xy = pl.format("{} {}", pl.element().arr.get(1), pl.element().arr.get(0))
    return pl.format("({})", shape.list.eval(xy).list.join(", "))


def with_geometry(rows: pl.DataFrame, config: str | Path | None = None) -> pl.DataFrame:
    """*rows* with ``geometry``: their road as WKT (lon lat), in the direction driven.

    A row with ``edge_id`` (e.g. :func:`~trucktrack.congestion.rush_speeds`)
    gets its edge as a LINESTRING; a row with ``edge_ids`` (e.g.
    :func:`~trucktrack.congestion.way_rush_speeds`) gets those edges as
    a MULTILINESTRING, a part each, so a way is drawn where it was
    measured.  Rows whose edges aren't in *config*'s tiles get null.
    QGIS, GeoPandas (``shapely.from_wkt``), R's sf and DuckDB read WKT.

    Edge ids mean nothing on other tiles, so rows whose
    ``tile_build_id`` isn't *config*'s raise ``ValueError``.
    """
    if "tile_build_id" in rows.columns:
        build_id = tile_build_id(config)
        builds = set(rows["tile_build_id"].unique())
        if builds - {build_id}:
            raise ValueError(
                f"rows are for tile builds {sorted(builds)}, tiles are {build_id}"
            )
    if "edge_ids" in rows.columns:
        indexed = rows.with_row_index("_row")
        parts = indexed.select("_row", edge_id="edge_ids").explode("edge_id")
        shapes = edge_shapes(parts["edge_id"].unique().to_list(), config)
        geometry = (
            parts.join(shapes, on="edge_id", maintain_order="left")
            .group_by("_row")
            .agg(
                geometry=pl.format(
                    "MULTILINESTRING ({})",
                    _wkt_points(pl.col("shape")).str.join(", "),
                )
            )
        )
        return indexed.join(
            geometry, on="_row", how="left", maintain_order="left"
        ).drop("_row")
    shapes = edge_shapes(rows["edge_id"].unique().to_list(), config)
    lines = shapes.select(
        "edge_id", geometry=pl.format("LINESTRING {}", _wkt_points(pl.col("shape")))
    )
    return rows.join(lines, on="edge_id", how="left", maintain_order="left")
