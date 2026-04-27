"""Bbox-pruned loader for ``way_counts.parquet`` (from way_id_counts.py).

The output parquet has ``bbox_xmin/ymin/xmax/ymax`` and ``centroid_x/y``
columns alongside ``way_id``, ``trip_count``, ``geometry``. This module
loads a bbox slice into a ``GeoDataFrame`` without decoding WKB for
ways outside the requested area.

Bbox filtering happens in WGS84 (the OSM source CRS). When you plot in
a different CRS — e.g. EPSG:3347 (Statistics Canada Lambert) — pass the
projected bbox via ``bbox_crs=...`` and the function transforms it
back to 4326 with edge densification (via
``pyproj.Transformer.transform_bounds``) so the curved CRS edges are
fully covered without you having to over-pad manually.

Requires (not in pyproject.toml)::

    pip install duckdb geopandas pyproj

Usage in a notebook::

    from way_counts_zoom import load_zoom

    # Plot in EPSG:3347; pass the projected bbox directly.
    gdf = load_zoom(
        "temp/way_counts.parquet",
        bbox=(7_000_000, 700_000, 8_000_000, 1_400_000),
        bbox_crs="EPSG:3347",
        min_trip_count=10,
        to_crs="EPSG:3347",
    )
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import duckdb
import geopandas as gpd
import pyproj

Bbox = tuple[float, float, float, float]


def load_zoom(
    parquet_path: Path | str,
    bbox: Bbox | None = None,
    bbox_crs: str | int = 4326,
    min_trip_count: int | None = None,
    top_n: int | None = None,
    to_crs: str | int | None = None,
) -> gpd.GeoDataFrame:
    """Load a bbox-pruned slice of ``way_counts.parquet``.

    Parameters
    ----------
    parquet_path:
        Output of ``way_id_counts.py`` — must contain ``bbox_*`` columns
        (run with ``--force-stage 3`` once if upgrading from an older
        output).
    bbox:
        ``(xmin, ymin, xmax, ymax)`` in ``bbox_crs`` coordinates.
    bbox_crs:
        CRS of ``bbox``. Anything other than EPSG:4326 is transformed
        back to 4326 with 21-point edge densification before filtering.
    min_trip_count:
        Drop ways below this trip count.
    top_n:
        Keep only the top N ways by trip_count after filtering.
    to_crs:
        If set, reproject the returned GeoDataFrame to this CRS.

    Returns
    -------
    GeoDataFrame with ``way_id``, ``trip_count``, ``geometry``.
    """
    src = Path(parquet_path)
    src_lit = "'" + str(src).replace("'", "''") + "'"

    clauses: list[str] = []
    params: list[Any] = []
    if bbox is not None:
        x0, y0, x1, y1 = _bbox_to_wgs84(bbox, bbox_crs)
        clauses.append(
            "bbox_xmax >= ? AND bbox_xmin <= ? AND bbox_ymax >= ? AND bbox_ymin <= ?"
        )
        params += [x0, x1, y0, y1]
    if min_trip_count is not None:
        clauses.append("trip_count >= ?")
        params.append(int(min_trip_count))
    where_sql = ("WHERE " + " AND ".join(clauses)) if clauses else ""

    limit_sql = ""
    if top_n is not None:
        limit_sql = "LIMIT ?"
        params.append(int(top_n))

    sql = f"""
        SELECT way_id, trip_count, geometry
        FROM read_parquet({src_lit})
        {where_sql}
        ORDER BY trip_count DESC
        {limit_sql}
    """

    con = duckdb.connect()
    arrow_tbl = con.execute(sql, params).arrow()
    df = arrow_tbl.to_pandas()
    geom = gpd.GeoSeries.from_wkb(df["geometry"], crs="EPSG:4326")
    gdf = gpd.GeoDataFrame(
        df.drop(columns=["geometry"]),
        geometry=geom,
        crs="EPSG:4326",
    )
    if to_crs is not None:
        gdf = gdf.to_crs(to_crs)
    return gdf


def _bbox_to_wgs84(bbox: Bbox, bbox_crs: str | int) -> Bbox:
    src = pyproj.CRS.from_user_input(bbox_crs)
    if src.equals(pyproj.CRS.from_epsg(4326)):
        return bbox
    transformer = pyproj.Transformer.from_crs(src, 4326, always_xy=True)
    x0, y0, x1, y1 = transformer.transform_bounds(*bbox, densify_pts=21)
    return (x0, y0, x1, y1)
