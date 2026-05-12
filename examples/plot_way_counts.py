"""Render a continent-scale map of the way-count parquet using plotnine.

Consumes the output of ``way_id_counts.py`` (way_id, trip_count,
geometry as WKB) and produces a PNG of the whole Canada + US matched
road network, coloured by log(trip_count) and projected to EPSG:3347
(Statistics Canada Lambert; conformal across most of North America,
with mild stretching in the southern US).

Usage::

    uv run --with geopandas --with plotnine --with shapely --with matplotlib \\
        python examples/plot_way_counts.py \\
        data/way_counts.parquet \\
        data/way_counts.png
"""

from __future__ import annotations

import sys
from pathlib import Path

import geopandas as gpd
import plotnine as pn
import polars as pl
from shapely import wkb


def load_gdf(parquet_path: Path) -> gpd.GeoDataFrame:
    df = (
        pl.scan_parquet(parquet_path)
        .select("trip_count", "geometry")
        .drop_nulls("geometry")
        .rename({"trip_count": "n"})
        .collect()
    )
    gdf = gpd.GeoDataFrame(
        df.drop("geometry").to_pandas(),
        geometry=[wkb.loads(b) for b in df["geometry"]],
        crs="EPSG:4326",
    ).to_crs(3347)
    gdf = gdf.explode(index_parts=False)
    return gdf[gdf.geom_type == "LineString"]


def build_plot(gdf: gpd.GeoDataFrame) -> pn.ggplot:
    return (
        pn.ggplot()
        + pn.geom_map(gdf.sort_values("n"), pn.aes(colour="n"), size=0.05)
        + pn.theme_void()
        + pn.scale_color_cmap(cmap_name="magma_r", trans="log")
        + pn.theme(legend_position="none")
        + pn.coord_fixed()
    )


def main() -> None:
    if len(sys.argv) != 3:
        print(
            f"Usage: {sys.argv[0]} <way_counts.parquet> <output.png>",
            file=sys.stderr,
        )
        sys.exit(1)
    parquet_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2])

    gdf = load_gdf(parquet_path)
    print(f"{len(gdf):,} LineString segments after explode")

    p = build_plot(gdf)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # CA + US in EPSG:3347 is ~1.3× wider than tall; coord_fixed handles
    # the exact aspect, so this just sets the bounding canvas.
    p.save(output_path, width=18, height=14, dpi=300)
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
