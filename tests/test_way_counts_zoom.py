"""Tests for examples/way_counts_zoom.py — bbox-pruned loader."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

pytest.importorskip("duckdb")
gpd = pytest.importorskip("geopandas")
pyproj = pytest.importorskip("pyproj")
pytest.importorskip("shapely")

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "examples"))

from shapely.geometry import LineString  # noqa: E402
from way_counts_zoom import load_zoom  # noqa: E402


@pytest.fixture
def synthetic_parquet(tmp_path: Path) -> Path:
    """One LineString in Toronto, one in Vancouver, with bbox/centroid cols."""
    gdf = gpd.GeoDataFrame(
        {"way_id": [1, 2], "trip_count": [100, 5]},
        geometry=[
            LineString([(-79.4, 43.6), (-79.3, 43.7)]),
            LineString([(-123.1, 49.2), (-123.0, 49.3)]),
        ],
        crs="EPSG:4326",
    )
    bounds = gdf.geometry.bounds
    gdf["bbox_xmin"] = bounds["minx"]
    gdf["bbox_ymin"] = bounds["miny"]
    gdf["bbox_xmax"] = bounds["maxx"]
    gdf["bbox_ymax"] = bounds["maxy"]
    # centroid_* columns exist in the real schema but load_zoom doesn't
    # read them, so we omit them here to avoid the geographic-CRS warning.

    path = tmp_path / "way_counts.parquet"
    gdf.to_parquet(path)
    return path


class TestLoadZoom:
    def test_no_filter_returns_all(self, synthetic_parquet: Path) -> None:
        gdf = load_zoom(synthetic_parquet)
        assert len(gdf) == 2
        assert sorted(gdf["way_id"].tolist()) == [1, 2]

    def test_returns_geodataframe_in_4326(self, synthetic_parquet: Path) -> None:
        gdf = load_zoom(synthetic_parquet)
        assert isinstance(gdf, gpd.GeoDataFrame)
        assert gdf.crs.to_epsg() == 4326

    def test_wgs84_bbox_filters(self, synthetic_parquet: Path) -> None:
        gdf = load_zoom(synthetic_parquet, bbox=(-79.5, 43.5, -79.2, 43.8))
        assert gdf["way_id"].tolist() == [1]

    def test_wgs84_bbox_excludes_outside(self, synthetic_parquet: Path) -> None:
        gdf = load_zoom(synthetic_parquet, bbox=(0.0, 0.0, 1.0, 1.0))
        assert len(gdf) == 0

    def test_projected_bbox_is_transformed(self, synthetic_parquet: Path) -> None:
        # Same Toronto window expressed in EPSG:3347.
        transformer = pyproj.Transformer.from_crs(4326, 3347, always_xy=True)
        x0, y0 = transformer.transform(-79.5, 43.5)
        x1, y1 = transformer.transform(-79.2, 43.8)
        gdf = load_zoom(
            synthetic_parquet,
            bbox=(min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)),
            bbox_crs="EPSG:3347",
        )
        assert gdf["way_id"].tolist() == [1]

    def test_min_trip_count(self, synthetic_parquet: Path) -> None:
        gdf = load_zoom(synthetic_parquet, min_trip_count=10)
        assert gdf["way_id"].tolist() == [1]

    def test_top_n(self, synthetic_parquet: Path) -> None:
        gdf = load_zoom(synthetic_parquet, top_n=1)
        assert gdf["way_id"].tolist() == [1]

    def test_to_crs_reprojects(self, synthetic_parquet: Path) -> None:
        gdf = load_zoom(synthetic_parquet, to_crs="EPSG:3347")
        assert gdf.crs.to_epsg() == 3347

    def test_results_sorted_by_trip_count_desc(self, synthetic_parquet: Path) -> None:
        gdf = load_zoom(synthetic_parquet)
        assert gdf["trip_count"].tolist() == sorted(
            gdf["trip_count"].tolist(), reverse=True
        )
