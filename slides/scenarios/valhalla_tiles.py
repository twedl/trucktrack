"""Extract Valhalla tile coverage from a trucktrack tile archive.

Valhalla partitions the world into a grid at three hierarchy levels:

    level 0 (highway):  4°    tiles, 90 × 45    =  4 050 tiles globally
    level 1 (arterial): 1°    tiles, 360 × 180  = 64 800 tiles globally
    level 2 (local):    0.25° tiles, 1440 × 720 = 1 036 800 tiles globally

Each tile is one .gph file inside the tar archive, named
``<level>/<aaa>/<bbb>[/<ccc>].gph``.  Concatenating the digits and
dropping leading zeros gives the tile ID; from the tile ID and level
size we recover the SW corner of the tile in (lat, lon).

This script reads the tar, builds a frame with one row per tile
(level, tile_id, lat0, lon0, lat1, lon1, bytes), and writes it to
slides/valhalla/data/tile_coverage.parquet.

    uv run python scenarios/valhalla_tiles.py
"""

from __future__ import annotations

import re
import tarfile
from pathlib import Path

import polars as pl

SLIDES = Path(__file__).resolve().parent.parent
REPO = SLIDES.parent
TILE_TAR = REPO / "valhalla_tiles" / "valhalla_tiles.tar"
OUT = SLIDES / "valhalla" / "data" / "tile_coverage.parquet"

# (size_deg, cols) per level — derived from Valhalla's hardcoded grid.
LEVELS = {
    0: (4.0, 90),
    1: (1.0, 360),
    2: (0.25, 1440),
}

PATH_RE = re.compile(r"^(\d)/(\d{3})/(\d{3})(?:/(\d{3}))?\.gph$")


def _parse(name: str) -> tuple[int, int] | None:
    m = PATH_RE.match(name)
    if not m:
        return None
    level = int(m.group(1))
    digits = m.group(2) + m.group(3) + (m.group(4) or "")
    return level, int(digits)


def _bbox(level: int, tile_id: int) -> tuple[float, float, float, float]:
    size, cols = LEVELS[level]
    col = tile_id % cols
    row = tile_id // cols
    lon0 = -180 + col * size
    lat0 = -90 + row * size
    return lat0, lon0, lat0 + size, lon0 + size


def main() -> None:
    rows: list[dict] = []
    with tarfile.open(TILE_TAR) as tar:
        for member in tar:
            if not member.isfile():
                continue
            parsed = _parse(member.name)
            if not parsed:
                continue
            level, tid = parsed
            lat0, lon0, lat1, lon1 = _bbox(level, tid)
            rows.append({
                "level": level,
                "tile_id": tid,
                "lat0": lat0,
                "lon0": lon0,
                "lat1": lat1,
                "lon1": lon1,
                "bytes": int(member.size),
            })

    df = pl.DataFrame(rows).sort("level", "tile_id")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(OUT)

    print(f"  {OUT.relative_to(SLIDES)}: {df.height} tiles")
    summary = (
        df.group_by("level")
        .agg([
            pl.len().alias("tiles"),
            (pl.col("bytes").sum() / 1024 / 1024).round(1).alias("mb"),
            pl.col("lat0").min().alias("lat_min"),
            pl.col("lat1").max().alias("lat_max"),
            pl.col("lon0").min().alias("lon_min"),
            pl.col("lon1").max().alias("lon_max"),
        ])
        .sort("level")
    )
    print()
    with pl.Config(tbl_rows=10, tbl_cols=10):
        print(summary)


if __name__ == "__main__":
    main()
