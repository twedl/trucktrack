"""Generate a fake truck trip, split, partition, and map-match it.

End-to-end example: synthesize a GPS trace between two Ontario points,
split it at observation gaps, partition into a hive layout, map-match
each segment, and print the OSM way IDs traversed.

Requires pyvalhalla and a tile extract built by scripts/setup_valhalla.py.

Usage::

    VALHALLA_TILE_EXTRACT=valhalla_tiles/valhalla_tiles.tar \
        uv run python examples/generate_and_match.py
"""

from __future__ import annotations

import os
import tempfile
from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl
from trucktrack import (
    generate_trace,
    partition_existing_parquet,
    read_parquet,
    split_by_observation_gap,
    traces_to_parquet,
)
from trucktrack.generate import TripConfig
from trucktrack.valhalla import map_match_ways

TILE_EXTRACT = os.environ.get(
    "VALHALLA_TILE_EXTRACT", "valhalla_tiles/valhalla_tiles.tar"
)

# Toronto → London, Ontario
ORIGIN = (43.65, -79.38)
DESTINATION = (42.98, -81.25)


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)

        # 1. Generate a synthetic truck trip.
        print("Generating trace...")
        config = TripConfig(
            origin=ORIGIN,
            destination=DESTINATION,
            departure_time=datetime(2025, 6, 15, 8, 0, tzinfo=UTC),
            seed=42,
            tile_extract=TILE_EXTRACT,
        )
        points = generate_trace(config)
        print(f"  {len(points)} trace points generated")

        # 2. Write to parquet so we can use the DataFrame pipeline.
        parquet_path = tmp_dir / "trace.parquet"
        traces_to_parquet([(points, config.trip_id)], str(parquet_path))
        df = read_parquet(str(parquet_path))
        print(f"  DataFrame: {df.shape[0]} rows, columns: {df.columns}")

        # 3. Split at 5-minute observation gaps.
        print("Splitting at observation gaps...")
        split = split_by_observation_gap(df, timedelta(minutes=5), min_length=3)
        n_segments = split["segment_id"].n_unique()
        print(f"  {n_segments} segment(s)")

        # 4. Partition into a hive layout.
        print("Partitioning...")
        split_path = tmp_dir / "split.parquet"
        split.write_parquet(split_path)
        partition_dir = tmp_dir / "partitioned"
        summary = partition_existing_parquet(split_path, partition_dir)
        for tier, n in sorted(summary.items()):
            print(f"  {tier}: {n} partition(s)")

        # 5. Map-match each segment and collect OSM way IDs.
        print("Map-matching...")
        all_ways: list[int] = []
        for pq in sorted(partition_dir.rglob("*.parquet")):
            chunk = pl.read_parquet(pq)
            for (seg_id,), seg in chunk.group_by("segment_id"):
                lats = seg["lat"].to_list()
                lons = seg["lon"].to_list()
                coords = list(zip(lats, lons, strict=True))
                ways = map_match_ways(coords, tile_extract=TILE_EXTRACT)
                all_ways.extend(ways)
                print(f"  segment {seg_id}: {len(seg)} pts, {len(ways)} OSM ways")

        # 6. Print the OSM way IDs.
        print(f"\nOSM way IDs ({len(all_ways)} ways):")
        for wid in all_ways[:10]:
            print(f"  {wid}")
        if len(all_ways) > 10:
            print(f"  ... ({len(all_ways) - 10} more)")


if __name__ == "__main__":
    main()
