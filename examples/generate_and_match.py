"""Generate a fake truck trip, split, partition, and map-match it.

End-to-end example: synthesize a GPS trace between two Ontario points,
split it at observation gaps, partition into a hive layout, map-match
each segment, and print the matched polyline.

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
from trucktrack.valhalla import map_match_dataframe

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

        # 5. Map-match each segment.
        print("Map-matching...")
        matched_parts = []
        for pq in sorted(partition_dir.rglob("*.parquet")):
            chunk = pl.read_parquet(pq)
            for seg_id, seg in chunk.group_by("segment_id"):
                matched = map_match_dataframe(seg, tile_extract=TILE_EXTRACT)
                matched_parts.append(matched)
                print(
                    f"  segment {seg_id[0]}: "
                    f"{len(seg)} pts, "
                    f"mean snap distance {matched['distance_from_trace'].mean():.1f} m"
                )

        result = pl.concat(matched_parts)
        print(f"\nResult: {result.shape[0]} matched points")
        print(result.select("lat", "lon", "matched_lat", "matched_lon", "distance_from_trace").head(10))

        # 6. Print the matched polyline as a coordinate list.
        coords = list(
            zip(
                result["matched_lat"].to_list(),
                result["matched_lon"].to_list(),
                strict=True,
            )
        )
        print(f"\nMatched polyline ({len(coords)} points):")
        for lat, lon in coords[:5]:
            print(f"  ({lat:.6f}, {lon:.6f})")
        if len(coords) > 5:
            print(f"  ... ({len(coords) - 5} more)")


if __name__ == "__main__":
    main()
