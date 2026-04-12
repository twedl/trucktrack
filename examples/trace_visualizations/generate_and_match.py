"""Generate a fake truck trip, split, partition, map-match, and visualize it.

End-to-end example: synthesize a GPS trace between two Ontario points,
split it at observation gaps, partition into a hive layout, map-match
each segment, and visualize all three stages on an interactive map.

Requires pyvalhalla and a tile extract built by scripts/setup_valhalla.py.

Usage::

    # Save to HTML file:
    VALHALLA_TILE_EXTRACT=valhalla_tiles/valhalla_tiles.tar \
        uv run python examples/trace_visualizations/generate_and_match.py

    # Serve via Flask (useful inside k8s notebooks):
    VALHALLA_TILE_EXTRACT=valhalla_tiles/valhalla_tiles.tar \
        uv run python examples/trace_visualizations/generate_and_match.py --serve
"""

from __future__ import annotations

import argparse
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
    split_by_stops,
    traces_to_parquet,
)
from trucktrack.generate import TripConfig
from trucktrack.valhalla import map_match_dataframe_full
from trucktrack.visualize import plot_trace_layers, save_map, serve_map

TILE_EXTRACT = os.environ.get(
    "VALHALLA_TILE_EXTRACT", "valhalla_tiles/valhalla_tiles.tar"
)
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", "examples/trace_visualizations/output"))

# FCA Brampton Assembly Plant → Comber Petro-Canada
ORIGIN = (43.7387, -79.7271)
DESTINATION = (42.2383, -82.5506)


def main(args: argparse.Namespace) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

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

        # 3. Split at 5-minute observation gaps, then detect stops.
        print("Splitting...")
        gap_split = split_by_observation_gap(df, timedelta(minutes=5), min_length=3)
        split = split_by_stops(
            gap_split,
            max_diameter=50.0,
            min_duration=timedelta(minutes=2),
        )
        n_segments = split["segment_id"].n_unique()
        n_stops = split.filter(pl.col("is_stop"))["segment_id"].n_unique()
        print(f"  {n_segments} segment(s), {n_stops} stop(s)")

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
        matched_parts = []
        all_ways: list[int] = []
        all_shapes: list[list[tuple[float, float]]] = []
        for pq in sorted(partition_dir.rglob("*.parquet")):
            chunk = pl.read_parquet(pq)
            for (seg_id,), seg in chunk.group_by("segment_id"):
                matched, ways, shape = map_match_dataframe_full(
                    seg, tile_extract=TILE_EXTRACT
                )
                matched_parts.append(matched)
                all_ways.extend(ways)
                if shape:
                    all_shapes.append(shape)
                print(f"  segment {seg_id}: {len(seg)} pts, {len(ways)} OSM ways")

        result = pl.concat(matched_parts)

        # 6. Visualize all stages on one map.
        print("Building map...")
        m = plot_trace_layers(
            raw=df, segments=split, matched=result, matched_shape=all_shapes or None
        )

        # 7. Print the OSM way IDs.
        print(f"\nOSM way IDs ({len(all_ways)} ways):")
        for wid in all_ways[:10]:
            print(f"  {wid}")
        if len(all_ways) > 10:
            print(f"  ... ({len(all_ways) - 10} more)")

        # 8. Save or serve the map (serve_map blocks until killed).
        if args.serve:
            serve_map(m, host="0.0.0.0", port=args.port)
        else:
            out_path = OUTPUT_DIR / "trace.html"
            save_map(m, out_path)
            print(f"  Saved {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--serve",
        action="store_true",
        help="Serve the map via Flask instead of saving to HTML",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=5000,
        help="Port for Flask server (default: 5000)",
    )
    args = parser.parse_args()
    main(args)
