"""Generate a Toronto -> Ottawa trace with a ~20 km geofence gap, then
split into trips/stops, map-match, and visualize.

The trace is generated with only the ``geofence_gap`` operational error
enabled, centered roughly on Kingston with a 10 km radius (20 km diameter).
All points whose location falls inside that circle are removed, leaving a
large geofencing gap along Hwy 401 for downstream map-matching to handle.

Usage::

    VALHALLA_TILE_EXTRACT=valhalla_tiles/valhalla_tiles.tar \
        uv run python examples/trace_visualizations/geofence_gap_toronto_ottawa.py

    # Or serve interactively:
    VALHALLA_TILE_EXTRACT=valhalla_tiles/valhalla_tiles.tar \
        uv run python examples/trace_visualizations/geofence_gap_toronto_ottawa.py --serve
"""

from __future__ import annotations

import argparse
import os
import tempfile
from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl
from trucktrack import (
    ErrorConfig,
    TripConfig,
    generate_trace,
    read_parquet,
    split_by_observation_gap,
    split_by_stops,
    traces_to_parquet,
)
from trucktrack.valhalla import map_match_dataframe_full
from trucktrack.visualize import plot_trace_layers, save_map, serve_map

TILE_EXTRACT = os.environ.get(
    "VALHALLA_TILE_EXTRACT", "valhalla_tiles/valhalla_tiles.tar"
)
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", "examples/trace_visualizations/output"))

# Toronto (CN Tower) -> Ottawa (Parliament Hill)
ORIGIN = (43.6426, -79.3871)
DESTINATION = (45.4236, -75.7009)

# Roughly on Hwy 401 near Kingston; 20 km radius removes ~40 km of route.
GEOFENCE_CENTER = (44.2312, -76.4860)
GEOFENCE_RADIUS_M = 20_000.0


def main(args: argparse.Namespace) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)

        # 1. Generate trace with a 20 km geofence gap near Kingston.
        config = TripConfig(
            origin=ORIGIN,
            destination=DESTINATION,
            departure_time=datetime(2025, 6, 15, 8, 0, tzinfo=UTC),
            seed=42,
            tile_extract=TILE_EXTRACT,
            errors=[
                ErrorConfig(
                    "geofence_gap",
                    probability=1.0,
                    params={
                        "center": GEOFENCE_CENTER,
                        "radius_m": GEOFENCE_RADIUS_M,
                    },
                ),
            ],
        )
        print(f"Generating Toronto -> Ottawa trace (seed={config.seed})...")
        points = generate_trace(config)
        print(f"  {len(points)} trace points after geofence gap")

        # 2. Write to parquet and reload as a DataFrame.
        parquet_path = tmp_dir / "trace.parquet"
        traces_to_parquet([(points, config.trip_id)], str(parquet_path))
        df = read_parquet(str(parquet_path))

        # 3. Split on the large observation gap, then detect stops.
        print("Splitting...")
        gap_split = split_by_observation_gap(df, timedelta(minutes=5), min_length=3)
        split = split_by_stops(
            gap_split,
            max_diameter=50.0,
            min_duration=timedelta(minutes=2),
        )
        n_segments = split["segment_id"].n_unique()
        n_stops = split.filter(pl.col("is_stop"))["segment_id"].n_unique()
        n_trips = n_segments - n_stops
        print(f"  {n_segments} segment(s): {n_trips} trip(s), {n_stops} stop(s)")

        # 4. Map-match each movement segment.
        print("Map-matching...")
        movement = split.filter(~pl.col("is_stop"))
        matched_parts: list[pl.DataFrame] = []
        all_ways: list[int] = []
        all_shapes: list[list[tuple[float, float]]] = []
        for (seg_id,), seg in movement.group_by("segment_id", maintain_order=True):
            matched, ways, shape = map_match_dataframe_full(
                seg, tile_extract=TILE_EXTRACT
            )
            matched_parts.append(matched)
            all_ways.extend(ways)
            if shape:
                all_shapes.append(shape)
            print(f"  segment {seg_id}: {len(seg)} pts, {len(ways)} OSM ways")

        result = pl.concat(matched_parts) if matched_parts else None

        # 5. Visualize all layers together.
        print("Building map...")
        m = plot_trace_layers(
            raw=df,
            segments=split,
            matched=result,
            matched_shape=all_shapes or None,
        )

        # 6. Save or serve.
        if args.serve:
            serve_map(m, host="0.0.0.0", port=args.port)
        else:
            out_path = OUTPUT_DIR / "geofence_gap_toronto_ottawa.html"
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
    main(parser.parse_args())
