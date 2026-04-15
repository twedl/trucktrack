"""Generate a trip with a simulated traffic jam, filter it out, and visualize.

Same as traffic_jam.py but applies filter_traffic_stops after stop
splitting, which should reclassify the traffic-jam stop as movement
and merge it back into a single trip segment.

Usage::

    uv run python examples/trace_visualizations/traffic_jam_filtered.py

Requires a ``valhalla.json`` in cwd.
"""

from __future__ import annotations

import argparse
import os
import tempfile
from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl
from traffic_jam import inject_traffic_jam
from trucktrack import (
    filter_traffic_stops,
    generate_trace,
    read_parquet,
    split_by_observation_gap,
    split_by_stops,
    traces_to_parquet,
)
from trucktrack.generate import TripConfig
from trucktrack.valhalla import find_config, map_match_dataframe_full
from trucktrack.visualize import plot_trace_layers, save_map, serve_map

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
            config=find_config(),
        )
        points = generate_trace(config)
        print(f"  {len(points)} trace points generated")

        # 2. Inject a traffic jam in the middle.
        print("Injecting traffic jam...")
        points = inject_traffic_jam(points, n_jam_points=20)
        print(f"  {len(points)} points after injection")

        # 3. Write to parquet.
        parquet_path = tmp_dir / "trace.parquet"
        traces_to_parquet([(points, config.trip_id)], str(parquet_path))
        df = read_parquet(str(parquet_path))

        # 4. Split at observation gaps, then detect stops.
        print("Splitting...")
        gap_split = split_by_observation_gap(df, timedelta(minutes=5), min_length=3)
        stop_split = split_by_stops(
            gap_split,
            max_diameter=50.0,
            min_duration=timedelta(minutes=2),
        )
        n_before = stop_split.filter(pl.col("is_stop"))["segment_id"].n_unique()
        print(f"  Before filter: {n_before} stop(s)")

        # 5. Apply the traffic filter to reclassify the jam as movement.
        split = filter_traffic_stops(stop_split)
        n_segments = split["segment_id"].n_unique()
        n_stops = split.filter(pl.col("is_stop"))["segment_id"].n_unique()
        n_trips = n_segments - n_stops
        print(
            f"  After filter: {n_segments} segment(s): "
            f"{n_trips} trip(s), {n_stops} stop(s)"
        )

        # 6. Map-match the movement segments.
        print("Map-matching...")
        movement = split.filter(~pl.col("is_stop"))
        matched_parts = []
        all_ways: list[int] = []
        all_shapes: list[list[tuple[float, float]]] = []
        for (seg_id,), seg in movement.group_by("segment_id", maintain_order=True):
            matched, ways, shape = map_match_dataframe_full(seg)
            matched_parts.append(matched)
            all_ways.extend(ways)
            all_shapes.extend(shape)
            print(f"  segment {seg_id}: {len(seg)} pts, {len(ways)} OSM ways")

        result = pl.concat(matched_parts) if matched_parts else None

        # 7. Visualize.
        print("Building map...")
        m = plot_trace_layers(
            raw=df,
            segments=split,
            matched=result,
            matched_shape=all_shapes or None,
        )

        # 8. Print summary.
        print(f"\nOSM way IDs ({len(all_ways)} ways):")
        for wid in all_ways[:10]:
            print(f"  {wid}")
        if len(all_ways) > 10:
            print(f"  ... ({len(all_ways) - 10} more)")

        # 9. Save or serve.
        if args.serve:
            serve_map(m, host="0.0.0.0", port=args.port)
        else:
            out_path = OUTPUT_DIR / "traffic_jam_filtered.html"
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
