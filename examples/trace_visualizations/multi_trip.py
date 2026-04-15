"""Generate three back-to-back trips with an observation gap, and visualize.

Simulates a truck driving between four random Ontario locations.
Between trip 1 and trip 2 there is a 5-hour observation gap (driver
rest / device off), which the gap splitter should detect.  All three
trips share a single vehicle ID so they appear in one DataFrame.

Usage::

    uv run python examples/trace_visualizations/multi_trip.py

Requires a ``valhalla.json`` in cwd.
"""

from __future__ import annotations

import argparse
import os
import random
import tempfile
from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl
from trucktrack import (
    generate_trace,
    read_parquet,
    split_by_observation_gap,
    split_by_stops,
    traces_to_parquet,
)
from trucktrack.generate import TripConfig
from trucktrack.generate.models import TracePoint
from trucktrack.valhalla import map_match_dataframe_full, map_match_route_shape
from trucktrack.valhalla._actor import _find_config
from trucktrack.visualize import plot_trace_layers, save_map, serve_map

OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", "examples/trace_visualizations/output"))

# Four Ontario locations (short hops to stay within Valhalla limits)
WAYPOINTS = [
    (43.7387, -79.7271),  # Brampton (FCA Assembly Plant)
    (43.2557, -79.8711),  # Hamilton (Stelco)
    (43.4643, -80.5204),  # Kitchener (Fairview Park Mall)
    (43.5448, -80.2482),  # Guelph (Stone Road Mall)
]


def generate_multi_trip(
    waypoints: list[tuple[float, float]],
    *,
    departure: datetime,
    gap_after_trip: int = 1,
    gap_duration: timedelta = timedelta(hours=5),
    seed: int = 42,
) -> list[TracePoint]:
    """Generate consecutive trips, inserting an observation gap after one trip."""
    rng = random.Random(seed)
    all_points: list[TracePoint] = []
    current_time = departure
    valhalla_config = _find_config()

    for i in range(len(waypoints) - 1):
        config = TripConfig(
            origin=waypoints[i],
            destination=waypoints[i + 1],
            departure_time=current_time,
            seed=rng.randint(0, 2**31),
            config=valhalla_config,
            gps_noise_meters=1.0,
            errors=[],
        )
        points = generate_trace(config)
        all_points.extend(points)
        print(f"  Trip {i + 1}: {waypoints[i]} → {waypoints[i + 1]}, {len(points)} pts")

        if points:
            current_time = points[-1].timestamp + timedelta(minutes=5)

        # Insert observation gap after the specified trip.
        if i + 1 == gap_after_trip:
            print(f"  [5-hour observation gap after trip {gap_after_trip}]")
            current_time += gap_duration

    return all_points


def main(args: argparse.Namespace) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)

        # 1. Generate three back-to-back trips.
        print("Generating trips...")
        all_points = generate_multi_trip(
            WAYPOINTS,
            departure=datetime(2025, 6, 15, 6, 0, tzinfo=UTC),
        )
        print(f"  {len(all_points)} total points")

        # 2. Write to parquet with a single vehicle ID.
        parquet_path = tmp_dir / "trace.parquet"
        traces_to_parquet([(all_points, "truck_01")], str(parquet_path))
        df = read_parquet(str(parquet_path))

        # 3. Split at observation gaps, then detect stops.
        print("Splitting...")
        gap_split = split_by_observation_gap(df, timedelta(minutes=3), min_length=3)
        n_gap_segs = gap_split["segment_id"].n_unique()
        print(f"  Gap split: {n_gap_segs} segment(s)")

        split = split_by_stops(
            gap_split,
            max_diameter=50.0,
            min_duration=timedelta(minutes=2),
        )
        n_segments = split["segment_id"].n_unique()
        n_stops = split.filter(pl.col("is_stop"))["segment_id"].n_unique()
        n_trips = n_segments - n_stops
        print(f"  Stop split: {n_trips} trip(s), {n_stops} stop(s)")

        # 4. Map-match the movement segments.
        print("Map-matching...")
        movement = split.filter(~pl.col("is_stop"))
        matched_parts = []
        all_ways: list[int] = []
        all_shapes: list[list[tuple[float, float]]] = []
        for (seg_id,), seg in movement.group_by("segment_id", maintain_order=True):
            matched, ways, _ = map_match_dataframe_full(seg)
            pts = list(zip(seg["lat"].to_list(), seg["lon"].to_list(), strict=True))
            shapes = map_match_route_shape(pts)
            matched_parts.append(matched)
            all_ways.extend(ways)
            all_shapes.extend(shapes)
            print(f"  segment {seg_id}: {len(seg)} pts, {len(ways)} OSM ways")

        result = pl.concat(matched_parts) if matched_parts else None

        # 5. Visualize.
        print("Building map...")
        m = plot_trace_layers(
            raw=df,
            segments=split,
            matched=result,
            matched_shape=all_shapes or None,
        )

        # 6. Print summary.
        print(f"\nOSM way IDs ({len(all_ways)} ways):")
        for wid in all_ways[:10]:
            print(f"  {wid}")
        if len(all_ways) > 10:
            print(f"  ... ({len(all_ways) - 10} more)")

        # 7. Save or serve.
        if args.serve:
            serve_map(m, host="0.0.0.0", port=args.port)
        else:
            out_path = OUTPUT_DIR / "multi_trip.html"
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
