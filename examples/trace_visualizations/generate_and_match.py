"""Generate a fake truck trip, split, map-match, and visualize it.

End-to-end example: synthesize a GPS trace between two Ontario points,
then drive the rest of the workflow through :mod:`trucktrack.inspect`:
split at observation gaps / stops, map-match each trip, evaluate match
quality, and plot all stages on one interactive map.

Requires pyvalhalla and a tile extract built by scripts/setup_valhalla.py.

Usage::

    # Save to HTML file:
    uv run python examples/trace_visualizations/generate_and_match.py

    # Serve via Flask (useful inside k8s notebooks):
    uv run python examples/trace_visualizations/generate_and_match.py --serve

Requires a ``valhalla.json`` in cwd.
"""

from __future__ import annotations

import argparse
import os
import tempfile
from datetime import UTC, datetime, timedelta
from pathlib import Path

from trucktrack import (
    generate_trace,
    read_parquet,
    traces_to_parquet,
)
from trucktrack import (
    inspect as tt_inspect,
)
from trucktrack.generate import TripConfig
from trucktrack.valhalla._actor import _find_config
from trucktrack.visualize import save_map, serve_map

OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", "examples/trace_visualizations/output"))

# FCA Brampton Assembly Plant → Comber Petro-Canada
ORIGIN = (43.7387, -79.7271)
DESTINATION = (42.2383, -82.5506)


def main(args: argparse.Namespace) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)

        print("Generating trace...")
        config = TripConfig(
            origin=ORIGIN,
            destination=DESTINATION,
            departure_time=datetime(2025, 6, 15, 8, 0, tzinfo=UTC),
            seed=42,
            config=_find_config(),
        )
        points = generate_trace(config)
        print(f"  {len(points)} trace points generated")

        parquet_path = tmp_dir / "trace.parquet"
        traces_to_parquet([(points, config.trip_id)], str(parquet_path))
        df = read_parquet(str(parquet_path))
        print(f"  DataFrame: {df.shape[0]} rows, columns: {df.columns}")

        print("Splitting...")
        split = tt_inspect.split_trips(
            df,
            gap=timedelta(minutes=5),
            stop_max_diameter=50.0,
            stop_min_duration=timedelta(minutes=2),
        )
        n_trips = split.filter(~split["is_stop"])["segment_id"].n_unique()
        n_stops = split.filter(split["is_stop"])["segment_id"].n_unique()
        print(f"  {n_trips} trip(s), {n_stops} stop(s)")

        print("Map-matching...")
        trips = tt_inspect.map_match_trips(split)
        for sid, tm in trips.items():
            print(
                f"  segment {sid}: {tm.matched_df.height} pts, "
                f"{len(tm.way_ids)} OSM ways"
            )

        quality = tt_inspect.evaluate_quality(split, trips=trips)
        print("\nQuality report:")
        print(quality)

        print("Building map...")
        m = tt_inspect.plot_inspection(df, split, trips)

        all_ways = [w for tm in trips.values() for w in tm.way_ids]
        print(f"\nOSM way IDs ({len(all_ways)} total):")
        for wid in all_ways[:10]:
            print(f"  {wid}")
        if len(all_ways) > 10:
            print(f"  ... ({len(all_ways) - 10} more)")

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
