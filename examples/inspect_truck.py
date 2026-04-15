"""Inspect a single truck over a date range end-to-end.

The workflow:
    1. Pick a truck_id and date range.
    2. Split at observation gaps / stops, filter traffic-jam stops.
    3. Map-match route shape + OSM way IDs per trip.
    4. Evaluate match quality per trip.
    5. Plot raw data, trips/stops, and map-matched shapes on one map.
    6. Save or serve the map.

Tune the parameters in CONFIG and re-run to iterate.

Usage (against the committed sample dataset at ``data/trucks/``)::

    uv run python examples/inspect_truck.py \
        --data-dir data/trucks \
        --truck-id aaaaaaaaaaaaaaaaaaaaaaaaaaaaaa01 \
        --start 2026-01-01 --end 2026-01-03

Requires a ``valhalla.json`` in cwd.  Regenerate or extend
``data/trucks/`` via ``scripts/generate_sample_trucks.py``.
"""

from __future__ import annotations

import argparse
from datetime import timedelta
from pathlib import Path

from trucktrack import inspect as tt
from trucktrack.visualize import save_map, serve_map

CONFIG = {
    "gap": timedelta(minutes=5),
    "stop_max_diameter": 50.0,
    "stop_min_duration": timedelta(minutes=2),
    "traffic_max_angle_change": 30.0,
    "traffic_min_distance": 10.0,
}

# Valhalla Meili defaults — tweak any key to change matching behaviour.
# See DEFAULT_TRACE_OPTIONS in trucktrack.valhalla.map_matching for source.
TRACE_OPTIONS: dict[str, object] = {
    "search_radius": 50,  # meters, per-point candidate search
    "gps_accuracy": 25,  # meters, emission noise (Meili sigma_z)
    "interpolation_distance": 20,  # meters, fold points closer than this
    "max_route_distance_factor": 10,  # max detour between consecutive fixes
    "max_route_time_factor": 10,
    "beta": 5,  # transition smoothing; higher = stickier to prev edge
    "turn_penalty_factor": 500,  # cost per U-turn; 0 = permissive
    # "breakage_distance": 3000,  # override Valhalla's default; leave commented
    # to keep the adaptive per-call value from
    # trucktrack.valhalla.map_matching
}


def main(args: argparse.Namespace) -> None:
    raw = tt.load_truck_trace(
        args.truck_id, args.start, args.end, data_dir=args.data_dir
    )
    print(f"loaded {raw.height} points")

    split = tt.split_trips(raw, **CONFIG)
    n_trips = split.filter(~split["is_stop"])["segment_id"].n_unique()
    n_stops = split.filter(split["is_stop"])["segment_id"].n_unique()
    print(f"{n_trips} trip(s), {n_stops} stop(s)")

    trips = tt.map_match_trips(split, trace_options=TRACE_OPTIONS)
    for sid, tm in trips.items():
        print(f"  trip {sid}: {tm.matched_df.height} pts, {len(tm.way_ids)} ways")

    # Cached path — reuses trips so Valhalla isn't called a second time.
    quality = tt.evaluate_quality(split, trips=trips)
    print("\nquality:")
    print(quality)

    m = tt.plot_inspection(raw, split, trips)

    if args.serve:
        serve_map(m, host=args.host, port=args.port)
    else:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        save_map(m, out)
        print(f"saved {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True, help="raw hive layout root")
    parser.add_argument("--truck-id", required=True)
    parser.add_argument("--start", required=True, help="ISO datetime, e.g. 2026-01-01")
    parser.add_argument("--end", required=True, help="ISO datetime (exclusive)")
    parser.add_argument("--output", default="inspect_truck.html")
    parser.add_argument("--serve", action="store_true")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5000)
    main(parser.parse_args())
