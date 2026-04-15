"""Experiment: iterate on meili parameters to minimize map-matching time.

Generates the same Toronto → Hamilton trip and times trace_route calls
with varying parameter combinations.

Usage::

    uv run python examples/trace_visualizations/experiment_meili.py

Requires a ``valhalla.json`` in cwd.
"""

from __future__ import annotations

import json
import tempfile
import time
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
from trucktrack.valhalla import find_config, get_actor
from trucktrack.valhalla._parsing import decode_polyline6

ORIGIN = (43.6532, -79.3832)  # Toronto
DESTINATION = (43.2557, -79.8711)  # Hamilton


def get_movement_points(
    *,
    gps_noise_meters: float = 1.0,
    use_errors: bool = False,
) -> list[tuple[float, float]]:
    """Generate trace and return the movement segment points."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        config = TripConfig(
            origin=ORIGIN,
            destination=DESTINATION,
            departure_time=datetime(2025, 6, 15, 8, 0, tzinfo=UTC),
            seed=42,
            config=find_config(),
            gps_noise_meters=gps_noise_meters,
            errors=None if use_errors else [],  # None = default profile
        )
        points = generate_trace(config)
        parquet_path = tmp_dir / "trace.parquet"
        traces_to_parquet([(points, config.trip_id)], str(parquet_path))
        df = read_parquet(str(parquet_path))
        gap_split = split_by_observation_gap(df, timedelta(minutes=5), min_length=3)
        split = split_by_stops(
            gap_split, max_diameter=50.0, min_duration=timedelta(minutes=2)
        )
        movement = split.filter(~pl.col("is_stop"))
        return list(
            zip(movement["lat"].to_list(), movement["lon"].to_list(), strict=True)
        )


def run_trace_route(
    actor,
    pts: list[tuple[float, float]],
    trace_options: dict[str, object],
) -> tuple[int, float]:
    """Run trace_route and return (n_fragments, elapsed_ms)."""
    body = {
        "shape": [{"lat": lat, "lon": lon} for lat, lon in pts],
        "costing": "auto",
        "shape_match": "map_snap",
        "trace_options": trace_options,
    }
    t0 = time.perf_counter()
    resp = json.loads(actor.trace_route(json.dumps(body)))
    elapsed_ms = (time.perf_counter() - t0) * 1000

    n_fragments = 1 if resp.get("trip", {}).get("legs") else 0
    n_fragments += len(resp.get("alternates", []))
    return n_fragments, elapsed_ms


def run_trace_attributes(
    actor,
    pts: list[tuple[float, float]],
    trace_options: dict[str, object],
) -> tuple[int, float]:
    """Run trace_attributes and return (n_unmatched, elapsed_ms)."""
    body = {
        "shape": [{"lat": lat, "lon": lon} for lat, lon in pts],
        "costing": "auto",
        "shape_match": "map_snap",
        "trace_options": trace_options,
    }
    t0 = time.perf_counter()
    resp = json.loads(actor.trace_attributes(json.dumps(body)))
    elapsed_ms = (time.perf_counter() - t0) * 1000

    matched = resp.get("matched_points", [])
    n_unmatched = sum(1 for mp in matched if mp.get("type") == "unmatched")
    return n_unmatched, elapsed_ms


EXPERIMENTS = [
    # (label, trace_options)
    # --- Breakage distance sweep (sr=150) ---
    ("bd=1000 sr=150", {"search_radius": 150, "gps_accuracy": 15, "breakage_distance": 1000, "interpolation_distance": 20, "max_route_distance_factor": 10, "max_route_time_factor": 10, "beta": 5}),
    ("bd=2000 sr=150", {"search_radius": 150, "gps_accuracy": 15, "breakage_distance": 2000, "interpolation_distance": 20, "max_route_distance_factor": 10, "max_route_time_factor": 10, "beta": 5}),
    ("bd=3000 sr=150", {"search_radius": 150, "gps_accuracy": 15, "breakage_distance": 3000, "interpolation_distance": 20, "max_route_distance_factor": 10, "max_route_time_factor": 10, "beta": 5}),
    ("bd=5000 sr=150", {"search_radius": 150, "gps_accuracy": 15, "breakage_distance": 5000, "interpolation_distance": 20, "max_route_distance_factor": 10, "max_route_time_factor": 10, "beta": 5}),
    ("bd=10000 sr=150", {"search_radius": 150, "gps_accuracy": 15, "breakage_distance": 10000, "interpolation_distance": 20, "max_route_distance_factor": 10, "max_route_time_factor": 10, "beta": 5}),
    ("bd=20000 sr=150", {"search_radius": 150, "gps_accuracy": 15, "breakage_distance": 20000, "interpolation_distance": 20, "max_route_distance_factor": 10, "max_route_time_factor": 10, "beta": 5}),
    # --- Search radius sweep (bd=5000) ---
    ("bd=5000 sr=25", {"search_radius": 25, "gps_accuracy": 15, "breakage_distance": 5000, "interpolation_distance": 20, "max_route_distance_factor": 10, "max_route_time_factor": 10, "beta": 5}),
    ("bd=5000 sr=50", {"search_radius": 50, "gps_accuracy": 15, "breakage_distance": 5000, "interpolation_distance": 20, "max_route_distance_factor": 10, "max_route_time_factor": 10, "beta": 5}),
    ("bd=5000 sr=75", {"search_radius": 75, "gps_accuracy": 15, "breakage_distance": 5000, "interpolation_distance": 20, "max_route_distance_factor": 10, "max_route_time_factor": 10, "beta": 5}),
    ("bd=5000 sr=100", {"search_radius": 100, "gps_accuracy": 15, "breakage_distance": 5000, "interpolation_distance": 20, "max_route_distance_factor": 10, "max_route_time_factor": 10, "beta": 5}),
    # --- Combined low ---
    ("bd=2000 sr=50", {"search_radius": 50, "gps_accuracy": 15, "breakage_distance": 2000, "interpolation_distance": 20, "max_route_distance_factor": 10, "max_route_time_factor": 10, "beta": 5}),
    ("bd=3000 sr=50", {"search_radius": 50, "gps_accuracy": 15, "breakage_distance": 3000, "interpolation_distance": 20, "max_route_distance_factor": 10, "max_route_time_factor": 10, "beta": 5}),
    ("bd=3000 sr=75", {"search_radius": 75, "gps_accuracy": 15, "breakage_distance": 3000, "interpolation_distance": 20, "max_route_distance_factor": 10, "max_route_time_factor": 10, "beta": 5}),
]


def run_suite(label: str, pts: list[tuple[float, float]]) -> None:
    actor = get_actor()

    # Warmup
    run_trace_route(actor, pts, EXPERIMENTS[0][1])

    n_runs = 3

    print(f"\n=== {label}: {len(pts)} points ===\n")
    print(
        f"{'experiment':<35} {'route_ms':>10} {'attr_ms':>10} "
        f"{'fragments':>10} {'unmatched':>10}"
    )
    print("-" * 80)

    for exp_label, opts in EXPERIMENTS:
        route_times = []
        attr_times = []
        for _ in range(n_runs):
            n_frag, rt = run_trace_route(actor, pts, opts)
            n_unm, at = run_trace_attributes(actor, pts, opts)
            route_times.append(rt)
            attr_times.append(at)

        avg_rt = sum(route_times) / n_runs
        avg_at = sum(attr_times) / n_runs

        print(
            f"{exp_label:<35} {avg_rt:>9.1f}ms {avg_at:>9.1f}ms "
            f"{n_frag:>10} {n_unm:>10}"
        )


def main() -> None:
    print("Generating clean trace...")
    clean_pts = get_movement_points(gps_noise_meters=1.0, use_errors=False)
    run_suite("Clean (noise=1m, no errors)", clean_pts)

    print("\nGenerating noisy trace...")
    noisy_pts = get_movement_points(gps_noise_meters=3.0, use_errors=True)
    run_suite("Noisy (noise=3m, default errors)", noisy_pts)


if __name__ == "__main__":
    main()
