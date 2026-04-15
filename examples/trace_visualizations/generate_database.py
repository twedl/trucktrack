"""Generate a database of N trucks × K trips with all error types.

Creates a hive-partitioned parquet dataset at::

    trucks/year=YYYY/chunk_id=XX/part-0.parquet

where *year* comes from the trip timestamp and *chunk_id* is the last
two hex characters of the truck UUID.  Every error type appears at
least once; remaining trips use the default error profile.

Usage::

    uv run python examples/trace_visualizations/generate_database.py

Requires a ``valhalla.json`` in cwd.
"""

from __future__ import annotations

import os
import random
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl
from trucktrack import generate_trace
from trucktrack.generate.gps_errors import GPS_ERRORS
from trucktrack.generate.models import ErrorConfig, TripConfig, default_error_profile
from trucktrack.generate.operational_errors import OPERATIONAL_ERRORS
from trucktrack.query import _CHUNK_ID_LEN, _chunk_id
from trucktrack.valhalla._actor import _find_config

OUTPUT_DIR = Path(
    os.environ.get("OUTPUT_DIR", "examples/trace_visualizations/output/trucks")
)

N_TRUCKS = int(os.environ.get("N_TRUCKS", "10"))
K_TRIPS = int(os.environ.get("K_TRIPS", "10"))

# Ontario waypoints — short hops that stay within Valhalla's 2000 km limit.
WAYPOINTS = [
    (43.7387, -79.7271),  # Brampton
    (43.2557, -79.8711),  # Hamilton
    (43.4643, -80.5204),  # Kitchener
    (43.5448, -80.2482),  # Guelph
    (43.6532, -79.3832),  # Toronto
    (43.4516, -80.4925),  # Waterloo
    (43.8561, -79.3370),  # Markham
    (43.1594, -79.2469),  # St. Catharines
    (44.2312, -76.4860),  # Kingston
    (43.0096, -81.2737),  # London
    (42.9849, -82.4066),  # Sarnia
    (43.1890, -80.3841),  # Brantford
    (43.9000, -78.8500),  # Oshawa
    (44.3894, -79.6903),  # Barrie
    (43.5168, -79.8711),  # Milton
]

# All error types sorted alphabetically for deterministic assignment.
ALL_ERROR_TYPES = sorted({*GPS_ERRORS, *OPERATIONAL_ERRORS})


def _guaranteed_error_configs() -> list[list[ErrorConfig]]:
    """Return one error profile per guaranteed error type (probability=1.0)."""
    profiles: list[list[ErrorConfig]] = []
    for etype in ALL_ERROR_TYPES:
        params: dict[str, object] = {}
        if etype == "geofence_gap":
            params = {"center": (44.25, -76.5), "radius_m": 2000}
        elif etype in ("multipath", "timestamp_glitch", "coordinate_corruption"):
            params = {"count": 1}
        profiles.append([ErrorConfig(etype, probability=1.0, params=params)])
    return profiles


def _make_trip_configs(
    truck_id: str,
    n_trips: int,
    start_time: datetime,
    rng: random.Random,
    error_overrides: dict[int, list[ErrorConfig]],
) -> list[TripConfig]:
    """Build TripConfig for each trip of a single truck."""
    configs: list[TripConfig] = []
    current_time = start_time
    valhalla_config = _find_config()
    for i in range(n_trips):
        origin = rng.choice(WAYPOINTS)
        dest = rng.choice([w for w in WAYPOINTS if w != origin])
        errors = error_overrides.get(i, default_error_profile())
        configs.append(
            TripConfig(
                origin=origin,
                destination=dest,
                departure_time=current_time,
                trip_id=truck_id,
                seed=rng.randint(0, 2**31),
                config=valhalla_config,
                errors=errors,
            )
        )
        # Next trip starts 1–8 hours later.
        current_time += timedelta(hours=rng.uniform(1, 8))
    return configs


def main() -> None:
    master_rng = random.Random(42)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Generate truck UUIDs.
    truck_ids = [uuid.UUID(int=master_rng.getrandbits(128)).hex for _ in range(N_TRUCKS)]

    # Assign guaranteed error profiles round-robin across (truck, trip) slots.
    guaranteed = _guaranteed_error_configs()
    error_overrides: dict[tuple[int, int], list[ErrorConfig]] = {}
    for idx, profile in enumerate(guaranteed):
        truck_idx = idx % N_TRUCKS
        trip_idx = idx // N_TRUCKS
        if trip_idx < K_TRIPS:
            error_overrides[(truck_idx, trip_idx)] = profile

    # Use a fixed year; stagger truck start dates across the year.
    year = 2025
    all_ids: list[str] = []
    all_lats: list[float] = []
    all_lons: list[float] = []
    all_speeds: list[float] = []
    all_headings: list[float] = []
    all_times: list[datetime] = []
    total_trips = 0

    for t_idx, truck_id in enumerate(truck_ids):
        rng = random.Random(master_rng.randint(0, 2**63))
        start_day = 1 + (t_idx * 30) % 360
        start_time = datetime(year, 1, 1, 6, 0, tzinfo=UTC) + timedelta(days=start_day)

        overrides = {
            trip_idx: profile
            for (ti, trip_idx), profile in error_overrides.items()
            if ti == t_idx
        }

        configs = _make_trip_configs(truck_id, K_TRIPS, start_time, rng, overrides)
        chunk_id = _chunk_id(truck_id)

        truck_points = 0
        for trip_num, config in enumerate(configs, 1):
            try:
                points = generate_trace(config)
            except Exception as e:
                print(f"  [WARN] truck {truck_id[:8]} trip {trip_num}: {e}")
                continue

            if not points:
                continue

            truck_points += len(points)
            total_trips += 1

            for pt in points:
                all_ids.append(truck_id)
                all_lats.append(pt.lat)
                all_lons.append(pt.lon)
                all_speeds.append(round(pt.speed_mph, 1))
                all_headings.append(round(pt.heading, 1))
                all_times.append(pt.timestamp)

        print(
            f"  Truck {truck_id[:8]}… chunk_id={chunk_id}: "
            f"{K_TRIPS} trips, {truck_points} points"
        )

    if not all_ids:
        print("No trips generated.")
        return

    full = pl.DataFrame(
        {
            "id": all_ids,
            "lat": all_lats,
            "lon": all_lons,
            "speed": all_speeds,
            "heading": all_headings,
            "time": all_times,
        }
    )

    # Derive partition columns.
    full = full.with_columns(
        pl.col("time").dt.year().alias("year"),
        pl.col("id").str.slice(-_CHUNK_ID_LEN).alias("chunk_id"),
    )

    # Write hive-partitioned parquet and collect stats.
    partition_stats: list[tuple[str, int, int]] = []
    for (yr, cid), group in full.group_by("year", "chunk_id", maintain_order=True):
        part_dir = OUTPUT_DIR / f"year={yr}" / f"chunk_id={cid}"
        part_dir.mkdir(parents=True, exist_ok=True)
        out_path = part_dir / "part-0.parquet"
        data = group.drop("year", "chunk_id")
        data.write_parquet(out_path)
        partition_stats.append(
            (f"year={yr}/chunk_id={cid}/part-0.parquet", data.height, data["id"].n_unique())
        )

    # Summary.
    print(f"\nDatabase written to {OUTPUT_DIR}/")
    print(f"  {N_TRUCKS} trucks, {total_trips} trips, {len(all_ids)} points")
    print(f"  {len(partition_stats)} partition(s)")
    print(f"  Error types guaranteed: {len(ALL_ERROR_TYPES)}")

    for path, n_rows, n_trucks in partition_stats:
        print(f"  {path}: {n_rows} rows, {n_trucks} truck(s)")


if __name__ == "__main__":
    main()
