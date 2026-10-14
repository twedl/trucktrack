"""Generate ``data/trucks/`` — a small road-aligned sample for demos.

Three trucks × two trips each, routed through Valhalla between Ontario
cities and decorated with the default GPS/operational error profile.
Laid out like the provider's raw input, which the pipeline doesn't
rely on (see :func:`trucktrack.bucket_by_truck`)::

    data/trucks/year=YYYY/chunk_id=XX/part-0.parquet

Truck UUIDs and seeds are fixed so successive runs produce identical
output — safe to commit the result.  Requires a discoverable
``valhalla.json`` (e.g. ``./valhalla.json`` or
``./valhalla_tiles/valhalla.json``)::

    uv run python scripts/generate_sample_trucks.py

``--trucks N`` swaps the three fixed trucks for N generated IDs,
``--random-sampling`` gives each truck its own reporting schedule (see
:func:`trucktrack.generate.random_sampling`) and records it in
``sampling.json`` next to the partitions, ``--trips N`` sets trips per
truck, and ``--weeks W`` spreads each truck's first departure uniformly
over W weeks from Monday 23 Feb 2026 (local time), so the trips cover
every hour of the week and the DST change on 8 March::

    uv run python scripts/generate_sample_trucks.py \\
        --trucks 200 --random-sampling --trips 6 --weeks 4 \\
        --output data/trucks_sampling
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl
from trucktrack import generate_trace
from trucktrack.generate.models import SamplingConfig, TripConfig, random_sampling
from trucktrack.valhalla import find_config

OUTPUT_DIR = Path(__file__).parent.parent / "data" / "trucks"
# The provider's chunk_id: the last characters of the truck id.
CHUNK_ID_LEN = 2

if find_config() is None:
    raise SystemExit(
        "No valhalla.json found. Place one at ./valhalla.json or "
        "./valhalla_tiles/valhalla.json."
    )

# Hand-picked 32-char hex IDs whose last two chars span three chunk
# partitions; stable across runs so the committed parquet has fixed
# truck_ids to document in example scripts.
TRUCKS: list[str] = [
    "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaa01",
    "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbb02",
    "cccccccccccccccccccccccccccccc03",
]

# Ontario waypoints — chosen so every pair is under ~120 km (well below
# Valhalla's default trace_max_distance so map-matching works out of the box).
WAYPOINTS: list[tuple[float, float]] = [
    (43.7387, -79.7271),  # Brampton
    (43.2557, -79.8711),  # Hamilton
    (43.4643, -80.5204),  # Kitchener
    # Toronto (Dundas St; City Hall itself has no truck route in)
    (43.6555, -79.3800),
    (43.9000, -78.8500),  # Oshawa
]

N_TRIPS_PER_TRUCK = 2
FIRST_DEPARTURE = datetime(2026, 1, 1, 8, 0, tzinfo=UTC)
# Monday 00:00 in Toronto; two weeks before the DST change.
SPREAD_START = datetime(2026, 2, 23, 5, 0, tzinfo=UTC)
# Shortest wait between a truck's trips when a long one runs late.
MIN_TURNAROUND = timedelta(minutes=30)
# Deterministic master seed → deterministic per-trip seeds.
MASTER_SEED = 20260101


def _build_trip_configs(
    truck_id: str,
    rng: random.Random,
    sampling: SamplingConfig,
    first_departure: datetime,
    n_trips: int,
) -> list[TripConfig]:
    configs: list[TripConfig] = []
    current_time = first_departure
    for _ in range(n_trips):
        origin = rng.choice(WAYPOINTS)
        dest = rng.choice([w for w in WAYPOINTS if w != origin])
        configs.append(
            TripConfig(
                origin=origin,
                destination=dest,
                departure_time=current_time,
                trip_id=truck_id,
                seed=rng.randint(0, 2**31),
                config=find_config(),
                sampling=sampling,
            )
        )
        current_time += timedelta(hours=rng.uniform(4, 8))
    return configs


def _collect_truck(
    truck_id: str,
    truck_rng: random.Random,
    sampling: SamplingConfig,
    first_departure: datetime,
    n_trips: int,
) -> pl.DataFrame:
    rows: list[dict[str, object]] = []
    not_before = first_departure
    for config in _build_trip_configs(
        truck_id, truck_rng, sampling, first_departure, n_trips
    ):
        # A truck is never on two trips at once: a trip that runs long
        # pushes the next one back, as does a yard dwell that starts the
        # next one before its departure.
        departure = max(config.departure_time, not_before)
        points = generate_trace(replace(config, departure_time=departure))
        if points and points[0].timestamp < not_before:
            departure += not_before - points[0].timestamp
            points = generate_trace(replace(config, departure_time=departure))
        if points:
            not_before = points[-1].timestamp + MIN_TURNAROUND
        for pt in points:
            rows.append(
                {
                    "id": truck_id,
                    "time": pt.timestamp,
                    "lat": pt.lat,
                    "lon": pt.lon,
                    "speed": round(pt.speed_kmh, 1),
                    "heading": round(pt.heading, 1),
                }
            )
    return pl.DataFrame(rows)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--trucks",
        type=int,
        default=None,
        help="generate this many trucks with random IDs (default: the 3 fixed ones)",
    )
    parser.add_argument(
        "--random-sampling",
        action="store_true",
        help="draw each truck's reporting schedule (default: fixed 60 s)",
    )
    parser.add_argument(
        "--trips",
        type=int,
        default=N_TRIPS_PER_TRUCK,
        help=f"trips per truck (default: {N_TRIPS_PER_TRUCK})",
    )
    parser.add_argument(
        "--weeks",
        type=float,
        default=None,
        help="spread first departures over this many weeks from "
        f"{SPREAD_START:%Y-%m-%d} (default: all at {FIRST_DEPARTURE:%Y-%m-%d %H:%M})",
    )
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    output_dir: Path = args.output
    master_rng = random.Random(MASTER_SEED)
    # Separate streams so no flag changes the trips themselves.
    id_rng = random.Random(f"{MASTER_SEED}:ids")
    sampling_rng = random.Random(f"{MASTER_SEED}:sampling")
    departure_rng = random.Random(f"{MASTER_SEED}:departures")
    output_dir.mkdir(parents=True, exist_ok=True)

    trucks = (
        TRUCKS
        if args.trucks is None
        else [f"{id_rng.getrandbits(128):032x}" for _ in range(args.trucks)]
    )

    parts: list[pl.DataFrame] = []
    samplings: dict[str, SamplingConfig] = {}
    for truck_id in trucks:
        truck_rng = random.Random(master_rng.randint(0, 2**63))
        sampling = (
            random_sampling(sampling_rng) if args.random_sampling else SamplingConfig()
        )
        samplings[truck_id] = sampling
        first_departure = (
            FIRST_DEPARTURE
            if args.weeks is None
            else SPREAD_START + timedelta(weeks=departure_rng.uniform(0, args.weeks))
        )
        df = _collect_truck(truck_id, truck_rng, sampling, first_departure, args.trips)
        parts.append(df)
        print(f"  {truck_id}: {df.height} points, every {sampling.interval_s:g} s")

    full = pl.concat(parts).with_columns(
        pl.col("time").dt.year().alias("year"),
        pl.col("id").str.slice(-CHUNK_ID_LEN).alias("chunk_id"),
    )

    for (yr, cid), group in full.group_by("year", "chunk_id", maintain_order=True):
        part_dir = output_dir / f"year={yr}" / f"chunk_id={cid}"
        part_dir.mkdir(parents=True, exist_ok=True)
        out = part_dir / "part-0.parquet"
        group.drop("year", "chunk_id").write_parquet(out)
        print(f"wrote {out} ({group.height} rows)")

    if args.random_sampling:
        out = output_dir / "sampling.json"
        out.write_text(
            json.dumps({k: asdict(v) for k, v in samplings.items()}, indent=2) + "\n"
        )
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
