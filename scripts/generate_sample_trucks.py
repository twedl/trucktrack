"""Generate ``data/trucks/`` — a small road-aligned sample for demos.

Three trucks × two trips each, routed through Valhalla between Ontario
cities and decorated with the default GPS/operational error profile.
Layout matches the pipeline's raw stage::

    data/trucks/year=YYYY/chunk_id=XX/part-0.parquet

Truck UUIDs and seeds are fixed so successive runs produce identical
output — safe to commit the result.  Requires a discoverable
``valhalla.json`` (e.g. ``./valhalla.json`` or
``./valhalla_tiles/valhalla.json``)::

    uv run python scripts/generate_sample_trucks.py

``--trucks N`` swaps the three fixed trucks for N generated IDs, and
``--random-sampling`` gives each truck its own reporting schedule (see
:func:`trucktrack.generate.random_sampling`) and records it in
``sampling.json`` next to the partitions::

    uv run python scripts/generate_sample_trucks.py \\
        --trucks 50 --random-sampling --output data/trucks_sampling
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict
from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl
from trucktrack import generate_trace
from trucktrack.generate.models import SamplingConfig, TripConfig, random_sampling
from trucktrack.query import _RAW_CHUNK_ID_LEN
from trucktrack.valhalla import find_config

OUTPUT_DIR = Path(__file__).parent.parent / "data" / "trucks"

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
# Deterministic master seed → deterministic per-trip seeds.
MASTER_SEED = 20260101


def _build_trip_configs(
    truck_id: str, rng: random.Random, sampling: SamplingConfig
) -> list[TripConfig]:
    configs: list[TripConfig] = []
    current_time = datetime(2026, 1, 1, 8, 0, tzinfo=UTC)
    for _ in range(N_TRIPS_PER_TRUCK):
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
    truck_id: str, truck_rng: random.Random, sampling: SamplingConfig
) -> pl.DataFrame:
    rows: list[dict[str, object]] = []
    for config in _build_trip_configs(truck_id, truck_rng, sampling):
        points = generate_trace(config)
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
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    output_dir: Path = args.output
    master_rng = random.Random(MASTER_SEED)
    # Separate streams so neither flag changes the trips themselves.
    id_rng = random.Random(f"{MASTER_SEED}:ids")
    sampling_rng = random.Random(f"{MASTER_SEED}:sampling")
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
        df = _collect_truck(truck_id, truck_rng, sampling)
        parts.append(df)
        print(f"  {truck_id}: {df.height} points, every {sampling.interval_s:g} s")

    full = pl.concat(parts).with_columns(
        pl.col("time").dt.year().alias("year"),
        pl.col("id").str.slice(-_RAW_CHUNK_ID_LEN).alias("chunk_id"),
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
