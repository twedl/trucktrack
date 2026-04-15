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
"""

from __future__ import annotations

import random
from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl
from trucktrack import generate_trace
from trucktrack.generate.models import TripConfig
from trucktrack.query import _CHUNK_ID_LEN
from trucktrack.valhalla._actor import _find_config

OUTPUT_DIR = Path(__file__).parent.parent / "data" / "trucks"


def _config_path() -> Path:
    found = _find_config()
    if found is None:
        raise SystemExit(
            "No valhalla.json found. Place one at ./valhalla.json or "
            "./valhalla_tiles/valhalla.json."
        )
    return found


CONFIG_PATH = _config_path()

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
    (43.6532, -79.3832),  # Toronto
    (43.9000, -78.8500),  # Oshawa
]

N_TRIPS_PER_TRUCK = 2
# Deterministic master seed → deterministic per-trip seeds.
MASTER_SEED = 20260101


def _build_trip_configs(truck_id: str, rng: random.Random) -> list[TripConfig]:
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
                config=CONFIG_PATH,
            )
        )
        current_time += timedelta(hours=rng.uniform(4, 8))
    return configs


def _collect_truck(truck_id: str, truck_rng: random.Random) -> pl.DataFrame:
    rows: list[dict[str, object]] = []
    for config in _build_trip_configs(truck_id, truck_rng):
        points = generate_trace(config)
        for pt in points:
            rows.append(
                {
                    "id": truck_id,
                    "time": pt.timestamp,
                    "lat": pt.lat,
                    "lon": pt.lon,
                    "speed": round(pt.speed_mph, 1),
                    "heading": round(pt.heading, 1),
                }
            )
    return pl.DataFrame(rows)


def main() -> None:
    master_rng = random.Random(MASTER_SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    parts: list[pl.DataFrame] = []
    for truck_id in TRUCKS:
        truck_rng = random.Random(master_rng.randint(0, 2**63))
        df = _collect_truck(truck_id, truck_rng)
        parts.append(df)
        print(f"  {truck_id}: {df.height} points")

    full = pl.concat(parts).with_columns(
        pl.col("time").dt.year().alias("year"),
        pl.col("id").str.slice(-_CHUNK_ID_LEN).alias("chunk_id"),
    )

    for (yr, cid), group in full.group_by("year", "chunk_id", maintain_order=True):
        part_dir = OUTPUT_DIR / f"year={yr}" / f"chunk_id={cid}"
        part_dir.mkdir(parents=True, exist_ok=True)
        out = part_dir / "part-0.parquet"
        group.drop("year", "chunk_id").write_parquet(out)
        print(f"wrote {out} ({group.height} rows)")


if __name__ == "__main__":
    main()
