"""Generate splitter_test_tracks.parquet for testing ObservationGapSplitter and StopSplitter."""

from datetime import datetime, timedelta
from pathlib import Path

import polars as pl

DATA_DIR = Path(__file__).parent.parent / "data"


def _truck_a() -> pl.DataFrame:
    """~30 rows, steady movement with a 10-minute gap between rows 15 and 16."""
    base = datetime(2024, 6, 10, 8, 0, 0)
    rows = []

    # First segment: rows 0-14, every 30s heading NE from (40.700, -74.000)
    for i in range(15):
        rows.append({
            "id": "truck_A",
            "time": base + timedelta(seconds=i * 30),
            "speed": 55.0 + i * 0.3,
            "heading": 45.0 + i * 0.2,
            "lat": 40.7000 + i * 0.0003,
            "lon": -74.0000 + i * 0.0004,
        })

    # 10-minute gap
    gap_start = rows[-1]["time"] + timedelta(minutes=10)

    # Second segment: rows 15-29, every 30s continuing
    for i in range(15):
        rows.append({
            "id": "truck_A",
            "time": gap_start + timedelta(seconds=i * 30),
            "speed": 58.0 + i * 0.2,
            "heading": 48.0 + i * 0.1,
            "lat": 40.7050 + i * 0.0003,
            "lon": -73.9940 + i * 0.0004,
        })

    return pl.DataFrame(rows)


def _truck_b() -> pl.DataFrame:
    """~30 rows, includes a stop: rows 10-18 within ~25m diameter over 5 minutes."""
    base = datetime(2024, 6, 10, 9, 0, 0)
    rows = []

    # Movement before stop: rows 0-9
    for i in range(10):
        rows.append({
            "id": "truck_B",
            "time": base + timedelta(seconds=i * 30),
            "speed": 60.0 + i * 0.5,
            "heading": 90.0 + i * 0.3,
            "lat": 40.7500 + i * 0.0004,
            "lon": -73.9800 + i * 0.0005,
        })

    # Stop: rows 10-18, clustered within ~25m diameter, 5 min (every ~33s)
    stop_center_lat = 40.7540
    stop_center_lon = -73.9750
    stop_start = rows[-1]["time"] + timedelta(seconds=30)
    # ~0.0001 degrees ≈ ~11m, so radius 0.00005 ≈ 5.5m, diameter ~11m well within 50m
    offsets = [
        (0.00002, 0.00001),
        (-0.00001, 0.00002),
        (0.00003, -0.00001),
        (-0.00002, 0.00003),
        (0.00001, -0.00002),
        (0.00000, 0.00001),
        (-0.00001, -0.00001),
        (0.00002, 0.00002),
        (-0.00003, 0.00000),
    ]
    for i in range(9):
        rows.append({
            "id": "truck_B",
            "time": stop_start + timedelta(seconds=i * 33),
            "speed": 0.5 + i * 0.1,  # near-zero speed
            "heading": 90.0,
            "lat": stop_center_lat + offsets[i][0],
            "lon": stop_center_lon + offsets[i][1],
        })

    # Movement after stop: rows 19-29
    resume_time = rows[-1]["time"] + timedelta(seconds=30)
    for i in range(11):
        rows.append({
            "id": "truck_B",
            "time": resume_time + timedelta(seconds=i * 30),
            "speed": 55.0 + i * 0.4,
            "heading": 180.0 + i * 0.5,
            "lat": 40.7540 - i * 0.0003,
            "lon": -73.9750 + i * 0.0004,
        })

    return pl.DataFrame(rows)


def _truck_c() -> pl.DataFrame:
    """~20 rows with both a 5-minute gap and a 3-minute stop within 20m."""
    base = datetime(2024, 6, 10, 10, 0, 0)
    rows = []

    # First segment: rows 0-6
    for i in range(7):
        rows.append({
            "id": "truck_C",
            "time": base + timedelta(seconds=i * 30),
            "speed": 50.0 + i * 0.5,
            "heading": 270.0 + i * 0.2,
            "lat": 40.8000 + i * 0.0003,
            "lon": -73.9500 + i * 0.0004,
        })

    # 5-minute gap
    gap_start = rows[-1]["time"] + timedelta(minutes=5)

    # Second segment with a stop in the middle: rows 7-12 movement, 13-18 stop, 19-22 movement
    for i in range(6):
        rows.append({
            "id": "truck_C",
            "time": gap_start + timedelta(seconds=i * 30),
            "speed": 52.0 + i * 0.3,
            "heading": 270.0,
            "lat": 40.8030 + i * 0.0003,
            "lon": -73.9460 + i * 0.0004,
        })

    # Stop: 6 rows within ~15m diameter, 3 minutes
    stop_lat = 40.8050
    stop_lon = -73.9430
    stop_start = rows[-1]["time"] + timedelta(seconds=30)
    for i in range(6):
        rows.append({
            "id": "truck_C",
            "time": stop_start + timedelta(seconds=i * 30),
            "speed": 0.2,
            "heading": 270.0,
            "lat": stop_lat + (i % 3 - 1) * 0.00001,
            "lon": stop_lon + (i % 2) * 0.00001,
        })

    # Movement after stop
    resume_time = rows[-1]["time"] + timedelta(seconds=30)
    for i in range(4):
        rows.append({
            "id": "truck_C",
            "time": resume_time + timedelta(seconds=i * 30),
            "speed": 48.0 + i * 0.5,
            "heading": 270.0,
            "lat": 40.8050 - i * 0.0003,
            "lon": -73.9430 - i * 0.0004,
        })

    return pl.DataFrame(rows)


def main() -> None:
    df = pl.concat([_truck_a(), _truck_b(), _truck_c()])
    out = DATA_DIR / "splitter_test_tracks.parquet"
    df.write_parquet(out)
    print(f"Wrote {len(df)} rows ({df['id'].n_unique()} vehicles) to {out}")
    print(df)


if __name__ == "__main__":
    main()
