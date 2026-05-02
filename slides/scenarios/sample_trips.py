"""Fetch sample multi-tier trips for the Valhalla deck.

trucktrack's partition module classifies a trip into one of three tiers
based on its bbox diagonal:

    local    < 100 km
    regional 100 – 800 km
    longhaul > 800 km

Our local Valhalla tile bundle only covers Ontario, so we can't route
a longhaul trip ourselves.  Instead, fetch one route per tier from
the OSRM public demo server and cache as parquet.

    uv run python scenarios/sample_trips.py
"""

from __future__ import annotations

import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

import polars as pl

SLIDES = Path(__file__).resolve().parent.parent
OUT = SLIDES / "valhalla" / "data" / "sample_trips.parquet"
OSRM_URL = "https://router.project-osrm.org/route/v1/driving"

# (id, tier, label, (start_lat, start_lon), (end_lat, end_lon))
TRIPS = [
    ("local",
     "local",
     "Toronto → Mississauga",
     (43.6532, -79.3832),
     (43.5890, -79.6441)),
    ("regional",
     "regional",
     "Toronto → Ottawa",
     (43.6532, -79.3832),
     (45.4215, -75.6972)),
    ("longhaul",
     "longhaul",
     "Toronto → Vancouver",
     (43.6532, -79.3832),
     (49.2827, -123.1207)),
]


def _fetch(start: tuple[float, float], end: tuple[float, float]) -> list[tuple[float, float]]:
    """Fetch a routed polyline from OSRM (lat, lon pairs in order)."""
    coords = f"{start[1]},{start[0]};{end[1]},{end[0]}"
    url = (
        f"{OSRM_URL}/{urllib.parse.quote(coords)}"
        "?overview=full&geometries=geojson"
    )
    with urllib.request.urlopen(url, timeout=30) as resp:
        body = json.loads(resp.read())
    if body.get("code") != "Ok":
        raise RuntimeError(f"OSRM error: {body.get('message')}")
    line = body["routes"][0]["geometry"]["coordinates"]
    # GeoJSON returns [lon, lat] — swap to (lat, lon).
    return [(lat, lon) for lon, lat in line]


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for trip_id, tier, label, start, end in TRIPS:
        print(f"  fetching {trip_id}: {label} …")
        coords = _fetch(start, end)
        for i, (lat, lon) in enumerate(coords):
            rows.append({
                "id": trip_id,
                "tier": tier,
                "label": label,
                "seq": i,
                "lat": lat,
                "lon": lon,
            })
        # Be polite to the public demo server.
        time.sleep(1.0)

    df = pl.DataFrame(rows).sort("id", "seq")
    df.write_parquet(OUT)

    summary = (
        df.group_by("id", "tier", "label")
        .agg([
            pl.len().alias("n_points"),
            pl.col("lat").min().alias("lat_min"),
            pl.col("lat").max().alias("lat_max"),
            pl.col("lon").min().alias("lon_min"),
            pl.col("lon").max().alias("lon_max"),
        ])
        .sort("n_points")
    )
    print()
    print(f"  {OUT.relative_to(SLIDES)}: {df.height} points across {len(TRIPS)} trips")
    with pl.Config(tbl_rows=10, tbl_cols=10, fmt_str_lengths=40):
        print(summary)


if __name__ == "__main__":
    main()
