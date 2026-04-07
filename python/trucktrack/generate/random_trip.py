"""Generate random viable truck trip endpoints within Ontario."""

from __future__ import annotations

import math
import random

import requests

from trucktrack.generate.models import DEFAULT_VALHALLA_URL

# (name, min_lat, max_lat, min_lon, max_lon, weight)
ZONES: list[tuple[str, float, float, float, float, int]] = [
    ("GTA", 43.50, 43.90, -79.85, -79.15, 20),
    ("Hamilton-Niagara", 43.05, 43.35, -79.90, -79.00, 10),
    ("Kitchener-Guelph", 43.35, 43.60, -80.65, -80.15, 8),
    ("London", 42.90, 43.10, -81.40, -81.10, 8),
    ("Windsor-Essex", 42.10, 42.40, -83.10, -82.70, 8),
    ("Ottawa-Gatineau", 45.25, 45.55, -75.90, -75.50, 10),
    ("Kingston", 44.15, 44.35, -76.65, -76.35, 5),
    ("Barrie-Orillia", 44.30, 44.70, -79.80, -79.30, 5),
    ("Peterborough", 44.20, 44.40, -78.45, -78.20, 3),
    ("401-GTA-Kingston", 43.85, 44.15, -78.80, -76.70, 8),
    ("401-London-KW", 43.10, 43.40, -81.10, -80.40, 5),
    ("401-Windsor-London", 42.20, 42.55, -82.70, -81.40, 5),
    ("Sudbury", 46.40, 46.60, -81.10, -80.80, 3),
    ("North Bay", 46.25, 46.40, -79.60, -79.35, 2),
    ("Sault Ste Marie", 46.45, 46.60, -84.45, -84.20, 2),
    ("Thunder Bay", 48.30, 48.50, -89.35, -89.15, 2),
]

ORIGIN_MANEUVERS = ["alley_dock", "straight_back", "blind_side"]
DEST_MANEUVERS = [
    "alley_dock",
    "straight_back",
    "blind_side",
    "pull_through",
    "angle_back",
]

MIN_TRIP_DISTANCE_KM = 50
MAX_LOCATE_RETRIES = 10


def _pick_zone(rng: random.Random) -> tuple[str, float, float, float, float]:
    total = sum(z[5] for z in ZONES)
    r = rng.uniform(0, total)
    cumulative = 0.0
    for name, min_lat, max_lat, min_lon, max_lon, w in ZONES:
        cumulative += w
        if r <= cumulative:
            return name, min_lat, max_lat, min_lon, max_lon
    last = ZONES[-1]
    return last[0], last[1], last[2], last[3], last[4]


def _random_point_in_zone(
    rng: random.Random,
    min_lat: float,
    max_lat: float,
    min_lon: float,
    max_lon: float,
) -> tuple[float, float]:
    return rng.uniform(min_lat, max_lat), rng.uniform(min_lon, max_lon)


def _snap_to_road(
    lat: float, lon: float, valhalla_url: str
) -> tuple[float, float] | None:
    body = {
        "locations": [{"lat": lat, "lon": lon, "search_cutoff": 2000}],
        "costing": "truck",
    }
    url = valhalla_url.rstrip("/") + "/locate"
    try:
        resp = requests.post(url, json=body, timeout=5)
        resp.raise_for_status()
        result = resp.json()[0]
        edges = result.get("edges")
        if not edges:
            return None
        edge = edges[0]
        return (edge["correlated_lat"], edge["correlated_lon"])
    except (requests.RequestException, KeyError, IndexError):
        return None


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    )
    return 6371 * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def generate_random_endpoints(
    rng: random.Random,
    min_distance_km: float = MIN_TRIP_DISTANCE_KM,
    valhalla_url: str = DEFAULT_VALHALLA_URL,
) -> tuple[tuple[float, float], tuple[float, float], str, str]:
    """Pick random origin/destination points snapped to truck-routable roads."""
    for _ in range(MAX_LOCATE_RETRIES):
        o_name, *o_bounds = _pick_zone(rng)
        d_name, *d_bounds = _pick_zone(rng)

        o_raw = _random_point_in_zone(rng, *o_bounds)
        d_raw = _random_point_in_zone(rng, *d_bounds)

        origin = _snap_to_road(*o_raw, valhalla_url=valhalla_url)
        dest = _snap_to_road(*d_raw, valhalla_url=valhalla_url)

        if origin is None or dest is None:
            continue

        if _haversine_km(*origin, *dest) < min_distance_km:
            continue

        return origin, dest, o_name, d_name

    raise RuntimeError(
        f"Could not find viable random endpoints after {MAX_LOCATE_RETRIES} attempts. "
        "Is Valhalla running?"
    )


def random_maneuvers(rng: random.Random) -> tuple[str, str]:
    return rng.choice(ORIGIN_MANEUVERS), rng.choice(DEST_MANEUVERS)
