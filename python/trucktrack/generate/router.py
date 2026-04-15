"""Route fetching via a Valhalla instance with truck costing."""

from __future__ import annotations

from pathlib import Path

import requests

from trucktrack.generate.models import DEFAULT_VALHALLA_URL, RouteSegment
from trucktrack.valhalla._actor import DEFAULT_TRUCK_COSTING
from trucktrack.valhalla._parsing import parse_valhalla_response


def fetch_route(
    origin: tuple[float, float],
    destination: tuple[float, float],
    valhalla_url: str = DEFAULT_VALHALLA_URL,
    config: str | Path | None = None,
) -> RouteSegment:
    """Fetch a truck route from Valhalla.

    If *config* is provided, uses local pyvalhalla bindings via the
    given ``valhalla.json``.  Otherwise falls back to the HTTP API at
    *valhalla_url*.
    """
    if config is not None:
        from trucktrack.valhalla.routing import route

        return route(origin, destination, config=config)

    body = {
        "locations": [
            {"lat": origin[0], "lon": origin[1]},
            {"lat": destination[0], "lon": destination[1]},
        ],
        "costing": "truck",
        "costing_options": {
            "truck": DEFAULT_TRUCK_COSTING,
        },
        "shape_match": "map_snap",
        "units": "km",
    }

    url = valhalla_url.rstrip("/") + "/route"
    resp = requests.post(url, json=body, timeout=30)
    resp.raise_for_status()
    return parse_valhalla_response(resp.json())
