"""Route fetching via a Valhalla instance with truck costing."""

from __future__ import annotations

import requests

from trucktrack.generate.models import DEFAULT_VALHALLA_URL, RouteSegment
from trucktrack.valhalla._parsing import parse_valhalla_response


def fetch_route(
    origin: tuple[float, float],
    destination: tuple[float, float],
    valhalla_url: str = DEFAULT_VALHALLA_URL,
    tile_extract: str | None = None,
) -> RouteSegment:
    """Fetch a truck route from Valhalla.

    If *tile_extract* is provided, uses local pyvalhalla bindings.
    Otherwise falls back to the HTTP API at *valhalla_url*.
    """
    if tile_extract is not None:
        from trucktrack.valhalla.routing import route

        return route(origin, destination, tile_extract)

    body = {
        "locations": [
            {"lat": origin[0], "lon": origin[1]},
            {"lat": destination[0], "lon": destination[1]},
        ],
        "costing": "truck",
        "costing_options": {
            "truck": {
                "height": 4.11,
                "width": 2.6,
                "length": 22.0,
                "weight": 36.287,
            }
        },
        "shape_match": "map_snap",
        "units": "km",
    }

    url = valhalla_url.rstrip("/") + "/route"
    resp = requests.post(url, json=body, timeout=30)
    resp.raise_for_status()
    return parse_valhalla_response(resp.json())
