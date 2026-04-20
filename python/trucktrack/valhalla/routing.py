"""Route fetching via local pyvalhalla with truck costing."""

from __future__ import annotations

from pathlib import Path

from trucktrack.generate.models import RouteSegment
from trucktrack.valhalla._actor import DEFAULT_TRUCK_COSTING, get_actor
from trucktrack.valhalla._json import dumps as _json_dumps
from trucktrack.valhalla._json import loads as _json_loads
from trucktrack.valhalla._parsing import parse_valhalla_response


def route(
    origin: tuple[float, float],
    destination: tuple[float, float],
    *,
    costing_options: dict[str, object] | None = None,
    config: str | Path | None = None,
) -> RouteSegment:
    """Fetch a truck route using local pyvalhalla bindings."""
    actor = get_actor(config=config)
    body = {
        "locations": [
            {"lat": origin[0], "lon": origin[1]},
            {"lat": destination[0], "lon": destination[1]},
        ],
        "costing": "truck",
        "costing_options": {
            "truck": costing_options or DEFAULT_TRUCK_COSTING,
        },
        "units": "km",
    }
    resp = _json_loads(actor.route(_json_dumps(body)))
    return parse_valhalla_response(resp)
