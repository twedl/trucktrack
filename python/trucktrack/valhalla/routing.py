"""Route fetching via local pyvalhalla with truck costing."""

from __future__ import annotations

from datetime import datetime
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
    depart_at: datetime | None = None,
) -> RouteSegment:
    """Fetch a truck route using local pyvalhalla bindings.

    *depart_at*, a naive clock time local to *origin*, makes the route
    time-dependent: edges with predicted traffic (see
    :mod:`trucktrack.valhalla.traffic`) take their speed for the time
    the truck reaches them.  Without it, they take their
    ``constrained_speed``.
    """
    if depart_at is not None and depart_at.tzinfo is not None:
        raise ValueError("depart_at is local clock time at the origin; drop its tzinfo")
    actor = get_actor(config=config)
    body: dict[str, object] = {
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
    if depart_at is not None:
        body["date_time"] = {"type": 1, "value": depart_at.strftime("%Y-%m-%dT%H:%M")}
    resp = _json_loads(actor.route(_json_dumps(body)))
    return parse_valhalla_response(resp)
