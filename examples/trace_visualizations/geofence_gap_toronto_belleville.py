"""Toronto -> Belleville trace with a ~60 km geofence gap near Cobourg.

Usage::

    uv run python examples/trace_visualizations/geofence_gap_toronto_belleville.py

Requires a ``valhalla.json`` in cwd.
"""

from __future__ import annotations

import argparse

from _geofence_gap_common import GeofenceGapExample, run

EXAMPLE = GeofenceGapExample(
    label="Toronto -> Belleville",
    origin=(43.6426, -79.3871),  # CN Tower
    destination=(44.1628, -77.3832),  # Belleville City Hall
    geofence_center=(43.9593, -78.1677),  # Hwy 401 near Cobourg
    geofence_radius_m=30_000.0,
    output_filename="geofence_gap_toronto_belleville.html",
)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serve", action="store_true")
    parser.add_argument("--port", type=int, default=5000)
    args = parser.parse_args()
    run(EXAMPLE, serve=args.serve, port=args.port)
