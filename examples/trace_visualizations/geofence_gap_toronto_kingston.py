"""Toronto -> Kingston trace with a ~50 km geofence gap near Belleville.

Usage::

    uv run python examples/trace_visualizations/geofence_gap_toronto_kingston.py

Requires a ``valhalla.json`` in cwd.
"""

from __future__ import annotations

import argparse

from _geofence_gap_common import GeofenceGapExample, run

EXAMPLE = GeofenceGapExample(
    label="Toronto -> Kingston",
    origin=(43.6426, -79.3871),  # CN Tower
    destination=(44.2312, -76.4860),  # Kingston City Hall
    geofence_center=(44.1628, -77.3832),  # Hwy 401 near Belleville
    geofence_radius_m=25_000.0,
    output_filename="geofence_gap_toronto_kingston.html",
)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serve", action="store_true")
    parser.add_argument("--port", type=int, default=5000)
    args = parser.parse_args()
    run(EXAMPLE, serve=args.serve, port=args.port)
