"""Toronto -> Ottawa trace with a ~40 km geofence gap near Kingston.

Usage::

    uv run python examples/trace_visualizations/geofence_gap_toronto_ottawa.py

Requires a ``valhalla.json`` in cwd.
"""

from __future__ import annotations

import argparse

from _geofence_gap_common import GeofenceGapExample, run

EXAMPLE = GeofenceGapExample(
    label="Toronto -> Ottawa",
    origin=(43.6426, -79.3871),  # CN Tower
    destination=(45.4236, -75.7009),  # Parliament Hill
    geofence_center=(44.2312, -76.4860),  # Hwy 401 near Kingston
    geofence_radius_m=20_000.0,
    output_filename="geofence_gap_toronto_ottawa.html",
)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serve", action="store_true")
    parser.add_argument("--port", type=int, default=5000)
    args = parser.parse_args()
    run(EXAMPLE, serve=args.serve, port=args.port)
