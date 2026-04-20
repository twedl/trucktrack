"""Generate a trace with a forced signal-dropout, run the bridging
orchestrator, and serve an interactive map showing raw GPS + matched
sub-segments + routed bridge geometry.

Usage::

    uv run python scripts/demo_bridges.py
"""

from __future__ import annotations

from datetime import datetime

import polars as pl

from trucktrack import generate_trace
from trucktrack.generate.models import ErrorConfig, TripConfig
from trucktrack.valhalla import (
    BridgeConfig,
    find_config,
    map_match_dataframe_with_bridges,
)
from trucktrack.visualize import plot_trace_layers, save_map, serve_map

OUTPUT_HTML = "bridge_demo.html"

valhalla_config = find_config()
if valhalla_config is None:
    raise SystemExit("No valhalla.json discoverable under cwd.")

# Toronto → Barrie: ~100 km of highway, plenty of room for a big gap.
trip = TripConfig(
    origin=(43.6532, -79.3832),
    destination=(44.3894, -79.6903),
    departure_time=datetime(2026, 4, 19, 8, 0),
    trip_id="demo_bridges",
    seed=42,
    config=valhalla_config,
    # Single forced signal dropout — removes 15 min of consecutive points
    # from the middle of the trace, creating a large spatial+temporal gap.
    errors=[ErrorConfig("signal_dropout", probability=1.0, params={"duration_s": 900})],
)

print("Generating trace...")
points = generate_trace(trip)
print(f"  {len(points)} points, {points[0].timestamp} -> {points[-1].timestamp}")

df = pl.DataFrame(
    {
        "id": [trip.trip_id] * len(points),
        "time": [p.timestamp for p in points],
        "lat": [p.lat for p in points],
        "lon": [p.lon for p in points],
    }
)

bridges = BridgeConfig()  # 5 km / 4 min / 1 km defaults
print(f"Running bridging orchestrator with {bridges}...")
result = map_match_dataframe_with_bridges(df, bridges=bridges, collect_shapes=True)

print("\n--- Result ---")
print(f"  n_bridges:       {len(result.fits)}")
print(f"  fallback_used:   {result.fallback_used}")
print(f"  shapes returned: {len(result.shapes)}")
print(f"  way_ids:         {len(result.way_ids)}")
for i, fit in enumerate(result.fits):
    print(
        f"  bridge[{i}]: straight={fit.straight_m:,.0f} m  "
        f"route={fit.route_m:,.0f} m  detour={fit.detour_ratio:.2f}x  "
        f"gap={fit.gap_seconds:.0f} s"
    )

print("\nBuilding map...")
m = plot_trace_layers(
    raw=df,
    matched_shape=result.shapes,
)
save_map(m, OUTPUT_HTML)
print(f"Saved to {OUTPUT_HTML}")
print(f"\nServing at http://127.0.0.1:5000 (Ctrl+C to stop)...")
serve_map(m)
