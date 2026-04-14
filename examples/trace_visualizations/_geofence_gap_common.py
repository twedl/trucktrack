"""Shared pipeline for the geofence-gap example scripts:
generate a trace with a single geofence gap, split into trips/stops,
map-match each movement segment, and render/serve a layered map.
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl
from trucktrack import (
    ErrorConfig,
    TripConfig,
    generate_trace,
    read_parquet,
    split_by_observation_gap,
    split_by_stops,
    traces_to_parquet,
)
from trucktrack.valhalla import map_match_dataframe_full
from trucktrack.valhalla.map_matching import _adaptive_breakage_distance
from trucktrack.visualize import plot_trace_layers, save_map, serve_map

TILE_EXTRACT = os.environ.get(
    "VALHALLA_TILE_EXTRACT", "valhalla_tiles/valhalla_tiles.tar"
)
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", "examples/trace_visualizations/output"))


@dataclass
class GeofenceGapExample:
    label: str
    origin: tuple[float, float]
    destination: tuple[float, float]
    geofence_center: tuple[float, float]
    geofence_radius_m: float
    output_filename: str


def run(example: GeofenceGapExample, *, serve: bool = False, port: int = 5000) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)

        config = TripConfig(
            origin=example.origin,
            destination=example.destination,
            departure_time=datetime(2025, 6, 15, 8, 0, tzinfo=UTC),
            seed=42,
            tile_extract=TILE_EXTRACT,
            errors=[
                ErrorConfig(
                    "geofence_gap",
                    probability=1.0,
                    params={
                        "center": example.geofence_center,
                        "radius_m": example.geofence_radius_m,
                    },
                ),
            ],
        )
        print(f"Generating {example.label} trace (seed={config.seed})...")
        points = generate_trace(config)
        print(f"  {len(points)} trace points after geofence gap")

        parquet_path = tmp_dir / "trace.parquet"
        traces_to_parquet([(points, config.trip_id)], str(parquet_path))
        df = read_parquet(str(parquet_path))

        print("Splitting...")
        gap_split = split_by_observation_gap(df, timedelta(minutes=5), min_length=3)
        split = split_by_stops(
            gap_split,
            max_diameter=50.0,
            min_duration=timedelta(minutes=2),
        )
        n_segments = split["segment_id"].n_unique()
        n_stops = split.filter(pl.col("is_stop"))["segment_id"].n_unique()
        n_trips = n_segments - n_stops
        print(f"  {n_segments} segment(s): {n_trips} trip(s), {n_stops} stop(s)")

        print("Map-matching...")
        movement = split.filter(~pl.col("is_stop"))
        matched_parts: list[pl.DataFrame] = []
        all_ways: list[int] = []
        all_shapes: list[list[tuple[float, float]]] = []
        for (seg_id,), seg in movement.group_by("segment_id", maintain_order=True):
            if len(seg) < 2:
                print(f"  segment {seg_id}: {len(seg)} pts — skipping (too short)")
                continue
            seg_points = list(zip(seg["lat"].to_list(), seg["lon"].to_list()))
            bd = _adaptive_breakage_distance(seg_points)
            matched, ways, shape = map_match_dataframe_full(
                seg, tile_extract=TILE_EXTRACT
            )
            matched_parts.append(matched)
            all_ways.extend(ways)
            if shape:
                all_shapes.append(shape)
            print(
                f"  segment {seg_id}: {len(seg)} pts, {len(ways)} OSM ways, "
                f"adaptive breakage_distance={bd:.0f} m"
            )

        result = pl.concat(matched_parts) if matched_parts else None

        print("Building map...")
        m = plot_trace_layers(
            raw=df,
            segments=split,
            matched=result,
            matched_shape=all_shapes or None,
        )

        if serve:
            serve_map(m, host="0.0.0.0", port=port)
        else:
            out_path = OUTPUT_DIR / example.output_filename
            save_map(m, out_path)
            print(f"  Saved {out_path}")
