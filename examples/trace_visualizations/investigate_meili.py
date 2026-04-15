"""Investigate map-matching gaps on a single Toronto → Hamilton trip.

Reproduces missing segments in the map-matched polyline returned by
Valhalla's trace_route endpoint.

Usage::

    uv run python examples/trace_visualizations/investigate_meili.py

Requires a ``valhalla.json`` in cwd.
"""

from __future__ import annotations

import os
import tempfile
from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl
from trucktrack import (
    generate_trace,
    read_parquet,
    split_by_observation_gap,
    split_by_stops,
    traces_to_parquet,
)
from trucktrack.generate import TripConfig
from trucktrack.valhalla import map_match_dataframe_full, map_match_route_shape
from trucktrack.valhalla._actor import _find_config
from trucktrack.visualize import plot_trace_layers, save_map

OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", "examples/trace_visualizations/output"))

# Toronto → Hamilton
ORIGIN = (43.6532, -79.3832)
DESTINATION = (43.2557, -79.8711)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)

        # 1. Generate trace.
        print("Generating trace...")
        config = TripConfig(
            origin=ORIGIN,
            destination=DESTINATION,
            departure_time=datetime(2025, 6, 15, 8, 0, tzinfo=UTC),
            seed=42,
            config=_find_config(),
        )
        points = generate_trace(config)
        print(f"  {len(points)} trace points")

        # 2. Write to parquet.
        parquet_path = tmp_dir / "trace.parquet"
        traces_to_parquet([(points, config.trip_id)], str(parquet_path))
        df = read_parquet(str(parquet_path))

        # 3. Split.
        print("Splitting...")
        gap_split = split_by_observation_gap(df, timedelta(minutes=5), min_length=3)
        split = split_by_stops(
            gap_split, max_diameter=50.0, min_duration=timedelta(minutes=2)
        )
        n_segments = split["segment_id"].n_unique()
        n_stops = split.filter(pl.col("is_stop"))["segment_id"].n_unique()
        print(f"  {n_segments} segment(s), {n_stops} stop(s)")

        # 4. Map-match movement segments.
        print("Map-matching...")
        movement = split.filter(~pl.col("is_stop"))
        matched_parts = []
        all_ways: list[int] = []
        all_shapes: list[list[tuple[float, float]]] = []
        for (seg_id,), seg in movement.group_by("segment_id", maintain_order=True):
            matched, ways, _ = map_match_dataframe_full(seg)
            pts = list(zip(seg["lat"].to_list(), seg["lon"].to_list(), strict=True))
            shapes = map_match_route_shape(pts)
            matched_parts.append(matched)
            all_ways.extend(ways)
            all_shapes.extend(shapes)
            print(
                f"  segment {seg_id}: {len(seg)} pts, {len(ways)} ways, "
                f"{len(shapes)} shape fragment(s)"
            )

        result = pl.concat(matched_parts) if matched_parts else None

        # 5. Visualize.
        print("Building map...")
        m = plot_trace_layers(
            raw=df,
            segments=split,
            matched=result,
            matched_shape=all_shapes or None,
        )

        out_path = OUTPUT_DIR / "investigate_meili.html"
        save_map(m, out_path)
        print(f"  Saved {out_path}")


if __name__ == "__main__":
    main()
