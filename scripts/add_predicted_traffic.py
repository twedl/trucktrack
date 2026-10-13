#!/usr/bin/env python3
"""Load truck speeds from map-matched observations into the Valhalla tiles.

Reads ``_observations/`` and ``_quality/`` from a
``run_map_matching(..., observations=ObservationConfig())`` output
directory, measures each edge's hourly speeds
(:func:`trucktrack.congestion.edge_cells`), fills in every hour of the
week (:func:`trucktrack.congestion.historical_speeds`,
:func:`trucktrack.congestion.flow_speeds`), and rebuilds the tile
extract with them as predicted traffic
(:func:`trucktrack.valhalla.traffic.add_predicted_traffic`)::

    uv run python scripts/add_predicted_traffic.py data/matched

The observations must be matched on the tiles being loaded (same tile
build ID).  Rerunning replaces the speeds; the pristine extract is kept
beside the tiles as ``valhalla_tiles.base.tar``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import polars as pl
from trucktrack.congestion import edge_cells, historical_speeds
from trucktrack.valhalla.traffic import add_predicted_traffic


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("matched", type=Path, help="run_map_matching output directory")
    parser.add_argument(
        "--config", type=Path, default=None, help="valhalla.json (default: discovered)"
    )
    args = parser.parse_args(argv)

    obs = pl.scan_parquet(args.matched / "_observations" / "**" / "*.parquet")
    quality = pl.scan_parquet(args.matched / "_quality" / "**" / "*.parquet")
    historical = historical_speeds(edge_cells(obs, quality=quality))
    sources = dict(historical["source"].value_counts().rows())
    print(
        f"{historical['edge_id'].n_unique()} edges; edge-hours by source: {sources}",
        file=sys.stderr,
    )
    counts = add_predicted_traffic(historical, config=args.config)
    print(
        f"Loaded {counts['edges']} edges from {counts['files']} tiles", file=sys.stderr
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
