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
build ID).  Edges no truck drove take their own Valhalla speed, scaled
to how fast trucks drove roads of their class and length, shaped by
their road class's week (where enough trucks measured the class),
unless ``--measured-only``.  Rerunning replaces the speeds; the pristine
extract is kept beside the tiles as ``valhalla_tiles.base.tar``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import polars as pl
from trucktrack.congestion import (
    edge_cells,
    fit_model,
    historical_speeds,
    speed_ratios,
)
from trucktrack.valhalla.traffic import add_predicted_traffic, graph_edges


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("matched", type=Path, help="run_map_matching output directory")
    parser.add_argument(
        "--config", type=Path, default=None, help="valhalla.json (default: discovered)"
    )
    parser.add_argument(
        "--measured-only",
        action="store_true",
        help="leave edges no truck drove at Valhalla's own speeds",
    )
    args = parser.parse_args(argv)

    obs = pl.scan_parquet(args.matched / "_observations" / "**" / "*.parquet")
    quality = pl.scan_parquet(args.matched / "_quality" / "**" / "*.parquet")
    cells = edge_cells(obs, quality=quality).collect()
    model = fit_model(cells)
    historical = historical_speeds(cells, model=model)
    sources = dict(historical["source"].value_counts().rows())
    print(
        f"{historical['edge_id'].n_unique()} measured edges; "
        f"edge-hours by source: {sources}",
        file=sys.stderr,
    )
    pattern = ratios = None
    if not args.measured_only:
        pattern = model[2]
        ratios = speed_ratios(model[0], cells, graph_edges(args.config))
        print(f"Truck speed / Valhalla speed: {ratios.rows()}", file=sys.stderr)
    counts = add_predicted_traffic(
        historical, config=args.config, pattern=pattern, ratios=ratios
    )
    print(
        f"Loaded {counts['edges']} measured and {counts['class_edges']} other edges "
        f"into {counts['files']} tiles",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
