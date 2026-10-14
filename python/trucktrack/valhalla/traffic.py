"""Load per-edge truck speeds into the Valhalla tiles as predicted traffic.

Valhalla keeps three speeds on an edge besides its default: a
predicted speed for each five minutes of the week, which it uses for
requests with a time; ``constrained_speed``, for daytime requests
without predicted speeds and for every request with no time; and
``freeflow_speed``, for night-time requests without predicted speeds.
:func:`~trucktrack.congestion.historical_speeds` and
:func:`~trucktrack.congestion.flow_speeds` give all three.

``valhalla_add_predicted_traffic`` reads them from CSVs laid out like
the tiles, one line per edge, and rewrites tiles in a tile directory.
It only adds: run on tiles that already carry traffic it would leave
the old speeds of edges it isn't given, so the tiles are always rebuilt
from the extract as first packed (see
:func:`~trucktrack.valhalla._build_id.pristine_extract`).  Edge GraphIds
don't change, so the tile build ID still holds.

From the command line, measure, impute and load the speeds in
``run_map_matching(..., observations=ObservationConfig())`` output::

    python -m trucktrack.valhalla.traffic data/matched

Edges no truck drove take their own Valhalla speed, scaled to how fast
trucks drove roads of their class and length, shaped by their road
class's week (where enough trucks measured the class), unless
``--measured-only``.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

import numpy as np
import polars as pl
from valhalla.baldr import GraphId
from valhalla.baldr.utils import (
    BUCKETS_PER_WEEK,
    SPEED_BUCKET_SIZE_MINUTES,
    compress_speed_buckets,
    encode_compressed_speeds,
)

from trucktrack.congestion import (
    HOURS_PER_WEEK,
    class_weeks,
    edge_cells,
    fit_model,
    flow_speeds,
    historical_speeds,
    length_band,
    speed_ratios,
)
from trucktrack.valhalla._actor import resolve_config
from trucktrack.valhalla._build_id import pristine_extract, tile_build_id
from trucktrack.valhalla._tiles import pack_tiles, read_edges, valhalla_bin

# Valhalla takes any speed of at least 1 km/h, and warns about speeds
# above the 140 km/h it assumes no road exceeds.
SPEED_RANGE_KMH = (1.0, 140.0)


def _bucket_weights() -> np.ndarray:
    """(168, 2016) weights taking hourly speeds to five-minute buckets.

    Hour 0 is Sunday 00:00 local, like Valhalla's week.  Each hour's
    speed sits at its middle, with straight lines between, around the
    week: Valhalla stores the week as 200 DCT coefficients, which can't
    hold the steps of an hourly staircase anyway.
    """
    hour_middles = np.arange(HOURS_PER_WEEK) * 60.0 + 30.0
    bucket_middles = (np.arange(BUCKETS_PER_WEEK) + 0.5) * SPEED_BUCKET_SIZE_MINUTES
    return np.stack(
        [
            np.interp(bucket_middles, hour_middles, hour, period=HOURS_PER_WEEK * 60)
            for hour in np.eye(HOURS_PER_WEEK)
        ]
    )


_BUCKET_WEIGHTS = _bucket_weights()


def speed_buckets(hourly: np.ndarray | list[float]) -> np.ndarray:
    """Expand 168 hourly speeds (or rows of them) to Valhalla's 2016 buckets.

    See :func:`_bucket_weights` for how hours become buckets.
    """
    speeds = np.asarray(hourly, dtype=np.float64)
    if speeds.shape[-1:] != (HOURS_PER_WEEK,):
        raise ValueError(f"need {HOURS_PER_WEEK} hourly speeds, got {speeds.shape}")
    out: np.ndarray = (speeds @ _BUCKET_WEIGHTS).astype(np.float32)
    return out


def encode_week(buckets: np.ndarray) -> str:
    """2016 five-minute speeds as Valhalla's base64-encoded predicted speeds."""
    encoded: str = encode_compressed_speeds(compress_speed_buckets(buckets))
    return encoded


def _encoded_weeks(
    historical: pl.DataFrame, flows: pl.DataFrame | None = None
) -> pl.DataFrame:
    """Each edge's week encoded for Valhalla, and its flow speeds in whole km/h.

    Speeds are clipped to ``SPEED_RANGE_KMH``.  Returns ``edge_id``,
    ``freeflow_speed``, ``constrained_speed`` and ``week``.
    """
    lo, hi = SPEED_RANGE_KMH
    flows = flow_speeds(historical) if flows is None else flows
    h = historical.sort("edge_id", "hour_of_week")
    ids = h["edge_id"].to_numpy()
    edge_ids = (
        ids.reshape(-1, HOURS_PER_WEEK) if len(ids) % HOURS_PER_WEEK == 0 else None
    )
    if edge_ids is None or not (edge_ids == edge_ids[:, :1]).all():
        raise ValueError(f"every edge needs {HOURS_PER_WEEK} hours")
    buckets = speed_buckets(
        h["speed"].clip(lo, hi).to_numpy().reshape(-1, HOURS_PER_WEEK)
    )
    return (
        pl.DataFrame({"edge_id": edge_ids[:, 0]})
        .join(
            flows.select("edge_id", "freeflow_speed", "constrained_speed"),
            on="edge_id",
            how="left",
            maintain_order="left",
        )
        .with_columns(
            pl.col("freeflow_speed", "constrained_speed")
            .clip(lo, hi)
            .round()
            .cast(pl.Int32),
            week=pl.Series([encode_week(b) for b in buckets], dtype=pl.String),
        )
    )


def _csv_lines(weeks: pl.DataFrame) -> pl.DataFrame:
    """Traffic CSV ``path`` and ``line`` for each edge of :func:`_encoded_weeks`."""
    graph_id = weeks.with_columns(
        level=pl.col("edge_id") % 8,
        tile=(pl.col("edge_id") // 8) % (1 << 22),
        index=pl.col("edge_id") // (1 << 25),
    )
    tiles = (
        graph_id.select("level", "tile")
        .unique()
        .with_columns(
            path=pl.struct("level", "tile").map_elements(
                lambda t: str(
                    Path(GraphId(t["tile"], t["level"], 0)).with_suffix(".csv")
                ),
                return_dtype=pl.String,
            )
        )
    )
    return graph_id.join(tiles, on=["level", "tile"]).select(
        "path",
        line=pl.format(
            "{}/{}/{},{},{},{}",
            "level",
            "tile",
            "index",
            "freeflow_speed",
            "constrained_speed",
            "week",
        ),
    )


def traffic_rows(
    historical: pl.DataFrame, flows: pl.DataFrame | None = None
) -> pl.DataFrame:
    """One line of traffic CSV per edge, and the CSV it belongs in.

    *historical* is :func:`~trucktrack.congestion.historical_speeds`,
    *flows* :func:`~trucktrack.congestion.flow_speeds` (by default, of
    *historical*).  Speeds are clipped to ``SPEED_RANGE_KMH``; flow
    speeds are whole km/h.  Returns ``path`` (the CSV, relative to the
    traffic directory) and ``line``.
    """
    return _csv_lines(_encoded_weeks(historical, flows))


def class_rows(weeks: pl.DataFrame, edges: pl.DataFrame) -> pl.DataFrame:
    """Traffic CSV lines for *edges* no truck measured, from their class's week.

    *weeks* is :func:`~trucktrack.congestion.class_weeks`; *edges* has
    ``edge_id``, ``road_class`` and ``speed`` (see :func:`read_edges`).
    Edges whose class has no week are left out.  Each distinct week is
    encoded once.
    """
    combos = (
        weeks.select("road_class", "speed")
        .unique(maintain_order=True)
        .with_row_index("combo")
        .with_columns(pl.col("combo").cast(pl.UInt64))
    )
    encoded = _encoded_weeks(
        weeks.join(combos, on=["road_class", "speed"]).select(
            pl.lit("").alias("tile_build_id"),
            pl.col("combo").alias("edge_id"),
            "hour_of_week",
            speed="week_speed",
        )
    ).rename({"edge_id": "combo"})
    return _csv_lines(
        edges.select("edge_id", pl.col("road_class").cast(pl.String), "speed")
        .join(combos, on=["road_class", "speed"])
        .join(encoded, on="combo")
        .select("edge_id", "freeflow_speed", "constrained_speed", "week")
    )


def graph_edges(config: str | Path | None = None) -> pl.DataFrame:
    """Every road edge a truck may drive in the tiles *config* points at.

    See :func:`~trucktrack.valhalla._tiles.read_edges`; read from the
    extract as first packed.
    """
    return read_edges(pristine_extract(resolve_config(config)))


def write_traffic_csvs(rows: pl.DataFrame, traffic_dir: str | Path) -> int:
    """Write :func:`traffic_rows` under *traffic_dir*; return the number of files."""
    traffic_dir = Path(traffic_dir)
    for (path,), group in rows.group_by("path"):
        out = traffic_dir / str(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("\n".join(group["line"]) + "\n")
    return rows["path"].n_unique()


def add_predicted_traffic(
    historical: pl.DataFrame,
    flows: pl.DataFrame | None = None,
    config: str | Path | None = None,
    *,
    pattern: pl.DataFrame | None = None,
    ratios: pl.DataFrame | None = None,
) -> dict[str, int]:
    """Rebuild the tile extract *config* points at with these speeds.

    Starts from the extract as first packed (see the module notes), so
    edges left out get Valhalla's own speeds back.  The speeds must come
    from observations matched on these tiles: their ``tile_build_id``
    must be the tiles'.  *flows* defaults to
    :func:`~trucktrack.congestion.flow_speeds` of *historical*.

    With *pattern* (:func:`~trucktrack.congestion.fit_model`'s), every
    other road edge a truck may drive takes its own Valhalla speed
    shaped by its class's week (:func:`~trucktrack.congestion.class_weeks`),
    so roads no truck drove slow down at rush hour too and routes don't
    escape onto them.  With *ratios*
    (:func:`~trucktrack.congestion.speed_ratios`), that speed is first
    scaled to how fast trucks were measured driving roads of the class
    and length; classes and bands without one keep Valhalla's.

    Returns counts of measured ``edges``, ``class_edges`` and traffic
    ``files`` written.  Restart any process holding the old extract
    open to see the new speeds.
    """
    config_path = resolve_config(config)
    build_id = tile_build_id(config_path)
    builds = set(historical["tile_build_id"])
    if builds != {build_id}:
        raise ValueError(
            f"speeds are for tile builds {sorted(builds)}, tiles are {build_id}"
        )
    base = pristine_extract(config_path)
    settings = json.loads(config_path.read_text())
    extract = Path(settings["mjolnir"]["tile_extract"])
    rows = traffic_rows(historical, flows)
    class_edges = 0
    if pattern is not None:
        edges = read_edges(base).join(
            historical.select("edge_id").unique(), on="edge_id", how="anti"
        )
        if ratios is not None:
            edges = (
                edges.with_columns(pl.col("road_class").cast(pl.String), length_band())
                .join(ratios, on=["road_class", "band"], how="left")
                .with_columns(speed=pl.col("speed") * pl.col("ratio").fill_null(1.0))
            )
        weeks = class_weeks(pattern, edges.select("road_class", "speed").unique())
        others = class_rows(weeks, edges)
        class_edges = others.height
        rows = pl.concat([rows, others])

    with tempfile.TemporaryDirectory(dir=extract.parent, prefix=".traffic-") as tmp:
        work = Path(tmp)
        tiles, traffic = work / "tiles", work / "traffic"
        with tarfile.open(base) as tar:
            tar.extractall(tiles, filter="data")
        files = write_traffic_csvs(rows, traffic)
        settings["mjolnir"]["tile_dir"] = str(tiles)
        settings["mjolnir"].pop("tile_extract", None)
        tool_config = work / "valhalla.json"
        tool_config.write_text(json.dumps(settings))
        subprocess.run(
            [
                str(valhalla_bin("valhalla_add_predicted_traffic")),
                "-c",
                str(tool_config),
                "-j",
                str(os.cpu_count() or 1),
                str(traffic),
            ],
            check=True,
            capture_output=True,
        )
        pack_tiles(tiles, extract)
    return {
        "edges": rows.height - class_edges,
        "class_edges": class_edges,
        "files": files,
    }


def main(argv: list[str] | None = None) -> int:
    """Measure, impute and load speeds: ``python -m trucktrack.valhalla.traffic``."""
    parser = argparse.ArgumentParser(
        prog="python -m trucktrack.valhalla.traffic",
        description="Load truck speeds from map-matched observations into the "
        "Valhalla tiles.",
    )
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
    level, depth, pattern = fit_model(cells)
    historical = historical_speeds(cells, model=(level, depth, pattern))
    sources = dict(historical["source"].value_counts().rows())
    print(
        f"{historical['edge_id'].n_unique()} measured edges; "
        f"edge-hours by source: {sources}",
        file=sys.stderr,
    )
    ratios = None
    if not args.measured_only:
        ratios = speed_ratios(level, cells, graph_edges(args.config))
        print(f"Truck speed / Valhalla speed: {ratios.rows()}", file=sys.stderr)
    counts = add_predicted_traffic(
        historical,
        config=args.config,
        pattern=None if args.measured_only else pattern,
        ratios=ratios,
    )
    print(
        f"Loaded {counts['edges']} measured and {counts['class_edges']} other edges "
        f"into {counts['files']} tiles",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
