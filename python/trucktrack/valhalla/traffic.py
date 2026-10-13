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
"""

from __future__ import annotations

import json
import os
import subprocess
import tarfile
import tempfile
from pathlib import Path

import numpy as np
import polars as pl
from valhalla.utils import (
    BUCKETS_PER_WEEK,
    SPEED_BUCKET_SIZE_MINUTES,
    GraphId,
    compress_speed_buckets,
    encode_compressed_speeds,
)

from trucktrack.congestion import HOURS_PER_WEEK, flow_speeds
from trucktrack.valhalla._actor import resolve_config
from trucktrack.valhalla._build_id import pristine_extract, tile_build_id
from trucktrack.valhalla._tiles import pack_tiles, valhalla_bin

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
    flow = (
        pl.DataFrame({"edge_id": edge_ids[:, 0]})
        .join(flows, on="edge_id", how="left", maintain_order="left")
        .select(
            pl.col("freeflow_speed", "constrained_speed")
            .clip(lo, hi)
            .round()
            .cast(pl.Int32)
        )
    )
    rows = []
    for edge_id, week, (free, constrained) in zip(
        edge_ids[:, 0], buckets, flow.iter_rows(), strict=True
    ):
        graph_id = GraphId(int(edge_id))
        rows.append(
            (
                str(Path(graph_id).with_suffix(".csv")),
                f"{graph_id},{free},{constrained},{encode_week(week)}",
            )
        )
    return pl.DataFrame(rows, schema=["path", "line"], orient="row")


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
) -> dict[str, int]:
    """Rebuild the tile extract *config* points at with these speeds.

    Starts from the extract as first packed (see the module notes), so
    edges left out of *historical* get Valhalla's own speeds back.  The
    speeds must come from observations matched on these tiles: their
    ``tile_build_id`` must be the tiles'.  *flows* defaults to
    :func:`~trucktrack.congestion.flow_speeds` of *historical*.
    Returns counts of ``edges`` and traffic ``files`` written.  Restart
    any process holding the old extract open to see the new speeds.
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
    return {"edges": rows.height, "files": files}
