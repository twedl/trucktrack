"""Space-mean speed per edge and time bin, from edge observations.

A cell is an edge (or a way, in one direction) over a time bin, by
default the local hour of the week.  Its speed is Σ speed·w / Σ w over
the cell's usable observations: Σ metres / Σ seconds, the space-mean
speed (see :func:`~trucktrack.congestion.time_weights`).

A cell is only as good as the trucks behind it.  A truck reporting
every minute gets a fix on a small share of its traversals of a short
edge, so many edge cells have no fix even though trucks drove them.
Such a cell can take its way's cell (the same OSM way, in the same
direction, over the same bin), which pools the way's edges.

Pooling along a way fills gaps but adds little evidence: one truck
drives the whole way, so the way's cell is mostly the edge cell's own
trucks on neighbouring edges.  It also only suits edges that share
the way's flow.  Measured on synthetic trucks against held-out trucks
reporting every 5 s, a way's hourly speed misses an edge of 200 m to
1 km by 6 km/h where the edge's own mean over the week misses by
11 km/h.  Under 200 m it misses by 17-19 km/h against 13: short edges
are junction pieces whose speed depends on whether a truck turns or
stops, not on the flow along the way.
"""

from __future__ import annotations

from collections.abc import Sequence

import polars as pl

from trucktrack.congestion.config import DEFAULT_AGGREGATE_CONFIG, AggregateConfig
from trucktrack.query import truck_id_expr

# Keys of an edge and of a way (in one direction), before the time bin.
# Edge ids are only valid within a tile build.
_EDGE: tuple[str, ...] = ("tile_build_id", "edge_id")
_WAY: tuple[str, ...] = ("tile_build_id", "way_id", "forward")
# Edge attributes carried from the observations to every edge cell.
_EDGE_ATTRIBUTES: tuple[str, ...] = ("way_id", "forward", "road_class", "edge_length")
_CELL_STATS: tuple[str, ...] = ("speed", "seconds", "fixes", "trucks")

# Local hours [start, end) in which Valhalla uses an edge's
# constrained_speed (and for requests with no time); freeflow_speed
# outside them.  Fixed in Valhalla (GraphTile::GetSpeed), not a setting.
VALHALLA_DAYTIME_HOURS = (7, 19)


def _local_time(tz: str) -> pl.Expr:
    """``time`` on the clock in *tz*; a tz-naive ``time`` is taken as UTC."""
    return pl.col("time").dt.convert_time_zone(tz)


def hour_of_week(tz: str = DEFAULT_AGGREGATE_CONFIG.local_tz) -> pl.Expr:
    """Hour of the week of ``time`` in *tz*, 0-167, counted from Sunday 00:00.

    Valhalla's predicted speeds run from Sunday 00:00 in the edge's
    local time, so bins follow the clock on the road: a weekday 08:00
    rush is hour 8 of its day in winter and in summer, across the DST
    change.  A tz-naive ``time`` is taken as UTC.
    """
    local = _local_time(tz)
    day = (local.dt.weekday() % 7).cast(pl.Int16)  # Monday is 1, Sunday 7
    return (day * 24 + local.dt.hour()).alias("hour_of_week")


def flow_window(tz: str = DEFAULT_AGGREGATE_CONFIG.local_tz) -> pl.Expr:
    """``"constrained"`` from 07:00 to 19:00 in *tz*, any day, else ``"freeflow"``.

    The windows in which Valhalla falls back on an edge's
    ``constrained_speed`` and ``freeflow_speed`` when it has no
    predicted speed (see :data:`VALHALLA_DAYTIME_HOURS`), as a bin for
    :func:`edge_cells`.
    """
    day = _local_time(tz).dt.hour().is_between(*VALHALLA_DAYTIME_HOURS, closed="left")
    return (
        pl.when(day)
        .then(pl.lit("constrained"))
        .otherwise(pl.lit("freeflow"))
        .alias("flow")
    )


def usable_observations(
    obs: pl.LazyFrame,
    quality: pl.LazyFrame | None = None,
    config: AggregateConfig = DEFAULT_AGGREGATE_CONFIG,
) -> pl.LazyFrame:
    """Drop observations that don't speak to an edge's traffic speed.

    ``dwell`` zeros are a truck parked, not queued; ``wrong_way`` fixes
    sit on the opposing edge; ``speed_mismatch`` speeds don't account
    for the ground covered.  Speeds above ``config.max_speed_kmh`` are
    glitches, and nothing else rules out a speed that is too high (see
    :func:`~trucktrack.congestion.flag_speed_mismatch`); the
    generator's multipath fixes, for one, carry 100-190 km/h.
    *quality*, the matcher's ``_quality/`` rows, also drops trips whose
    match failed its checks (``ok`` is false): a match that detours or
    doubles back puts fixes on edges the truck never drove.
    """
    flagged = pl.col("dwell") | pl.col("wrong_way") | pl.col("speed_mismatch")
    obs = obs.filter(~flagged & (pl.col("speed") <= config.max_speed_kmh))
    if quality is not None:
        bad = quality.filter(~pl.col("ok")).select("id").unique()
        obs = obs.join(bad, on="id", how="anti")
    return obs


def cell_speeds(
    obs: pl.LazyFrame, by: Sequence[str], truck: pl.Expr | None = None
) -> pl.LazyFrame:
    """Space-mean speed per cell of *by*, and what's behind it.

    Returns *by* plus ``speed`` (Σ speed·w / Σ w, km/h), ``seconds``
    (Σ w), ``fixes``, and ``trucks``: the effective number of trucks,
    (Σ_k W_k)² / Σ_k W_k² over each truck's summed weight W_k.  It is
    the count when every truck stands for the same time, and nears 1
    when one truck stands for most of it, whatever the rest add.
    *truck* defaults to the trip id's truck (see
    :func:`~trucktrack.query.truck_id_from_trip`).
    """
    truck = truck_id_expr() if truck is None else truck
    per_truck = obs.group_by(*by, truck.alias("_truck")).agg(
        _vw=(pl.col("speed") * pl.col("w")).sum(),
        _w=pl.col("w").sum(),
        _n=pl.len(),
    )
    return per_truck.group_by(by).agg(
        speed=pl.col("_vw").sum() / pl.col("_w").sum(),
        seconds=pl.col("_w").sum(),
        fixes=pl.col("_n").sum(),
        trucks=pl.col("_w").sum() ** 2 / (pl.col("_w") ** 2).sum(),
    )


def _prefixed(cells: pl.LazyFrame, prefix: str) -> pl.LazyFrame:
    return cells.rename({c: f"{prefix}{c}" for c in _CELL_STATS})


def _edge_rows(lf: pl.LazyFrame) -> pl.LazyFrame:
    """One row per edge with its attributes.

    Valhalla gives the first and last edge of a match the length
    driven on it, so an edge's own length is the longest seen.
    """
    return lf.group_by(_EDGE).agg(
        pl.col(_EDGE_ATTRIBUTES[:-1]).first(), pl.col("edge_length").max()
    )


def edge_cells(
    obs: pl.LazyFrame,
    *,
    by: pl.Expr | None = None,
    quality: pl.LazyFrame | None = None,
    config: AggregateConfig = DEFAULT_AGGREGATE_CONFIG,
) -> pl.LazyFrame:
    """Each edge's measured space-mean speed per time bin, from its own fixes.

    *obs* and *quality* are as for :func:`edge_speeds`; *by* names the
    bin (default: :func:`hour_of_week` in ``config.local_tz``).  Rows
    carry the edge's attributes, the bin, and the cell statistics of
    :func:`cell_speeds`, one per edge and bin with a usable fix.
    Nothing is borrowed from other edges or bins: a bin the trucks
    didn't measure has no row.

    ``by=flow_window()`` measures Valhalla's ``constrained_speed`` and
    ``freeflow_speed``.
    """
    by = hour_of_week(config.local_tz) if by is None else by
    bin_col = by.meta.output_name()
    lf = usable_observations(obs, quality, config).with_columns(by)
    return (
        _edge_rows(lf)
        .join(cell_speeds(lf, [*_EDGE, bin_col]), on=_EDGE)
        .sort(*_EDGE, bin_col)
    )


def edge_speeds(
    obs: pl.LazyFrame,
    *,
    by: pl.Expr | None = None,
    quality: pl.LazyFrame | None = None,
    config: AggregateConfig = DEFAULT_AGGREGATE_CONFIG,
) -> pl.LazyFrame:
    """Space-mean speed per edge and time bin, pooling thin cells along the way.

    *obs* are edge observations as written to ``_observations/``
    (:data:`OBSERVATION_COLUMNS` plus ``tile_build_id``);
    :func:`usable_observations` drops the ones that don't measure
    traffic, using *quality* if given.  *by* names the time bin
    (default: :func:`hour_of_week` in ``config.local_tz``).

    An edge cell with ``config.min_trucks`` effective trucks or more
    (see :func:`cell_speeds`) gives its own speed.  A thinner one, on
    an edge at least ``config.min_pool_edge_m`` long, takes its way's
    cell (the edge's OSM way, same direction, same bin) if that has
    enough.  Otherwise it has no speed: a short edge's own speed over
    other bins is a better guess than its way's (see the module
    notes), and that is for the next step to weigh.  ``source`` says
    which: ``"edge"``, ``"way"`` or null.  Every edge seen at all gets a
    row for each bin its way was seen in, so an edge with no fix in a
    bin can still take its way's speed.

    Rows carry the edge's attributes (``way_id``, ``forward``,
    ``road_class``, and ``edge_length``, the longest length seen) and
    both cells' statistics,
    prefixed ``edge_`` and ``way_`` (``speed``, ``seconds``, ``fixes``,
    ``trucks``), so later steps can weigh them differently.  Each
    ``tile_build_id`` is aggregated separately.
    """
    by = hour_of_week(config.local_tz) if by is None else by
    bin_col = by.meta.output_name()
    # Derive each fix's truck once, for both cell levels.
    lf = usable_observations(obs, quality, config).with_columns(
        by, _truck=truck_id_expr().cast(pl.Categorical)
    )
    truck = pl.col("_truck")

    edges = _edge_rows(lf)
    way_cells = _prefixed(cell_speeds(lf, [*_WAY, bin_col], truck), "way_")
    own_cells = _prefixed(cell_speeds(lf, [*_EDGE, bin_col], truck), "edge_")

    edge_ok = pl.col("edge_trucks") >= config.min_trucks
    way_ok = (pl.col("way_trucks") >= config.min_trucks) & (
        pl.col("edge_length") >= config.min_pool_edge_m
    )
    return (
        edges.join(way_cells, on=_WAY)
        .join(own_cells, on=[*_EDGE, bin_col], how="left")
        .with_columns(
            speed=pl.when(edge_ok)
            .then(pl.col("edge_speed"))
            .when(way_ok)
            .then(pl.col("way_speed")),
            source=pl.when(edge_ok)
            .then(pl.lit("edge"))
            .when(way_ok)
            .then(pl.lit("way")),
        )
        .select(
            *_EDGE,
            *_EDGE_ATTRIBUTES,
            bin_col,
            "speed",
            "source",
            *(f"edge_{c}" for c in _CELL_STATS),
            *(f"way_{c}" for c in _CELL_STATS),
        )
        .sort(*_EDGE, bin_col)
    )
