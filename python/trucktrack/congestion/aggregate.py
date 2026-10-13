"""Space-mean speed per edge and time bin, from edge observations.

A cell is an edge over a time bin, by default the local hour of the
week.  Its speed is Σ speed·w / Σ w over the cell's usable
observations: Σ metres / Σ seconds, the space-mean speed (see
:func:`~trucktrack.congestion.time_weights`).  This is measurement
only: each cell is what trucks saw on that edge in that bin.  Filling
in the bins they didn't see, and steadying thin cells, is
:mod:`trucktrack.congestion.impute`.
"""

from __future__ import annotations

from collections.abc import Sequence

import polars as pl

from trucktrack.congestion.config import DEFAULT_AGGREGATE_CONFIG, AggregateConfig
from trucktrack.query import truck_id_expr

# An edge's key; edge ids are only valid within a tile build.
_EDGE: tuple[str, ...] = ("tile_build_id", "edge_id")
# Edge attributes carried from the observations to every edge cell.
_EDGE_ATTRIBUTES: tuple[str, ...] = ("way_id", "forward", "road_class", "edge_length")

# Bins of hour_of_week, from Sunday 00:00.
HOURS_PER_WEEK = 168

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

    *obs* are edge observations as written to ``_observations/``
    (:data:`OBSERVATION_COLUMNS` plus ``tile_build_id``);
    :func:`usable_observations` drops the ones that don't measure
    traffic, using *quality* if given.  *by* names the bin (default:
    :func:`hour_of_week` in ``config.local_tz``).

    Rows carry the edge's attributes (``way_id``, ``forward``,
    ``road_class``, and ``edge_length``, the longest length seen), the
    bin, and the cell statistics of :func:`cell_speeds`, one per edge
    and bin with a usable fix.  Nothing is borrowed from other edges or
    bins: a bin the trucks didn't measure has no row.  Each
    ``tile_build_id`` is aggregated separately.

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
