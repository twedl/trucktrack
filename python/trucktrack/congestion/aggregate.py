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
# A directed OSM way's: its edges run along it (forward) or against it.
_WAY: tuple[str, ...] = ("tile_build_id", "way_id", "forward")
# Edge attributes carried from the observations to every edge cell.
_EDGE_ATTRIBUTES: tuple[str, ...] = ("way_id", "forward", "road_class", "edge_length")

# Bins of hour_of_week, from Sunday 00:00.
HOURS_PER_WEEK = 168

# Local hours [start, end) in which Valhalla uses an edge's
# constrained_speed (and for requests with no time); freeflow_speed
# outside them.  Fixed in Valhalla (GraphTile::GetSpeed), not a setting.
VALHALLA_DAYTIME_HOURS = (7, 19)

# rush_window's defaults, as local hours [start, end): the weekday
# morning and evening rush, and the night, when roads run free.
AM_RUSH_HOURS = (7, 9)
PM_RUSH_HOURS = (16, 18)
NIGHT_HOURS = (0, 5)
# Cell speeds vary in the last bits from run to run (the order Polars
# sums them in), so windows are compared to this many decimals and a
# tie goes to the morning.
_TIE_DIGITS = 9


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


def rush_window(
    tz: str = DEFAULT_AGGREGATE_CONFIG.local_tz,
    am: tuple[int, int] = AM_RUSH_HOURS,
    pm: tuple[int, int] = PM_RUSH_HOURS,
    night: tuple[int, int] = NIGHT_HOURS,
) -> pl.Expr:
    """``"am"``, ``"pm"`` or ``"night"``: the windows :func:`rush_speeds` compares.

    Each is [start, end) local hours in *tz*: *am* and *pm* Monday to
    Friday, *night* any day.  Other times are null.  A bin for
    :func:`edge_cells`.
    """
    local = _local_time(tz)
    hour = local.dt.hour()
    weekday = local.dt.weekday() <= 5  # Monday is 1
    return (
        pl.when(weekday & hour.is_between(*am, closed="left"))
        .then(pl.lit("am"))
        .when(weekday & hour.is_between(*pm, closed="left"))
        .then(pl.lit("pm"))
        .when(hour.is_between(*night, closed="left"))
        .then(pl.lit("night"))
        .alias("window")
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


def rush_speeds(
    cells: pl.LazyFrame | pl.DataFrame, min_trucks: float = 1.0
) -> pl.LazyFrame:
    """Each edge's rush-hour speed against its free-flow speed.

    *cells* are :func:`edge_cells` by :func:`rush_window`.
    ``free_speed`` is the edge's night cell, and ``rush_speed`` the
    slower of its morning and evening cells, with ``rush_window``
    saying which: an edge into a city slows in the morning, out of it
    in the evening.  ``ratio`` is rush_speed / free_speed, the free
    travel time over the rush one, and ``delay_s`` the seconds rush
    hour adds to driving the whole edge, which add up along a route as
    ratios don't.  ``free_trucks`` and ``rush_trucks`` are the cells'
    effective trucks.  An edge gets a row only with a night cell and a
    rush cell, each of at least *min_trucks* and faster than 0: a cell
    whose every fix is stopped (mostly one sparse fix at a light, on a
    short edge) caught a stop, not the edge's speed.

    On synthetic trucks, against held-out trucks' 5 s data, ``ratio``
    is off by 0.08-0.09 on edges of 200 m or more (the road class's
    mean ratio, 0.13) and by 0.4 under 200 m (the class's, 0.24-0.30).
    """
    cells = cells.lazy().filter(pl.col("trucks") >= min_trucks, pl.col("speed") > 0)
    free = cells.filter(pl.col("window") == "night").select(
        *_EDGE, *_EDGE_ATTRIBUTES, free_speed="speed", free_trucks="trucks"
    )
    rush = (
        cells.filter(pl.col("window").is_in(["am", "pm"]))
        .group_by(_EDGE)
        .agg(
            pl.col("window", "speed", "trucks")
            .sort_by(pl.col("speed").round(_TIE_DIGITS), "window")
            .first()
        )
        .select(*_EDGE, rush_window="window", rush_speed="speed", rush_trucks="trucks")
    )
    hours = pl.col("edge_length") / 1000.0
    return (
        free.join(rush, on=_EDGE)
        .with_columns(
            ratio=pl.col("rush_speed") / pl.col("free_speed"),
            delay_s=3600.0
            * (hours / pl.col("rush_speed") - hours / pl.col("free_speed")),
        )
        .sort(_EDGE)
    )


def way_rush_speeds(
    cells: pl.LazyFrame | pl.DataFrame, min_trucks: float = 1.0
) -> pl.LazyFrame:
    """Each directed OSM way's rush-hour speed against its free-flow speed.

    :func:`rush_speeds` for a way in one direction (``way_id`` and
    ``forward``) instead of one of its Valhalla edges: way ids outlast
    a tile build and join to OpenStreetMap.  For each rush window, the
    way's edges with a cell in it and a night cell give their summed
    length over their summed travel time in each, so both speeds come
    from the same stretch of road.  ``rush_window`` is the window with
    the lower ratio, and ``length`` and ``edge_ids`` the edges behind
    it.  ``free_trucks`` and ``rush_trucks`` are the fewest effective
    trucks on any of them.  Other columns are as in :func:`rush_speeds`,
    with the same *min_trucks* and stopped cells left out.
    """
    cells = cells.lazy().filter(pl.col("trucks") >= min_trucks, pl.col("speed") > 0)
    hours = pl.col("edge_length") / 1000.0 / pl.col("speed")
    free = cells.filter(pl.col("window") == "night").select(
        *_EDGE, free_h=hours, free_trucks="trucks"
    )
    windows = (
        cells.filter(pl.col("window").is_in(["am", "pm"]))
        .join(free, on=_EDGE)
        .group_by(*_WAY, "window")
        .agg(
            pl.col("road_class").first(),
            length=pl.col("edge_length").sum(),
            edge_ids=pl.col("edge_id").sort(),
            free_h=pl.col("free_h").sum(),
            rush_h=hours.sum(),
            free_trucks=pl.col("free_trucks").min(),
            rush_trucks=pl.col("trucks").min(),
        )
    )
    km = pl.col("length") / 1000.0
    return (
        windows.group_by(_WAY)
        .agg(
            pl.all()
            .sort_by((pl.col("free_h") / pl.col("rush_h")).round(_TIE_DIGITS), "window")
            .first()
        )
        .select(
            *_WAY,
            "road_class",
            "length",
            "edge_ids",
            free_speed=km / pl.col("free_h"),
            free_trucks="free_trucks",
            rush_window="window",
            rush_speed=km / pl.col("rush_h"),
            rush_trucks="rush_trucks",
            ratio=pl.col("free_h") / pl.col("rush_h"),
            delay_s=3600.0 * (pl.col("rush_h") - pl.col("free_h")),
        )
        .sort(_WAY)
    )
