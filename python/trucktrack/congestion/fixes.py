"""Per-trip operations on a truck's GPS fixes, before map matching.

Both functions look at a fix's neighbours in time, so they run on a
whole trip before any rows are dropped (stop rows, unmatched points).
"""

from __future__ import annotations

import polars as pl


def _seconds(duration: pl.Expr) -> pl.Expr:
    return duration.dt.total_microseconds() / 1e6


def time_weights[FrameT: (pl.DataFrame, pl.LazyFrame)](
    frame: FrameT,
    *,
    max_half_gap_s: float = 30.0,
    id_col: str = "id",
    time_col: str = "time",
    weight_col: str = "w",
) -> FrameT:
    """Add the time, in seconds, that each fix stands for.

    A fix's speed is instantaneous, so it stands for the time around
    it: half the gap back to the previous fix plus half the gap on to
    the next, each side capped at *max_half_gap_s*.  A trip's first and
    last fixes get only their inner side.

    The weighted mean of fix speeds on an edge, Σ speed·w / Σ w, is the
    edge's space-mean speed (Σ metres / Σ seconds).  Even when fixes
    are sparser than edges, a traversal gets a fix with probability
    proportional to its time on the edge, and the weight undoes each
    truck's reporting rate.  An unweighted mean weights trucks by how
    often they report: for the same time on an edge, a 1 s truck
    contributes sixty times the fixes of a 60 s truck.

    The cap keeps a fix next to a long reporting gap from standing in
    for minutes of driving it didn't see.  Trucks reporting less often
    than every ``2 * max_half_gap_s`` count for less than their time on
    the road, which costs nothing unless they drive differently.

    Rows repeating a trip's timestamp get ``w = 0`` and gaps are
    measured between distinct timestamps, so a duplicated fix is
    dropped rather than counted twice.  The first row, in input order,
    keeps the weight.  The result is sorted by trip, then time.
    """
    t = pl.col(time_col)
    first = pl.struct(id_col, time_col).is_first_distinct()
    distinct_t = pl.when(first).then(t)
    prev_t = distinct_t.shift(1).forward_fill()
    next_t = distinct_t.shift(-1).backward_fill()

    def half(gap: pl.Expr) -> pl.Expr:
        return (_seconds(gap) / 2).clip(upper_bound=max_half_gap_s).fill_null(0.0)

    w = pl.when(first).then(half(t - prev_t) + half(next_t - t)).otherwise(0.0)
    return frame.sort(id_col, time_col, maintain_order=True).with_columns(
        w.over(id_col).alias(weight_col)
    )


def flag_dwell[FrameT: (pl.DataFrame, pl.LazyFrame)](
    frame: FrameT,
    *,
    stationary_kmh: float = 2.0,
    max_stationary_step_m: float = 30.0,
    max_queue_stop_s: float = 180.0,
    id_col: str = "id",
    time_col: str = "time",
    lat_col: str = "lat",
    lon_col: str = "lon",
    speed_col: str = "speed",
    weight_col: str = "w",
    flag_col: str = "dwell",
) -> FrameT:
    """Flag stationary fixes that belong to a dwell rather than a queue.

    A zero-speed fix in a queue or at a signal is part of the delay an
    edge's speed should reflect.  A zero from a truck parked at a curb,
    a weigh station or a fuel stop isn't.  Stops shorter than the stop
    splitter's minimum, and longer stops the traffic filter
    reclassified as movement, stay inside trips with their zeros.

    A stationary run is consecutive fixes below *stationary_kmh*, each
    within *max_stationary_step_m* of the one before.  It is a dwell
    when its fixes stand for more than *max_queue_stop_s* in total, by
    their time weights (run :func:`time_weights` first).

    The step limit matters for sparse trucks: two zeros a minute apart
    and a few hundred metres down the road are two signal stops, not
    one long wait.  It also lets a queue that creeps forward between
    cycles break into short runs.

    The summed weight estimates how long the truck stood still whatever
    its reporting rate.  The span from first to last zero would not: it
    falls short by about one reporting interval, so a sparse truck's
    idle would pass for a queue where a dense truck's was a dwell.  A
    lone zero from a truck reporting every 30 min is capped (see
    :func:`time_weights`) and counts as a queue.

    A null speed is not stationary.  The result is sorted by trip, then
    time.
    """
    lat = pl.col(lat_col).radians()
    lon = pl.col(lon_col).radians()
    step_m = (
        6_371_000.0
        * ((lat - lat.shift(1)) ** 2 + ((lon - lon.shift(1)) * lat.cos()) ** 2).sqrt()
    )
    stationary = (pl.col(speed_col) < stationary_kmh).fill_null(False)
    run_start = (
        stationary.ne(stationary.shift(1)) | (step_m > max_stationary_step_m)
    ).fill_null(True)
    run_s = pl.col(weight_col).sum().over(id_col, "_run")
    return (
        frame.sort(id_col, time_col, maintain_order=True)
        .with_columns(run_start.cum_sum().over(id_col).alias("_run"))
        .with_columns((stationary & (run_s > max_queue_stop_s)).alias(flag_col))
        .drop("_run")
    )
