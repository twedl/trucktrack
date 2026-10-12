"""Per-trip operations on a truck's GPS fixes, before map matching.

Both functions look at a fix's neighbours in time, so they run on a
whole trip before any rows are dropped (stop rows, unmatched points).
They read ``id``, ``time``, ``lat``, ``lon`` and ``speed`` (km/h) and
sort by trip, then time.
"""

from __future__ import annotations

import polars as pl

from trucktrack.congestion.config import DEFAULT_CONFIG, ObservationConfig


def _step_m() -> pl.Expr:
    """Metres from the previous fix (flat-earth; fine at fix spacing)."""
    lat = pl.col("lat").radians()
    lon = pl.col("lon").radians()
    return (
        6_371_000.0
        * ((lat - lat.shift(1)) ** 2 + ((lon - lon.shift(1)) * lat.cos()) ** 2).sqrt()
    )


def _seconds(duration: pl.Expr) -> pl.Expr:
    return duration.dt.total_microseconds() / 1e6


def time_weights[FrameT: (pl.DataFrame, pl.LazyFrame)](
    frame: FrameT, config: ObservationConfig = DEFAULT_CONFIG
) -> FrameT:
    """Add ``w``: the time, in seconds, that each fix stands for.

    A fix's speed is instantaneous, so it stands for the time around
    it: half the gap back to the previous fix plus half the gap on to
    the next, each side capped at ``config.max_half_gap_s``.  A trip's
    first and last fixes get only their inner side.

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
    keeps the weight.
    """
    t = pl.col("time")
    first = t.ne(t.shift(1)).fill_null(True)
    distinct_t = pl.when(first).then(t)
    prev_t = distinct_t.shift(1).forward_fill()
    next_t = distinct_t.shift(-1).backward_fill()

    def half(gap: pl.Expr) -> pl.Expr:
        return (
            (_seconds(gap) / 2).clip(upper_bound=config.max_half_gap_s).fill_null(0.0)
        )

    w = pl.when(first).then(half(t - prev_t) + half(next_t - t)).otherwise(0.0)
    return frame.sort("id", "time", maintain_order=True).with_columns(
        w.over("id").alias("w")
    )


def flag_dwell[FrameT: (pl.DataFrame, pl.LazyFrame)](
    frame: FrameT, config: ObservationConfig = DEFAULT_CONFIG
) -> FrameT:
    """Add ``dwell``: stationary fixes from a dwell rather than a queue.

    A zero-speed fix in a queue or at a signal is part of the delay an
    edge's speed should reflect.  A zero from a truck parked at a curb,
    a weigh station or a fuel stop isn't.  Stops shorter than the stop
    splitter's minimum, and longer stops the traffic filter
    reclassified as movement, stay inside trips with their zeros.

    A stationary run is consecutive fixes below
    ``config.stationary_kmh``, each within
    ``config.max_stationary_step_m`` of the one before.  It is a dwell
    when its fixes stand for more than ``config.max_queue_stop_s`` in
    total, by their time weights (run :func:`time_weights` first).

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

    A null speed is not stationary.
    """
    stationary = (pl.col("speed") < config.stationary_kmh).fill_null(False)
    run_start = (
        stationary.ne(stationary.shift(1)) | (_step_m() > config.max_stationary_step_m)
    ).fill_null(True)
    run_s = pl.col("w").sum().over("id", "_run")
    return (
        frame.sort("id", "time", maintain_order=True)
        .with_columns(run_start.cum_sum().over("id").alias("_run"))
        .with_columns((stationary & (run_s > config.max_queue_stop_s)).alias("dwell"))
        .drop("_run")
    )


def _reach_m(v: pl.Expr, dt: pl.Expr, config: ObservationConfig) -> pl.Expr:
    """Furthest a truck at *v* m/s gets in *dt* s, speeding up as hard as it can.

    Speeds up at ``max_accel_mps2`` until ``max_truck_kmh`` (or *v*, if
    faster), then holds.
    """
    a = config.max_accel_mps2
    top = pl.max_horizontal(pl.lit(config.max_truck_kmh / 3.6), v)
    t_top = ((top - v) / a).clip(0.0, dt)
    return (v + (v + a * t_top)) / 2 * t_top + top * (dt - t_top)


def flag_speed_mismatch[FrameT: (pl.DataFrame, pl.LazyFrame)](
    frame: FrameT, config: ObservationConfig = DEFAULT_CONFIG
) -> FrameT:
    """Add ``speed_mismatch``: reported speeds too low for the ground covered.

    Two consecutive fixes are a mismatch when they are further apart,
    by more than ``config.position_noise_m``, than a truck could get
    between them even leaving at the faster of their two speeds and
    speeding up at ``config.max_accel_mps2`` to
    ``config.max_truck_kmh``.  Neither speed explains the movement, so
    both fixes are flagged.

    This catches a speed stuck at zero while the truck drives on, a
    device fault that would put false queue time on the edges it
    passes.  A pair where either speed accounts for the distance
    passes, so a real stop never trips it, nor does a lone stuck zero
    between moving fixes.  The denser the fixes, the tighter the bound:
    zeros 20 s apart are caught above about 40 km/h, zeros a minute
    apart only above about 85 km/h, since slower than that they could
    be stop-and-go traffic.  A position glitch also trips it.  A speed
    too high for the ground covered isn't checked: the road can be
    longer than the straight line.
    """
    v = pl.col("speed") / 3.6
    dt = _seconds(pl.col("time") - pl.col("time").shift(1))
    reach = _reach_m(pl.max_horizontal(v, v.shift(1)), dt, config)
    too_far = (dt > 0) & (_step_m() > reach + config.position_noise_m)
    too_far = too_far.fill_null(False)
    return frame.sort("id", "time", maintain_order=True).with_columns(
        (too_far | too_far.shift(-1).fill_null(False))
        .over("id")
        .alias("speed_mismatch")
    )
