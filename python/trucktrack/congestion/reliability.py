"""Truck travel times along a corridor, and their reliability by FHWA's measure.

A pass is one truck driving a stretch of a corridor (see
:mod:`trucktrack.valhalla.corridor`): its fixes on the corridor's
edges, in time order, with no fix off it and no long gap between.  The
time a truck crosses a point is interpolated between the fixes either
side of it, along the corridor's distance.  Trucks, not trips: a trip
split during a long stop on the highway is still one pass.

Reliability follows the FHWA's Truck Travel Time Reliability measure
(23 CFR 490.611): each stretch's passes are averaged per 15 minutes,
to the second, and the averages ranked within five periods of the
week; ``tttr`` is the 95th percentile over the 50th.  FHWA defines it
on short reporting segments; :func:`travel_time_reliability` gives the
same for any stretch, up to the whole corridor.
"""

from __future__ import annotations

import polars as pl

from trucktrack.congestion.config import DEFAULT_AGGREGATE_CONFIG
from trucktrack.query import truck_id_expr

# FHWA's periods (23 CFR 490.611(a)(1)), by local time: weekdays 6-10,
# 10-16 and 16-20, weekends 6-20, and every night 20-6.
PERIODS = ("AM peak", "Midday", "PM peak", "Weekend", "Overnight")
_SPAN = ("direction", "kind", "name", "begin_m", "end_m")
PASS_SCHEMA = {
    "direction": pl.String,
    "kind": pl.String,
    "name": pl.String,
    "begin_m": pl.Float64,
    "end_m": pl.Float64,
    "truck": pl.String,
    "entry": pl.Datetime("us", "UTC"),
    "travel_s": pl.Float64,
}


def fhwa_period(time: pl.Expr, tz: str = DEFAULT_AGGREGATE_CONFIG.local_tz) -> pl.Expr:
    """Which of FHWA's :data:`PERIODS` *time* falls in, on *tz*'s clock.

    A tz-naive *time* is taken as UTC.
    """
    local = time.dt.convert_time_zone(tz)
    hour = local.dt.hour()
    weekday = local.dt.weekday() <= 5  # Monday is 1
    day = hour.is_between(6, 20, closed="left")
    return (
        pl.when(~day)
        .then(pl.lit("Overnight"))
        .when(~weekday)
        .then(pl.lit("Weekend"))
        .when(hour < 10)
        .then(pl.lit("AM peak"))
        .when(hour < 16)
        .then(pl.lit("Midday"))
        .otherwise(pl.lit("PM peak"))
        .cast(pl.Enum(PERIODS))
        .alias("period")
    )


def corridor_passes(
    obs: pl.LazyFrame,
    edges: pl.DataFrame,
    spans: pl.DataFrame,
    max_gap_s: float = 600.0,
    backtrack_m: float = 200.0,
) -> pl.LazyFrame:
    """Every pass of every truck over every span, with its entry time and travel time.

    *obs* are edge observations (``_observations/``); *edges* and
    *spans* come from a :class:`~trucktrack.valhalla.corridor.Corridor`.
    A truck's fixes on the corridor in one direction make a run until a
    fix lands off it, the next comes more than *max_gap_s* later, or
    the truck goes back more than *backtrack_m* (a bad match).  A run
    crosses a point if it has fixes on both sides of it (or one on it);
    the crossing time is interpolated on chainage between them.  A run
    that crosses both ends of a span makes a pass: ``entry`` is when it
    crossed the start and ``travel_s`` the seconds to the end.
    """
    chainage = pl.col("begin_m") + pl.col("distance_along_edge") * (
        pl.col("end_m") - pl.col("begin_m")
    )
    fixes = (
        obs.select(
            truck=truck_id_expr(),
            t=pl.col("time").dt.convert_time_zone("UTC").dt.epoch("us") / 1e6,
            edge_id="edge_id",
            distance_along_edge="distance_along_edge",
        )
        .join(edges.lazy(), on="edge_id", how="left")
        .with_columns(ch=chainage)
        .sort("truck", "t")
    )
    new_run = (
        pl.col("direction").is_null()
        | pl.col("prev_direction").is_null()
        | (pl.col("direction") != pl.col("prev_direction"))
        | (pl.col("t") - pl.col("prev_t") > max_gap_s)
        | (pl.col("ch") < pl.col("prev_ch") - backtrack_m)
    )
    runs = (
        # The fix before in its own step: polars 1.31 can't nest windows.
        fixes.with_columns(
            pl.col(c).shift(1).over("truck").name.prefix("prev_")
            for c in ("direction", "t", "ch")
        )
        .with_columns(run=new_run.cum_sum().over("truck"))
        .filter(pl.col("direction").is_not_null())
        .with_columns(x=pl.col("ch").cum_max().over("truck", "run"))
        .select("truck", "run", "direction", "x", "t")
    )
    extents = runs.group_by("truck", "run", "direction").agg(
        lo=pl.col("x").min(), hi=pl.col("x").max()
    )
    points = pl.concat(
        [
            spans.lazy().select("direction", c="begin_m"),
            spans.lazy().select("direction", c="end_m"),
        ]
    ).unique()
    crossings = (
        extents.join(points, on="direction")
        .filter(pl.col("c").is_between(pl.col("lo"), pl.col("hi")))
        .select("truck", "run", "direction", "c")
        .sort("truck", "run", "c")
    )
    by = ["truck", "run"]
    after = runs.select(*by, x1="x", t1="t")
    before = runs.select(*by, x0="x", t0="t")
    crossed = (
        crossings.join_asof(
            after,
            left_on="c",
            right_on="x1",
            by=by,
            strategy="forward",
            check_sortedness=False,
        )
        .join_asof(
            before,
            left_on="c",
            right_on="x0",
            by=by,
            strategy="backward",
            allow_exact_matches=False,
            check_sortedness=False,
        )
        .with_columns(
            t=pl.when(pl.col("x1") == pl.col("c"))
            .then(pl.col("t1"))
            .otherwise(
                pl.col("t0")
                + (pl.col("c") - pl.col("x0"))
                / (pl.col("x1") - pl.col("x0"))
                * (pl.col("t1") - pl.col("t0"))
            )
        )
        .drop_nulls("t")
        .select("truck", "run", "direction", "c", "t")
    )
    return (
        spans.lazy()
        .select(*_SPAN)
        .join(crossed, left_on=["direction", "begin_m"], right_on=["direction", "c"])
        .join(
            crossed.rename({"t": "t_end"}),
            left_on=["direction", "end_m", "truck", "run"],
            right_on=["direction", "c", "truck", "run"],
        )
        .select(
            *_SPAN,
            "truck",
            entry=pl.from_epoch(
                (pl.col("t") * 1e6).round().cast(pl.Int64), "us"
            ).dt.replace_time_zone("UTC"),
            travel_s=pl.col("t_end") - pl.col("t"),
        )
        .sort(*_SPAN, "entry")
    )


def travel_time_reliability(
    passes: pl.LazyFrame | pl.DataFrame,
    tz: str = DEFAULT_AGGREGATE_CONFIG.local_tz,
) -> pl.LazyFrame:
    """Each span's travel time distribution and TTTR in each FHWA period.

    *passes* are :func:`corridor_passes`.  As FHWA's truck travel time
    data set: passes are averaged per 15 minutes of entry (to the
    second), and each average falls in the :func:`fhwa_period` of its
    15 minutes.  Over a period's averages: ``bins`` and the ``passes``
    behind them, ``mean_s`` and ``p10_s`` to ``p95_s`` (linear between
    ranks, as a spreadsheet's PERCENTILE), ``tttr`` (95th / 50th, to
    the hundredth) and ``buffer_index`` ((95th - mean) / mean).
    """
    bins = (
        passes.lazy()
        .with_columns(bin=pl.col("entry").dt.truncate("15m"))
        .group_by(*_SPAN, "bin")
        .agg(travel_s=pl.col("travel_s").mean().round(), passes=pl.len())
        .with_columns(fhwa_period(pl.col("bin"), tz))
    )
    t = pl.col("travel_s")
    pct = {f"p{q}_s": t.quantile(q / 100, "linear").round() for q in (10, 50, 80, 95)}
    return (
        bins.group_by(*_SPAN, "period")
        .agg(
            bins=pl.len(),
            passes=pl.col("passes").sum(),
            mean_s=t.mean(),
            **pct,
        )
        .with_columns(
            tttr=(pl.col("p95_s") / pl.col("p50_s")).round(2),
            buffer_index=(pl.col("p95_s") - pl.col("mean_s")) / pl.col("mean_s"),
        )
        .sort(*_SPAN, "period")
    )


def tttr_index(reliability: pl.LazyFrame | pl.DataFrame) -> pl.LazyFrame:
    """FHWA's TTTR Index of each direction: segments' worst TTTR, length-weighted.

    Over :func:`travel_time_reliability`'s ``"segment"`` rows: each
    segment's highest TTTR of the five periods, weighted by its length
    (23 CFR 490.613).  ``segments`` counts those with any period.
    """
    worst = (
        reliability.lazy()
        .filter(pl.col("kind") == "segment")
        .group_by(*_SPAN)
        .agg(tttr=pl.col("tttr").max())
        .with_columns(length_m=pl.col("end_m") - pl.col("begin_m"))
    )
    return (
        worst.group_by("direction")
        .agg(
            segments=pl.len(),
            tttr_index=(pl.col("tttr") * pl.col("length_m")).sum()
            / pl.col("length_m").sum(),
        )
        .sort("direction")
    )
