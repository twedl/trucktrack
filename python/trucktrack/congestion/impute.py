"""Every edge's speed in every hour of the week, from its measured cells.

:func:`~trucktrack.congestion.edge_cells` measures what trucks saw: an
edge's space-mean speed in the hours they drove it, nothing more.  Most
edge-hours have no truck, and most that do have one.  Imputation fills
them in from a model of the edge,

    speed(e, h) = level(e) * pattern(h | class) ** depth(e, time of day),

and steadies the measured ones by blending them with it.

- ``pattern`` is the road class's hour-of-week shape, 1 at night: when
  roads of that kind slow down, and by how much on average.  Monday to
  Friday share one shape, so an hour's pattern doesn't hang on which
  edges trucks happened to drive that one weekday.  It is the one thing
  borrowed from other edges.
- ``level`` is the edge's own night-time speed, from all its cells,
  each deflated by the model at its hour.
- ``depth`` scales the class's dips to the edge's, in log terms: 0 is
  flat, 1 the class's, 2 twice as deep.  Deeper near a city core,
  shallower out of it, and separately for the morning peak, midday and
  the evening peak: a road into a city slows in the morning, out of it
  in the evening, and both ways at midday.  An edge with little
  rush-hour time stays near the class's 1.

A cell with ``trucks`` effective trucks gets ``trucks / (trucks + k)``
of its own speed and the rest from the model.  The more trucks behind
it, the less the class matters.

Measured on synthetic trucks against held-out trucks' 5 s data (km/h,
weighted by driving time): hours an edge measured, 5.0 from the cell
alone, 4.2 blended; hours it didn't, 14.4 from the edge's all-week
mean, 5.4 imputed (6.3 with a class pattern for each hour of the week
and an edge's depths split at noon).  Borrowing the edge's OSM way's
cell for the hour, as a speed or as a ratio to the model, does worse
(11.2 vs 13.9 where the way has one): one truck drives the whole way,
and short edges are junction pieces that don't share its flow.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import polars as pl

from trucktrack.congestion.aggregate import (
    _EDGE,
    HOURS_PER_WEEK,
    VALHALLA_DAYTIME_HOURS,
)

_PATTERN = ("road_class", "hour_of_week")
# The class pattern is 1 here: [start, end) local hours, any day.
_NIGHT_HOURS = (0, 5)
# Monday 00:00 to Friday 24:00, as [start, end) hours of the week.
_WEEKDAYS = (24, 6 * 24)
_ITERATIONS = 5
# A class-hour of stopped trucks or a few odd cells can't take the
# pattern to 0 (log(0) in the depth fit) or past twice the night speed;
# no edge's dips run deeper than three times the class's, in log terms.
_PATTERN_RANGE = (0.05, 2.0)
_DEPTH_RANGE = (0.0, 3.0)
_SOURCES = pl.Enum(["imputed", "blended", "measured"])
# class_weeks uses a class's pattern with at least this much measured
# driving behind it (about 14 minutes for each hour of the week), and
# only if it dips more than MIN_CLASS_DIP below night speed somewhere.
MIN_CLASS_HOURS = 40.0
MIN_CLASS_DIP = 0.05
# Edge lengths (m) splitting the bands speed_ratios measures in, and the
# seconds of driving at which a band's own ratio counts as much as its
# class's.
LENGTH_BANDS = (50.0, 200.0, 1000.0)
RATIO_SECONDS = 3600.0
# Model = (level, depth, pattern), as fit_model returns them.
Model = tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]


@dataclass(frozen=True)
class ImputeConfig:
    """Settings for :func:`historical_speeds`.

    *cell_trucks* is ``k``: the effective trucks at which a cell's own
    speed and the model's count equally.  *depth_seconds* is how much
    of an edge's own evidence on its depth (seconds of driving, times
    the squared log of the class pattern at the time) counts as much as
    the class's depth of 1.  An edge's depth changes at each local hour
    of *depth_split_hours*: by default one before 10:00 (the morning
    peak), one to 15:00 (midday) and one after (the evening peak), as a
    road into a city slows in the morning, out of it in the evening,
    and both ways at midday; ``()`` gives it one.
    *pattern_seconds* pulls each class-hour's pattern towards 1 (no
    congestion): with W seconds of driving behind it, its log is scaled
    by W / (W + pattern_seconds), so a class few trucks drove stays
    nearly flat.  *weekday_seconds* is the driving a weekday hour needs
    for its own shape to count as much as the shape Monday to Friday
    share at that hour of the day; by default (infinite) they share it
    outright.  A fleet with many trucks in every hour of the week can
    lower it to keep, say, Friday's lighter morning.

    Defaults are the best of those tried against held-out trucks'
    5 s data on the synthetic sample.
    """

    cell_trucks: float = 1.0
    depth_seconds: float = 10.0
    depth_split_hours: tuple[int, ...] = (10, 15)
    pattern_seconds: float = 600.0
    weekday_seconds: float = math.inf


DEFAULT_IMPUTE_CONFIG = ImputeConfig()


def _window(config: ImputeConfig) -> pl.Expr:
    """``window``: which of an edge's depths applies at ``hour_of_week``."""
    hour = pl.col("hour_of_week") % 24
    splits = (hour >= h for h in config.depth_split_hours)
    return pl.sum_horizontal(pl.lit(0), *splits).cast(pl.Int8).alias("window")


def _class_pattern(g: pl.DataFrame, config: ImputeConfig) -> pl.DataFrame:
    """Each class's pattern in all 168 hours, from cells and their edge's level.

    An hour's own pattern is Σ speed·w / Σ level·w over the class's
    cells in it, over the same at night.  A weekday hour takes the
    pattern of its hour of the day over Monday to Friday, its log
    pulled towards 0 by ``config.pattern_seconds``, mixed with its own
    by ``config.weekday_seconds``; a weekend hour its own, pulled
    towards 1.  Hours no truck drove get 1, or the weekdays' shape.
    """
    hour = pl.col("hour_of_week")
    cells = g.group_by(_PATTERN).agg(
        s=pl.col("s").sum(),
        lw=(pl.col("level") * pl.col("w")).sum(),
        w=pl.col("w").sum(),
    )
    night = (
        cells.filter((hour % 24).is_between(*_NIGHT_HOURS, closed="left"))
        .group_by("road_class")
        .agg(n=pl.col("s").sum() / pl.col("lw").sum())
    )
    weekdays = (
        cells.filter(hour.is_between(*_WEEKDAYS, closed="left"))
        .group_by("road_class", hour % 24)
        .agg(ds=pl.col("s").sum(), dlw=pl.col("lw").sum(), dw=pl.col("w").sum())
    )
    hours = pl.DataFrame(
        {"hour_of_week": pl.int_range(HOURS_PER_WEEK, dtype=pl.Int16, eager=True)}
    )

    def log_ratio(s: str, lw: str) -> pl.Expr:
        return (pl.col(s) / pl.col(lw) / pl.col("n")).clip(*_PATTERN_RANGE).log()

    def pull(x: pl.Expr, toward: pl.Expr | float, w: str, seconds: float) -> pl.Expr:
        """*x* pulled towards *toward* by *seconds* against *w*; else *toward*."""
        weight = pl.col(w) / (pl.col(w) + seconds)
        return (toward + weight * (x - toward)).fill_null(toward)

    own = log_ratio("s", "lw")  # null in hours without cells
    shared = pull(log_ratio("ds", "dlw"), 0.0, "dw", config.pattern_seconds)
    weekday = pull(own, shared, "w", config.weekday_seconds)
    return (
        cells.select("road_class")
        .unique()
        .join(hours, how="cross")
        .join(cells, on=_PATTERN, how="left")
        .join(night, on="road_class", how="left")
        .with_columns(pl.col("n").fill_null(1.0))
        .join(
            weekdays,
            left_on=["road_class", hour % 24],
            right_on=["road_class", "hour_of_week"],
            how="left",
        )
        .select(
            *_PATTERN,
            pattern=pl.when(hour.is_between(*_WEEKDAYS, closed="left"))
            .then(weekday)
            .otherwise(pull(own, 0.0, "w", config.pattern_seconds))
            .exp(),
            seconds=pl.col("w").fill_null(0.0),
        )
    )


def fit_model(
    cells: pl.DataFrame, config: ImputeConfig = DEFAULT_IMPUTE_CONFIG
) -> Model:
    """Fit each edge's ``level`` and ``depth``, and each class's ``pattern``.

    *cells* are hour-of-week :func:`~trucktrack.congestion.edge_cells`.
    The fit alternates, on the cells' time-weighted speeds:

    - level: Σ speed·w / Σ w·pattern**depth over the edge's cells;
    - pattern: Σ speed·w / Σ w·level over the class's cells in the
      hour, scaled to 1 at night (00:00-05:00) and pulled towards 1 by
      ``config.pattern_seconds``; Monday to Friday share one per hour
      of the day (see ``config.weekday_seconds``);
    - depth: the slope of log(speed / level) on log(pattern), weighted
      by seconds, shrunk towards 1 by ``config.depth_seconds``, within
      0 to 3.  Cell speeds under 1 km/h count as 1.

    Returns ``level`` per edge, ``depth`` per edge and time of day
    (``window``, see ``config.depth_split_hours``) it has cells in, and
    ``pattern`` per class and hour of the week, all 168, with the
    ``seconds`` of the class's own driving in the hour.  An edge only
    ever seen stopped has no level, and gets no speeds.
    """
    depth_key = [*_EDGE, "window"]
    c = cells.select(
        *_EDGE,
        *_PATTERN,
        _window(config),
        s=pl.col("speed") * pl.col("seconds"),  # Σ speed·w
        w=pl.col("seconds"),
        log_v=pl.col("speed").clip(lower_bound=1.0).log(),
    )
    depth_keys = c.select(depth_key).unique()
    edges = depth_keys.with_columns(depth=pl.lit(1.0))
    pattern = c.select(_PATTERN).unique().with_columns(pattern=pl.lit(1.0))
    tau = config.depth_seconds
    for _ in range(_ITERATIONS):
        g = c.join(edges, on=depth_key).join(pattern, on=_PATTERN)
        level = g.group_by(_EDGE).agg(
            level=pl.col("s").sum()
            / (pl.col("pattern").pow(pl.col("depth")) * pl.col("w")).sum()
        )
        level = level.filter(pl.col("level") > 0)
        g = g.drop("pattern").join(level, on=_EDGE)
        pattern = _class_pattern(g, config)
        x = pl.col("pattern").log()
        r = pl.col("log_v") - pl.col("level").log()
        fitted = (
            g.join(pattern.drop("seconds"), on=_PATTERN)
            .group_by(depth_key)
            .agg(sxy=(pl.col("w") * x * r).sum(), sxx=(pl.col("w") * x * x).sum())
            .select(*depth_key, depth=((pl.col("sxy") + tau) / (pl.col("sxx") + tau)))
        )
        edges = depth_keys.join(fitted, on=depth_key, how="left").with_columns(
            pl.col("depth").fill_null(1.0).clip(*_DEPTH_RANGE)
        )
    return level, edges, pattern


def historical_speeds(
    cells: pl.LazyFrame | pl.DataFrame,
    config: ImputeConfig = DEFAULT_IMPUTE_CONFIG,
    model: Model | None = None,
) -> pl.DataFrame:
    """Every measured edge's speed in each of the 168 local hours of the week.

    *cells* are hour-of-week :func:`~trucktrack.congestion.edge_cells`.
    Each edge gets a row per hour, in edge then hour order, with
    ``speed`` (km/h), ``own_weight``, the share of it from the hour's
    measurement, and ``source``: ``"imputed"`` for hours no truck
    measured, ``"blended"`` for the rest (``"measured"`` when
    ``config.cell_trucks`` is 0).  The model behind it is *model*, or
    :func:`fit_model` of *cells*.  Edges no truck measured, or only
    ever saw stopped, get no rows.
    """
    measured = cells.lazy().collect()
    level, depth, pattern = model or fit_model(measured, config)
    pattern = pattern.drop("seconds")
    edges = measured.select(*_EDGE, "road_class").unique().sort(_EDGE)
    hours = pl.DataFrame(
        {"hour_of_week": pl.int_range(HOURS_PER_WEEK, dtype=pl.Int16, eager=True)}
    ).with_columns(_window(config))
    k = config.cell_trucks
    n = pl.col("trucks").fill_null(0.0)
    own = pl.when(n > 0).then(n / (n + k)).otherwise(0.0)
    prior = pl.col("level") * pl.col("pattern").pow(pl.col("depth").fill_null(1.0))
    return (
        edges.join(hours, how="cross", maintain_order="left_right")
        .join(level, on=_EDGE, maintain_order="left")
        .join(depth, on=[*_EDGE, "window"], how="left", maintain_order="left")
        .join(pattern, on=_PATTERN, how="left", maintain_order="left")
        .join(
            measured.select(*_EDGE, "hour_of_week", "trucks", y="speed"),
            on=[*_EDGE, "hour_of_week"],
            how="left",
            maintain_order="left",
        )
        .with_columns(own_weight=own)
        .select(
            *_EDGE,
            "road_class",
            "hour_of_week",
            speed=pl.col("own_weight") * pl.col("y").fill_null(0.0)
            + (1.0 - pl.col("own_weight")) * prior,
            own_weight="own_weight",
            source=pl.when(n == 0)
            .then(pl.lit("imputed"))
            .otherwise(pl.lit("blended" if k > 0 else "measured"))
            .cast(_SOURCES),
        )
    )


def length_band(length: str | pl.Expr = "length") -> pl.Expr:
    """``band``: 0 under 50 m, 1 to 200 m, 2 to 1 km, 3 beyond.

    Short edges are mostly junction pieces, where trucks turn and stop,
    so they run further below a road's free speed than mid-block ones.
    """
    m = pl.col(length) if isinstance(length, str) else length
    return (
        pl.sum_horizontal(m >= edge for edge in LENGTH_BANDS)
        .cast(pl.Int8)
        .alias("band")
    )


def speed_ratios(
    level: pl.DataFrame,
    cells: pl.DataFrame,
    edges: pl.DataFrame,
    ratio_seconds: float = RATIO_SECONDS,
) -> pl.DataFrame:
    """How fast trucks drive each road class and length band, against base speed.

    *level* is :func:`fit_model`'s, *cells* the hourly cells it was fit
    on, and *edges* every edge's ``road_class``, ``length`` and
    ``speed`` without traffic (e.g. Valhalla's).  A class's ratio is
    Σ level·s / Σ speed·s over its measured edges, s the seconds of
    driving behind each; a band's is the same over the band's edges,
    pulled towards the class's by *ratio_seconds*.  Returns
    ``road_class``, ``band`` and ``ratio``.

    On synthetic trucks, motorways run at 0.98 of Valhalla's speed at
    every length, arterials at 0.4-0.5 under 200 m and 0.6-0.8 beyond.
    """
    measured = (
        level.join(cells.group_by(_EDGE).agg(s=pl.col("seconds").sum()), on=_EDGE)
        .join(edges.select("edge_id", "road_class", "length", "speed"), on="edge_id")
        .with_columns(pl.col("road_class").cast(pl.String), length_band())
    )

    def ratio(keys: list[str], name: str) -> pl.DataFrame:
        return measured.group_by(keys).agg(
            **{
                name: (pl.col("level") * pl.col("s")).sum()
                / (pl.col("speed") * pl.col("s")).sum()
            },
            **{f"{name}_s": pl.col("s").sum()},
        )

    k = ratio_seconds
    return (
        ratio(["road_class", "band"], "rb")
        .join(ratio(["road_class"], "rc"), on="road_class")
        .select(
            "road_class",
            "band",
            ratio=(pl.col("rb_s") * pl.col("rb") + k * pl.col("rc"))
            / (pl.col("rb_s") + k),
        )
        .sort("road_class", "band")
    )


def class_weeks(
    pattern: pl.DataFrame,
    speeds: pl.DataFrame,
    min_class_hours: float = MIN_CLASS_HOURS,
    min_dip: float = MIN_CLASS_DIP,
) -> pl.DataFrame:
    """Weeks for edges no truck measured: their own speed shaped by their class.

    *pattern* is :func:`fit_model`'s; *speeds* has ``road_class`` and
    ``speed`` (an edge's speed without traffic: Valhalla's, or that
    scaled by :func:`speed_ratios`) for each combination wanted.  Each
    gets ``speed * pattern`` for every hour of the week, as
    ``week_speed``, keyed by ``road_class``,
    ``speed`` and ``hour_of_week``.  A road class gets none if less
    than *min_class_hours* of measured driving is behind its pattern
    (it would be mostly noise), or if the pattern never dips more than
    *min_dip* below night speed (its edges would be as they were, at
    the cost of a predicted-speed record each).

    On synthetic trucks, for edges half the trucks never drove, scored
    against the other half's 5 s data, this takes Valhalla's flat speed
    from 19.9 km/h off to 10.5 (motorways 22.1 to 8.2).
    """
    supported = (
        pattern.group_by("road_class")
        .agg(hours=pl.col("seconds").sum() / 3600.0, low=pl.col("pattern").min())
        .filter(pl.col("hours") >= min_class_hours, pl.col("low") < 1.0 - min_dip)
        .select("road_class")
    )
    shape = supported.join(pattern, on="road_class").select(
        pl.col("road_class").cast(pl.String), "hour_of_week", "pattern"
    )
    return (
        speeds.select(pl.col("road_class").cast(pl.String), "speed")
        .unique()
        .join(shape, on="road_class")
        .select(
            "road_class",
            "speed",
            "hour_of_week",
            week_speed=pl.col("speed") * pl.col("pattern"),
        )
        .sort("road_class", "speed", "hour_of_week")
    )


def flow_speeds(historical: pl.DataFrame) -> pl.DataFrame:
    """Each edge's ``freeflow_speed`` and ``constrained_speed`` from its 168 hours.

    Valhalla uses ``constrained_speed`` from 07:00 to 19:00 (and for
    requests with no time) and ``freeflow_speed`` otherwise, when it has
    no predicted speed (see
    :data:`~trucktrack.congestion.VALHALLA_DAYTIME_HOURS`).  Each is the
    speed that gives the edge's mean travel time over the hours of its
    window, every day weighing the same: the harmonic mean of
    :func:`historical_speeds` there.
    """
    day = (pl.col("hour_of_week") % 24).is_between(
        *VALHALLA_DAYTIME_HOURS, closed="left"
    )

    def harmonic(window: pl.Expr) -> pl.Expr:
        return 1.0 / (1.0 / pl.col("speed")).filter(window).mean()

    return (
        historical.group_by(_EDGE)
        .agg(freeflow_speed=harmonic(~day), constrained_speed=harmonic(day))
        .sort(_EDGE)
    )
