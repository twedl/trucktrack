"""Every edge's speed in every hour of the week, from its measured cells.

:func:`~trucktrack.congestion.edge_cells` measures what trucks saw: an
edge's space-mean speed in the hours they drove it, nothing more.  Most
edge-hours have no truck, and most that do have one.  Imputation fills
them in from a model of the edge,

    speed(e, h) = level(e) * pattern(h | class) ** depth(e, half of day),

and steadies the measured ones by blending them with it.

- ``pattern`` is the road class's hour-of-week shape, 1 at night: when
  roads of that kind slow down, and by how much on average.  It is the
  one thing borrowed from other edges.
- ``level`` is the edge's own night-time speed, from all its cells,
  each deflated by the model at its hour.
- ``depth`` scales the class's dips to the edge's, in log terms: 0 is
  flat, 1 the class's, 2 twice as deep.  Deeper near a city core,
  shallower out of it, and separately before and after noon: a road
  into a city slows in the morning, out of it in the evening.  An edge
  with little rush-hour time stays near the class's 1.

A cell with ``trucks`` effective trucks gets ``trucks / (trucks + k)``
of its own speed and the rest from the model.  The more trucks behind
it, the less the class matters.

Measured on synthetic trucks against held-out trucks' 5 s data (km/h,
weighted by driving time): hours an edge measured, 5.0 from the cell
alone, 4.5 blended; hours it didn't, 14.4 from the edge's all-week
mean, 6.5 imputed.  Borrowing the edge's OSM way's cell for the hour,
as a speed or as a ratio to the model, does worse (11.2 vs 13.9 where
the way has one): one truck drives the whole way, and short edges are
junction pieces that don't share its flow.
"""

from __future__ import annotations

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
_ITERATIONS = 5
# A class-hour of stopped trucks or a few odd cells can't take the
# pattern to 0 (log(0) in the depth fit) or past twice the night speed;
# no edge's dips run deeper than three times the class's, in log terms.
_PATTERN_RANGE = (0.05, 2.0)
_DEPTH_RANGE = (0.0, 3.0)
_SOURCES = pl.Enum(["imputed", "blended", "measured"])


@dataclass(frozen=True)
class ImputeConfig:
    """Settings for :func:`historical_speeds`.

    *cell_trucks* is ``k``: the effective trucks at which a cell's own
    speed and the model's count equally.  *depth_seconds* is how much
    of an edge's own evidence on its depth (seconds of driving, times
    the squared log of the class pattern at the time) counts as much as
    the class's depth of 1.  An edge has one depth before local hour
    *depth_split_hour* and another from it on (a road into a city slows
    in the morning, out of it in the evening); 24 gives it one.

    Defaults are the best of those tried against held-out trucks'
    5 s data on the synthetic sample.
    """

    cell_trucks: float = 1.0
    depth_seconds: float = 10.0
    depth_split_hour: int = 12


DEFAULT_IMPUTE_CONFIG = ImputeConfig()


def _half_day(config: ImputeConfig) -> pl.Expr:
    """Which of an edge's depths applies at ``hour_of_week``."""
    return ((pl.col("hour_of_week") % 24) >= config.depth_split_hour).alias("half")


def fit_model(
    cells: pl.DataFrame, config: ImputeConfig = DEFAULT_IMPUTE_CONFIG
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Fit each edge's ``level`` and ``depth``, and each class's ``pattern``.

    *cells* are hour-of-week :func:`~trucktrack.congestion.edge_cells`.
    The fit alternates, on the cells' time-weighted speeds:

    - level: Σ speed·w / Σ w·pattern**depth over the edge's cells;
    - pattern: Σ speed·w / Σ w·level over the class's cells in the
      hour, scaled to 1 at night (00:00-05:00);
    - depth: the slope of log(speed / level) on log(pattern), weighted
      by seconds, shrunk towards 1 by ``config.depth_seconds``, within
      0 to 3.  Cell speeds under 1 km/h count as 1.

    Returns ``level`` per edge, ``depth`` per edge and half of the day
    (``half``, see ``config.depth_split_hour``) it has cells in, and
    ``pattern`` per class and hour.  An edge only ever seen stopped has
    no level, and gets no speeds.
    """
    depth_key = [*_EDGE, "half"]
    c = cells.select(
        *_EDGE,
        *_PATTERN,
        _half_day(config),
        s=pl.col("speed") * pl.col("seconds"),  # Σ speed·w
        w=pl.col("seconds"),
        log_v=pl.col("speed").clip(lower_bound=1.0).log(),
    )
    depth_keys = c.select(depth_key).unique()
    edges = depth_keys.with_columns(depth=pl.lit(1.0))
    pattern = c.select(_PATTERN).unique().with_columns(pattern=pl.lit(1.0))
    night = (pl.col("hour_of_week") % 24).is_between(*_NIGHT_HOURS, closed="left")
    tau = config.depth_seconds
    for _ in range(_ITERATIONS):
        g = c.join(edges, on=depth_key).join(pattern, on=_PATTERN)
        level = g.group_by(_EDGE).agg(
            level=pl.col("s").sum()
            / (pl.col("pattern").pow(pl.col("depth")) * pl.col("w")).sum()
        )
        level = level.filter(pl.col("level") > 0)
        g = g.drop("pattern").join(level, on=_EDGE)
        pattern = g.group_by(_PATTERN).agg(
            pattern=pl.col("s").sum() / (pl.col("level") * pl.col("w")).sum(),
            pw=(pl.col("level") * pl.col("w")).sum(),
        )
        night_mean = (
            pattern.filter(night)
            .group_by("road_class")
            .agg(n=(pl.col("pattern") * pl.col("pw")).sum() / pl.col("pw").sum())
        )
        pattern = pattern.join(night_mean, on="road_class", how="left").select(
            *_PATTERN,
            pattern=(pl.col("pattern") / pl.col("n").fill_null(1.0)).clip(
                *_PATTERN_RANGE
            ),
        )
        x = pl.col("pattern").log()
        r = pl.col("log_v") - pl.col("level").log()
        fitted = (
            g.join(pattern, on=_PATTERN)
            .group_by(depth_key)
            .agg(sxy=(pl.col("w") * x * r).sum(), sxx=(pl.col("w") * x * x).sum())
            .select(*depth_key, depth=((pl.col("sxy") + tau) / (pl.col("sxx") + tau)))
        )
        edges = depth_keys.join(fitted, on=depth_key, how="left").with_columns(
            pl.col("depth").fill_null(1.0).clip(*_DEPTH_RANGE)
        )
    return level, edges, pattern


def historical_speeds(
    cells: pl.LazyFrame | pl.DataFrame, config: ImputeConfig = DEFAULT_IMPUTE_CONFIG
) -> pl.DataFrame:
    """Every measured edge's speed in each of the 168 local hours of the week.

    *cells* are hour-of-week :func:`~trucktrack.congestion.edge_cells`.
    Each edge gets a row per hour, in edge then hour order, with
    ``speed`` (km/h), ``own_weight``, the share of it from the hour's
    measurement, and ``source``: ``"imputed"`` for hours no truck
    measured, ``"blended"`` for the rest (``"measured"`` when
    ``config.cell_trucks`` is 0).  :func:`fit_model` gives the model
    behind it.  Edges no truck measured, or only ever saw stopped, get
    no rows.
    """
    measured = cells.lazy().collect()
    level, depth, pattern = fit_model(measured, config)
    edges = measured.select(*_EDGE, "road_class").unique().sort(_EDGE)
    hours = pl.DataFrame(
        {"hour_of_week": pl.int_range(HOURS_PER_WEEK, dtype=pl.Int16, eager=True)}
    ).with_columns(_half_day(config))
    k = config.cell_trucks
    n = pl.col("trucks").fill_null(0.0)
    own = pl.when(n > 0).then(n / (n + k)).otherwise(0.0)
    model = pl.col("level") * pl.col("pattern").fill_null(1.0).pow(
        pl.col("depth").fill_null(1.0)
    )
    return (
        edges.join(hours, how="cross", maintain_order="left_right")
        .join(level, on=_EDGE, maintain_order="left")
        .join(depth, on=[*_EDGE, "half"], how="left", maintain_order="left")
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
            + (1.0 - pl.col("own_weight")) * model,
            own_weight="own_weight",
            source=pl.when(n == 0)
            .then(pl.lit("imputed"))
            .otherwise(pl.lit("blended" if k > 0 else "measured"))
            .cast(_SOURCES),
        )
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
