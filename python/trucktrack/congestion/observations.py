"""Edge speed observations: one row per GPS fix matched to a graph edge.

Weights and dwell flags come from whole trips before matching
(:func:`prepare_fixes`).  The matcher then adds where each fix landed
(see :func:`trucktrack.valhalla.parse_fix_matches`) and
:func:`edge_observations` keeps the fixes that tell us about an edge.
Rows are flagged, not dropped, so aggregation can choose what to
exclude without re-matching.
"""

from __future__ import annotations

import polars as pl

from trucktrack.congestion.config import DEFAULT_CONFIG, ObservationConfig
from trucktrack.congestion.fixes import flag_dwell, time_weights

# Columns the fixes carry into matching.
FIX_COLUMNS: tuple[str, ...] = ("id", "time", "speed", "heading", "w", "dwell")

OBSERVATION_COLUMNS: tuple[str, ...] = (
    *FIX_COLUMNS,
    "edge_id",
    "way_id",
    "distance_along_edge",
    "match_type",
    "distance_from_trace",
    "heading_diff",
    "wrong_way",
)


def prepare_fixes[FrameT: (pl.DataFrame, pl.LazyFrame)](
    frame: FrameT, config: ObservationConfig = DEFAULT_CONFIG
) -> FrameT:
    """Add ``w`` and ``dwell`` to a frame of whole trips, before matching."""
    return flag_dwell(time_weights(frame, config), config)


def _wrap180(deg: pl.Expr) -> pl.Expr:
    """Map an angle difference of either sign into [-180, 180)."""
    return ((deg + 540.0) % 360.0) - 180.0


def edge_observations(
    matched: pl.DataFrame, config: ObservationConfig = DEFAULT_CONFIG
) -> pl.DataFrame:
    """Keep the matched fixes that tell us about an edge, and check headings.

    *matched* has :data:`FIX_COLUMNS` plus the per-fix match columns of
    :func:`trucktrack.valhalla.parse_fix_matches`.  Returns
    :data:`OBSERVATION_COLUMNS` for fixes that landed on an edge, have a
    speed, and carry weight: unmatched fixes and repeated timestamps
    tell us nothing about an edge.
    """
    begin = pl.col("begin_heading")
    turn = _wrap180(pl.col("end_heading") - begin)
    edge_bearing = begin + pl.col("distance_along_edge") * turn
    heading_diff = pl.when(turn.abs() <= config.max_edge_turn_deg).then(
        _wrap180(pl.col("heading") - edge_bearing).abs()
    )
    wrong_way = (
        (pl.col("heading_diff") > config.wrong_way_deg)
        & (pl.col("speed") >= config.heading_min_speed_kmh)
    ).fill_null(False)
    return (
        matched.filter(
            pl.col("edge_id").is_not_null()
            & pl.col("speed").is_not_null()
            & (pl.col("w") > 0)
        )
        .with_columns(heading_diff.alias("heading_diff"))
        .with_columns(wrong_way.alias("wrong_way"))
        .select(OBSERVATION_COLUMNS)
    )
