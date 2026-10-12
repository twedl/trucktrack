"""Edge speed observations: one row per GPS fix matched to a graph edge.

The map-matching pipeline asks ``trace_attributes`` for
:data:`OBSERVATION_ATTRIBUTES` in the same call that yields way IDs,
and :func:`observations_from_trace_attributes` joins the response back
onto the fixes.  Weights and dwell flags come from the whole trip
before matching (:func:`prepare_fixes`).  Rows are flagged, not
dropped, so aggregation can choose what to exclude without re-matching.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import polars as pl

from trucktrack.congestion.fixes import flag_dwell, time_weights

OBSERVATION_ATTRIBUTES: tuple[str, ...] = (
    "edge.id",
    "edge.way_id",
    "edge.begin_heading",
    "edge.end_heading",
    "matched.type",
    "matched.edge_index",
    "matched.distance_along_edge",
    "matched.distance_from_trace_point",
)

# Columns the fixes need on the way into matching.
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

_MATCH_SCHEMA = {
    "edge_index": pl.Int64,
    "match_type": pl.Utf8,
    "distance_along_edge": pl.Float64,
    "distance_from_trace": pl.Float64,
}
_EDGE_SCHEMA = {
    "edge_index": pl.Int64,
    "edge_id": pl.UInt64,
    "way_id": pl.Int64,
    "begin_heading": pl.Float64,
    "end_heading": pl.Float64,
}


@dataclass(frozen=True)
class ObservationConfig:
    """Thresholds for turning matched fixes into edge speed observations.

    *max_half_gap_s* caps each side of a fix's time weight (see
    :func:`~trucktrack.congestion.time_weights`).  *stationary_kmh*,
    *max_stationary_step_m* and *max_queue_stop_s* separate dwell zeros
    from queue zeros (see :func:`~trucktrack.congestion.flag_dwell`).

    A fix is ``wrong_way`` when its heading is more than
    *wrong_way_deg* off the matched edge's bearing and it is moving at
    *heading_min_speed_kmh* or more.  Below that, GPS course is noise.
    The edge's bearing at the fix is interpolated between its begin and
    end headings.  That only holds on edges that turn by at most
    *max_edge_turn_deg* (a cloverleaf loop doesn't), so other edges get
    no heading check.  Parallel roadways in the same direction (a
    collector beside an express lane) look alike and pass.
    """

    max_half_gap_s: float = 30.0
    stationary_kmh: float = 2.0
    max_stationary_step_m: float = 30.0
    max_queue_stop_s: float = 180.0
    heading_min_speed_kmh: float = 15.0
    wrong_way_deg: float = 90.0
    max_edge_turn_deg: float = 90.0


def prepare_fixes[FrameT: (pl.DataFrame, pl.LazyFrame)](
    frame: FrameT, config: ObservationConfig
) -> FrameT:
    """Add ``w`` and ``dwell`` to a frame of whole trips, before matching."""
    frame = time_weights(frame, max_half_gap_s=config.max_half_gap_s)
    return flag_dwell(
        frame,
        stationary_kmh=config.stationary_kmh,
        max_stationary_step_m=config.max_stationary_step_m,
        max_queue_stop_s=config.max_queue_stop_s,
    )


def _wrap180(deg: pl.Expr) -> pl.Expr:
    """Map an angle difference of either sign into [-180, 180)."""
    return ((deg + 540.0) % 360.0) - 180.0


def _edge_indices(matched: list[dict[str, Any]], n_edges: int) -> list[int | None]:
    """Each matched point's index into ``edges``, or ``None`` when it has none.

    Valhalla reports a point with no edge by omitting ``edge_index`` or
    by setting it to its invalid sentinel (2**64 - 1).  It does that to
    some ``interpolated`` points that it did place along an edge:
    fixes within ``interpolation_distance`` of the last, most often a
    truck standing in a queue.  Their ``distance_along_edge`` carries
    on from the previous point's, so such a point takes the previous
    point's edge as long as it doesn't move backwards along it.
    Unmatched points break the chain.  Without this, trucks reporting
    every few seconds lose their queue time and every queue looks
    shorter than it was.
    """
    out: list[int | None] = []
    last: tuple[int, float] | None = (
        None  # (edge index, distance along) of last placed point
    )
    for p in matched:
        i = p.get("edge_index")
        if i is not None and i >= n_edges:
            i = None
        along = p.get("distance_along_edge")
        if (
            i is None
            and p.get("type") == "interpolated"
            and last is not None
            and along is not None
            and along >= last[1]
        ):
            i = last[0]
        out.append(i)
        if p.get("type") == "unmatched":
            last = None
        elif i is not None and along is not None:
            last = (i, along)
    return out


def observations_from_trace_attributes(
    response: dict[str, Any],
    fixes: pl.DataFrame,
    config: ObservationConfig,
) -> pl.DataFrame:
    """Join a ``trace_attributes`` response onto the fixes it matched.

    *fixes* are the rows sent to the matcher, in order, with
    :data:`FIX_COLUMNS`.  *response* must have been requested with
    :data:`OBSERVATION_ATTRIBUTES`.  Returns :data:`OBSERVATION_COLUMNS`
    for fixes that landed on an edge, have a speed, and carry weight.
    Unmatched fixes and repeated timestamps tell us nothing about an
    edge.  Interpolated fixes Valhalla left without an edge index are
    put back on their edge (see :func:`_edge_indices`).
    """
    matched: list[dict[str, Any]] = response.get("matched_points", [])
    edges: list[dict[str, Any]] = response.get("edges", [])
    if len(matched) != len(fixes):
        raise ValueError(
            f"{len(matched)} matched points for {len(fixes)} fixes; "
            "the response must cover exactly the fixes passed"
        )

    match_df = pl.DataFrame(
        {
            "edge_index": _edge_indices(matched, len(edges)),
            "match_type": [p.get("type") for p in matched],
            "distance_along_edge": [p.get("distance_along_edge") for p in matched],
            "distance_from_trace": [
                p.get("distance_from_trace_point") for p in matched
            ],
        },
        schema=_MATCH_SCHEMA,
    )
    edge_df = pl.DataFrame(
        {
            "edge_index": range(len(edges)),
            "edge_id": [e.get("id") for e in edges],
            "way_id": [e.get("way_id") for e in edges],
            "begin_heading": [e.get("begin_heading") for e in edges],
            "end_heading": [e.get("end_heading") for e in edges],
        },
        schema=_EDGE_SCHEMA,
    )

    begin = pl.col("begin_heading")
    turn = _wrap180(pl.col("end_heading") - begin)
    edge_bearing = begin + pl.col("distance_along_edge") * turn
    heading_diff = (
        pl.when(turn.abs() <= config.max_edge_turn_deg)
        .then(_wrap180(pl.col("heading") - edge_bearing).abs())
        .alias("heading_diff")
    )
    wrong_way = (
        (pl.col("heading_diff") > config.wrong_way_deg)
        & (pl.col("speed") >= config.heading_min_speed_kmh)
    ).fill_null(False)

    return (
        pl.concat([fixes.select(FIX_COLUMNS), match_df], how="horizontal")
        .join(edge_df, on="edge_index", how="left", maintain_order="left")
        .filter(
            pl.col("edge_id").is_not_null()
            & pl.col("speed").is_not_null()
            & (pl.col("w") > 0)
        )
        .with_columns(heading_diff)
        .with_columns(wrong_way.alias("wrong_way"))
        .select(OBSERVATION_COLUMNS)
    )
