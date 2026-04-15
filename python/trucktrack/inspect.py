"""REPL-friendly helpers for the end-to-end truck inspection workflow.

One function per workflow step, all returning plain Python / Polars objects.
No disk caching — each helper takes its input and returns its output so
callers can iterate on parameters cheaply::

    import trucktrack as tt
    from datetime import datetime, timedelta

    raw = tt.inspect.load_truck_trace(
        truck_id, datetime(2026, 1, 1), datetime(2026, 1, 8),
        data_dir="data/raw",
    )
    split = tt.inspect.split_trips(
        raw,
        gap=timedelta(minutes=5),
        stop_max_diameter=50.0,
        stop_min_duration=timedelta(minutes=2),
    )
    trips = tt.inspect.map_match_trips(split)
    quality = tt.inspect.evaluate_quality(split, trips=trips)
    m = tt.inspect.plot_inspection(raw, split, trips)
    tt.visualize.serve_map(m)
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import polars as pl

from trucktrack.query import ChunkIndex, scan_raw_truck
from trucktrack.splitters import (
    filter_traffic_stops,
    split_by_observation_gap,
    split_by_stops,
)
from trucktrack.valhalla.map_matching import map_match_dataframe_full
from trucktrack.valhalla.quality import (
    MapMatchQuality,
    evaluate_map_match,
    path_quality,
)
from trucktrack.visualize import plot_trace_layers

__all__ = [
    "TripMatch",
    "evaluate_quality",
    "load_truck_trace",
    "map_match_trips",
    "plot_inspection",
    "split_trips",
]


@dataclass(frozen=True)
class TripMatch:
    """Map-match result for a single trip segment.

    ``matched_df`` is the trip's original rows augmented with
    ``matched_lat``, ``matched_lon``, and ``distance_from_trace``.
    ``shape`` is a list of road-snapped polylines — one per route
    segment Valhalla returned.  When the matcher breaks the trace,
    disjoint segments arrive separately so renderers never bridge them
    with a straight chord.  Empty when Valhalla returns no geometry.
    """

    segment_id: int
    matched_df: pl.DataFrame
    way_ids: list[int]
    shape: list[list[tuple[float, float]]]


def load_truck_trace(
    truck_id: str,
    start: datetime | str,
    end: datetime | str,
    *,
    data_dir: str | Path | None = None,
    index: ChunkIndex | None = None,
    time_col: str = "time",
) -> pl.DataFrame:
    """Load a truck's raw points within ``[start, end)``, sorted by time.

    Provide either *data_dir* (raw hive layout) or a pre-built *index*.
    ``ChunkIndex`` is best-suited to the partitioned layout; for raw
    traces pass ``data_dir`` directly.

    String timestamps are parsed with :func:`datetime.fromisoformat`.
    Naive timestamps are coerced to UTC when the underlying column is
    tz-aware.  Raises ``ValueError`` when the filtered result is empty
    so notebooks fail fast instead of propagating empty frames.
    """
    if index is not None:
        lf = index.scan_truck(truck_id)
    elif data_dir is not None:
        lf = scan_raw_truck(data_dir, truck_id)
    else:
        raise ValueError("provide data_dir or index")

    start_dt = _coerce_datetime(start)
    end_dt = _coerce_datetime(end)

    tcol_dtype = lf.collect_schema().get(time_col)
    if tcol_dtype is not None and getattr(tcol_dtype, "time_zone", None) is not None:
        if start_dt.tzinfo is None:
            start_dt = start_dt.replace(tzinfo=UTC)
        if end_dt.tzinfo is None:
            end_dt = end_dt.replace(tzinfo=UTC)

    df = (
        lf.filter(
            (pl.col(time_col) >= start_dt) & (pl.col(time_col) < end_dt),
        )
        .sort(time_col)
        .collect()
    )
    if df.is_empty():
        raise ValueError(f"no rows for truck_id={truck_id!r} in [{start_dt}, {end_dt})")
    return df


def _coerce_datetime(value: datetime | str) -> datetime:
    if isinstance(value, datetime):
        return value
    return datetime.fromisoformat(value)


def split_trips(
    df: pl.DataFrame,
    *,
    gap: timedelta,
    stop_max_diameter: float,
    stop_min_duration: timedelta,
    traffic_max_angle_change: float | None = 30.0,
    traffic_min_distance: float = 10.0,
    min_length: int = 3,
) -> pl.DataFrame:
    """Apply gap split, stop split, and (optionally) the traffic filter.

    Returns the input rows annotated with ``segment_id`` and ``is_stop``.
    Pass ``traffic_max_angle_change=None`` to skip the traffic filter
    (useful for with/without comparisons).
    """
    gapped = split_by_observation_gap(df, gap, min_length=min_length)
    stopped = split_by_stops(
        gapped, stop_max_diameter, stop_min_duration, min_length=min_length
    )
    if traffic_max_angle_change is None:
        return stopped
    return filter_traffic_stops(
        stopped,
        max_angle_change=traffic_max_angle_change,
        min_distance=traffic_min_distance,
    )


def map_match_trips(
    split_df: pl.DataFrame,
    *,
    costing: str = "auto",
    costing_options: dict[str, Any] | None = None,
    trace_options: dict[str, Any] | None = None,
    config: str | Path | None = None,
    skip_stops: bool = True,
    min_points: int = 2,
) -> dict[int, TripMatch]:
    """Map-match each non-stop segment, keyed by ``segment_id``.

    Segments shorter than *min_points* are skipped.  Valhalla errors
    propagate — run :func:`evaluate_quality` without a ``trips`` argument
    if you want failures captured as rows instead.
    """
    df = _non_stop(split_df, skip_stops)
    out: dict[int, TripMatch] = {}
    for (sid,), sub in df.sort("time").group_by("segment_id", maintain_order=True):
        if sub.height < min_points:
            continue
        matched_df, way_ids, shape = map_match_dataframe_full(
            sub,
            costing=costing,
            costing_options=costing_options,
            trace_options=trace_options,
            config=config,
        )
        out[int(sid)] = TripMatch(
            segment_id=int(sid),
            matched_df=matched_df,
            way_ids=way_ids,
            shape=shape,
        )
    return out


_QUALITY_SCHEMA: dict[str, pl.DataType] = {
    "segment_id": pl.Int64(),
    "n_points": pl.Int64(),
    "ok": pl.Boolean(),
    "error": pl.Utf8(),
    "path_length_ratio": pl.Float64(),
    "heading_reversals": pl.Int64(),
    "has_issues": pl.Boolean(),
}


def evaluate_quality(
    split_df: pl.DataFrame,
    *,
    trips: dict[int, TripMatch] | None = None,
    costing: str = "auto",
    costing_options: dict[str, Any] | None = None,
    trace_options: dict[str, Any] | None = None,
    config: str | Path | None = None,
    skip_stops: bool = True,
    min_points: int = 2,
) -> pl.DataFrame:
    """Return one quality row per trip.

    When *trips* is provided, ``path_length_ratio`` and
    ``heading_reversals`` are computed directly from the cached match
    output (no second Valhalla call).  ``shape_gaps`` / ``n_polylines``
    are unavailable on the cached path — pass ``trips=None`` to let
    Valhalla compute them (requires a discoverable ``valhalla.json`` or
    an explicit ``config=`` path).
    """
    df = _non_stop(split_df, skip_stops)
    rows: list[dict[str, Any]] = []
    for (sid,), sub in df.sort("time").group_by("segment_id", maintain_order=True):
        sid_int = int(sid)
        if sub.height < min_points:
            continue
        if trips is not None and sid_int in trips:
            rows.append(_cached_quality_row(sid_int, sub, trips[sid_int]))
        else:
            pts = list(zip(sub["lat"].to_list(), sub["lon"].to_list(), strict=True))
            q = evaluate_map_match(
                str(sid_int),
                pts,
                costing=costing,
                costing_options=costing_options,
                config=config,
                trace_options=trace_options,
            )
            rows.append(_row_from_quality(sid_int, q))
    return pl.DataFrame(rows, schema=_QUALITY_SCHEMA)


def _cached_quality_row(
    sid: int, sub_df: pl.DataFrame, tm: TripMatch
) -> dict[str, Any]:
    pts = list(zip(sub_df["lat"].to_list(), sub_df["lon"].to_list(), strict=True))
    ratio, reversals = path_quality(pts, tm.shape)
    q = MapMatchQuality(
        trip_id=str(sid),
        ok=True,
        n_points=len(pts),
        path_length_ratio=ratio,
        heading_reversals=reversals,
    )
    q.ok = not q.has_issues
    return _row_from_quality(sid, q)


def _row_from_quality(sid: int, q: MapMatchQuality) -> dict[str, Any]:
    return {
        "segment_id": sid,
        "n_points": q.n_points,
        "ok": q.ok,
        "error": q.error,
        "path_length_ratio": q.path_length_ratio,
        "heading_reversals": q.heading_reversals,
        "has_issues": q.has_issues,
    }


def plot_inspection(
    raw_df: pl.DataFrame,
    split_df: pl.DataFrame,
    trips: dict[int, TripMatch],
    **plot_kwargs: Any,
) -> Any:
    """Render raw, segmented, and map-matched layers on one folium map."""
    if trips:
        matched = pl.concat(
            [tm.matched_df for tm in trips.values()], how="vertical_relaxed"
        )
        matched_shape = [seg for tm in trips.values() for seg in tm.shape] or None
    else:
        matched = None
        matched_shape = None
    return plot_trace_layers(
        raw=raw_df,
        segments=split_df,
        matched=matched,
        matched_shape=matched_shape,
        **plot_kwargs,
    )


def _non_stop(split_df: pl.DataFrame, skip_stops: bool) -> pl.DataFrame:
    if skip_stops and "is_stop" in split_df.columns:
        return split_df.filter(~pl.col("is_stop"))
    return split_df
