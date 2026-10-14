"""One-call helpers for inspecting trucks and trips on an interactive map."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

import polars as pl

from trucktrack.query import scan_trip, scan_truck, truck_id_from_trip
from trucktrack.visualize._map import plot_trace, plot_trace_layers, serve_map


def _apply_date_filter(
    lf: pl.LazyFrame,
    date_range: tuple[date, date] | None,
) -> pl.LazyFrame:
    """Push date range filter into the lazy plan if applicable."""
    if date_range is None:
        return lf
    start, end = date_range
    trip_date = pl.col("time").dt.date()
    return lf.filter((trip_date >= start) & (trip_date <= end))


def _resolve_data(
    data_dir: str | Path,
    date_range: tuple[date, date] | None = None,
    *,
    truck_id: str | None = None,
    trip_ids: str | list[str] | None = None,
) -> pl.DataFrame:
    """Load some trips' rows, if given, else a truck's, from one pipeline stage."""
    if trip_ids is not None:
        lf = scan_trip(data_dir, trip_ids)
    elif truck_id is not None:
        lf = scan_truck(data_dir, truck_id)
    else:
        raise ValueError("Provide truck_id or trip_ids")
    return _apply_date_filter(lf, date_range).collect()


def _plot_and_serve(
    df: pl.DataFrame,
    serve: bool,
    host: str,
    port: int,
    plot_kwargs: dict[str, Any],
) -> Any:
    """Plot a trace and optionally serve it."""
    m = plot_trace(df, **plot_kwargs)
    if serve:
        serve_map(m, host=host, port=port)
    return m


def inspect_truck(
    data_dir: str | Path,
    truck_id: str,
    *,
    date_range: tuple[date, date] | None = None,
    host: str = "127.0.0.1",
    port: int = 5000,
    serve: bool = True,
    **plot_kwargs: Any,
) -> Any:
    """Load all trips for a truck, plot them, and optionally serve the map.

    Parameters
    ----------
    data_dir
        Root of any stage: raw, bucketed, split, or matched (see
        :func:`~trucktrack.query.scan_truck`).
    truck_id
        Full truck UUID.
    date_range
        Optional ``(start, end)`` date range to filter trips.
    host, port
        Passed to :func:`~trucktrack.visualize.serve_map`.
    serve
        If ``True`` (default), start a Flask server. If ``False``,
        return the folium Map without serving.
    **plot_kwargs
        Extra keyword arguments forwarded to
        :func:`~trucktrack.visualize.plot_trace`.

    Returns
    -------
    folium.Map
    """
    df = _resolve_data(data_dir, date_range, truck_id=truck_id)
    return _plot_and_serve(df, serve, host, port, plot_kwargs)


def inspect_trip(
    data_dir: str | Path,
    trip_id: str | list[str],
    *,
    host: str = "127.0.0.1",
    port: int = 5000,
    serve: bool = True,
    **plot_kwargs: Any,
) -> Any:
    """Load one or more trips, plot them, and optionally serve the map.

    Parameters
    ----------
    data_dir
        Root of the split or matched stage.
    trip_id
        A single composite trip ID or a list of them.
    host, port
        Passed to :func:`~trucktrack.visualize.serve_map`.
    serve
        If ``True`` (default), start a Flask server. If ``False``,
        return the folium Map without serving.
    **plot_kwargs
        Extra keyword arguments forwarded to
        :func:`~trucktrack.visualize.plot_trace`.

    Returns
    -------
    folium.Map
    """
    df = _resolve_data(data_dir, trip_ids=trip_id)
    return _plot_and_serve(df, serve, host, port, plot_kwargs)


def inspect_pipeline(
    truck_id: str | None = None,
    *,
    trip_id: str | list[str] | None = None,
    raw_dir: str | Path | None = None,
    split_dir: str | Path | None = None,
    matched_dir: str | Path | None = None,
    date_range: tuple[date, date] | None = None,
    host: str = "127.0.0.1",
    port: int = 5000,
    serve: bool = True,
    **plot_kwargs: Any,
) -> Any:
    """Overlay raw, segmented, and map-matched layers for one truck.

    Loads data from up to three pipeline stages and plots them on a
    single map with togglable layers via
    :func:`~trucktrack.visualize.plot_trace_layers`.

    Provide at least one of *raw_dir*, *split_dir*, or
    *matched_dir*. Each layer is skipped if its directory is not given.

    Parameters
    ----------
    truck_id
        Full truck UUID.  Inferred from *trip_id* when not provided.
    trip_id
        Optional trip ID or list of trip IDs.  When provided, the
        split and matched layers are scoped to these trips
        only.  The raw layer is filtered to the time range covered
        by the trips so only the relevant GPS points appear.
    raw_dir
        Root of the raw input, in any layout, or of its bucketed copy.
    split_dir
        Root of the split dataset.
    matched_dir
        Root of the map-matched dataset.
    date_range
        Optional ``(start, end)`` date range to filter all layers.
    host, port
        Passed to :func:`~trucktrack.visualize.serve_map`.
    serve
        If ``True`` (default), start a Flask server. If ``False``,
        return the folium Map without serving.
    **plot_kwargs
        Extra keyword arguments forwarded to
        :func:`~trucktrack.visualize.plot_trace_layers`.

    Returns
    -------
    folium.Map
    """
    # Normalise trip_id into a list (or None).
    trip_ids: list[str] | None = None
    if isinstance(trip_id, str):
        trip_ids = [trip_id]
    elif trip_id is not None:
        trip_ids = list(trip_id)

    # Derive truck_id from trip_ids when not provided explicitly.
    if truck_id is None:
        if trip_ids is None:
            raise ValueError("Provide at least one of truck_id or trip_id")
        truck_id = truck_id_from_trip(trip_ids[0])

    # --- split / matched: scope to trips when given ---------------------
    segments_df = (
        _resolve_data(split_dir, date_range, truck_id=truck_id, trip_ids=trip_ids)
        if split_dir is not None
        else None
    )
    matched_df = (
        _resolve_data(matched_dir, date_range, truck_id=truck_id, trip_ids=trip_ids)
        if matched_dir is not None
        else None
    )

    # --- raw: auto-derive date range from trip data ----------------------
    raw_date_range = date_range
    if raw_date_range is None and trip_ids is not None:
        # Use the time span of the loaded trip data to filter raw points.
        ref = segments_df if segments_df is not None else matched_df
        if ref is not None and len(ref) > 0:
            raw_date_range = (
                ref["time"].min().date(),  # type: ignore[union-attr]
                ref["time"].max().date(),  # type: ignore[union-attr]
            )

    raw_df = None
    if raw_dir is not None:
        raw_df = _resolve_data(raw_dir, raw_date_range, truck_id=truck_id)

    m = plot_trace_layers(
        raw=raw_df,
        segments=segments_df,
        matched=matched_df,
        **plot_kwargs,
    )
    if serve:
        serve_map(m, host=host, port=port)
    return m
