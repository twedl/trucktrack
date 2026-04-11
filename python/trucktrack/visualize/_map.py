"""Core map-building logic for trace visualization."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import polars as pl

from trucktrack.visualize._convert import tracepoints_to_dataframe

if TYPE_CHECKING:
    from trucktrack.generate.models import TracePoint

# Qualitative palette for distinguishing segments.
_SEGMENT_COLORS = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
]


def _import_folium() -> tuple[Any, Any]:
    try:
        import branca.colormap as cm
        import folium
    except ImportError as exc:
        raise ImportError(
            "folium is required for visualization. "
            "Install it with: pip install trucktrack[viz]"
        ) from exc
    return folium, cm


def _downsample(df: pl.DataFrame, max_points: int) -> pl.DataFrame:
    """Take every Nth row, preserving segment boundaries."""
    if df.height <= max_points:
        return df
    if "segment_id" in df.columns:
        # Keep first and last row of each segment, sample in between.
        parts = []
        for _, seg in df.group_by("segment_id", maintain_order=True):
            if seg.height <= 2:
                parts.append(seg)
                continue
            # Compute step for this segment's share of the budget.
            budget = max(2, int(max_points * seg.height / df.height))
            step = max(1, seg.height // budget)
            indices = list(range(0, seg.height, step))
            if indices[-1] != seg.height - 1:
                indices.append(seg.height - 1)
            parts.append(seg[indices])
        return pl.concat(parts)
    step = max(1, df.height // max_points)
    indices = list(range(0, df.height, step))
    if indices[-1] != df.height - 1:
        indices.append(df.height - 1)
    return df[indices]


def _add_polyline(
    folium: Any,
    fg: Any,
    lats: list[float],
    lons: list[float],
    color: str = "#1f77b4",
    weight: int = 4,
    dash_array: str | None = None,
    popup: str | None = None,
) -> None:
    locations = list(zip(lats, lons, strict=True))
    if len(locations) < 2:
        return
    kwargs: dict[str, Any] = {"color": color, "weight": weight}
    if dash_array:
        kwargs["dash_array"] = dash_array
    if popup:
        kwargs["popup"] = folium.Popup(popup, max_width=300)
    folium.PolyLine(locations, **kwargs).add_to(fg)


def _add_colored_line(
    folium: Any,
    cm: Any,
    fg: Any,
    lats: list[float],
    lons: list[float],
    values: list[float],
    vmin: float,
    vmax: float,
    colormap: Any,
    weight: int = 4,
) -> None:
    """Draw a polyline colored segment-by-segment by values."""
    for i in range(len(lats) - 1):
        color = colormap(values[i])
        folium.PolyLine(
            [(lats[i], lons[i]), (lats[i + 1], lons[i + 1])],
            color=color,
            weight=weight,
        ).add_to(fg)


def _add_stop_markers(
    folium: Any,
    fg: Any,
    df: pl.DataFrame,
    stop_color: str,
    stop_radius: int,
) -> None:
    """Add circle markers for stop segments."""
    stops = df.filter(pl.col("is_stop"))
    if stops.height == 0:
        return
    for _, group in stops.group_by("segment_id", maintain_order=True):
        lat = group["lat"].mean()
        lon = group["lon"].mean()
        n_pts = group.height
        time_col = "time" if "time" in group.columns else None
        popup_text = f"Stop: {n_pts} points"
        if time_col and group[time_col].dtype != pl.Null:
            t_min = group[time_col].min()
            t_max = group[time_col].max()
            if t_min is not None and t_max is not None:
                duration = t_max - t_min
                popup_text += f"<br>Duration: {duration}"
        folium.CircleMarker(
            location=(lat, lon),
            radius=stop_radius,
            color=stop_color,
            fill=True,
            fill_color=stop_color,
            fill_opacity=0.6,
            popup=folium.Popup(popup_text, max_width=300),
        ).add_to(fg)


def plot_trace(
    data: pl.DataFrame | list[TracePoint],
    *,
    color_by: str | None = None,
    tile_layer: str = "OpenStreetMap",
    width: str | int = "100%",
    height: str | int = 600,
    max_points: int | None = 5000,
    stop_color: str = "red",
    stop_radius: int = 8,
    matched_color: str = "#1f77b4",
    trace_color: str = "gray",
) -> Any:
    """Plot a trace on an interactive Leaflet map.

    Auto-detects the data stage based on columns present:

    - **Raw**: ``lat``, ``lon`` → single polyline
    - **Gap-split**: ``segment_id`` → distinct color per segment
    - **Stop-split**: ``segment_id`` + ``is_stop`` → polylines for movement,
      circle markers for stops
    - **Map-matched**: ``matched_lat`` + ``matched_lon`` → dashed original
      trace overlaid with solid matched trace

    Parameters
    ----------
    data
        A Polars DataFrame or a ``list[TracePoint]``.
    color_by
        Column name to color movement segments by (e.g. ``"speed"``).
    tile_layer
        Folium tile layer name.
    width, height
        Map dimensions.
    max_points
        Downsample traces with more points than this.  ``None`` to disable.
    stop_color
        Color for stop markers.
    stop_radius
        Radius of stop circle markers in pixels.
    matched_color
        Color for the map-matched polyline.
    trace_color
        Color for the original trace when shown alongside matched.

    Returns
    -------
    folium.Map
        An interactive map that renders in Jupyter or can be saved to HTML
        via :func:`save_map`.
    """
    folium, cm = _import_folium()

    # Normalize input.
    if isinstance(data, list):
        from trucktrack.generate.models import TracePoint as TP

        if data and isinstance(data[0], TP):
            df = tracepoints_to_dataframe(data)
        else:
            msg = "data must be a Polars DataFrame or list[TracePoint]"
            raise TypeError(msg)
    else:
        df = data

    if max_points is not None:
        df = _downsample(df, max_points)

    # Detect mode.
    has_matched = "matched_lat" in df.columns and "matched_lon" in df.columns
    has_is_stop = "is_stop" in df.columns
    has_segment_id = "segment_id" in df.columns

    # Map setup.
    center_lat = df["lat"].mean()
    center_lon = df["lon"].mean()
    m = folium.Map(
        location=(center_lat, center_lon),
        tiles=tile_layer,
        width=width,
        height=height,
    )

    # Fit bounds.
    bounds = [
        [df["lat"].min(), df["lon"].min()],
        [df["lat"].max(), df["lon"].max()],
    ]
    m.fit_bounds(bounds)

    # --- Map-matched mode ---
    if has_matched:
        # Original trace as dashed line.
        trace_fg = folium.FeatureGroup(name="Original trace")
        _add_polyline(
            folium,
            trace_fg,
            df["lat"].to_list(),
            df["lon"].to_list(),
            color=trace_color,
            weight=3,
            dash_array="8 4",
        )
        trace_fg.add_to(m)

        # Matched trace.
        matched_fg = folium.FeatureGroup(name="Matched trace")
        m_lats = df["matched_lat"].to_list()
        m_lons = df["matched_lon"].to_list()

        if color_by == "distance_from_trace" and "distance_from_trace" in df.columns:
            vals = df["distance_from_trace"].to_list()
            vmin = min(vals)
            vmax = max(vals) if max(vals) > vmin else vmin + 1
            colormap = cm.LinearColormap(
                ["green", "yellow", "red"], vmin=vmin, vmax=vmax
            )
            colormap.caption = "distance_from_trace (m)"
            _add_colored_line(
                folium, cm, matched_fg, m_lats, m_lons, vals, vmin, vmax, colormap
            )
            colormap.add_to(m)
        elif color_by is not None and color_by in df.columns:
            vals = df[color_by].cast(pl.Float64).to_list()
            vmin = min(vals)
            vmax = max(vals) if max(vals) > vmin else vmin + 1
            colormap = cm.LinearColormap(
                ["green", "yellow", "red"], vmin=vmin, vmax=vmax
            )
            colormap.caption = color_by
            _add_colored_line(
                folium, cm, matched_fg, m_lats, m_lons, vals, vmin, vmax, colormap
            )
            colormap.add_to(m)
        else:
            _add_polyline(
                folium, matched_fg, m_lats, m_lons, color=matched_color, weight=4
            )
        matched_fg.add_to(m)

        # Stops if present.
        if has_is_stop and has_segment_id:
            stop_fg = folium.FeatureGroup(name="Stops")
            _add_stop_markers(folium, stop_fg, df, stop_color, stop_radius)
            stop_fg.add_to(m)

        folium.LayerControl().add_to(m)
        return m

    # --- Stop-split mode ---
    if has_is_stop and has_segment_id:
        movement = df.filter(~pl.col("is_stop"))
        move_fg = folium.FeatureGroup(name="Movement")
        _render_segments(folium, cm, move_fg, movement, color_by, matched_color)
        move_fg.add_to(m)

        stop_fg = folium.FeatureGroup(name="Stops")
        _add_stop_markers(folium, stop_fg, df, stop_color, stop_radius)
        stop_fg.add_to(m)

        folium.LayerControl().add_to(m)
        return m

    # --- Gap-split mode ---
    if has_segment_id:
        seg_fg = folium.FeatureGroup(name="Segments")
        _render_segments(folium, cm, seg_fg, df, color_by, matched_color)
        seg_fg.add_to(m)
        return m

    # --- Raw mode ---
    fg = folium.FeatureGroup(name="Trace")
    if color_by is not None and color_by in df.columns:
        vals = df[color_by].cast(pl.Float64).to_list()
        vmin = min(vals)
        vmax = max(vals) if max(vals) > vmin else vmin + 1
        colormap = cm.LinearColormap(["green", "yellow", "red"], vmin=vmin, vmax=vmax)
        colormap.caption = color_by
        _add_colored_line(
            folium,
            cm,
            fg,
            df["lat"].to_list(),
            df["lon"].to_list(),
            vals,
            vmin,
            vmax,
            colormap,
        )
        colormap.add_to(m)
    else:
        _add_polyline(
            folium,
            fg,
            df["lat"].to_list(),
            df["lon"].to_list(),
            color=matched_color,
        )
    fg.add_to(m)
    return m


def _render_segments(
    folium: Any,
    cm: Any,
    fg: Any,
    df: pl.DataFrame,
    color_by: str | None,
    default_color: str,
) -> None:
    """Render segments as polylines, optionally colored by a column."""
    if color_by is not None and color_by in df.columns:
        vals = df[color_by].cast(pl.Float64).to_list()
        vmin = min(vals)
        vmax = max(vals) if max(vals) > vmin else vmin + 1
        colormap = cm.LinearColormap(["green", "yellow", "red"], vmin=vmin, vmax=vmax)
        colormap.caption = color_by
        _add_colored_line(
            folium,
            cm,
            fg,
            df["lat"].to_list(),
            df["lon"].to_list(),
            vals,
            vmin,
            vmax,
            colormap,
        )
        # Attach colormap to parent map via fg — caller must add to map.
        fg._colormap = colormap
        return

    seg_ids = df["segment_id"].unique(maintain_order=True).to_list()
    for i, sid in enumerate(seg_ids):
        seg = df.filter(pl.col("segment_id") == sid)
        color = _SEGMENT_COLORS[i % len(_SEGMENT_COLORS)]
        _add_polyline(
            folium,
            fg,
            seg["lat"].to_list(),
            seg["lon"].to_list(),
            color=color,
            popup=f"Segment {sid}",
        )


def save_map(m: Any, path: str | Path) -> None:
    """Save a folium Map to an HTML file."""
    m.save(str(Path(path)))
