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


def _sort_by_time(df: pl.DataFrame | None) -> pl.DataFrame | None:
    """Sort a DataFrame by (id, time) if a time column is present."""
    if df is None or "time" not in df.columns:
        return df
    sort_cols = ["id", "time"] if "id" in df.columns else ["time"]
    return df.sort(sort_cols)


def _normalize_input(
    data: pl.DataFrame | list[Any],
    param_name: str = "data",
) -> pl.DataFrame:
    """Convert list[TracePoint] to DataFrame, or pass through a DataFrame."""
    if isinstance(data, list):
        from trucktrack.generate.models import TracePoint as TP

        if data and isinstance(data[0], TP):
            return tracepoints_to_dataframe(data)
        msg = f"{param_name} must be a Polars DataFrame or list[TracePoint]"
        raise TypeError(msg)
    return data


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
        parts = []
        for _, seg in df.group_by("segment_id", maintain_order=True):
            if seg.height <= 2:
                parts.append(seg)
                continue
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
    fg: Any,
    lats: list[float],
    lons: list[float],
    values: list[float],
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


def _make_colormap(
    cm: Any,
    values: list[float],
    caption: str,
) -> Any:
    """Build a LinearColormap from a list of values."""
    vmin = min(values)
    raw_max = max(values)
    vmax = raw_max if raw_max > vmin else vmin + 1
    colormap = cm.LinearColormap(["green", "yellow", "red"], vmin=vmin, vmax=vmax)
    colormap.caption = caption
    return colormap


def _add_colored_layer(
    folium: Any,
    cm: Any,
    fg: Any,
    m: Any,
    lats: list[float],
    lons: list[float],
    values: list[float],
    caption: str,
) -> None:
    """Build a colormap, draw colored segments, and add the legend to the map."""
    colormap = _make_colormap(cm, values, caption)
    _add_colored_line(folium, fg, lats, lons, values, colormap)
    colormap.add_to(m)


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
        popup_text = f"Stop: {n_pts} points"
        if "time" in group.columns and group["time"].dtype != pl.Null:
            t_min = group["time"].min()
            t_max = group["time"].max()
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
    tile_layer: str = "CartoDB Positron",
    width: str | int = "100%",
    height: str | int = "100%",
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

    df = _normalize_input(data)

    if df.height == 0:
        m = folium.Map(tiles=tile_layer, width=width, height=height)
        return m

    if "time" in df.columns:
        sort_cols = ["id", "time"] if "id" in df.columns else ["time"]
        df = df.sort(sort_cols)

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

    bounds = [
        [df["lat"].min(), df["lon"].min()],
        [df["lat"].max(), df["lon"].max()],
    ]
    m.fit_bounds(bounds)

    # --- Map-matched mode ---
    if has_matched:
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

        matched_fg = folium.FeatureGroup(name="Matched trace")
        m_lats = df["matched_lat"].to_list()
        m_lons = df["matched_lon"].to_list()

        if color_by is not None and color_by in df.columns:
            vals = df[color_by].cast(pl.Float64).to_list()
            caption = (
                f"{color_by} (m)" if color_by == "distance_from_trace" else color_by
            )
            _add_colored_layer(folium, cm, matched_fg, m, m_lats, m_lons, vals, caption)
        else:
            _add_polyline(
                folium, matched_fg, m_lats, m_lons, color=matched_color, weight=4
            )
        matched_fg.add_to(m)

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
        _render_segments(folium, cm, move_fg, m, movement, color_by, matched_color)
        move_fg.add_to(m)

        stop_fg = folium.FeatureGroup(name="Stops")
        _add_stop_markers(folium, stop_fg, df, stop_color, stop_radius)
        stop_fg.add_to(m)

        folium.LayerControl().add_to(m)
        return m

    # --- Gap-split mode ---
    if has_segment_id:
        seg_fg = folium.FeatureGroup(name="Segments")
        _render_segments(folium, cm, seg_fg, m, df, color_by, matched_color)
        seg_fg.add_to(m)
        return m

    # --- Raw mode ---
    fg = folium.FeatureGroup(name="Trace")
    if color_by is not None and color_by in df.columns:
        vals = df[color_by].cast(pl.Float64).to_list()
        _add_colored_layer(
            folium, cm, fg, m, df["lat"].to_list(), df["lon"].to_list(), vals, color_by
        )
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
    m: Any,
    df: pl.DataFrame,
    color_by: str | None,
    default_color: str,
) -> None:
    """Render segments as polylines, optionally colored by a column."""
    if color_by is not None and color_by in df.columns:
        vals = df[color_by].cast(pl.Float64).to_list()
        _add_colored_layer(
            folium, cm, fg, m, df["lat"].to_list(), df["lon"].to_list(), vals, color_by
        )
        return

    for i, (_, seg) in enumerate(df.group_by("segment_id", maintain_order=True)):
        color = _SEGMENT_COLORS[i % len(_SEGMENT_COLORS)]
        _add_polyline(
            folium,
            fg,
            seg["lat"].to_list(),
            seg["lon"].to_list(),
            color=color,
            popup=f"Segment {seg['segment_id'][0]}",
        )


def plot_trace_layers(
    *,
    raw: pl.DataFrame | list[TracePoint] | None = None,
    segments: pl.DataFrame | None = None,
    matched: pl.DataFrame | None = None,
    matched_shape: list[list[tuple[float, float]]] | None = None,
    tile_layer: str = "CartoDB Positron",
    width: str | int = "100%",
    height: str | int = "100%",
    max_points: int | None = 5000,
    raw_color: str = "black",
    matched_color: str = "#e31a1c",
    stop_color: str = "red",
    stop_radius: int = 8,
) -> Any:
    """Plot multiple pipeline stages on a single map with togglable layers.

    Each non-``None`` argument becomes a ``FeatureGroup`` that can be
    toggled on or off via the layer control.

    Parameters
    ----------
    raw
        Raw trace as a DataFrame or ``list[TracePoint]``.
        Rendered as a dashed polyline.
    segments
        Gap-split or stop-split DataFrame (must have ``segment_id``).
        Each segment is drawn in a distinct color; stops (if ``is_stop``
        is present) are shown as circle markers.
    matched
        Map-matched DataFrame (must have ``matched_lat``, ``matched_lon``).
        Used for bounds calculation.  Ignored for rendering when
        ``matched_shape`` is provided.
    matched_shape
        Road-snapped polyline as ``(lat, lon)`` tuples, e.g. from
        :func:`~trucktrack.valhalla.map_match_full`.  When provided,
        this is drawn instead of straight lines between matched points.
    tile_layer
        Folium tile layer name.
    width, height
        Map dimensions.
    max_points
        Downsample each layer independently.  ``None`` to disable.
    raw_color
        Color for the raw trace polyline.
    matched_color
        Color for the map-matched polyline.
    stop_color
        Color for stop circle markers.
    stop_radius
        Radius of stop circle markers in pixels.

    Returns
    -------
    folium.Map
        An interactive map with a layer control.
    """
    folium, cm = _import_folium()

    raw_df = _normalize_input(raw, "raw") if raw is not None else None

    # Sort each layer by time so polylines render in chronological order.
    raw_df = _sort_by_time(raw_df)
    segments = _sort_by_time(segments)
    matched = _sort_by_time(matched)

    # Compute bounds using Polars-native aggregations (no list materialization).
    lat_min = float("inf")
    lat_max = float("-inf")
    lon_min = float("inf")
    lon_max = float("-inf")
    has_data = False
    for df in (raw_df, segments, matched):
        if df is not None and df.height > 0:
            lat_min = min(lat_min, df["lat"].min())  # type: ignore[arg-type]
            lat_max = max(lat_max, df["lat"].max())  # type: ignore[arg-type]
            lon_min = min(lon_min, df["lon"].min())  # type: ignore[arg-type]
            lon_max = max(lon_max, df["lon"].max())  # type: ignore[arg-type]
            has_data = True

    if not has_data:
        return folium.Map(tiles=tile_layer, width=width, height=height)

    center_lat = (lat_min + lat_max) / 2
    center_lon = (lon_min + lon_max) / 2
    m = folium.Map(
        location=(center_lat, center_lon),
        tiles=tile_layer,
        width=width,
        height=height,
    )
    m.fit_bounds([[lat_min, lon_min], [lat_max, lon_max]])

    # Layer: raw trace.
    if raw_df is not None and raw_df.height > 0:
        df = raw_df
        if max_points is not None:
            df = _downsample(df, max_points)
        fg = folium.FeatureGroup(name="Raw trace")
        for lat, lon in zip(df["lat"].to_list(), df["lon"].to_list(), strict=True):
            folium.CircleMarker(
                location=(lat, lon),
                radius=5,
                color=raw_color,
                fill=True,
                fill_color=raw_color,
                fill_opacity=0.7,
                weight=1,
            ).add_to(fg)
        fg.add_to(m)

    # Layer: segments.
    if segments is not None and segments.height > 0:
        seg_df = segments
        if max_points is not None:
            seg_df = _downsample(seg_df, max_points)

        has_is_stop = "is_stop" in seg_df.columns
        movement = seg_df.filter(~pl.col("is_stop")) if has_is_stop else seg_df

        seg_fg = folium.FeatureGroup(name="Segments")
        _render_segments(folium, cm, seg_fg, m, movement, None, raw_color)
        seg_fg.add_to(m)

        if has_is_stop:
            stop_fg = folium.FeatureGroup(name="Stops")
            _add_stop_markers(folium, stop_fg, seg_df, stop_color, stop_radius)
            stop_fg.add_to(m)

    # Layer: map-matched trace.
    if matched_shape:
        fg = folium.FeatureGroup(name="Map-matched")
        for shape in matched_shape:
            lats = [p[0] for p in shape]
            lons = [p[1] for p in shape]
            _add_polyline(folium, fg, lats, lons, color=matched_color, weight=4)
        fg.add_to(m)
    elif matched is not None and matched.height > 0:
        m_df = matched
        if max_points is not None:
            m_df = _downsample(m_df, max_points)
        fg = folium.FeatureGroup(name="Map-matched")
        _add_polyline(
            folium,
            fg,
            m_df["matched_lat"].to_list(),
            m_df["matched_lon"].to_list(),
            color=matched_color,
            weight=4,
        )
        fg.add_to(m)

    folium.LayerControl().add_to(m)
    return m


def save_map(m: Any, path: str | Path) -> None:
    """Save a folium Map to an HTML file."""
    m.save(str(Path(path)))


def serve_map(m: Any, *, host: str = "127.0.0.1", port: int = 5000) -> None:
    """Serve a folium Map via Flask.

    Useful inside k8s notebooks or other environments where static HTML
    files cannot be opened directly in a browser.

    Parameters
    ----------
    m
        A folium Map (returned by :func:`plot_trace` or
        :func:`plot_trace_layers`).
    host
        Bind address.  Use ``"0.0.0.0"`` to make the server reachable
        from outside the container.
    port
        Port to listen on.
    """
    try:
        from flask import Flask
    except ImportError as exc:
        raise ImportError(
            "flask is required to serve maps. Install it with: pip install flask"
        ) from exc

    app = Flask(__name__)
    html = m.get_root().render()

    @app.route("/")
    def index() -> str:
        return html

    print(f"Serving map at http://{host}:{port}/")
    app.run(host=host, port=port)
