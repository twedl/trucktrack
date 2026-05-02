"""Render per-filter before/after PNGs for the slide deck.

Every figure is a square map zoomed to the region where the filter's
error lives, rendered without axes or surrounding whitespace:

- basemap: CartoDB Positron
- pings:   black dots, alpha 0.75
- matched polyline: red
"""

from __future__ import annotations

import math
from datetime import timedelta
from pathlib import Path

import contextily as cx
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import trucktrack as tt
from matplotlib.collections import LineCollection

# Mirrors trucktrack.visualize's segment palette so the slide figures track
# the live viz when it cycles colours per movement segment.  Private import
# is intentional: drift guard, not a stable API contract.
from trucktrack.visualize._map import _SEGMENT_COLORS as SEGMENT_PALETTE

SLIDES = Path(__file__).resolve().parent.parent
DATA_DIR = SLIDES / "filters" / "data"
FIGDIR = SLIDES / "filters" / "figures"

PING_SIZE = 40
MATCH_LW = 3.5
MATCH_ALPHA = 0.95

# Single OSM Mapnik palette.  All figures render against the colorful
# OSM basemap on a white card so they read consistently on the dark
# slide background.
PALETTES: dict[str, dict] = {
    "light": dict(
        basemap=cx.providers.OpenStreetMap.Mapnik,
        ping="black",
        ping_alpha=0.75,
        match="#1d4ed8",   # bright saturated blue — pops against OSM
        text="black",
        text_bg="white",
        text_bg_alpha=0.9,
        ring="#ff7f0e",
        leg1="#1d4ed8",   # leg 1 reuses the route blue
        leg2="#ff6f00",   # leg 2 contrasts in saturated orange
        fig_bg="white",
        hl_a="#bbdefb",   # pale blue — source / "original"
        hl_b="#ffcdd2",   # pale red  — stale / "duplicate"
    ),
}


def _palette(key: str) -> dict:
    return PALETTES[key]


def _suffix(key: str) -> str:
    return "" if key == "light" else "-dark"


# ── figure primitives ──────────────────────────────────────────────────

def _square_fig() -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_axis_off()
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    return fig, ax


def _focus(ax: plt.Axes, lat_c: float, lon_c: float, radius_m: float) -> None:
    """Zoom to a square region of ``radius_m`` around (lat_c, lon_c)."""
    dlat = radius_m / 111_000
    dlon = radius_m / (111_000 * math.cos(math.radians(lat_c)))
    ax.set_xlim(lon_c - dlon, lon_c + dlon)
    ax.set_ylim(lat_c - dlat, lat_c + dlat)
    ax.set_aspect(1 / math.cos(math.radians(lat_c)), adjustable="datalim")


def _save(fig: plt.Figure, name: str, *, tight: bool = False) -> None:
    FIGDIR.mkdir(parents=True, exist_ok=True)
    path = FIGDIR / name
    fig.savefig(
        path, dpi=240,
        bbox_inches="tight" if tight else None,
        pad_inches=0.05 if tight else 0,
        facecolor=fig.get_facecolor(),
    )
    plt.close(fig)
    print(f"  wrote {path.relative_to(SLIDES)}")


def _add_basemap(ax: plt.Axes, pal: dict, *, zoom: int | str = "auto") -> None:
    try:
        cx.add_basemap(
            ax, crs="EPSG:4326", source=pal["basemap"], attribution=False, zoom=zoom
        )
    except Exception as e:
        print(f"    (basemap skipped: {e})")


def _draw_pings(ax: plt.Axes, df: pl.DataFrame, pal: dict) -> None:
    ax.scatter(
        df["lon"], df["lat"],
        s=PING_SIZE, color=pal["ping"], alpha=pal["ping_alpha"],
        edgecolor="none", zorder=3,
    )


def _draw_labeled_pings(
    ax: plt.Axes,
    df: pl.DataFrame,
    pal: dict,
    *,
    label_col: str = "label",
    offset_overrides: dict[int, tuple[int, int]] | None = None,
) -> None:
    offset_overrides = offset_overrides or {}
    for row in df.iter_rows(named=True):
        lon, lat = row["lon"], row["lat"]
        label = row[label_col]
        ax.scatter(
            lon, lat,
            s=PING_SIZE, color=pal["ping"], alpha=pal["ping_alpha"],
            edgecolor="none", zorder=3,
        )
        x_off, y_off = offset_overrides.get(int(label), (8, 8))
        ax.annotate(
            str(label),
            (lon, lat),
            textcoords="offset points",
            xytext=(x_off, y_off),
            fontsize=12,
            fontweight="bold",
            color=pal["text"],
            bbox=dict(
                boxstyle="round,pad=0.2",
                facecolor=pal["text_bg"],
                edgecolor="none",
                alpha=0.85,
            ),
            zorder=4,
        )


def _draw_matched(
    ax: plt.Axes,
    filename: str,
    pal: dict,
    *,
    reveal_retrace: bool = False,
    shape_colors: dict[int, str] | None = None,
) -> None:
    """Draw matched-route polylines.  Defaults to a single palette colour
    (``palette[0]``) for every shape, since matcher-side fragmentation
    isn't a user-applied segment.  Pass ``shape_colors`` to colour shapes
    individually — used when matched-shape index *N* should track the
    *N*-th movement segment from a trucktrack splitter."""
    path = DATA_DIR / filename
    if not path.exists():
        return
    df = pl.read_parquet(path)
    default_color = _segment_color(0)
    for sid in sorted(df["shape_id"].unique()):
        sub = df.filter(pl.col("shape_id") == sid)
        lons = sub["lon"].to_numpy()
        lats = sub["lat"].to_numpy()
        sid_int = int(sid)
        color = (
            shape_colors[sid_int]
            if shape_colors and sid_int in shape_colors
            else default_color
        )
        if reveal_retrace and len(lons) >= 2:
            pts = np.column_stack([lons, lats]).reshape(-1, 1, 2)
            segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
            lc = LineCollection(
                segs, colors=color, linewidth=5.0, alpha=0.28,
                zorder=2, capstyle="round",
            )
            ax.add_collection(lc)
        else:
            ax.plot(
                lons, lats, color=color, linewidth=MATCH_LW,
                alpha=MATCH_ALPHA, zorder=2,
            )


def _segment_color(sid: int) -> str:
    """Palette colour for ``segment_id``."""
    return SEGMENT_PALETTE[sid % len(SEGMENT_PALETTE)]


def _movement_shape_colors(seg: pl.DataFrame) -> dict[int, str]:
    """Map matched-shape index → segment-palette colour, mirroring
    ``trucktrack.visualize._add_matched_segments``: shape *N* corresponds
    to the *N*-th movement segment in time order; stops are skipped."""
    if "segment_id" not in seg.columns:
        return {}
    movement = seg.filter(~pl.col("is_stop")) if "is_stop" in seg.columns else seg
    sids = (
        movement.sort("time")
        .select("segment_id")
        .unique(maintain_order=True)
        .to_series()
        .to_list()
    )
    return {idx: _segment_color(int(sid)) for idx, sid in enumerate(sids)}


def _draw_segments_palette(ax: plt.Axes, df: pl.DataFrame, pal: dict) -> None:
    """Palette-cycled polyline per movement segment plus the standard
    black raw-trace dots.  Stop segments are skipped —
    ``_draw_all_stop_decorations`` handles them."""
    if "segment_id" in df.columns:
        has_is_stop = "is_stop" in df.columns
        for _, group in df.sort("time").group_by("segment_id", maintain_order=True):
            sid = int(group["segment_id"][0])
            if has_is_stop and bool(group["is_stop"][0]):
                continue
            if group.height >= 2:
                ax.plot(
                    group["lon"], group["lat"],
                    color=_segment_color(sid), linewidth=1.8, alpha=0.55, zorder=3,
                )
    _draw_pings(ax, df, pal)


def _draw_stop_decoration(
    ax: plt.Axes,
    stop_rows: pl.DataFrame,
    prev_pt: dict | None,
    next_pt: dict | None,
    color: str,
) -> None:
    """Mirror trucktrack.visualize stop rendering: interior polyline through
    the time-ordered stop pings, a centroid marker, and dashed connectors
    to the surrounding movement segments."""
    lons = stop_rows["lon"].to_list()
    lats = stop_rows["lat"].to_list()
    centroid_lon = sum(lons) / len(lons)
    centroid_lat = sum(lats) / len(lats)

    if len(lons) >= 2:
        ax.plot(lons, lats, color=color, linewidth=2.0, alpha=0.6, zorder=4)

    ax.scatter(
        [centroid_lon], [centroid_lat],
        s=PING_SIZE + 120, color=color, alpha=0.65,
        edgecolor=color, linewidths=2, zorder=5,
    )

    if prev_pt is not None:
        ax.plot(
            [prev_pt["lon"], lons[0]], [prev_pt["lat"], lats[0]],
            color=color, linewidth=1.2, alpha=0.55,
            linestyle=(0, (4, 4)), zorder=4,
        )
    if next_pt is not None:
        ax.plot(
            [lons[-1], next_pt["lon"]], [lats[-1], next_pt["lat"]],
            color=color, linewidth=1.2, alpha=0.55,
            linestyle=(0, (4, 4)), zorder=4,
        )


def _draw_all_stop_decorations(ax: plt.Axes, seg: pl.DataFrame) -> None:
    """Render the stop decoration (interior polyline + centroid + dashed
    connectors) for every is_stop segment in ``seg``, palette-coloured
    by segment_id."""
    if "is_stop" not in seg.columns or "segment_id" not in seg.columns:
        return
    seg_sorted = seg.sort("time")
    stop_sids = (
        seg_sorted.filter(pl.col("is_stop"))
        .select("segment_id")
        .unique(maintain_order=True)
        .to_series()
        .to_list()
    )
    for sid_val in stop_sids:
        sid = int(sid_val)
        stop_rows = seg_sorted.filter(pl.col("segment_id") == sid)
        prev_seg = seg_sorted.filter(pl.col("segment_id") == sid - 1)
        next_seg = seg_sorted.filter(pl.col("segment_id") == sid + 1)
        prev_pt = prev_seg.row(prev_seg.height - 1, named=True) if prev_seg.height else None
        next_pt = next_seg.row(0, named=True) if next_seg.height else None
        _draw_stop_decoration(ax, stop_rows, prev_pt, next_pt, _segment_color(sid))


# ── helpers for picking the focus area ─────────────────────────────────

def _anti_join(raw: pl.DataFrame, kept: pl.DataFrame) -> pl.DataFrame:
    return raw.join(kept.select("id", "time"), on=["id", "time"], how="anti")


def _bbox_focus(df: pl.DataFrame, pad_m: float) -> tuple[float, float, float]:
    """Center + radius covering every point in ``df`` plus ``pad_m``."""
    lat_min, lat_max = float(df["lat"].min()), float(df["lat"].max())
    lon_min, lon_max = float(df["lon"].min()), float(df["lon"].max())
    lat_c = (lat_min + lat_max) / 2
    lon_c = (lon_min + lon_max) / 2
    dlat_m = (lat_max - lat_min) * 111_000 / 2
    dlon_m = (lon_max - lon_min) * 111_000 * math.cos(math.radians(lat_c)) / 2
    radius = max(dlat_m, dlon_m) + pad_m
    return lat_c, lon_c, radius


def _time_neighbors(raw: pl.DataFrame, dropped: pl.DataFrame, window: int = 4) -> pl.DataFrame:
    """Kept rows temporally adjacent to dropped rows — used when dropped
    rows have unusable coords (coordinate-corruption spikes)."""
    raw_sorted = raw.sort("id", "time").with_row_index("_rid")
    dropped_ids = raw_sorted.join(
        dropped.select("id", "time"), on=["id", "time"], how="inner"
    )["_rid"].to_list()
    wanted: set[int] = set()
    for rid in dropped_ids:
        wanted.update(range(max(0, rid - window), rid + window + 1))
    return raw_sorted.filter(pl.col("_rid").is_in(list(wanted))).drop("_rid")


# ── per-filter builds ──────────────────────────────────────────────────

def _find_stale_indices(df: pl.DataFrame) -> tuple[int, int]:
    """Return (source_idx, stale_idx) for the unique stale re-emission."""
    ordered = df.sort("time")
    seen: dict[tuple, int] = {}
    for i, row in enumerate(ordered.to_dicts()):
        key = (row["lat"], row["lon"], row["speed"], row["heading"])
        if key in seen:
            return seen[key], i
        seen[key] = i
    raise ValueError("no duplicate (lat, lon, speed, heading) tuple found in trace")


def build_stale() -> None:
    raw = pl.read_parquet(DATA_DIR / "stale_trace.parquet").sort("time")
    raw = raw.with_columns(pl.col("time").diff().dt.total_seconds().alias("dt"))
    clean = tt.filter_stale_pings(raw.drop("dt")).sort("time")
    clean = clean.with_columns(pl.col("time").diff().dt.total_seconds().alias("dt"))

    src_idx, stale_idx = _find_stale_indices(raw.drop("dt"))

    # Map view: focus tight around the matcher's detour loop.  Filter the
    # matched-before polyline to points near the stale (excluding the rest
    # of the trip which dwarfs the loop) and bbox those.
    import math as _math
    stale_row = raw.drop("dt").row(stale_idx, named=True)
    src_lat, src_lon = float(stale_row["lat"]), float(stale_row["lon"])
    matched_before = pl.read_parquet(DATA_DIR / "stale_matched_before.parquet")
    near_radius_deg_lat = 800 / 111_000
    near_radius_deg_lon = 800 / (111_000 * _math.cos(_math.radians(src_lat)))
    loop_pts = matched_before.filter(
        (pl.col("lat").is_between(src_lat - near_radius_deg_lat, src_lat + near_radius_deg_lat))
        & (pl.col("lon").is_between(src_lon - near_radius_deg_lon, src_lon + near_radius_deg_lon))
    )
    if loop_pts.is_empty():
        loop_pts = matched_before
    lat_c, lon_c, radius_auto = _bbox_focus(loop_pts, pad_m=80)
    map_radius_m = max(radius_auto, 250)

    # Label every ping in the raw (with-stale) sequence by its time-order
    # index.  The clean (after-filter) sequence reuses the same labels —
    # the stale row's label is just absent.
    raw_labeled = raw.drop("dt").sort("time").with_row_index("label")
    stale_time_val = raw_labeled.row(stale_idx, named=True)["time"]
    clean_labeled = raw_labeled.filter(pl.col("time") != stale_time_val)
    # Place stale's label below the source's so both stay readable.
    overrides = {stale_idx: (8, -16)}

    for stem, matched, pings, override_dict in [
        ("stale-map-before", "stale_matched_before.parquet", raw_labeled, overrides),
        ("stale-map-after", "stale_matched_after.parquet", clean_labeled, {}),
    ]:
        for palette_key, pal in PALETTES.items():
            fig, ax = _square_fig()
            _focus(ax, lat_c, lon_c, map_radius_m)
            _draw_matched(ax, matched, pal)
            _draw_labeled_pings(ax, pings, pal, offset_overrides=override_dict)
            _add_basemap(ax, pal, zoom=17)
            _save(fig, f"{stem}{_suffix(palette_key)}.png")

    # Window: one row before source, then source, T2, stale, one row after.
    # That's 5 rows showing T0 → T1(src) → T2 → T3(=T1) → T4.
    take_start = max(0, src_idx - 1)
    before_window = raw.slice(take_start, 5)
    # Highlight T1 (source) and T3 (stale copy) — the two bit-identical rows.
    src_data_row = src_idx - take_start
    stale_data_row = stale_idx - take_start
    _polars_table(
        _slim_table(before_window),
        "stale-before{suffix}.png",
        highlights={src_data_row: "hl_a", stale_data_row: "hl_b"},
    )

    # After: same 5-row window with the stale row blanked to "---" so the
    # red highlight visually shows what the filter removed.  Both blue and
    # red highlights are kept to read against stale-before side-by-side.
    after_window = _blank_row(_slim_table(before_window), stale_data_row)
    _polars_table(
        after_window,
        "stale-after{suffix}.png",
        highlights={src_data_row: "hl_a", stale_data_row: "hl_b"},
    )


def build_speed() -> None:
    raw = pl.read_parquet(DATA_DIR / "speed_trace.parquet").sort("time")
    clean = tt.filter_impossible_speeds(raw, max_speed_kmh=200.0).sort("time")
    raw_labeled = raw.with_row_index("label")
    dropped = _anti_join(raw_labeled, clean).sort("time")
    if dropped.is_empty():
        print("    (speed: nothing dropped — skipping figures)")
        return

    spike_row = dropped.row(0, named=True)
    spike_lat = float(spike_row["lat"])
    spike_lon = float(spike_row["lon"])
    spike_label = int(spike_row["label"])

    # Carry the labels into the cleaned df by matching on time.
    clean_labeled = raw_labeled.filter(pl.col("time") != spike_row["time"])

    # Focus tight around the spike + a few surrounding kept pings so the
    # off-route position is unmistakable but we still see the road.
    nearby_kept = clean_labeled.filter(
        (pl.col("label") >= spike_label - 3) & (pl.col("label") <= spike_label + 3)
    )
    focus_pts = pl.concat(
        [
            nearby_kept.select("lat", "lon"),
            pl.DataFrame({"lat": [spike_lat], "lon": [spike_lon]}),
        ]
    )
    lat_c, lon_c, radius_auto = _bbox_focus(focus_pts, pad_m=400)
    map_radius_m = max(radius_auto, 800)

    for palette_key, pal in PALETTES.items():
        # Before — the spike is still in the trace.  Draw the raw trip
        # polyline through every ping in time order (including the
        # off-route spike at point 25); skip the map-matched route since
        # it would either fragment around the spike or quietly hide it.
        fig, ax = _square_fig()
        _focus(ax, lat_c, lon_c, map_radius_m)
        ax.plot(
            raw_labeled["lon"], raw_labeled["lat"],
            color=_segment_color(0), linewidth=MATCH_LW,
            alpha=MATCH_ALPHA, zorder=2,
        )
        _draw_labeled_pings(ax, raw_labeled, pal)
        ax.scatter(
            [spike_lon], [spike_lat],
            s=220, facecolors="none", edgecolors=_segment_color(0),
            linewidths=2.5, zorder=5,
        )
        _add_basemap(ax, pal, zoom=14)
        _save(fig, f"speed-before{_suffix(palette_key)}.png")

        # After — spike removed; matched route shows the clean routed line.
        fig, ax = _square_fig()
        _focus(ax, lat_c, lon_c, map_radius_m)
        _draw_matched(ax, "speed_matched_after.parquet", pal)
        _draw_labeled_pings(ax, clean_labeled, pal)
        _add_basemap(ax, pal, zoom=14)
        _save(fig, f"speed-after{_suffix(palette_key)}.png")


def _blank_row(df: pl.DataFrame, row_idx: int, placeholder: str = "---") -> pl.DataFrame:
    """Cast every column to string and overwrite one row with ``placeholder``.

    Used to render the stale-after table: same shape and positions as
    stale-before, but the dropped row is shown as ``---`` so the highlight
    visually marks what the filter removed.
    """
    str_df = df.select([pl.col(c).cast(pl.Utf8) for c in df.columns])
    if not (0 <= row_idx < str_df.height):
        return str_df
    cols: dict[str, list] = {}
    for c in str_df.columns:
        vals = str_df[c].to_list()
        vals[row_idx] = placeholder
        cols[c] = vals
    return pl.DataFrame(cols)


def _slim_table(df: pl.DataFrame) -> pl.DataFrame:
    """Cosmetic transforms for the rendered tables: blank out the noisy
    truck id, drop the date from the timestamp, and shorten segment_id."""
    out = df
    if "id" in out.columns:
        out = out.with_columns(pl.lit("...").alias("id"))
    if "time" in out.columns and out.schema["time"] == pl.Datetime:
        out = out.with_columns(pl.col("time").dt.time())
    if "segment_id" in out.columns:
        out = out.rename({"segment_id": "seg_id"})
    return out


def _polars_table(
    df: pl.DataFrame,
    filename_template: str,
    *,
    fontsize: int = 16,
    highlights: dict[int, str] | None = None,
) -> None:
    """Render a polars DataFrame as monospace text.  Writes light + dark
    variants; ``filename_template`` must contain ``{suffix}``.

    ``highlights`` maps a 0-based data-row index to a palette color key
    (e.g. ``"hl_a"``) — that row gets a colored background rectangle.
    """
    highlights = highlights or {}
    with pl.Config(
        tbl_rows=df.height,
        tbl_cols=df.width,
        fmt_str_lengths=60,
        tbl_hide_dataframe_shape=False,
        tbl_hide_column_data_types=False,
    ):
        text = str(df)

    lines = text.splitlines()
    n_lines = len(lines)
    longest = max(len(line) for line in lines)

    char_w_in = (fontsize * 0.6) / 72
    line_h_in = (fontsize * 1.25) / 72
    margin_in = 0.4
    fig_w = longest * char_w_in + 2 * margin_in
    fig_h = n_lines * line_h_in + 2 * margin_in

    # Polars table layout (per str(df)):
    #   line 0: "shape: (N, M)"
    #   line 1: ┌─...─┐    (top border)
    #   line 2: │ name │…   (header)
    #   line 3: │ --- │…
    #   line 4: │ dtype │…
    #   line 5: ╞═...═╡    (header/data separator)
    #   line 6: │ data row 0 │
    #   line 7: │ data row 1 │ …
    DATA_ROW_LINE_OFFSET = 6

    for palette_key, pal in PALETTES.items():
        fig = plt.figure(figsize=(fig_w, fig_h), facecolor=pal["fig_bg"])
        ax = fig.add_axes([0, 0, 1, 1])
        ax.set_facecolor(pal["fig_bg"])
        ax.set_axis_off()
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        line_h_axes = line_h_in / fig_h
        text_top_axes = 1 - margin_in / fig_h
        x_left = margin_in / fig_w
        x_right = (margin_in + longest * char_w_in) / fig_w

        # Highlight rectangles — drawn first so text reads on top.  Each
        # rectangle aligns 1:1 with a per-line text artist below; both
        # use the same y_step (line_h_axes), so they line up exactly.
        for data_row, color_key in highlights.items():
            line_idx = DATA_ROW_LINE_OFFSET + data_row
            y_top = text_top_axes - line_idx * line_h_axes
            y_bottom = y_top - line_h_axes
            rect = plt.Rectangle(
                (x_left, y_bottom),
                x_right - x_left,
                line_h_axes,
                facecolor=pal[color_key],
                edgecolor="none",
                alpha=0.55,
                zorder=0,
            )
            ax.add_patch(rect)

        # Render each line separately at a known y so text occupies
        # exactly the slot the highlight rectangle covers.  Using va="top"
        # places the line's top at y_top; ha="left" anchors it at x_left.
        for i, line in enumerate(lines):
            y_top = text_top_axes - i * line_h_axes
            ax.text(
                x_left, y_top, line,
                family="monospace", fontsize=fontsize,
                color=pal["text"], va="top", ha="left",
                zorder=1,
            )

        _save(fig, filename_template.format(suffix=_suffix(palette_key)), tight=True)


def build_gap() -> None:
    raw = pl.read_parquet(DATA_DIR / "gap_trace.parquet").sort("time")
    raw = raw.with_columns(pl.col("time").diff().dt.total_seconds().alias("dt"))
    seg = tt.split_by_observation_gap(raw.drop("dt"), timedelta(minutes=2))
    seg_with_dt = seg.sort("time").with_columns(
        pl.col("time").diff().dt.total_seconds().alias("dt")
    )

    gap_idx = int(raw["dt"].arg_max())
    take_start = max(0, gap_idx - 2)
    before_window = raw.slice(take_start, 5)
    after_window = seg_with_dt.slice(take_start, 5)
    # Row inside the 5-row window where the big dt sits — that's the
    # "observation where the gap is measured" in both tables.  In the
    # after-window the same row also gets a fresh segment_id, so the
    # highlight links the dt jump to the new segment boundary.
    gap_data_row = gap_idx - take_start

    _polars_table(
        _slim_table(before_window),
        "gap-before{suffix}.png",
        highlights={gap_data_row: "hl_b"},
    )
    _polars_table(
        _slim_table(after_window),
        "gap-after{suffix}.png",
        highlights={gap_data_row: "hl_b"},
    )


def build_gap_map() -> None:
    """Two zoomed-in views of the same A → near-B → A trip with a 10 h
    shutoff between the legs:

    1. ``gap-map-before`` — matched as one continuous route (no splitter).
       Meili has to invent a U-turn through the next interchange to
       connect the two pings on opposite carriageways.
    2. ``gap-map-after`` — split first with ``split_by_observation_gap``,
       then matched per segment.  Two clean routes ending / starting on
       the highway exactly where the device fell silent / resumed.
    """
    from matplotlib.patches import Patch

    raw = pl.read_parquet(DATA_DIR / "gap_uturn_trace.parquet").sort("time")
    matched_before = pl.read_parquet(DATA_DIR / "gap_uturn_matched_before.parquet")
    matched_after = pl.read_parquet(DATA_DIR / "gap_uturn_matched_after.parquet")
    seg = tt.split_by_observation_gap(raw, timedelta(minutes=2)).sort("time")

    out_last = seg.filter(pl.col("segment_id") == 0).sort("time").row(-1, named=True)
    back_first = seg.filter(pl.col("segment_id") == 1).sort("time").row(0, named=True)
    gap_lat_c = (out_last["lat"] + back_first["lat"]) / 2
    gap_lon_c = (out_last["lon"] + back_first["lon"]) / 2
    radius_m = 2500   # ~2.5 km half-side — fits the interchange + a
                     # handful of pings on each leg

    # Match the trucktrack.visualize segment palette so the after view
    # tracks the live viz: matched-shape N gets the colour of the N-th
    # movement segment.
    seg_colors = [_segment_color(0), _segment_color(1)]

    # Pings inside the visible window, numbered 1..N in time order.
    dlat_deg = radius_m / 111_000
    dlon_deg = radius_m / (111_000 * math.cos(math.radians(gap_lat_c)))
    visible = (
        seg.filter(
            (pl.col("lat").is_between(gap_lat_c - dlat_deg, gap_lat_c + dlat_deg))
            & (pl.col("lon").is_between(gap_lon_c - dlon_deg, gap_lon_c + dlon_deg))
        )
        .sort("time")
        .with_row_index("n")
        .with_columns((pl.col("n") + 1).alias("n"))
    )
    # Find the labels of the two gap endpoints inside the numbered set.
    gap_out_n = int(visible.filter(
        (pl.col("time") == out_last["time"]) & (pl.col("segment_id") == 0)
    ).row(0, named=True)["n"])
    gap_in_n = int(visible.filter(
        (pl.col("time") == back_first["time"]) & (pl.col("segment_id") == 1)
    ).row(0, named=True)["n"])

    def _draw_pings(ax, pal):
        """All trace pings as black dots, with a small number label
        next to each ping showing its time-order position in the
        visible window."""
        ax.scatter(
            seg["lon"], seg["lat"],
            s=70, color="black", edgecolor="none",
            alpha=1.0, zorder=5,
        )
        for row in visible.iter_rows(named=True):
            ax.annotate(
                str(row["n"]),
                (row["lon"], row["lat"]),
                textcoords="offset points", xytext=(8, 8),
                fontsize=12, fontweight="bold", color="black",
                bbox=dict(boxstyle="round,pad=0.18",
                          facecolor="white", edgecolor="none",
                          alpha=0.85),
                zorder=6,
            )

    def _draw_gap_callout(ax, pal, label: str):
        """Numbered callout label centered on the gap midpoint."""
        ax.annotate(
            label,
            (gap_lon_c, gap_lat_c),
            textcoords="offset points", xytext=(0, 70),
            ha="center",
            fontsize=14, fontweight="bold", color="black",
            bbox=dict(boxstyle="round,pad=0.35",
                      facecolor=pal["text_bg"], edgecolor="black",
                      linewidth=1.5, alpha=0.95),
            zorder=8,
        )

    # ── Map 1: no splitter — Meili matches the whole trace as one shape.
    for palette_key, pal in PALETTES.items():
        fig, ax = _square_fig()
        _focus(ax, gap_lat_c, gap_lon_c, radius_m)

        # Single matched polyline — palette[0] for visual consistency
        # with the rest of the deck (no segments yet, so just the first
        # palette colour).
        before_color = _segment_color(0)
        for sid in sorted(matched_before["shape_id"].unique().to_list()):
            sub = matched_before.filter(pl.col("shape_id") == sid)
            ax.plot(
                sub["lon"], sub["lat"],
                color=before_color, linewidth=MATCH_LW,
                alpha=MATCH_ALPHA, zorder=2,
            )

        _draw_pings(ax, pal)
        _draw_gap_callout(
            ax, pal,
            f"10 h gap between ({gap_out_n}) and ({gap_in_n})",
        )

        ax.legend(
            handles=[
                Patch(facecolor=before_color, edgecolor="none",
                      label="matched route (no splitter)"),
                Patch(facecolor="black", edgecolor="none",
                      label="trace pings"),
            ],
            loc="upper left", fontsize=11, framealpha=0.95,
            facecolor=pal["text_bg"], edgecolor="none",
        ).set_zorder(20)

        _add_basemap(ax, pal, zoom=15)
        _save(fig, f"gap-map-before{_suffix(palette_key)}.png")

    # ── Map 2: split first, match each segment — two clean routes.
    for palette_key, pal in PALETTES.items():
        fig, ax = _square_fig()
        _focus(ax, gap_lat_c, gap_lon_c, radius_m)

        for sid in (0, 1):
            sub = matched_after.filter(pl.col("shape_id") == sid)
            if sub.is_empty():
                continue
            ax.plot(
                sub["lon"], sub["lat"],
                color=seg_colors[sid], linewidth=MATCH_LW,
                alpha=MATCH_ALPHA, zorder=2 + sid,
            )

        _draw_pings(ax, pal)
        _draw_gap_callout(
            ax, pal,
            f"10 h gap between ({gap_out_n}) and ({gap_in_n})\n"
            "splitter ends one trip, starts another",
        )

        ax.legend(
            handles=[
                Patch(facecolor=seg_colors[0], edgecolor="none",
                      label="leg 1: westbound (out)"),
                Patch(facecolor=seg_colors[1], edgecolor="none",
                      label="leg 2: eastbound (return)"),
                Patch(facecolor="black", edgecolor="none",
                      label="trace pings"),
            ],
            loc="upper left", fontsize=11, framealpha=0.95,
            facecolor=pal["text_bg"], edgecolor="none",
        ).set_zorder(20)

        _add_basemap(ax, pal, zoom=15)
        _save(fig, f"gap-map-after{_suffix(palette_key)}.png")


def _first_stop_segment(seg: pl.DataFrame) -> pl.DataFrame:
    """Earliest is_stop segment — skips origin/destination maneuvers since
    those come first/last in time and we want the injected mid-route stop."""
    stops = seg.filter(pl.col("is_stop")).sort("time")
    first_sid = int(stops.row(0, named=True)["segment_id"])
    return seg.filter(pl.col("segment_id") == first_sid)


def build_stop() -> None:
    raw = pl.read_parquet(DATA_DIR / "stop_trace.parquet").sort("time")
    seg = (
        tt.split_by_stops(raw, max_diameter=50.0, min_duration=timedelta(minutes=2))
        .sort("time")
    )

    # The first is_stop segment (the gas-station dwell — destination
    # maneuver may also flag but isn't the demo).
    first_stop_rows = _first_stop_segment(seg).filter(pl.col("is_stop"))
    n_dwell = first_stop_rows.height
    first_sid = int(first_stop_rows["segment_id"][0])

    # Wide enough to show the off-ramp / gas station / on-ramp context.
    lat_c, lon_c, radius_auto = _bbox_focus(first_stop_rows, pad_m=1500)
    map_radius_m = max(radius_auto, 1500)

    matched_shape_colors = _movement_shape_colors(seg)
    dwell_label_color = _segment_color(first_sid)

    def _label_dwell(ax, pal, *, color: str) -> None:
        anchor = first_stop_rows.row(0, named=True)
        ax.annotate(
            f"{n_dwell} pings, ~{n_dwell - 1} min dwell",
            (anchor["lon"], anchor["lat"]),
            textcoords="offset points",
            xytext=(14, -55),
            fontsize=12,
            color=color,
            bbox=dict(
                boxstyle="round,pad=0.25",
                facecolor=pal["text_bg"], edgecolor="none", alpha=pal["text_bg_alpha"],
            ),
            zorder=4,
        )

    for palette_key, pal in PALETTES.items():
        # Before — raw pings, single matched colour (no segments yet).
        fig, ax = _square_fig()
        _focus(ax, lat_c, lon_c, map_radius_m)
        _draw_matched(ax, "stop_matched.parquet", pal)
        _draw_pings(ax, raw, pal)
        _label_dwell(ax, pal, color=pal["text"])
        _add_basemap(ax, pal, zoom=15)
        _save(fig, f"stop-before{_suffix(palette_key)}.png")

        # After — segment-palette throughout: per-segment polyline, stop
        # decoration with palette colour, matched route palette-coloured.
        fig, ax = _square_fig()
        _focus(ax, lat_c, lon_c, map_radius_m)
        _draw_matched(
            ax, "stop_matched.parquet", pal,
            shape_colors=matched_shape_colors,
        )
        _draw_segments_palette(ax, seg, pal)
        _draw_all_stop_decorations(ax, seg)
        _label_dwell(ax, pal, color=dwell_label_color)
        _add_basemap(ax, pal, zoom=15)
        _save(fig, f"stop-after{_suffix(palette_key)}.png")


def build_traffic() -> None:
    raw = pl.read_parquet(DATA_DIR / "traffic_trace.parquet").sort("time")
    seg = (
        tt.split_by_stops(raw, max_diameter=50.0, min_duration=timedelta(minutes=2))
        .sort("time")
    )
    cleaned = tt.filter_traffic_stops(seg, max_angle_change=30.0, min_distance=10.0).sort("time")

    # Reclassified jam = rows that were is_stop in seg but not is_stop in cleaned.
    jam = seg.join(
        cleaned.filter(pl.col("is_stop")).select("id", "time"),
        on=["id", "time"], how="anti",
    ).filter(pl.col("is_stop"))
    if jam.is_empty():
        jam = _first_stop_segment(seg).filter(pl.col("is_stop"))

    lat_c, lon_c, radius_auto = _bbox_focus(jam, pad_m=18000)
    map_radius_m = max(radius_auto, 18000)
    n_jam = jam.height
    anchor = jam.row(0, named=True)

    # Segment palette: BEFORE shows two matched legs flanking the
    # jam-as-stop (segments 0 and 2); AFTER reclassifies the jam back to
    # movement, leaving one matched shape on segment 0.
    before_shape_colors = _movement_shape_colors(seg)
    after_shape_colors = _movement_shape_colors(cleaned)

    def _draw_two_leg_matched(ax, pal) -> None:
        _draw_matched(
            ax, "traffic_matched.parquet", pal,
            shape_colors=before_shape_colors,
        )

    def _draw_continuous_matched(ax, pal) -> None:
        _draw_matched(
            ax, "traffic_matched_after.parquet", pal,
            shape_colors=after_shape_colors,
        )

    jam_sid = int(jam["segment_id"][0]) if "segment_id" in jam.columns else None

    def _annotate(ax, pal, *, jam_label: bool) -> None:
        color = _segment_color(jam_sid) if (jam_label and jam_sid is not None) else pal["text"]
        ax.annotate(
            f"{n_jam} pings, ~{n_jam - 1} min stuck"
            + ("" if jam_label else "\n→ reclassified as movement"),
            (anchor["lon"], anchor["lat"]),
            textcoords="offset points",
            xytext=(14, -10),
            fontsize=12,
            color=color,
            bbox=dict(
                boxstyle="round,pad=0.25",
                facecolor=pal["text_bg"], edgecolor="none", alpha=pal["text_bg_alpha"],
            ),
            zorder=4,
        )

    for palette_key, pal in PALETTES.items():
        for after, stem in [(False, "traffic-before"), (True, "traffic-after")]:
            fig, ax = _square_fig()
            _focus(ax, lat_c, lon_c, map_radius_m)
            if after:
                _draw_continuous_matched(ax, pal)
                _draw_segments_palette(ax, cleaned, pal)
                _annotate(ax, pal, jam_label=False)
            else:
                _draw_two_leg_matched(ax, pal)
                _draw_segments_palette(ax, seg, pal)
                _draw_all_stop_decorations(ax, seg)
                _annotate(ax, pal, jam_label=True)
            _add_basemap(ax, pal, zoom=12)
            _save(fig, f"{stem}{_suffix(palette_key)}.png")

    # Companion slide: real stop preserved.  Reuse the stop scenario's
    # destination dwell — bearing changes (highway approach → dock turn)
    # so the traffic filter leaves it as a stop.
    stop_raw = pl.read_parquet(DATA_DIR / "stop_trace.parquet").sort("time")
    stop_seg = tt.split_by_stops(stop_raw, max_diameter=50.0, min_duration=timedelta(minutes=2))
    stop_after = tt.filter_traffic_stops(stop_seg, max_angle_change=30.0, min_distance=10.0)
    focus_rows = _first_stop_segment(stop_after).filter(pl.col("is_stop"))
    if focus_rows.is_empty():
        focus_rows = _first_stop_segment(stop_seg).filter(pl.col("is_stop"))
    lat_c, lon_c, radius_auto = _bbox_focus(focus_rows, pad_m=1500)
    map_radius_m = max(radius_auto, 1500)
    n_kept = focus_rows.height
    anchor = focus_rows.row(0, named=True)
    kept_sid = int(focus_rows["segment_id"][0])
    real_stop_shape_colors = _movement_shape_colors(stop_after)

    for palette_key, pal in PALETTES.items():
        fig, ax = _square_fig()
        _focus(ax, lat_c, lon_c, map_radius_m)
        _draw_matched(
            ax, "stop_matched.parquet", pal,
            shape_colors=real_stop_shape_colors,
        )
        _draw_segments_palette(ax, stop_after, pal)
        _draw_all_stop_decorations(ax, stop_after)
        ax.annotate(
            f"{n_kept} pings, ~{n_kept - 1} min dwell\n→ kept as stop",
            (anchor["lon"], anchor["lat"]),
            textcoords="offset points",
            xytext=(14, -65),
            fontsize=12,
            color=_segment_color(kept_sid),
            bbox=dict(
                boxstyle="round,pad=0.25",
                facecolor=pal["text_bg"], edgecolor="none", alpha=pal["text_bg_alpha"],
            ),
            zorder=4,
        )
        _add_basemap(ax, pal, zoom=15)
        _save(fig, f"traffic-real-stop{_suffix(palette_key)}.png")


def main() -> None:
    print(f"building figures → {FIGDIR.relative_to(SLIDES)}")
    build_stale()
    build_speed()
    build_gap()
    build_gap_map()
    build_stop()
    build_traffic()
    print("done.")


if __name__ == "__main__":
    main()
