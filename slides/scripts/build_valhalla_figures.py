"""Render figures for the Valhalla deck (slides/valhalla/figures).

Three figures, light + dark variants each:

    1. hierarchy-grid:    one map of southern Ontario with all three
                          tile-grid levels superimposed (mirrors the
                          Valhalla docs' Germany / NYC / PA panels)
    2. tile-zoom:         a single level-2 tile near Toronto, zoomed
                          tight so the basemap road network shows what
                          a tile actually contains
    3. trip-footprint:    the Mississauga → Cambridge trip from the
                          filter scenarios, overlaid with only the
                          tiles its pings fall into at each level
"""

from __future__ import annotations

import math
from pathlib import Path

import contextily as cx
import matplotlib.pyplot as plt
import polars as pl
from matplotlib.patches import Patch, Rectangle

# Mirror trucktrack.visualize's segment palette so trip / matched route
# colours track the live viz.  Private import is a deliberate drift guard.
from trucktrack.visualize._map import _SEGMENT_COLORS as SEGMENT_PALETTE

SLIDES = Path(__file__).resolve().parent.parent
VALHALLA_DATA = SLIDES / "valhalla" / "data"
FILTER_DATA = SLIDES / "filters" / "data"
FIGDIR = SLIDES / "valhalla" / "figures"

# Valhalla docs use light blue / light green / light red for levels
# 0 / 1 / 2.  We follow the same convention so readers cross-referencing
# the docs see the same colors.
PALETTES: dict[str, dict] = {
    "light": dict(
        basemap=cx.providers.OpenStreetMap.Mapnik,
        labels=None,
        text="black",
        text_bg="white",
        fig_bg="white",
        l0="#1976d2",
        l1="#388e3c",
        l2="#d32f2f",
        route=SEGMENT_PALETTE[0],
        ping="black",
    ),
}

LEVEL_SIZE = {0: 4.0, 1: 1.0, 2: 0.25}
LEVEL_COLS = {0: 90, 1: 360, 2: 1440}
LEVEL_KEY = {0: "l0", 1: "l1", 2: "l2"}

# trucktrack tier classification thresholds (km of bbox diagonal).
LOCAL_KM = 100.0
REGIONAL_KM = 800.0

# Tiers reuse the Valhalla level colors mapped by the tile size each
# tier buckets into:  local → 1° (l1 green), regional → 4° (l0 blue),
# longhaul → 8° (custom; takes the remaining l2 red).
TIER_LEVEL_KEY = {
    "local":    "l2",   # red
    "regional": "l1",   # green
    "longhaul": "l0",   # blue
}


def _suffix(key: str) -> str:
    return "" if key == "light" else "-dark"


def _save(fig: plt.Figure, name: str) -> None:
    FIGDIR.mkdir(parents=True, exist_ok=True)
    path = FIGDIR / name
    fig.savefig(path, dpi=240, bbox_inches=None, pad_inches=0,
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  wrote {path.relative_to(SLIDES)}")


def _square_fig(pal: dict) -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=(6, 6), facecolor=pal["fig_bg"])
    ax.set_facecolor(pal["fig_bg"])
    ax.set_axis_off()
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    return fig, ax


def _focus(ax, lat_c, lon_c, dlat, dlon) -> None:
    ax.set_xlim(lon_c - dlon, lon_c + dlon)
    ax.set_ylim(lat_c - dlat, lat_c + dlat)
    ax.set_aspect(1 / math.cos(math.radians(lat_c)), adjustable="datalim")


def _add_basemap(ax, pal, *, zoom) -> None:
    try:
        cx.add_basemap(ax, crs="EPSG:4326", source=pal["basemap"],
                       attribution=False, zoom=zoom)
    except Exception as e:
        print(f"    (basemap skipped: {e})")
        return
    # Optional brightened-labels overlay (dark theme).  CartoDB's
    # DarkMatter labels are very dim; we fetch the OnlyLabels tiles
    # for the same view and brighten them to ~1.7× before drawing on
    # top of the no-labels base.
    if pal.get("labels") is None:
        return
    try:
        import numpy as np
        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()
        img, ext = cx.bounds2img(
            xmin, ymin, xmax, ymax,
            zoom=zoom, source=pal["labels"], ll=True,
        )
        img = img.astype(np.float32)
        # Brighten labels.  Alpha-weighted: only boost where label
        # pixels exist so we don't lift the underlying transparent
        # background into a haze.
        boost = 2.0
        img[..., :3] = np.clip(img[..., :3] * boost, 0, 255)
        img = img.astype(np.uint8)
        # bounds2img returns extent in Web Mercator metres regardless
        # of the ``ll`` input flag — convert to lon/lat for our axes.
        R = 6378137.0
        x0, x1, y0, y1 = ext
        lon0 = math.degrees(x0 / R)
        lon1 = math.degrees(x1 / R)
        lat0 = math.degrees(math.atan(math.sinh(y0 / R)))
        lat1 = math.degrees(math.atan(math.sinh(y1 / R)))
        ax.imshow(img, extent=(lon0, lon1, lat0, lat1),
                  interpolation="bilinear", origin="upper", zorder=1.5)
    except Exception as e:
        print(f"    (label overlay skipped: {e})")


def _tile_id(level: int, lat: float, lon: float) -> int:
    size = LEVEL_SIZE[level]
    cols = LEVEL_COLS[level]
    col = int((lon + 180) / size)
    row = int((lat + 90) / size)
    return row * cols + col


def _tile_bbox(level: int, tile_id: int) -> tuple[float, float, float, float]:
    size = LEVEL_SIZE[level]
    cols = LEVEL_COLS[level]
    col = tile_id % cols
    row = tile_id // cols
    lon0 = -180 + col * size
    lat0 = -90 + row * size
    return lat0, lon0, lat0 + size, lon0 + size


def _full_grid_in_view(level: int, lat_c: float, lon_c: float,
                       dlat: float, dlon: float):
    """Yield (lat0, lon0) corners of every tile that intersects the view."""
    size = LEVEL_SIZE[level]
    lon_min, lon_max = lon_c - dlon, lon_c + dlon
    lat_min, lat_max = lat_c - dlat, lat_c + dlat
    col0 = int((lon_min + 180) / size)
    col1 = int((lon_max + 180) / size) + 1
    row0 = int((lat_min + 90) / size)
    row1 = int((lat_max + 90) / size) + 1
    for col in range(col0, col1):
        for row in range(row0, row1):
            yield -90 + row * size, -180 + col * size


def _draw_grid(ax, level: int, color: str, lat_c, lon_c, dlat, dlon,
               *, lw: float = 1.2, alpha: float = 0.85) -> None:
    """Draw outline-only rectangles for every tile in the visible area."""
    size = LEVEL_SIZE[level]
    for lat0, lon0 in _full_grid_in_view(level, lat_c, lon_c, dlat, dlon):
        ax.add_patch(Rectangle(
            (lon0, lat0), size, size,
            facecolor="none", edgecolor=color,
            linewidth=lw, alpha=alpha, zorder=3 + level,
        ))


def _draw_tile_boxes(ax, tiles: list[tuple[int, int]], color: str,
                     *, fill_alpha: float = 0.18, edge_alpha: float = 0.9,
                     lw: float = 1.4) -> None:
    """Draw filled rectangles for the given (level, tile_id) pairs."""
    for level, tid in tiles:
        lat0, lon0, lat1, lon1 = _tile_bbox(level, tid)
        ax.add_patch(Rectangle(
            (lon0, lat0), lon1 - lon0, lat1 - lat0,
            facecolor=color, alpha=fill_alpha,
            edgecolor=color, linewidth=lw,
            zorder=3 + level,
        ))
        ax.add_patch(Rectangle(
            (lon0, lat0), lon1 - lon0, lat1 - lat0,
            facecolor="none", edgecolor=color,
            linewidth=lw, alpha=edge_alpha, zorder=4 + level,
        ))


def _legend_box(ax, pal: dict, lines: list[tuple[str, str]],
                *, loc: str = "upper left") -> None:
    """Standard matplotlib legend with colored squares for each entry."""
    handles = [
        Patch(facecolor=pal[color_key], edgecolor="none", label=label)
        for color_key, label in lines
    ]
    leg = ax.legend(
        handles=handles, loc=loc, fontsize=12,
        labelcolor=pal["text"], facecolor=pal["text_bg"],
        edgecolor="none", framealpha=0.9, borderpad=0.6,
        handlelength=1.2, handleheight=1.0, borderaxespad=0.5,
    )
    leg.set_zorder(20)


# ── figure 1: nested-tile hierarchy ───────────────────────────────────

# Mirrors the Valhalla docs' explanatory figure: one level-0 tile with
# generous empty space around it, two adjacent level-1 tiles drawn
# inside it, and six level-2 tiles forming a strip across those.

# Level-0 tile covering southern Ontario (lat 42-46, lon -80 to -76).
HIER_L0 = (42.0, -80.0)         # SW corner
# Two adjacent level-1 tiles in the bottom-center of L0.
HIER_L1 = [
    (43.0, -80.0),              # 43-44 N, -80 to -79 W
    (43.0, -79.0),              # 43-44 N, -79 to -78 W
]
# Six level-2 tiles in the NW corner of the west-most level-1 tile —
# a 2-row × 3-col block hugging the upper-left.
HIER_L2 = [
    (43.75, -80.00), (43.75, -79.75), (43.75, -79.50),
    (43.50, -80.00), (43.50, -79.75), (43.50, -79.50),
]


def _draw_box(ax, sw_lat: float, sw_lon: float, size: float,
              edge: str, *, fill: str | None = None, lw: float = 2.0,
              fill_alpha: float = 0.0, edge_alpha: float = 1.0,
              zorder: int = 5) -> None:
    if fill is not None and fill_alpha > 0:
        ax.add_patch(Rectangle(
            (sw_lon, sw_lat), size, size,
            facecolor=fill, edgecolor="none",
            alpha=fill_alpha, zorder=zorder,
        ))
    ax.add_patch(Rectangle(
        (sw_lon, sw_lat), size, size,
        facecolor="none", edgecolor=edge,
        linewidth=lw, alpha=edge_alpha, zorder=zorder + 1,
    ))


def build_hierarchy_grid() -> None:
    # Center the view on the level-0 tile and add ~50% padding around
    # so the lone tile reads against lots of empty basemap.
    l0_lat0, l0_lon0 = HIER_L0
    lat_c = l0_lat0 + LEVEL_SIZE[0] / 2
    lon_c = l0_lon0 + LEVEL_SIZE[0] / 2
    half = LEVEL_SIZE[0] * 0.95

    for palette_key, pal in PALETTES.items():
        fig, ax = _square_fig(pal)
        _focus(ax, lat_c, lon_c, half, half)

        # Level-0: outline + faint fill.
        _draw_box(ax, l0_lat0, l0_lon0, LEVEL_SIZE[0],
                  pal["l0"], fill=pal["l0"], fill_alpha=0.10,
                  lw=3.0, zorder=4)

        # Level-1: stronger fill, thicker outline (within L0).
        for lat0, lon0 in HIER_L1:
            _draw_box(ax, lat0, lon0, LEVEL_SIZE[1],
                      pal["l1"], fill=pal["l1"], fill_alpha=0.20,
                      lw=2.4, zorder=6)

        # Level-2: most prominent fill (within the L1s).
        for lat0, lon0 in HIER_L2:
            _draw_box(ax, lat0, lon0, LEVEL_SIZE[2],
                      pal["l2"], fill=pal["l2"], fill_alpha=0.50,
                      lw=1.4, zorder=8)

        _add_basemap(ax, pal, zoom=7)
        _legend_box(ax, pal, [
            ("l0", "level 0 — 4° (highway)"),
            ("l1", "level 1 — 1° (arterial)"),
            ("l2", "level 2 — 0.25° (local)"),
        ])
        _save(fig, f"hierarchy-grid{_suffix(palette_key)}.png")


# ── figure 2: single-tile zoom ────────────────────────────────────────

# A level-2 tile that covers part of central Mississauga / west Toronto.
# Picked so the basemap shows the dense road network of the GTA inside
# the 0.25° box.
ZOOM_LEVEL = 2
ZOOM_LAT = 43.65
ZOOM_LON = -79.57


def build_tile_zoom() -> None:
    tid = _tile_id(ZOOM_LEVEL, ZOOM_LAT, ZOOM_LON)
    lat0, lon0, lat1, lon1 = _tile_bbox(ZOOM_LEVEL, tid)
    lat_c = (lat0 + lat1) / 2
    lon_c = (lon0 + lon1) / 2
    # 25% padding so the bbox doesn't touch the figure edge.
    dlat = (lat1 - lat0) / 2 * 1.25
    dlon = (lon1 - lon0) / 2 * 1.25
    for palette_key, pal in PALETTES.items():
        fig, ax = _square_fig(pal)
        _focus(ax, lat_c, lon_c, dlat, dlon)
        _draw_tile_boxes(ax, [(ZOOM_LEVEL, tid)], pal["l2"],
                         fill_alpha=0.0, edge_alpha=1.0, lw=3.0)
        _legend_box(ax, pal, [
            ("l2", f"level 2 tile · 2/{tid // 1_000_000:03d}/"
                   f"{(tid // 1000) % 1000:03d}/{tid % 1000:03d}.gph"),
        ])
        _add_basemap(ax, pal, zoom=12)
        _save(fig, f"tile-zoom{_suffix(palette_key)}.png")


# ── figure 3: trip → tile footprint ───────────────────────────────────

def _touched_tiles(df: pl.DataFrame, level: int) -> list[int]:
    """Unique tile IDs at ``level`` for every (lat, lon) in df."""
    seen: set[int] = set()
    for lat, lon in zip(df["lat"].to_list(), df["lon"].to_list(), strict=True):
        seen.add(_tile_id(level, lat, lon))
    return sorted(seen)


def build_trip_footprint() -> None:
    raw = pl.read_parquet(FILTER_DATA / "traffic_trace.parquet").sort("time")
    matched = pl.read_parquet(FILTER_DATA / "traffic_matched_after.parquet")

    by_level: dict[int, list[int]] = {
        level: _touched_tiles(raw, level) for level in (0, 1, 2)
    }

    # Figure focus: bbox of touched level-2 tiles + small pad.
    lats: list[float] = []
    lons: list[float] = []
    for tid in by_level[2]:
        lat0, lon0, lat1, lon1 = _tile_bbox(2, tid)
        lats += [lat0, lat1]
        lons += [lon0, lon1]
    lat_c = (min(lats) + max(lats)) / 2
    lon_c = (min(lons) + max(lons)) / 2
    dlat = (max(lats) - min(lats)) / 2 + 0.4
    dlon = (max(lons) - min(lons)) / 2 + 0.4

    for palette_key, pal in PALETTES.items():
        fig, ax = _square_fig(pal)
        _focus(ax, lat_c, lon_c, dlat, dlon)
        # Draw level 2 (most granular) first so coarser tiles sit on
        # top and remain visible.
        for level in (2, 1, 0):
            _draw_tile_boxes(
                ax,
                [(level, tid) for tid in by_level[level]],
                pal[LEVEL_KEY[level]],
                fill_alpha=0.0 if level < 2 else 0.18,
                edge_alpha=0.95,
                lw=2.4 if level == 0 else 1.6 if level == 1 else 1.0,
            )
        # Route polyline + ping markers, drawn on top.
        ax.plot(matched["lon"], matched["lat"],
                color=pal["route"], linewidth=2.0, alpha=0.9, zorder=10)
        ax.scatter(raw["lon"], raw["lat"],
                   s=24, color=pal["ping"], alpha=0.85,
                   edgecolor="none", zorder=11)
        _legend_box(ax, pal, [
            ("l0", f"level 0 — {len(by_level[0])} tiles"),
            ("l1", f"level 1 — {len(by_level[1])} tiles"),
            ("l2", f"level 2 — {len(by_level[2])} tiles"),
        ])
        _add_basemap(ax, pal, zoom=9)
        _save(fig, f"trip-footprint{_suffix(palette_key)}.png")


def _bbox_diag_km(lats: list[float], lons: list[float]) -> float:
    import math
    lat1, lat2 = min(lats), max(lats)
    lon1, lon2 = min(lons), max(lons)
    R = 6371.0
    p = math.pi / 180.0
    a = (math.sin((lat2 - lat1) * p / 2) ** 2
         + math.cos(lat1 * p) * math.cos(lat2 * p)
         * math.sin((lon2 - lon1) * p / 2) ** 2)
    return R * 2 * math.asin(math.sqrt(a))


def _trip_summary(df: pl.DataFrame) -> dict[str, dict]:
    """One row per trip with bbox + classification."""
    out: dict[str, dict] = {}
    for tid in df["id"].unique().to_list():
        sub = df.filter(pl.col("id") == tid).sort("seq")
        lats = sub["lat"].to_list()
        lons = sub["lon"].to_list()
        diag = _bbox_diag_km(lats, lons)
        if diag < LOCAL_KM:
            tier = "local"
        elif diag < REGIONAL_KM:
            tier = "regional"
        else:
            tier = "longhaul"
        out[tid] = dict(
            tier=tier,
            label=sub["label"][0],
            diag_km=diag,
            lats=lats,
            lons=lons,
            lat0=min(lats), lat1=max(lats),
            lon0=min(lons), lon1=max(lons),
        )
    return out


def build_tier_classification() -> None:
    """Three trips on US+Canada showing how bbox diagonal → tier."""
    df = pl.read_parquet(VALHALLA_DATA / "sample_trips.parquet")
    summary = _trip_summary(df)

    # Center the view on the bbox of all three trips combined,
    # with generous padding so routes aren't hugging the edges.
    all_lats = [v for s in summary.values() for v in s["lats"]]
    all_lons = [v for s in summary.values() for v in s["lons"]]
    lat_c = (min(all_lats) + max(all_lats)) / 2
    lon_c = (min(all_lons) + max(all_lons)) / 2
    dlat = (max(all_lats) - min(all_lats)) / 2 + 4
    dlon = (max(all_lons) - min(all_lons)) / 2 + 4

    for palette_key, pal in PALETTES.items():
        fig, ax = _square_fig(pal)
        _focus(ax, lat_c, lon_c, dlat, dlon)

        # Draw each trip in the order longhaul → regional → local so
        # the smaller routes draw on top and remain visible.
        for tid in ("longhaul", "regional", "local"):
            s = summary[tid]
            color = pal[TIER_LEVEL_KEY[s["tier"]]]
            # Bounding-box rectangle (light fill, colored edge).
            ax.add_patch(Rectangle(
                (s["lon0"], s["lat0"]),
                s["lon1"] - s["lon0"], s["lat1"] - s["lat0"],
                facecolor=color, alpha=0.10,
                edgecolor=color, linewidth=1.5, zorder=4,
            ))
            ax.add_patch(Rectangle(
                (s["lon0"], s["lat0"]),
                s["lon1"] - s["lon0"], s["lat1"] - s["lat0"],
                facecolor="none", edgecolor=color,
                linewidth=1.5, alpha=0.85, zorder=5,
            ))
            ax.plot(s["lons"], s["lats"],
                    color=color, linewidth=2.4, alpha=0.95, zorder=6)
            # If the trip's bbox is small enough that it would vanish
            # at this zoom, ring its centroid so it's findable.
            bbox_span = max(s["lat1"] - s["lat0"], s["lon1"] - s["lon0"])
            if bbox_span < 1.0:
                cx_lat = (s["lat0"] + s["lat1"]) / 2
                cx_lon = (s["lon0"] + s["lon1"]) / 2
                ax.scatter([cx_lon], [cx_lat], s=320,
                           facecolors="none", edgecolors=color,
                           linewidths=2.5, zorder=7)

        legend_lines = []
        for tid in ("local", "regional", "longhaul"):
            s = summary[tid]
            legend_lines.append((
                TIER_LEVEL_KEY[s["tier"]],
                f"{s['label']:<26s} {s['diag_km']:>5.0f} km · {s['tier']}",
            ))
        _legend_box(ax, pal, legend_lines, loc="lower left")
        _add_basemap(ax, pal, zoom=5)
        _save(fig, f"tier-classification{_suffix(palette_key)}.png")


def _grid_cell_for(level_size: float, lat: float, lon: float
                   ) -> tuple[float, float]:
    """SW corner of the level-size grid cell that contains (lat, lon)."""
    col = math.floor((lon + 180.0) / level_size)
    row = math.floor((lat + 90.0) / level_size)
    return -90 + row * level_size, -180 + col * level_size


def build_partition_cells() -> None:
    """One trip with the 1°/4°/8° cells its centroid falls into.

    The chosen cell — matching the trip's tier — is filled; the other
    two are outlines only.  Together they show how the same point
    lands in different bucket sizes per tier and that trucktrack picks
    the cell whose size matches the bbox-diagonal tier."""
    df = pl.read_parquet(VALHALLA_DATA / "sample_trips.parquet")
    sub = df.filter(pl.col("id") == "regional").sort("seq")
    lats = sub["lat"].to_list()
    lons = sub["lon"].to_list()
    cen_lat = sum(lats) / len(lats)
    cen_lon = sum(lons) / len(lons)
    diag = _bbox_diag_km(lats, lons)
    chosen_tier = "regional"   # 100 < 366 < 800

    # Nested grid cells.
    cells = {
        "local":    (1.0, _grid_cell_for(1.0, cen_lat, cen_lon)),
        "regional": (4.0, _grid_cell_for(4.0, cen_lat, cen_lon)),
        "longhaul": (8.0, _grid_cell_for(8.0, cen_lat, cen_lon)),
    }

    # View covers the 8° cell with a small pad.
    lh_size, (lh_lat, lh_lon) = cells["longhaul"]
    lat_c = lh_lat + lh_size / 2
    lon_c = lh_lon + lh_size / 2
    half = lh_size / 2 * 1.15

    for palette_key, pal in PALETTES.items():
        fig, ax = _square_fig(pal)
        _focus(ax, lat_c, lon_c, half, half)

        # Draw cells largest → smallest so smaller ones sit on top.
        for tier_name in ("longhaul", "regional", "local"):
            size, (lat0, lon0) = cells[tier_name]
            color = pal[TIER_LEVEL_KEY[tier_name]]
            is_chosen = tier_name == chosen_tier
            _draw_box(
                ax, lat0, lon0, size, color,
                fill=color if is_chosen else None,
                fill_alpha=0.20 if is_chosen else 0.0,
                edge_alpha=1.0,
                lw=3.0 if is_chosen else 2.0,
                zorder=4 + (3 - {"longhaul": 0, "regional": 1, "local": 2}[tier_name]),
            )

        # Trip route (drawn over cells) and centroid.
        ax.plot(lons, lats, color=pal["text"], linewidth=2.4,
                alpha=0.95, zorder=10)
        ax.scatter([cen_lon], [cen_lat], s=120,
                   color=pal["text"], edgecolor="white",
                   linewidths=1.5, zorder=11)

        legend_lines = [
            ("l0", "longhaul — 8°"),
            ("l1", "regional — 4°  ← chosen"),
            ("l2", "local — 1°"),
        ]
        _legend_box(ax, pal, legend_lines, loc="upper left")
        # Tier-and-diagonal annotation in the bottom corner.
        ax.text(
            0.02, 0.02,
            f"trip diagonal = {diag:.0f} km  →  tier = {chosen_tier}",
            transform=ax.transAxes, va="bottom", ha="left",
            fontsize=12, color=pal["text"],
            bbox=dict(boxstyle="round,pad=0.4",
                      facecolor=pal["text_bg"], edgecolor="none", alpha=0.9),
            zorder=20,
        )
        _add_basemap(ax, pal, zoom=7)
        _save(fig, f"partition-cells{_suffix(palette_key)}.png")


# trucktrack's Hilbert grid bounding box (matches src/partition.rs).
HILBERT_LAT_MIN, HILBERT_LAT_MAX = 24.0, 84.0
HILBERT_LON_MIN, HILBERT_LON_MAX = -141.0, -52.0
HILBERT_ORDER = 12


def build_hilbert_locality() -> None:
    """Draw the Hilbert curve at trucktrack's actual order (12) over a
    small viewport so the path is visible.  Each segment is colored by
    its position along the curve — the line winds through neighbors,
    rarely making long jumps, which is the locality-preserving property
    of the curve."""
    import numpy as np
    import trucktrack
    from matplotlib.collections import LineCollection
    from matplotlib import colormaps

    # Anchor a contiguous stretch of the curve over the GTA.  We
    # compute Hilbert indices for a wider pad, then keep only the
    # block of N consecutive curve-cells starting at the index near
    # central Toronto.  This gives one continuous winding line with
    # a clean color gradient end-to-end.
    anchor_lat, anchor_lon = 43.65, -79.55
    n_curve_cells = 256

    n = (1 << HILBERT_ORDER) - 1  # 4095
    lat_range = HILBERT_LAT_MAX - HILBERT_LAT_MIN  # 60.0
    lon_range = HILBERT_LON_MAX - HILBERT_LON_MIN  # 89.0

    def lon_to_x(lon: float) -> int:
        return int((lon - HILBERT_LON_MIN) / lon_range * n)

    def lat_to_y(lat: float) -> int:
        return int((lat - HILBERT_LAT_MIN) / lat_range * n)

    def x_to_lon(x: int) -> float:
        return HILBERT_LON_MIN + (x + 0.5) / n * lon_range

    def y_to_lat(y: int) -> float:
        return HILBERT_LAT_MIN + (y + 0.5) / n * lat_range

    # Compute Hilbert for a wide pad around the anchor so we have
    # enough cells to find a contiguous stretch of the curve.
    pad_cells = 40
    x_anchor = lon_to_x(anchor_lon)
    y_anchor = lat_to_y(anchor_lat)
    x_min = max(0, x_anchor - pad_cells)
    x_max = min(n, x_anchor + pad_cells)
    y_min = max(0, y_anchor - pad_cells)
    y_max = min(n, y_anchor + pad_cells)

    cell_x: list[int] = []
    cell_y: list[int] = []
    cell_lats: list[float] = []
    cell_lons: list[float] = []
    for y in range(y_min, y_max + 1):
        for x in range(x_min, x_max + 1):
            cell_x.append(x)
            cell_y.append(y)
            cell_lons.append(x_to_lon(x))
            cell_lats.append(y_to_lat(y))

    h_idx = trucktrack._core.hilbert_indices(cell_lats, cell_lons)

    order = sorted(range(len(h_idx)), key=lambda i: h_idx[i])
    sorted_x = np.array([cell_x[i] for i in order])
    sorted_y = np.array([cell_y[i] for i in order])
    sorted_lats = np.array([cell_lats[i] for i in order])
    sorted_lons = np.array([cell_lons[i] for i in order])
    sorted_h = np.array([h_idx[i] for i in order], dtype=np.float64)

    # Keep only consecutive pairs that are true Hilbert neighbors
    # (Manhattan distance = 1 in grid coords) — drops jumps where the
    # curve leaves and re-enters the padded area.
    dx = np.abs(sorted_x[1:] - sorted_x[:-1])
    dy = np.abs(sorted_y[1:] - sorted_y[:-1])
    is_neighbor = (dx + dy) == 1

    # Find the longest run of true-neighbor pairs starting near the
    # anchor cell.  Take n_curve_cells from that run.
    anchor_pos_arr = np.where(
        (sorted_x == x_anchor) & (sorted_y == y_anchor)
    )[0]
    anchor_pos = int(anchor_pos_arr[0]) if anchor_pos_arr.size else 0
    start = max(0, anchor_pos - n_curve_cells // 2)
    end = min(len(sorted_x), start + n_curve_cells)
    run_neighbors = is_neighbor[start : end - 1]

    pts = np.stack([sorted_lons, sorted_lats], axis=1)[start:end]
    segs = np.stack([pts[:-1], pts[1:]], axis=1)[run_neighbors]
    seg_h = sorted_h[start : end - 1][run_neighbors]
    h_min, h_max = float(seg_h.min()), float(seg_h.max())
    norm = (seg_h - h_min) / (h_max - h_min)

    # Set viewport to bbox of the kept cells with a small pad.
    vis_lats = sorted_lats[start:end]
    vis_lons = sorted_lons[start:end]
    lat_view_min = float(vis_lats.min())
    lat_view_max = float(vis_lats.max())
    lon_view_min = float(vis_lons.min())
    lon_view_max = float(vis_lons.max())

    cmap = colormaps["turbo"]

    for palette_key, pal in PALETTES.items():
        fig, ax = _square_fig(pal)
        lat_c = (lat_view_min + lat_view_max) / 2
        lon_c = (lon_view_min + lon_view_max) / 2
        dlat = max((lat_view_max - lat_view_min) / 2, 0.001)
        dlon = max((lon_view_max - lon_view_min) / 2, 0.001)
        _focus(ax, lat_c, lon_c, dlat * 1.15, dlon * 1.15)

        lc = LineCollection(
            segs, cmap=cmap, array=norm,
            linewidth=1.6, alpha=0.95, zorder=6,
        )
        ax.add_collection(lc)

        # Colorbar inset.
        grad = np.linspace(0, 1, 256).reshape(1, -1)
        cax = ax.inset_axes((0.04, 0.05, 0.40, 0.025))
        cax.imshow(grad, aspect="auto", cmap=cmap)
        cax.set_xticks([0, 255])
        cax.set_xticklabels(
            [f"{int(h_min):,}", f"{int(h_max):,}"],
            color=pal["text"], fontsize=9,
        )
        cax.set_yticks([])
        for spine in cax.spines.values():
            spine.set_edgecolor(pal["text"])
            spine.set_alpha(0.6)
        cax.tick_params(colors=pal["text"], length=2)
        cax.set_title("hilbert index along this stretch of curve",
                      color=pal["text"], fontsize=10, pad=4, loc="left")

        ax.text(
            0.02, 0.98,
            f"one stretch of the order-{HILBERT_ORDER} hilbert curve",
            transform=ax.transAxes, va="top", ha="left",
            fontsize=12, color=pal["text"],
            bbox=dict(boxstyle="round,pad=0.4",
                      facecolor=pal["text_bg"], edgecolor="none", alpha=0.9),
            zorder=20,
        )
        _add_basemap(ax, pal, zoom=14)
        _save(fig, f"hilbert-locality{_suffix(palette_key)}.png")


# ── bridge orchestration figures ──────────────────────────────────────

# Shared focus for all three bridge figures so they read as a triptych.
# Toronto → Windsor along 401, with ~10 % padding on every side.
BRIDGE_LAT_C = 43.00
BRIDGE_LON_C = -81.20
BRIDGE_DLAT = 0.85
BRIDGE_DLON = 2.10
BRIDGE_FIG_W = 10.0   # inches
BRIDGE_FIG_H = 4.5

# Within-figure colors.  Matched segments share the trip's palette colour
# (palette[0] for the single-trip bridge demos).  Bridges keep a distinct
# red so the slide can teach the reader what a /route bridge looks like —
# the live viz colours bridges the same as the parent trip, but the deck
# trades visual unity for pedagogical contrast.
SEGMENT_COLOR = SEGMENT_PALETTE[0]   # palette[0] — deep pink
BRIDGE_COLOR = "#dc2626"             # bright red — /route bridges (dotted)


def _landscape_fig(pal: dict) -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=(BRIDGE_FIG_W, BRIDGE_FIG_H),
                           facecolor=pal["fig_bg"])
    ax.set_facecolor(pal["fig_bg"])
    ax.set_axis_off()
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    return fig, ax


def _bridge_focus(ax) -> None:
    _focus(ax, BRIDGE_LAT_C, BRIDGE_LON_C, BRIDGE_DLAT, BRIDGE_DLON)


def _draw_shape_polylines(ax, df: pl.DataFrame, color: str,
                          lw: float = 3.2, alpha: float = 0.95,
                          zorder: int = 6,
                          kind_filter: str | None = None) -> None:
    """Plot one polyline per shape_id (skipping connections between)."""
    if kind_filter is not None and "kind" in df.columns:
        df = df.filter(pl.col("kind") == kind_filter)
    for sid in df["shape_id"].unique().to_list():
        sub = df.filter(pl.col("shape_id") == sid)
        ax.plot(sub["lon"].to_list(), sub["lat"].to_list(),
                color=color, linewidth=lw, alpha=alpha, zorder=zorder)


def _draw_pings(ax, df: pl.DataFrame, color: str, alpha: float = 0.55,
                size: float = 6, zorder: int = 4) -> None:
    ax.scatter(df["lon"], df["lat"], s=size, color=color, alpha=alpha,
               edgecolor="none", zorder=zorder)


def _annotate_gap(ax, lat: float, lon: float, label: str, pal: dict,
                  *, dx: float = 0.0, dy: float = 0.18) -> None:
    ax.annotate(
        label,
        (lon, lat),
        xytext=(lon + dx, lat + dy),
        textcoords="data",
        fontsize=9, color=pal["text"],
        ha="center", va="bottom",
        bbox=dict(boxstyle="round,pad=0.25",
                  facecolor=pal["text_bg"], edgecolor="none", alpha=0.9),
        arrowprops=dict(arrowstyle="-", color=pal["text"], alpha=0.6,
                        lw=0.8, shrinkA=0, shrinkB=4),
        zorder=15,
    )


def _bbox_label(ax, pal: dict, text: str, *, loc: str = "upper left") -> None:
    x, y, va, ha = {
        "upper left":  (0.01, 0.97, "top", "left"),
        "upper right": (0.99, 0.97, "top", "right"),
        "lower left":  (0.01, 0.03, "bottom", "left"),
        "lower right": (0.99, 0.03, "bottom", "right"),
    }[loc]
    ax.text(
        x, y, text,
        transform=ax.transAxes, va=va, ha=ha,
        fontsize=12, color=pal["text"],
        bbox=dict(boxstyle="round,pad=0.4",
                  facecolor=pal["text_bg"], edgecolor="none", alpha=0.92),
        zorder=20,
    )


def _gap_endpoints(gappy: pl.DataFrame) -> list[tuple[float, float, float, float, float]]:
    """For each gap in the gappy trace, return (lat0, lon0, lat1, lon1, dist_m)."""
    lats = gappy["lat"].to_numpy()
    lons = gappy["lon"].to_numpy()
    times = gappy["time"].to_numpy()
    out = []
    import numpy as np
    R = 6371000.0
    p = math.pi / 180.0
    for i in range(len(lats) - 1):
        a = ((math.sin((lats[i + 1] - lats[i]) * p / 2)) ** 2
             + math.cos(lats[i] * p) * math.cos(lats[i + 1] * p)
             * math.sin((lons[i + 1] - lons[i]) * p / 2) ** 2)
        d = R * 2 * math.asin(math.sqrt(a))
        dt = (times[i + 1] - times[i]) / np.timedelta64(1, "s")
        if d > 5000 or (d > 1000 and dt > 240):
            out.append((float(lats[i]), float(lons[i]),
                        float(lats[i + 1]), float(lons[i + 1]),
                        float(d)))
    return out


def build_bridge_clean() -> None:
    raw = pl.read_parquet(VALHALLA_DATA / "bridge_clean_trace.parquet")
    matched = pl.read_parquet(VALHALLA_DATA / "bridge_clean_matched.parquet")
    for palette_key, pal in PALETTES.items():
        fig, ax = _landscape_fig(pal)
        _bridge_focus(ax)
        _draw_shape_polylines(ax, matched, SEGMENT_COLOR)
        _draw_pings(ax, raw, color=pal["text"])
        _bbox_label(
            ax, pal,
            f"clean trip · {raw.height} pings · "
            f"{matched['shape_id'].n_unique()} matched shape",
        )
        _add_basemap(ax, pal, zoom=9)
        _save(fig, f"bridge-clean{_suffix(palette_key)}.png")


def build_bridge_no_bridge() -> None:
    raw = pl.read_parquet(VALHALLA_DATA / "bridge_gappy_trace.parquet")
    matched = pl.read_parquet(VALHALLA_DATA / "bridge_no_bridge_matched.parquet")
    gaps = _gap_endpoints(raw.sort("time"))
    for palette_key, pal in PALETTES.items():
        fig, ax = _landscape_fig(pal)
        _bridge_focus(ax)
        _draw_shape_polylines(ax, matched, SEGMENT_COLOR)
        _draw_pings(ax, raw, color=pal["text"])
        for lat0, lon0, lat1, lon1, dist in gaps:
            mid_lat = (lat0 + lat1) / 2
            mid_lon = (lon0 + lon1) / 2
            _annotate_gap(ax, mid_lat, mid_lon,
                          f"{dist / 1000:.0f} km gap", pal)
        _bbox_label(
            ax, pal,
            f"gaps · breakage_distance = 3000 m · no bridge\n"
            f"→ {matched['shape_id'].n_unique()} disjoint matched shapes",
        )
        _add_basemap(ax, pal, zoom=9)
        _save(fig, f"bridge-no-bridge{_suffix(palette_key)}.png")


def build_bridge_with_bridge() -> None:
    raw = pl.read_parquet(VALHALLA_DATA / "bridge_gappy_trace.parquet")
    matched = pl.read_parquet(VALHALLA_DATA / "bridge_with_bridge_matched.parquet")
    n_segments = matched.filter(pl.col("kind") == "segment")["shape_id"].n_unique()
    n_bridges = matched.filter(pl.col("kind") == "bridge")["shape_id"].n_unique()
    for palette_key, pal in PALETTES.items():
        fig, ax = _landscape_fig(pal)
        _bridge_focus(ax)
        # Sub-segments first (blue), then bridges on top (red, dotted).
        _draw_shape_polylines(ax, matched, SEGMENT_COLOR,
                              kind_filter="segment")
        for sid in matched.filter(pl.col("kind") == "bridge")["shape_id"].unique().to_list():
            sub = matched.filter((pl.col("shape_id") == sid)
                                 & (pl.col("kind") == "bridge"))
            ax.plot(sub["lon"].to_list(), sub["lat"].to_list(),
                    color=BRIDGE_COLOR, linewidth=2.6, alpha=0.95,
                    linestyle=(0, (1, 2.2)), zorder=6)
        _draw_pings(ax, raw, color=pal["text"])
        _bbox_label(
            ax, pal,
            f"gaps · bridge orchestrator on\n"
            f"→ {n_segments} matched segments + {n_bridges} routed bridges (red, dotted)",
        )
        _add_basemap(ax, pal, zoom=9)
        _save(fig, f"bridge-with-bridge{_suffix(palette_key)}.png")


def main() -> None:
    print(f"building figures → {FIGDIR.relative_to(SLIDES)}")
    build_hierarchy_grid()
    build_tile_zoom()
    build_trip_footprint()
    build_tier_classification()
    build_partition_cells()
    build_hilbert_locality()
    build_bridge_clean()
    build_bridge_no_bridge()
    build_bridge_with_bridge()
    print("done.")


if __name__ == "__main__":
    main()
