"""A highway corridor: truck routes through named points, and the edges along them.

Each direction is its own route, which fixes a distance along the
corridor (chainage) in that direction.  Any motorway edge that runs
along a route, within *max_offset_m* of it, belongs to the corridor in
that direction, so parallel carriageways (express and collector lanes)
and the ramps between them count as the highway.
:mod:`trucktrack.congestion.reliability` times trucks between points
on it.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl

from trucktrack.valhalla._actor import DEFAULT_TRUCK_COSTING, get_actor
from trucktrack.valhalla._json import dumps as _json_dumps
from trucktrack.valhalla._json import loads as _json_loads
from trucktrack.valhalla._parsing import decode_polyline6
from trucktrack.valhalla.shapes import edge_shapes

_EARTH_M = 6_371_000.0
# FHWA reporting segments (23 CFR 490.103(e)): at most 1 mile in
# urbanized areas and 10 miles elsewhere.
URBAN_SEGMENT_M = 1609.344
RURAL_SEGMENT_M = 16093.44
# An edge runs along a route if its chainage grows by at least this
# share of its length.
_ALIGNED = 0.8
# Urban areas are found along a route at this spacing.
_URBAN_STEP_M = 100.0
# Express and collector lanes split up to ~200 m apart at interchanges.
MAX_OFFSET_M = 300.0
# A highway corridor; roads beside it that aren't one don't count.
ROAD_CLASSES = ("motorway",)
# A point is matched to edges this close that head within the tolerance
# of the way to the next point: the carriageway going that way.
_SNAP_M = 100
_HEADING_TOLERANCE = 90

SPAN_SCHEMA = {
    "direction": pl.String,
    "kind": pl.String,
    "name": pl.String,
    "begin_m": pl.Float64,
    "end_m": pl.Float64,
}
EDGE_SCHEMA = {
    "edge_id": pl.UInt64,
    "direction": pl.String,
    "begin_m": pl.Float64,
    "end_m": pl.Float64,
}


def _bearing(a: tuple[float, float], b: tuple[float, float]) -> float:
    """Compass bearing from (lat, lon) *a* to *b*, degrees clockwise from north."""
    la, lb = math.radians(a[0]), math.radians(b[0])
    dlon = math.radians(b[1] - a[1])
    x = math.sin(dlon) * math.cos(lb)
    y = math.cos(la) * math.sin(lb) - math.sin(la) * math.cos(lb) * math.cos(dlon)
    return math.degrees(math.atan2(x, y)) % 360.0


def _haversine_m(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """Distances between consecutive points (degrees), in metres."""
    la, lo = np.radians(lat), np.radians(lon)
    a = (
        np.sin(np.diff(la) / 2) ** 2
        + np.cos(la[:-1]) * np.cos(la[1:]) * np.sin(np.diff(lo) / 2) ** 2
    )
    out: np.ndarray = 2 * _EARTH_M * np.arcsin(np.sqrt(a))
    return out


@dataclass(frozen=True, eq=False)
class Route:
    """One direction of a corridor: the truck route through its named points.

    ``lat``, ``lon`` are the route's road geometry and ``chainage`` the
    metres along it at each of those points; ``stops`` is the chainage
    of each of ``names``, in driving order.
    """

    names: tuple[str, ...]
    lat: np.ndarray
    lon: np.ndarray
    chainage: np.ndarray
    stops: np.ndarray

    @property
    def direction(self) -> str:
        return f"{self.names[0]} to {self.names[-1]}"

    def spans(
        self,
        urban: Sequence[tuple[float, float, float]] = (),
        urban_segment_m: float = URBAN_SEGMENT_M,
        rural_segment_m: float = RURAL_SEGMENT_M,
    ) -> pl.DataFrame:
        """This direction's rows of :meth:`Corridor.spans`."""
        rows: list[tuple[str, str, float, float]] = [
            ("corridor", self.direction, float(self.stops[0]), float(self.stops[-1]))
        ]
        for i in range(len(self.names) - 1):
            a, b = float(self.stops[i]), float(self.stops[i + 1])
            section = f"{self.names[i]} to {self.names[i + 1]}"
            rows.append(("section", section, a, b))
            pieces = self._segments(a, b, urban, urban_segment_m, rural_segment_m)
            for k, (s, e) in enumerate(pieces, 1):
                rows.append(("segment", f"{section} {k}/{len(pieces)}", s, e))
        return pl.DataFrame(
            [(self.direction, *row) for row in rows], schema=SPAN_SCHEMA, orient="row"
        )

    def _segments(
        self,
        a: float,
        b: float,
        urban: Sequence[tuple[float, float, float]],
        urban_m: float,
        rural_m: float,
    ) -> list[tuple[float, float]]:
        """[a, b] cut where it enters or leaves an urban circle, then evenly."""
        steps = np.linspace(a, b, max(2, math.ceil((b - a) / _URBAN_STEP_M) + 1))
        lat = np.interp(steps, self.chainage, self.lat)
        lon = np.interp(steps, self.chainage, self.lon)
        inside = np.zeros(len(steps), dtype=bool)
        for c_lat, c_lon, radius in urban:
            north = np.radians(lat - c_lat) * _EARTH_M
            east = np.radians(lon - c_lon) * _EARTH_M * math.cos(math.radians(c_lat))
            inside |= np.hypot(north, east) <= radius
        change = np.flatnonzero(np.diff(inside.astype(int))) + 1
        cuts = [a, *steps[change].tolist(), b]
        out: list[tuple[float, float]] = []
        for s, e, is_urban in zip(
            cuts[:-1], cuts[1:], [inside[0], *inside[change]], strict=True
        ):
            cap = urban_m if is_urban else rural_m
            ends = np.linspace(s, e, max(1, math.ceil((e - s) / cap - 1e-9)) + 1)
            out += list(zip(ends[:-1].tolist(), ends[1:].tolist(), strict=True))
        return out

    def locate(
        self,
        shapes: pl.DataFrame,
        max_offset_m: float = MAX_OFFSET_M,
        road_classes: Sequence[str] = ROAD_CLASSES,
    ) -> pl.DataFrame:
        """This direction's rows of :func:`locate_edges`."""
        points = (
            shapes.filter(pl.col("road_class").cast(pl.String).is_in(road_classes))
            .select("edge_id", "shape")
            .explode("shape")
            .drop_nulls("shape")
        )
        if points.height == 0:
            return pl.DataFrame(schema=EDGE_SCHEMA)
        lat = points["shape"].arr.get(0).to_numpy()
        lon = points["shape"].arr.get(1).to_numpy()
        chainage, offset = _project(self, lat, lon, max(200.0, max_offset_m))
        # Distance from the point before; an edge's first point has none.
        step = np.concatenate([[0.0], _haversine_m(lat, lon)])
        per_edge = (
            points.select("edge_id")
            .with_columns(ch=chainage, offset=offset, step=step)
            .group_by("edge_id", maintain_order=True)
            .agg(
                begin_m=pl.col("ch").first(),
                end_m=pl.col("ch").last(),
                offset=pl.col("offset").max(),
                length=pl.col("step").slice(1).sum(),
            )
        )
        return per_edge.filter(
            pl.col("offset") <= max_offset_m,
            pl.col("length") > 0,
            pl.col("end_m") - pl.col("begin_m") >= _ALIGNED * pl.col("length"),
        ).select(
            pl.col("edge_id").cast(pl.UInt64),
            direction=pl.lit(self.direction),
            begin_m="begin_m",
            end_m="end_m",
        )


@dataclass(frozen=True, eq=False)
class Corridor:
    """A highway between named points, both ways: a :class:`Route` each."""

    routes: tuple[Route, ...]

    @property
    def directions(self) -> tuple[str, ...]:
        """``"<first> to <last>"`` and back."""
        return tuple(r.direction for r in self.routes)

    def spans(
        self,
        urban: Sequence[tuple[float, float, float]] = (),
        urban_segment_m: float = URBAN_SEGMENT_M,
        rural_segment_m: float = RURAL_SEGMENT_M,
    ) -> pl.DataFrame:
        """The stretches to time, in each direction, by ``kind``.

        ``"corridor"``: first point to last.  ``"section"``: each named
        point to the next.  ``"segment"``: FHWA reporting segments, each
        section cut into equal pieces of at most *urban_segment_m*
        inside the *urban* circles ((lat, lon, radius in m)) and
        *rural_segment_m* outside them.  ``begin_m`` and ``end_m`` are
        chainage along the direction's route.
        """
        return pl.concat(
            r.spans(urban, urban_segment_m, rural_segment_m) for r in self.routes
        ).sort("direction", "kind", "begin_m")

    def edges(
        self,
        edge_ids: Iterable[int],
        config: str | Path | None = None,
        max_offset_m: float = MAX_OFFSET_M,
        road_classes: Sequence[str] = ROAD_CLASSES,
    ) -> pl.DataFrame:
        """Which of *edge_ids* (in *config*'s tiles) belong to the corridor.

        See :func:`locate_edges`.
        """
        shapes = edge_shapes(edge_ids, config)
        return locate_edges(self, shapes, max_offset_m, road_classes)


def build_corridor(
    points: Mapping[str, tuple[float, float]],
    config: str | Path | None = None,
    *,
    before: tuple[float, float] | None = None,
    after: tuple[float, float] | None = None,
) -> Corridor:
    """Truck routes through *points* ({name: (lat, lon)}, in driving order) and back.

    Put each point on the highway itself: the route passes through it
    without stopping, and it marks where one section ends and the next
    begins.  Check the routes stay on the highway between points (a
    point off it pulls them off too).

    A truck is timed across a point only if it is seen on the corridor
    on both sides of it, so the routes should run on past the first and
    last points: between *before* and *after*, (lat, lon) further along
    the highway.  Put the first and last points far enough from where
    trucks join the highway that they report a fix between (a minute's
    driving).
    """
    if len(points) < 2:
        raise ValueError("a corridor needs at least two points")
    forward = _route(tuple(points.items()), before, after, config)
    back = _route(tuple(points.items())[::-1], after, before, config)
    return Corridor((forward, back))


def _route(
    points: tuple[tuple[str, tuple[float, float]], ...],
    before: tuple[float, float] | None,
    after: tuple[float, float] | None,
    config: str | Path | None,
) -> Route:
    stops_at = [p for _, p in points]
    route = [*([before] if before else []), *stops_at, *([after] if after else [])]
    locations = [
        {
            "lat": lat,
            "lon": lon,
            "type": "break" if i in (0, len(route) - 1) else "break_through",
            # Snap to the highway, not a ramp, on the carriageway heading
            # roughly from the point before to the point after.
            "search_filter": {"exclude_ramp": True},
            "radius": _SNAP_M,
            "heading": round(
                _bearing(route[max(i - 1, 0)], route[min(i + 1, len(route) - 1)])
            ),
            "heading_tolerance": _HEADING_TOLERANCE,
        }
        for i, (lat, lon) in enumerate(route)
    ]
    body = {
        "locations": locations,
        "costing": "truck",
        "costing_options": {"truck": DEFAULT_TRUCK_COSTING},
        "directions_type": "none",
    }
    legs = _json_loads(get_actor(config).route(_json_dumps(body)))["trip"]["legs"]
    lat: list[float] = []
    lon: list[float] = []
    ends = [0]
    for leg in legs:
        shape = decode_polyline6(leg["shape"])
        if lat:
            shape = shape[1:]  # each leg starts where the last ended
        lat += [p[0] for p in shape]
        lon += [p[1] for p in shape]
        ends.append(len(lat) - 1)
    lat_a, lon_a = np.array(lat), np.array(lon)
    chainage = np.concatenate([[0.0], np.cumsum(_haversine_m(lat_a, lon_a))])
    stops = ends[1 if before else 0 : len(ends) - 1 if after else len(ends)]
    names = tuple(name for name, _ in points)
    return Route(names, lat_a, lon_a, chainage, chainage[stops])


def locate_edges(
    corridor: Corridor,
    shapes: pl.DataFrame,
    max_offset_m: float = MAX_OFFSET_M,
    road_classes: Sequence[str] = ROAD_CLASSES,
) -> pl.DataFrame:
    """The edges in *shapes* that run along *corridor*, with their chainage.

    *shapes* are :func:`~trucktrack.valhalla.edge_shapes` rows.  An edge
    belongs to a direction if it is one of *road_classes* (ramps take
    their highway's class), every point of it lies within
    *max_offset_m* of that direction's route, and it runs the route's
    way.  Returns ``edge_id``, ``direction``, and ``begin_m`` and
    ``end_m``: the chainage of its start and end.
    """
    return pl.concat(
        r.locate(shapes, max_offset_m, road_classes) for r in corridor.routes
    ).sort("direction", "begin_m")


def _project(
    route: Route, lat: np.ndarray, lon: np.ndarray, cell_m: float
) -> tuple[np.ndarray, np.ndarray]:
    """Chainage of the nearest point of *route* to each (lat, lon), and how far.

    Points more than *cell_m* from the route get offset inf.  The route
    is bucketed into *cell_m* squares, so each point is compared with
    the few route segments near it.
    """
    lat0 = math.radians(float(np.mean(route.lat)))

    def xy(la: np.ndarray, lo: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return np.radians(lo) * _EARTH_M * math.cos(lat0), np.radians(la) * _EARTH_M

    rx, ry = xy(route.lat, route.lon)
    qx, qy = xy(lat, lon)
    ax, ay, bx, by = rx[:-1], ry[:-1], rx[1:], ry[1:]
    # Every cell within cell_m of each route segment's bounding box.
    lo_x = np.floor((np.minimum(ax, bx) - cell_m) / cell_m).astype(np.int64)
    hi_x = np.floor((np.maximum(ax, bx) + cell_m) / cell_m).astype(np.int64)
    lo_y = np.floor((np.minimum(ay, by) - cell_m) / cell_m).astype(np.int64)
    hi_y = np.floor((np.maximum(ay, by) + cell_m) / cell_m).astype(np.int64)
    seg_ids, cx, cy = [], [], []
    for i in range(len(ax)):
        gx, gy = np.meshgrid(
            np.arange(lo_x[i], hi_x[i] + 1), np.arange(lo_y[i], hi_y[i] + 1)
        )
        seg_ids.append(np.full(gx.size, i))
        cx.append(gx.ravel())
        cy.append(gy.ravel())
    cells = pl.DataFrame(
        {
            "seg": np.concatenate(seg_ids),
            "cx": np.concatenate(cx),
            "cy": np.concatenate(cy),
        }
    )
    queries = pl.DataFrame(
        {
            "q": np.arange(len(qx)),
            "cx": np.floor(qx / cell_m).astype(np.int64),
            "cy": np.floor(qy / cell_m).astype(np.int64),
        }
    )
    pairs = queries.join(cells, on=["cx", "cy"])
    q, s = pairs["q"].to_numpy(), pairs["seg"].to_numpy()
    dx, dy = bx[s] - ax[s], by[s] - ay[s]
    seg_len2 = dx * dx + dy * dy
    t = np.where(
        seg_len2 > 0,
        ((qx[q] - ax[s]) * dx + (qy[q] - ay[s]) * dy)
        / np.where(seg_len2 > 0, seg_len2, 1.0),
        0.0,
    ).clip(0.0, 1.0)
    dist = np.hypot(qx[q] - (ax[s] + t * dx), qy[q] - (ay[s] + t * dy))
    ch = route.chainage[s] + t * np.diff(route.chainage)[s]
    best = (
        pl.DataFrame({"q": q, "dist": dist, "ch": ch})
        .sort("dist")
        .group_by("q")
        .first()
    )
    chainage = np.full(len(qx), np.nan)
    offset = np.full(len(qx), np.inf)
    idx = best["q"].to_numpy()
    chainage[idx] = best["ch"].to_numpy()
    offset[idx] = best["dist"].to_numpy()
    return chainage, offset
