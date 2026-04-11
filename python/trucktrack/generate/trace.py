"""Orchestrator: chain parking, driving, and noise phases into a complete trace."""

from __future__ import annotations

import csv
import random
from datetime import datetime
from io import BytesIO, StringIO

from trucktrack.generate.gps_errors import GPS_ERRORS
from trucktrack.generate.interpolator import bearing, interpolate_route, resample_trace
from trucktrack.generate.models import ErrorConfig, RouteSegment, TracePoint, TripConfig
from trucktrack.generate.noise import apply_noise
from trucktrack.generate.operational_errors import OPERATIONAL_ERRORS
from trucktrack.generate.parking import (
    ManeuverType,
    generate_arrival_maneuver,
    generate_departure_maneuver,
)
from trucktrack.generate.router import fetch_route


def generate_trace(config: TripConfig) -> list[TracePoint]:
    """Generate a complete GPS trace for a truck delivery trip."""
    rng = random.Random(config.seed)

    route = fetch_route(
        config.origin, config.destination, config.valhalla_url, config.tile_extract
    )

    origin_heading = _route_start_heading(route)
    dest_heading = _route_end_heading(route)

    origin_maneuver = ManeuverType(config.origin_maneuver)
    dest_maneuver = ManeuverType(config.destination_maneuver)

    all_points: list[TracePoint] = []

    departure_pts = generate_departure_maneuver(
        dock_lat=config.origin[0],
        dock_lon=config.origin[1],
        dock_heading=origin_heading,
        start_time=config.departure_time,
        rng=rng,
        maneuver_type=origin_maneuver,
    )
    departure_pts = resample_trace(departure_pts)
    all_points.extend(departure_pts)

    driving_start = (
        departure_pts[-1].timestamp if departure_pts else config.departure_time
    )

    driving_pts = interpolate_route(route, driving_start, rng)
    all_points.extend(driving_pts)

    arrival_start = driving_pts[-1].timestamp if driving_pts else driving_start

    arrival_pts = generate_arrival_maneuver(
        dock_lat=config.destination[0],
        dock_lon=config.destination[1],
        dock_heading=dest_heading,
        start_time=arrival_start,
        rng=rng,
        maneuver_type=dest_maneuver,
    )
    arrival_pts = resample_trace(arrival_pts)
    all_points.extend(arrival_pts)

    if config.errors:
        all_points = _apply_errors(all_points, config.errors, rng)

    return apply_noise(all_points, config.gps_noise_meters, rng)


def _apply_errors(
    points: list[TracePoint],
    errors: list[ErrorConfig],
    rng: random.Random,
) -> list[TracePoint]:
    """Apply error injectors: operational patterns first, then GPS errors."""
    operational: list[ErrorConfig] = []
    gps: list[ErrorConfig] = []
    unknown: list[str] = []
    for e in errors:
        if e.error_type in OPERATIONAL_ERRORS:
            operational.append(e)
        elif e.error_type in GPS_ERRORS:
            gps.append(e)
        else:
            unknown.append(e.error_type)
    if unknown:
        raise ValueError(f"Unknown error types: {unknown}")

    for spec in operational + gps:
        if rng.random() < spec.probability:
            fn = OPERATIONAL_ERRORS.get(spec.error_type) or GPS_ERRORS[spec.error_type]
            points = fn(points, rng, **spec.params)
    return points


def _csv_rows(points: list[TracePoint], trip_id: str) -> list[list[str]]:
    return [
        [
            trip_id,
            f"{pt.lat:.6f}",
            f"{pt.lon:.6f}",
            f"{pt.speed_mph:.1f}",
            f"{pt.heading:.1f}",
            pt.timestamp.strftime("%Y-%m-%dT%H:%M:%SZ"),
        ]
        for pt in points
    ]


def traces_to_csv(
    trips: list[tuple[list[TracePoint], str]], output_path: str | None = None
) -> str:
    """Write multiple trips to a single CSV. Each trip is a (points, trip_id) tuple."""
    buf = StringIO()
    writer = csv.writer(buf)
    writer.writerow(["id", "lat", "lon", "speed", "heading", "time"])
    for points, trip_id in trips:
        writer.writerows(_csv_rows(points, trip_id))
    csv_str = buf.getvalue()

    if output_path:
        with open(output_path, "w") as f:
            f.write(csv_str)

    return csv_str


def traces_to_parquet(
    trips: list[tuple[list[TracePoint], str]], output_path: str | None = None
) -> bytes:
    """Write multiple trips to a single Parquet file. Returns the bytes."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    ids: list[str] = []
    lats: list[float] = []
    lons: list[float] = []
    speeds: list[float] = []
    headings: list[float] = []
    timestamps: list[datetime] = []
    for points, trip_id in trips:
        for pt in points:
            ids.append(trip_id)
            lats.append(pt.lat)
            lons.append(pt.lon)
            speeds.append(round(pt.speed_mph, 1))
            headings.append(round(pt.heading, 1))
            timestamps.append(pt.timestamp)

    # Column name `time` matches the schema used elsewhere in trucktrack
    # (see splitters.py defaults), so generated parquets feed straight into
    # split-stops / split-gap without requiring --time-col overrides.
    table = pa.table(
        {
            "id": ids,
            "lat": lats,
            "lon": lons,
            "speed": speeds,
            "heading": headings,
            "time": timestamps,
        }
    )

    buf = BytesIO()
    pq.write_table(table, buf)
    data = buf.getvalue()

    if output_path:
        with open(output_path, "wb") as f:
            f.write(data)

    return data


def _route_start_heading(route: RouteSegment) -> float:
    if len(route.coords) < 2:
        return 0.0
    return bearing(
        route.coords[0][0],
        route.coords[0][1],
        route.coords[1][0],
        route.coords[1][1],
    )


def _route_end_heading(route: RouteSegment) -> float:
    if len(route.coords) < 2:
        return 0.0
    return bearing(
        route.coords[-2][0],
        route.coords[-2][1],
        route.coords[-1][0],
        route.coords[-1][1],
    )
