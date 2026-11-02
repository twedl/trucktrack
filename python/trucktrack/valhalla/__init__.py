"""Valhalla routing and map-matching via local pyvalhalla bindings."""

from trucktrack.valhalla._actor import DEFAULT_TRUCK_COSTING, find_config, get_actor
from trucktrack.valhalla._bridge import (
    BridgeConfig,
    BridgedMatchResult,
    BridgeFailure,
    BridgeFit,
    bridge_gap,
    map_match_dataframe_with_bridges,
)
from trucktrack.valhalla._build_id import tile_build_id, write_tile_build_id
from trucktrack.valhalla._parsing import decode_polyline6, parse_valhalla_response
from trucktrack.valhalla.corridor import Corridor, Route, build_corridor, locate_edges
from trucktrack.valhalla.map_matching import (
    MatchedPoint,
    map_match,
    map_match_dataframe,
    map_match_dataframe_full,
    map_match_full,
    map_match_route_shape,
    map_match_ways,
    parse_fix_matches,
    trace_attributes,
)
from trucktrack.valhalla.pipeline import map_match_trip, run_map_matching
from trucktrack.valhalla.quality import evaluate_map_match_with_bridges
from trucktrack.valhalla.routing import route
from trucktrack.valhalla.shapes import edge_shapes, with_geometry

__all__ = [
    "DEFAULT_TRUCK_COSTING",
    "BridgeConfig",
    "BridgeFailure",
    "BridgeFit",
    "BridgedMatchResult",
    "Corridor",
    "MatchedPoint",
    "Route",
    "bridge_gap",
    "build_corridor",
    "decode_polyline6",
    "edge_shapes",
    "evaluate_map_match_with_bridges",
    "find_config",
    "get_actor",
    "locate_edges",
    "map_match",
    "map_match_dataframe",
    "map_match_dataframe_full",
    "map_match_dataframe_with_bridges",
    "map_match_full",
    "map_match_route_shape",
    "map_match_trip",
    "map_match_ways",
    "parse_fix_matches",
    "parse_valhalla_response",
    "route",
    "run_map_matching",
    "tile_build_id",
    "trace_attributes",
    "with_geometry",
    "write_tile_build_id",
]
