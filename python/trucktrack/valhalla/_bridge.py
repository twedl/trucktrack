"""Gap-split + /route bridge orchestration for slow map-match calls.

When a trip contains internal gaps that are large in distance or in
time (with a minimum distance floor), the Meili HMM spends most of its
time searching candidate routes between the pre- and post-gap points.
This module splits the trace at those gaps, map-matches each
sub-segment with a normal ``trace_attributes`` call, bridges the gap
via a single ``/route`` call plus ``trace_attributes`` in ``edge_walk``
mode (to recover way IDs along the bridge), and stitches the results
into a continuous way-ID sequence.

The ``/route`` bridge assumes the truck took a shortest/fastest path
through the gap.  True detours will be invisible; the per-bridge
``BridgeFit`` reports detour length and ratio so downstream can filter.

On any bridge failure the orchestrator falls back to a single full-HMM
call with ``breakage_distance`` pinned to the base (non-adaptive)
value, letting Meili itself break the trace at the gap.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import polars as pl

from trucktrack.generate.interpolator import haversine_m
from trucktrack.valhalla._actor import get_actor
from trucktrack.valhalla._json import dumps as _json_dumps
from trucktrack.valhalla._json import loads as _json_loads
from trucktrack.valhalla._parsing import concat_leg_shapes
from trucktrack.valhalla.map_matching import (
    _BASE_BREAKAGE_DISTANCE,
    _parse_way_ids,
    map_match_dataframe_full,
    map_match_ways,
)


@dataclass(frozen=True)
class BridgeConfig:
    """Gap-detection thresholds for the bridging orchestrator.

    A split is triggered when the inter-point distance exceeds
    *max_dist_m* OR when the inter-point time exceeds *time_s* AND
    the inter-point distance exceeds *min_dist_m*.  The ``min_dist_m``
    floor on the time rule prevents red-light stalls from triggering.
    """

    max_dist_m: float = 5000.0
    time_s: float = 240.0
    min_dist_m: float = 1000.0


_DEFAULT_BRIDGES = BridgeConfig()


@dataclass(frozen=True)
class BridgeFit:
    """Per-bridge quality signal for a routed gap segment."""

    straight_m: float  # haversine from a to b
    route_m: float  # /route length
    detour_ratio: float  # route_m / max(straight_m, 1.0)
    gap_seconds: float


class BridgeFailure(RuntimeError):
    """Raised when ``/route`` or ``edge_walk`` fails to bridge a gap."""


@dataclass
class BridgedMatchResult:
    """Return value of :func:`map_match_dataframe_with_bridges`."""

    matched_df: pl.DataFrame
    way_ids: list[int]
    shapes: list[list[tuple[float, float]]]
    fits: list[BridgeFit] = field(default_factory=list)
    fallback_used: bool = False


def _find_gap_indices(
    df: pl.DataFrame,
    *,
    bridges: BridgeConfig,
    lat_col: str = "lat",
    lon_col: str = "lon",
    time_col: str = "time",
) -> list[int]:
    """Indices i where the gap between row i and row i+1 should split the trace.

    See :class:`BridgeConfig` for the split rule.  Vectorized via numpy.
    """
    n = len(df)
    if n < 2:
        return []
    lat = np.radians(df[lat_col].to_numpy())
    lon = np.radians(df[lon_col].to_numpy())
    t = df[time_col].to_numpy()
    dlat = np.diff(lat)
    dlon = np.diff(lon)
    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat[:-1]) * np.cos(lat[1:]) * np.sin(dlon / 2) ** 2
    )
    dist_m = 2 * 6_371_000.0 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))
    dt_s = (t[1:] - t[:-1]) / np.timedelta64(1, "s")
    is_gap = (dist_m > bridges.max_dist_m) | (
        (dt_s > bridges.time_s) & (dist_m > bridges.min_dist_m)
    )
    return [int(i) for i in np.where(is_gap)[0]]


def _extend_dedup(acc: list[int], new: list[int]) -> None:
    """Concat *new* onto *acc*, dropping new[0] if it duplicates acc[-1]."""
    if not new:
        return
    if acc and acc[-1] == new[0]:
        new = new[1:]
    acc.extend(new)


def _split_at(df: pl.DataFrame, gap_idxs: list[int]) -> list[pl.DataFrame]:
    """Split *df* into sub-DataFrames at each gap index.

    gap_idxs[k] is the last row of the k-th sub-segment.
    """
    boundaries = [0, *(i + 1 for i in gap_idxs), len(df)]
    return [
        df.slice(boundaries[k], boundaries[k + 1] - boundaries[k])
        for k in range(len(boundaries) - 1)
    ]


def _with_costing(
    body: dict[str, object],
    costing: str,
    costing_options: dict[str, object] | None,
) -> dict[str, object]:
    body["costing"] = costing
    if costing_options is not None:
        body["costing_options"] = {costing: costing_options}
    return body


def bridge_gap(
    a: tuple[float, float],
    b: tuple[float, float],
    *,
    gap_seconds: float,
    costing: str,
    costing_options: dict[str, object] | None,
    config: str | Path | None,
) -> tuple[list[tuple[float, float]], list[int], BridgeFit]:
    """Bridge a gap from *a* to *b* via ``/route`` + ``edge_walk``.

    Returns the routed shape, the way-ID sequence along it, and a
    :class:`BridgeFit` with straight-line and route distances.

    Raises :exc:`BridgeFailure` when ``/route`` returns no path or when
    ``edge_walk`` fails to walk the returned shape.
    """
    actor = get_actor(config=config)

    route_body = _with_costing(
        {
            "locations": [
                {"lat": a[0], "lon": a[1]},
                {"lat": b[0], "lon": b[1]},
            ],
            "units": "km",
        },
        costing,
        costing_options,
    )
    try:
        route_resp = _json_loads(actor.route(_json_dumps(route_body)))
    except Exception as exc:
        raise BridgeFailure(f"route {a} -> {b}: {type(exc).__name__}: {exc}") from exc

    shape = concat_leg_shapes(route_resp.get("trip", {}).get("legs", []))
    if not shape:
        raise BridgeFailure(f"route {a} -> {b}: empty shape")
    route_m = float(route_resp["trip"]["summary"].get("length", 0.0)) * 1000.0

    walk_body = _with_costing(
        {
            "shape": [{"lat": lat, "lon": lon} for lat, lon in shape],
            "shape_match": "edge_walk",
            "filters": {"attributes": ["edge.way_id"], "action": "include"},
        },
        costing,
        costing_options,
    )
    try:
        walk_resp = _json_loads(actor.trace_attributes(_json_dumps(walk_body)))
    except Exception as exc:
        raise BridgeFailure(
            f"edge_walk {a} -> {b}: {type(exc).__name__}: {exc}"
        ) from exc

    way_ids = _parse_way_ids(walk_resp)
    straight_m = haversine_m(a[0], a[1], b[0], b[1])
    fit = BridgeFit(
        straight_m=straight_m,
        route_m=route_m,
        detour_ratio=route_m / max(straight_m, 1.0),
        gap_seconds=gap_seconds,
    )
    return shape, way_ids, fit


def _null_matched_frame(seg: pl.DataFrame) -> pl.DataFrame:
    """Copy *seg* with null matched_lat / matched_lon / distance_from_trace.

    Used for sub-segments too short to map-match (len < 2); preserves
    row count so the concatenated matched_df has one row per input.
    """
    n = len(seg)
    return seg.with_columns(
        pl.Series("matched_lat", [None] * n, dtype=pl.Float64),
        pl.Series("matched_lon", [None] * n, dtype=pl.Float64),
        pl.Series("distance_from_trace", [None] * n, dtype=pl.Float64),
    )


def map_match_dataframe_with_bridges(
    df: pl.DataFrame,
    *,
    bridges: BridgeConfig = _DEFAULT_BRIDGES,
    lat_col: str = "lat",
    lon_col: str = "lon",
    time_col: str = "time",
    costing: str = "auto",
    costing_options: dict[str, object] | None = None,
    config: str | Path | None = None,
    trace_options: dict[str, object] | None = None,
    collect_shapes: bool = False,
) -> BridgedMatchResult:
    """Map-match a trip, bridging large time/distance gaps with ``/route``.

    The matcher runs on each sub-segment separately; gaps are bridged
    by a ``/route`` call plus an ``edge_walk`` to recover way IDs.
    Way-ID sequences are concatenated with seam-dedup.

    When *collect_shapes* is false (the default), each sub-segment uses
    a single ``trace_attributes`` call (``map_match_ways``) — one
    Valhalla call per segment plus two per bridge.  ``matched_df`` and
    ``shapes`` come back empty.

    When *collect_shapes* is true, each sub-segment uses
    ``map_match_dataframe_full`` (``trace_attributes`` + ``trace_route``)
    so ``matched_df`` carries snapped coordinates and ``shapes`` carries
    road geometry — useful for visualization but doubles the Valhalla
    call count per segment.

    On any bridging failure (``/route`` returns no path, ``edge_walk``
    fails), the function falls back to a single full-HMM call with
    ``breakage_distance`` pinned to the base value — Meili breaks the
    trace at the gap rather than searching across it, and
    ``fallback_used=True`` is set on the result.

    The adaptive breakage-distance cap is tied to ``bridges.max_dist_m``
    so the matcher never searches further than the splitter would allow.
    """
    df = df.sort(time_col)

    def _segment_points(seg: pl.DataFrame) -> list[tuple[float, float]]:
        return list(zip(seg[lat_col].to_list(), seg[lon_col].to_list(), strict=True))

    def _match_full(
        frame: pl.DataFrame, opts: dict[str, object] | None
    ) -> tuple[pl.DataFrame, list[int], list[list[tuple[float, float]]]]:
        return map_match_dataframe_full(
            frame,
            lat_col=lat_col,
            lon_col=lon_col,
            costing=costing,
            costing_options=costing_options,
            config=config,
            trace_options=opts,
            max_breakage_m=bridges.max_dist_m,
        )

    def _match_ways(frame: pl.DataFrame, opts: dict[str, object] | None) -> list[int]:
        return map_match_ways(
            _segment_points(frame),
            costing=costing,
            costing_options=costing_options,
            config=config,
            trace_options=opts,
            max_breakage_m=bridges.max_dist_m,
        )

    gap_idxs = _find_gap_indices(
        df,
        bridges=bridges,
        lat_col=lat_col,
        lon_col=lon_col,
        time_col=time_col,
    )
    if not gap_idxs:
        if collect_shapes:
            matched_df, ways, shapes = _match_full(df, trace_options)
            return BridgedMatchResult(
                matched_df=matched_df, way_ids=ways, shapes=shapes
            )
        ways = _match_ways(df, trace_options)
        return BridgedMatchResult(matched_df=df.head(0), way_ids=ways, shapes=[])

    segments = _split_at(df, gap_idxs)
    all_ways: list[int] = []
    all_shapes: list[list[tuple[float, float]]] = []
    matched_frames: list[pl.DataFrame] = []
    fits: list[BridgeFit] = []

    try:
        for k, seg in enumerate(segments):
            if len(seg) < 2:
                if collect_shapes:
                    matched_frames.append(_null_matched_frame(seg))
            elif collect_shapes:
                seg_matched, seg_ways, seg_shapes = _match_full(seg, trace_options)
                matched_frames.append(seg_matched)
                _extend_dedup(all_ways, seg_ways)
                all_shapes.extend(seg_shapes)
            else:
                seg_ways = _match_ways(seg, trace_options)
                _extend_dedup(all_ways, seg_ways)

            if k < len(segments) - 1:
                next_seg = segments[k + 1]
                a = (seg[lat_col][-1], seg[lon_col][-1])
                b = (next_seg[lat_col][0], next_seg[lon_col][0])
                gap_s = (next_seg[time_col][0] - seg[time_col][-1]).total_seconds()
                bridge_shape, bridge_ways, fit = bridge_gap(
                    a,
                    b,
                    gap_seconds=gap_s,
                    costing=costing,
                    costing_options=costing_options,
                    config=config,
                )
                _extend_dedup(all_ways, bridge_ways)
                if collect_shapes:
                    all_shapes.append(bridge_shape)
                fits.append(fit)
    except Exception:
        fallback_opts = dict(trace_options or {})
        fallback_opts["breakage_distance"] = _BASE_BREAKAGE_DISTANCE
        if collect_shapes:
            matched_df, ways, shapes = _match_full(df, fallback_opts)
        else:
            matched_df = df.head(0)
            ways = _match_ways(df, fallback_opts)
            shapes = []
        return BridgedMatchResult(
            matched_df=matched_df,
            way_ids=ways,
            shapes=shapes,
            fits=[],
            fallback_used=True,
        )

    matched_df = (
        pl.concat(matched_frames, how="vertical_relaxed")
        if collect_shapes and matched_frames
        else df.head(0)
    )
    return BridgedMatchResult(
        matched_df=matched_df,
        way_ids=all_ways,
        shapes=all_shapes,
        fits=fits,
        fallback_used=False,
    )
