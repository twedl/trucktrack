"""Benchmark the bridging orchestrator vs. the plain pipeline matcher.

Generates a grid of synthetic trips with varying lengths and dropout
sizes, then map-matches each one both ways and reports per-trip wall
clock, way-ID count, and — for the bridging strategy — number of
bridges actually fired.

The "baseline" is ``evaluate_map_match_ways`` (single
``trace_attributes`` call with adaptive breakage_distance capped at
60 km) — what ``run_map_matching`` does without ``bridges=``.

The "bridges" strategy is ``evaluate_map_match_with_bridges`` with
``collect_shapes=False`` (what ``run_map_matching(bridges=...)``
does).

Usage::

    uv run python scripts/bench_bridges.py
    uv run python scripts/bench_bridges.py --iterations 5
    uv run python scripts/bench_bridges.py --output bench.parquet
"""

from __future__ import annotations

import argparse
import statistics
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from time import perf_counter

import polars as pl

from trucktrack import filter_impossible_speeds, generate_trace
from trucktrack.generate.models import ErrorConfig, TripConfig
from trucktrack.valhalla import BridgeConfig, find_config
from trucktrack.valhalla.quality import (
    evaluate_map_match_ways,
    evaluate_map_match_with_bridges,
)

# (lat, lon) — points across Ontario, progressively further from Toronto.
TORONTO = (43.6532, -79.3832)
MISSISSAUGA = (43.5890, -79.6441)  # ~25 km
BARRIE = (44.3894, -79.6903)  # ~90 km
KINGSTON = (44.2312, -76.4860)  # ~260 km
OTTAWA = (45.4215, -75.6972)  # ~450 km


@dataclass
class TripSpec:
    name: str
    origin: tuple[float, float]
    destination: tuple[float, float]
    dropout_s: int  # 0 for no forced dropout


TRIPS: list[TripSpec] = [
    TripSpec("short_nogap", TORONTO, MISSISSAUGA, 0),
    TripSpec("short_gap300", TORONTO, MISSISSAUGA, 300),
    TripSpec("short_gap600", TORONTO, MISSISSAUGA, 600),
    TripSpec("medium_nogap", TORONTO, BARRIE, 0),
    TripSpec("medium_gap300", TORONTO, BARRIE, 300),
    TripSpec("medium_gap900", TORONTO, BARRIE, 900),
    TripSpec("medium_gap1800", TORONTO, BARRIE, 1800),
    TripSpec("long_nogap", TORONTO, KINGSTON, 0),
    TripSpec("long_gap900", TORONTO, KINGSTON, 900),
    TripSpec("long_gap2400", TORONTO, KINGSTON, 2400),
    TripSpec("xlong_nogap", TORONTO, OTTAWA, 0),
    TripSpec("xlong_gap1800", TORONTO, OTTAWA, 1800),
]


def _build_trip(
    spec: TripSpec, config_path: Path, max_speed_kmh: float
) -> pl.DataFrame:
    errors = (
        [
            ErrorConfig(
                "signal_dropout",
                probability=1.0,
                params={"duration_s": spec.dropout_s},
            )
        ]
        if spec.dropout_s > 0
        else []
    )
    trip = TripConfig(
        origin=spec.origin,
        destination=spec.destination,
        departure_time=datetime(2026, 4, 20, 8, 0),
        trip_id=spec.name,
        seed=42,
        config=config_path,
        errors=errors,
    )
    points = generate_trace(trip)
    df = pl.DataFrame(
        {
            "id": [spec.name] * len(points),
            "time": [p.timestamp for p in points],
            "lat": [p.lat for p in points],
            "lon": [p.lon for p in points],
        }
    )
    # Drop points whose implied speed from the previous kept fix exceeds
    # *max_speed_kmh* — removes generator artifacts where the interpolator
    # produces multi-km jumps in a single 60 s sample.
    return filter_impossible_speeds(df, max_speed_kmh=max_speed_kmh)


def _time_baseline(df: pl.DataFrame, config_path: Path) -> tuple[float, int, bool]:
    points = list(zip(df["lat"].to_list(), df["lon"].to_list(), strict=True))
    t0 = perf_counter()
    q = evaluate_map_match_ways(df["id"][0], points, config=config_path)
    return perf_counter() - t0, len(q.way_ids), q.error is None


def _time_bridges(
    df: pl.DataFrame, config_path: Path, bridges: BridgeConfig
) -> tuple[float, int, int, bool, bool]:
    t0 = perf_counter()
    q = evaluate_map_match_with_bridges(
        df["id"][0],
        df,
        bridges=bridges,
        config=config_path,
        collect_shapes=False,
    )
    elapsed = perf_counter() - t0
    return elapsed, len(q.way_ids), q.n_bridges, q.any_bridge_failed, q.error is None


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return float("nan")
    values_sorted = sorted(values)
    k = (len(values_sorted) - 1) * pct
    lo = int(k)
    hi = min(lo + 1, len(values_sorted) - 1)
    return values_sorted[lo] + (values_sorted[hi] - values_sorted[lo]) * (k - lo)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--bridges-max-dist-m", type=float, default=5000.0)
    parser.add_argument("--bridges-time-s", type=float, default=240.0)
    parser.add_argument("--bridges-min-dist-m", type=float, default=1000.0)
    parser.add_argument(
        "--max-speed-kmh",
        type=float,
        default=140.0,
        help="Pre-filter threshold for impossible-speed generator artifacts.",
    )
    args = parser.parse_args()

    config_path = find_config()
    if config_path is None:
        raise SystemExit("No valhalla.json discoverable under cwd.")

    bridges = BridgeConfig(
        max_dist_m=args.bridges_max_dist_m,
        time_s=args.bridges_time_s,
        min_dist_m=args.bridges_min_dist_m,
    )
    print(f"Bridges: {bridges}")
    print(f"Max speed filter: {args.max_speed_kmh} km/h")
    print(f"Iterations per trip/strategy: {args.iterations}")

    # Warm the actor so tile-load latency doesn't bias the first trip.
    print("Warming actor with a throwaway match...")
    warm_df = _build_trip(TRIPS[0], config_path, args.max_speed_kmh)
    _time_baseline(warm_df, config_path)
    _time_bridges(warm_df, config_path, bridges)

    rows: list[dict[str, object]] = []

    print("\nGenerating traces + running matches...")
    for spec in TRIPS:
        df = _build_trip(spec, config_path, args.max_speed_kmh)
        base_times: list[float] = []
        bridge_times: list[float] = []
        base_ways = 0
        bridge_ways = 0
        n_bridges = 0
        any_failed = False
        base_ok = True
        bridge_ok = True
        for _ in range(args.iterations):
            bt, bw, bok = _time_baseline(df, config_path)
            base_times.append(bt)
            base_ways = bw
            base_ok = base_ok and bok
            gt, gw, nb, af, gok = _time_bridges(df, config_path, bridges)
            bridge_times.append(gt)
            bridge_ways = gw
            n_bridges = nb
            any_failed = any_failed or af
            bridge_ok = bridge_ok and gok

        base_median = statistics.median(base_times)
        bridge_median = statistics.median(bridge_times)
        speedup = base_median / bridge_median if bridge_median > 0 else float("nan")

        rows.append(
            {
                "trip": spec.name,
                "n_points": len(df),
                "n_bridges": n_bridges,
                "base_s_median": base_median,
                "bridge_s_median": bridge_median,
                "base_s_p90": _percentile(base_times, 0.9),
                "bridge_s_p90": _percentile(bridge_times, 0.9),
                "speedup": speedup,
                "base_ways": base_ways,
                "bridge_ways": bridge_ways,
                "any_bridge_failed": any_failed,
                "base_ok": base_ok,
                "bridge_ok": bridge_ok,
            }
        )
        print(
            f"  {spec.name:<18} n={len(df):>4}  "
            f"base={base_median * 1000:>7.1f}ms  "
            f"bridge={bridge_median * 1000:>7.1f}ms  "
            f"speedup={speedup:>4.2f}x  "
            f"bridges={n_bridges}  "
            f"ways base/bridge={base_ways}/{bridge_ways}"
        )

    out = pl.DataFrame(rows)
    print("\n--- Summary ---")
    with pl.Config(tbl_rows=30, tbl_cols=13):
        print(out)

    if args.output is not None:
        out.write_parquet(args.output)
        print(f"\nWrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
