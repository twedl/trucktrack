"""Benchmark trucktrack.split_by_stops against movingpandas.StopSplitter.

Measures wall-clock time and peak memory for both implementations on the same
dataset and parameters, in two regimes:

  1. Splitter-only ("algorithm"): movingpandas TrajectoryCollection is built
     and projected to UTM *before* the timed region. Compares the two stop
     detection algorithms head-to-head.
  2. End-to-end ("operational"): starts from the polars DataFrame loaded from
     parquet. For movingpandas this includes pandas/GeoPandas conversion and
     UTM reprojection; this is the real cost of using it from a polars
     pipeline.

Caveats:
  - tracemalloc only sees Python allocations. trucktrack's Rust allocations
    are invisible to it, so the tracemalloc number understates trucktrack's
    real memory use. The RSS delta from getrusage is reported as a coarse
    OS-level cross-check.
  - At the default --repeat-dataset 1 the dataset is small (1,884 rows) and
    trucktrack's splitter-only time may round to 0 ms. Pass e.g.
    --repeat-dataset 100 for stable numbers.

Requires (not in pyproject.toml):
    pip install movingpandas geopandas shapely pyproj pyarrow
"""

from __future__ import annotations

import argparse
import gc
import resource
import statistics
import sys
import time
import tracemalloc
from datetime import timedelta
from pathlib import Path
from typing import Callable

import geopandas as gpd
import movingpandas as mpd
import polars as pl
import trucktrack
from shapely.geometry import Point

DATA = Path(__file__).parent.parent / "data" / "route.parquet"


def tile_dataset(df: pl.DataFrame, k: int) -> pl.DataFrame:
    """Replicate df K times with offset trip ids and offset timestamps.

    Each replica gets ids suffixed with __{n} and timestamps shifted forward
    so trips never interleave across replicas.
    """
    if k <= 1:
        return df

    span = df["timestamp"].max() - df["timestamp"].min()
    # Step well past the original span so replicas are clearly disjoint.
    step = span + timedelta(days=1)

    parts: list[pl.DataFrame] = []
    for n in range(k):
        parts.append(
            df.with_columns(
                (pl.col("id") + f"__{n}").alias("id"),
                (pl.col("timestamp") + step * n).alias("timestamp"),
            )
        )
    return pl.concat(parts)


def build_mpd_collection(df: pl.DataFrame) -> mpd.TrajectoryCollection:
    """polars DataFrame -> projected movingpandas TrajectoryCollection."""
    pdf = df.to_pandas()
    pdf["geometry"] = [Point(xy) for xy in zip(pdf["lon"], pdf["lat"])]
    gdf = gpd.GeoDataFrame(pdf, geometry="geometry", crs="EPSG:4326")
    gdf = gdf.to_crs(gdf.estimate_utm_crs())
    return mpd.TrajectoryCollection(gdf, traj_id_col="id", t="timestamp")


# ── Timed callables ─────────────────────────────────────────────────────


def make_tt_splitter_only(
    df: pl.DataFrame, max_diameter: float, min_duration: timedelta
) -> Callable[[], object]:
    def run() -> object:
        return trucktrack.split_by_stops(
            df,
            max_diameter=max_diameter,
            min_duration=min_duration,
            id_col="id",
            time_col="timestamp",
            lat_col="lat",
            lon_col="lon",
        )

    return run


def make_mpd_splitter_only(
    tc: mpd.TrajectoryCollection, max_diameter: float, min_duration: timedelta
) -> Callable[[], object]:
    def run() -> object:
        return mpd.StopSplitter(tc).split(
            max_diameter=max_diameter, min_duration=min_duration
        )

    return run


def make_tt_end_to_end(
    df: pl.DataFrame, max_diameter: float, min_duration: timedelta
) -> Callable[[], object]:
    # Same as splitter-only — trucktrack consumes the polars df directly.
    return make_tt_splitter_only(df, max_diameter, min_duration)


def make_mpd_end_to_end(
    df: pl.DataFrame, max_diameter: float, min_duration: timedelta
) -> Callable[[], object]:
    def run() -> object:
        tc = build_mpd_collection(df)
        return mpd.StopSplitter(tc).split(
            max_diameter=max_diameter, min_duration=min_duration
        )

    return run


# ── Measurement ─────────────────────────────────────────────────────────


def _rss_kib() -> int:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss


def _rss_to_mib(kib_delta: int) -> float:
    # On Linux ru_maxrss is KiB; on macOS it is bytes. Heuristic: if the
    # raw RSS is implausibly large for a Python process in KiB (>100 GiB)
    # treat it as bytes.
    raw = _rss_kib()
    if raw > 100 * 1024 * 1024:  # >100 GiB in KiB → must be bytes
        return kib_delta / (1024 * 1024)
    return kib_delta / 1024


class Result:
    def __init__(self, name: str) -> None:
        self.name = name
        self.times_s: list[float] = []
        self.tracemalloc_peak_mib: float = float("nan")
        self.rss_delta_mib: float = float("nan")

    def add_time(self, t: float) -> None:
        self.times_s.append(t)

    def time_stats_ms(self) -> tuple[float, float, float, float]:
        ts = [t * 1000 for t in self.times_s]
        return (
            min(ts),
            statistics.mean(ts),
            statistics.median(ts),
            statistics.stdev(ts) if len(ts) > 1 else 0.0,
        )


def measure(name: str, fn: Callable[[], object], repeats: int) -> Result:
    res = Result(name)

    # Warm-up (untimed).
    fn()
    gc.collect()

    # Timing-only passes.
    for _ in range(repeats):
        gc.collect()
        t0 = time.perf_counter()
        out = fn()
        t1 = time.perf_counter()
        res.add_time(t1 - t0)
        del out

    # Single traced pass for memory.
    gc.collect()
    rss_before = _rss_kib()
    tracemalloc.start()
    out = fn()
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    rss_after = _rss_kib()
    del out
    res.tracemalloc_peak_mib = peak_bytes / (1024 * 1024)
    res.rss_delta_mib = _rss_to_mib(max(rss_after - rss_before, 0))

    return res


# ── Reporting ───────────────────────────────────────────────────────────


HEADER = (
    f"  {'':<14}"
    f"{'min':>10}{'mean':>10}{'median':>10}{'stdev':>10}"
    f"{'RSS Δ':>14}{'tracemalloc':>14}"
)


def _fmt_row(r: Result) -> str:
    mn, mean, med, sd = r.time_stats_ms()
    return (
        f"  {r.name:<14}"
        f"{mn:>8.2f}ms{mean:>8.2f}ms{med:>8.2f}ms{sd:>8.2f}ms"
        f"{r.rss_delta_mib:>10.1f}MiB{r.tracemalloc_peak_mib:>10.1f}MiB"
    )


def report_section(title: str, tt: Result, mp: Result) -> None:
    print(title)
    print(HEADER)
    print(_fmt_row(tt))
    print(_fmt_row(mp))
    tt_min = tt.time_stats_ms()[0]
    mp_min = mp.time_stats_ms()[0]
    if tt_min > 0:
        print(f"  speedup (mp / tt, by min): {mp_min / tt_min:.1f}×")
    else:
        print("  speedup (mp / tt, by min): n/a (tt below timer resolution)")
    print()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--max-diameter", type=float, default=250.0, help="meters")
    ap.add_argument("--min-duration", type=float, default=300.0, help="seconds")
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument(
        "--repeat-dataset",
        type=int,
        default=1,
        help="tile route.parquet K times with disjoint ids/times "
        "(try 100 or 1000 for stable numbers)",
    )
    args = ap.parse_args()

    min_duration = timedelta(seconds=args.min_duration)
    df = pl.read_parquet(DATA)
    df = tile_dataset(df, args.repeat_dataset)

    print(
        f"Dataset: {DATA.name} × {args.repeat_dataset} replicas "
        f"= {len(df):,} rows, {df['id'].n_unique():,} trips"
    )
    print(
        f"Parameters: max_diameter={args.max_diameter} m, "
        f"min_duration={min_duration}"
    )
    print(
        f"Repeats: {args.repeats} (+ 1 warm-up, + 1 traced run for memory)"
    )
    print()

    # Splitter-only: pre-build the movingpandas collection so it is excluded
    # from the timed region.
    tc = build_mpd_collection(df)
    tt_algo = measure(
        "trucktrack",
        make_tt_splitter_only(df, args.max_diameter, min_duration),
        args.repeats,
    )
    mp_algo = measure(
        "movingpandas",
        make_mpd_splitter_only(tc, args.max_diameter, min_duration),
        args.repeats,
    )
    report_section("Splitter-only (algorithm)", tt_algo, mp_algo)

    # End-to-end: start from the polars DataFrame.
    tt_e2e = measure(
        "trucktrack",
        make_tt_end_to_end(df, args.max_diameter, min_duration),
        args.repeats,
    )
    mp_e2e = measure(
        "movingpandas",
        make_mpd_end_to_end(df, args.max_diameter, min_duration),
        args.repeats,
    )
    report_section("End-to-end (from polars DataFrame)", tt_e2e, mp_e2e)

    print(
        "Notes: tracemalloc sees Python allocations only — trucktrack's Rust "
        "allocations are not counted, so its tracemalloc figure understates "
        "true memory use. RSS Δ is a coarse OS-level cross-check."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
