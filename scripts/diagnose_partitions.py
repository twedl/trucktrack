#!/usr/bin/env python3
"""Throwaway: per-trip wall-time benchmark across selected partitions.

Picks a subset of partitions (explicit keys or top-N per tier by row count),
runs ``map_match_trip`` single-threaded against each trip, and writes a
parquet with per-trip ``wall_ms`` plus ``tier``/``partition_id``/``trip_index``
for cold-vs-warm analysis.  Does not write into the normal ``matched/``
layout — results go to a single timing parquet so the main pipeline's
resume logic is untouched.

Usage::

    uv run python scripts/diagnose_partitions.py data/partitioned \\
        --top-per-tier 3

    uv run python scripts/diagnose_partitions.py data/partitioned \\
        --partition regional/42 --partition longhaul/7

The output parquet can then be analyzed with polars to see which
partitions are slow per trip vs. merely having more trips.
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path

import polars as pl
from tqdm import tqdm

from trucktrack.valhalla.pipeline import _silence_stdout, map_match_trip


def _partition_key(pdir: Path) -> tuple[str, int]:
    tier = pdir.parent.name.removeprefix("tier=")
    pid = int(pdir.name.removeprefix("partition_id="))
    return (tier, pid)


def _discover(input_dir: Path) -> list[Path]:
    seen: set[Path] = set()
    for p in input_dir.rglob("*.parquet"):
        if p.parent.name.startswith("partition_id="):
            seen.add(p.parent)
    return sorted(seen, key=_partition_key)


def _row_count(pdir: Path) -> int:
    return int(
        pl.scan_parquet(str(pdir / "*.parquet")).select(pl.len()).collect().item()
    )


def _select_partitions(
    all_partitions: list[Path],
    explicit: list[str],
    top_per_tier: int | None,
) -> list[Path]:
    if explicit:
        wanted: set[tuple[str, int]] = set()
        for key in explicit:
            tier, pid = key.split("/", 1)
            wanted.add((tier, int(pid)))
        return [p for p in all_partitions if _partition_key(p) in wanted]

    if top_per_tier is None:
        return all_partitions

    by_tier: dict[str, list[tuple[int, Path]]] = defaultdict(list)
    for p in all_partitions:
        tier, _ = _partition_key(p)
        by_tier[tier].append((_row_count(p), p))

    selected: list[Path] = []
    for items in by_tier.values():
        items.sort(reverse=True, key=lambda x: x[0])
        selected.extend(p for _, p in items[:top_per_tier])
    return sorted(selected, key=_partition_key)


def _run_partition(pdir: Path, config: Path | None) -> list[dict[str, object]]:
    tier, pid = _partition_key(pdir)
    rows: list[dict[str, object]] = []
    trip_index = 0
    for chunk_path in sorted(pdir.glob("*.parquet")):
        df = pl.read_parquet(chunk_path)
        if "is_stop" in df.columns:
            df = df.filter(~pl.col("is_stop"))
        if df.is_empty():
            continue
        # Stable group order so trip_index is reproducible across runs.
        df = df.sort("id", "time")
        for _, trip in df.group_by("id", maintain_order=True):
            t0 = time.perf_counter_ns()
            _, quality_df = map_match_trip(trip, config=config)
            wall_ms = (time.perf_counter_ns() - t0) / 1e6
            q = quality_df.row(0, named=True)
            rows.append(
                {
                    "tier": tier,
                    "partition_id": pid,
                    "chunk": chunk_path.name,
                    "trip_index": trip_index,
                    "id": q["id"],
                    "n_points": q["n_points"],
                    "wall_ms": wall_ms,
                    "ok": q["ok"],
                    "error": q["error"],
                    "n_polylines": q["n_polylines"],
                    "path_length_ratio": q["path_length_ratio"],
                    "heading_reversals": q["heading_reversals"],
                }
            )
            trip_index += 1
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Per-trip wall-time benchmark across selected partitions."
    )
    ap.add_argument("input_dir", type=Path, help="e.g. data/partitioned")
    ap.add_argument(
        "--partition",
        action="append",
        default=[],
        metavar="TIER/PID",
        help="Partition key, e.g. regional/42. Repeatable.",
    )
    ap.add_argument(
        "--top-per-tier",
        type=int,
        default=None,
        help="Auto-select top-N partitions per tier by row count.",
    )
    ap.add_argument(
        "--output",
        type=Path,
        default=Path("scripts/output/partition_timing.parquet"),
    )
    ap.add_argument("--config", type=Path, default=None)
    ap.add_argument(
        "--verbose",
        action="store_true",
        help="Show Valhalla's C++ stdout (default: suppressed).",
    )
    args = ap.parse_args()

    all_parts = _discover(args.input_dir)
    if not all_parts:
        print(f"No partitions under {args.input_dir}", file=sys.stderr)
        sys.exit(1)

    selected = _select_partitions(all_parts, args.partition, args.top_per_tier)
    if not selected:
        print("No partitions matched selection", file=sys.stderr)
        sys.exit(1)

    print(
        f"Benchmarking {len(selected)} partition(s) of {len(all_parts)} discovered",
        file=sys.stderr,
    )
    for p in selected:
        tier, pid = _partition_key(p)
        print(f"  tier={tier} partition_id={pid}", file=sys.stderr)

    args.output.parent.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict[str, object]] = []
    silencer = nullcontext() if args.verbose else _silence_stdout()
    with silencer:
        for pdir in tqdm(selected, unit="partition", desc="Timing"):
            all_rows.extend(_run_partition(pdir, args.config))

    if not all_rows:
        print("No trips matched — nothing to write.", file=sys.stderr)
        sys.exit(1)

    df = pl.DataFrame(all_rows)
    df.write_parquet(args.output)
    print(f"\nWrote {len(df):,} trip timings -> {args.output}", file=sys.stderr)

    summary = (
        df.group_by("tier", "partition_id")
        .agg(
            trips=pl.len(),
            total_s=pl.col("wall_ms").sum() / 1000,
            median_ms=pl.col("wall_ms").median(),
            p95_ms=pl.col("wall_ms").quantile(0.95),
            median_pts=pl.col("n_points").median(),
            ms_per_point=pl.col("wall_ms").sum() / pl.col("n_points").sum(),
            fail_rate=(~pl.col("ok")).mean(),
        )
        .sort("tier", "partition_id")
    )
    with pl.Config(tbl_rows=100, tbl_width_chars=200):
        print(summary, file=sys.stderr)


if __name__ == "__main__":
    main()
