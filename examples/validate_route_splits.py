"""Validate that the gap splitter recovers the trips in route.parquet.

Assigns all rows a single dummy ID so the splitter sees one long sequence,
then checks whether the inferred segment_id groups match the original id column.
"""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import polars as pl
import trucktrack

DATA = Path(__file__).parent.parent / "data" / "route.parquet"


def main() -> None:
    df = pl.read_parquet(DATA)
    print(f"Loaded {len(df)} rows, {df['id'].n_unique()} original trips")
    print(f"Original trip IDs: {sorted(df['id'].unique().to_list())}")
    print()

    # Assign a single dummy ID so the splitter treats all rows as one sequence.
    df_combined = df.with_columns(pl.lit("all").alias("_dummy_id"))

    # Split using a 2-minute observation gap threshold.
    result = trucktrack.split_by_observation_gap(
        df_combined,
        timedelta(minutes=2),
        id_col="_dummy_id",
        time_col="timestamp",
    )

    n_segments = result["segment_id"].n_unique()
    n_original = df["id"].n_unique()
    print(f"Splitter found {n_segments} segments (expected {n_original})")
    print()

    # Build a mapping: for each segment, which original IDs appear?
    segment_to_ids = (
        result.group_by("segment_id")
        .agg(
            pl.col("id").unique().alias("original_ids"),
            pl.col("id").n_unique().alias("n_ids"),
            pl.len().alias("n_rows"),
        )
        .sort("segment_id")
    )
    print("Segment breakdown:")
    print(segment_to_ids)
    print()

    # Check 1: each segment contains exactly one original ID.
    mixed = segment_to_ids.filter(pl.col("n_ids") > 1)
    if len(mixed) > 0:
        print("FAIL: some segments contain multiple original IDs:")
        print(mixed)
    else:
        print("PASS: every segment maps to exactly one original trip ID")

    # Check 2: each original ID maps to exactly one segment.
    id_to_segments = (
        result.group_by("id")
        .agg(
            pl.col("segment_id").unique().alias("segments"),
            pl.col("segment_id").n_unique().alias("n_segments"),
        )
        .sort("id")
    )
    split_ids = id_to_segments.filter(pl.col("n_segments") > 1)
    if len(split_ids) > 0:
        print("WARN: some original IDs were split across multiple segments:")
        print(split_ids)
    else:
        print("PASS: every original trip ID maps to exactly one segment")

    # Summary
    print()
    all_ok = len(mixed) == 0 and len(split_ids) == 0
    if all_ok and n_segments == n_original:
        print("SUCCESS: algorithm segments match original trip IDs perfectly")
    elif all_ok:
        print(
            f"PARTIAL: segments are pure but count differs "
            f"({n_segments} segments vs {n_original} trips)"
        )
    else:
        print("MISMATCH: algorithm segments do not align with original trip IDs")


if __name__ == "__main__":
    main()
