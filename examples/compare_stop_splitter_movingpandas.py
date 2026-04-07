"""Compare trucktrack.split_by_stops against movingpandas.StopSplitter.

Loads data/route.parquet, runs both implementations with the same parameters,
and reports any disagreement in per-trip segment counts and per-row stop/move
classification. Exits non-zero on mismatch.

Requires (not in pyproject.toml):
    pip install movingpandas geopandas shapely pyproj
"""

from __future__ import annotations

import argparse
import sys
from datetime import timedelta
from pathlib import Path

import geopandas as gpd
import movingpandas as mpd
import polars as pl
import trucktrack
from shapely.geometry import Point

DATA = Path(__file__).parent.parent / "data" / "route.parquet"


def trucktrack_segments(
    df: pl.DataFrame, max_diameter: float, min_duration: timedelta
) -> pl.DataFrame:
    """Run trucktrack and return (id, timestamp, segment_index) per moving row.

    segment_index numbers movement segments per id starting at 0.
    """
    out = trucktrack.split_by_stops(
        df,
        max_diameter=max_diameter,
        min_duration=min_duration,
        id_col="id",
        time_col="timestamp",
        lat_col="lat",
        lon_col="lon",
    )
    # segment_id from trucktrack increments globally across stops; renumber
    # per-id so 0..N-1 within each trip in time order.
    return (
        out.sort(["id", "timestamp"])
        .with_columns(
            (pl.col("segment_id").rle_id().over("id")).alias("segment_index")
        )
        .select(["id", "timestamp", "segment_index"])
    )


def movingpandas_segments(
    df: pl.DataFrame, max_diameter: float, min_duration: timedelta
) -> pl.DataFrame:
    """Run movingpandas StopSplitter and return same shape as trucktrack_segments."""
    pdf = df.to_pandas()
    pdf["geometry"] = [Point(xy) for xy in zip(pdf["lon"], pdf["lat"])]
    gdf = gpd.GeoDataFrame(pdf, geometry="geometry", crs="EPSG:4326")

    # Project to a metric CRS so max_diameter is in meters.
    utm = gdf.estimate_utm_crs()
    gdf = gdf.to_crs(utm)

    tc = mpd.TrajectoryCollection(gdf, traj_id_col="id", t="timestamp")
    splitter = mpd.StopSplitter(tc)
    split_tc = splitter.split(max_diameter=max_diameter, min_duration=min_duration)

    rows: list[tuple[str, object, int]] = []
    # Group sub-trajectories by their original parent id, ordered in time, and
    # number them 0..N-1.
    by_parent: dict[str, list[mpd.Trajectory]] = {}
    for traj in split_tc.trajectories:
        # movingpandas names sub-trajectories like "<parent>_<n>"; recover parent
        parent = str(traj.id).rsplit("_", 1)[0]
        by_parent.setdefault(parent, []).append(traj)

    for parent, trajs in by_parent.items():
        trajs.sort(key=lambda t: t.df.index.min())
        for idx, traj in enumerate(trajs):
            for ts in traj.df.index:
                rows.append((parent, ts.to_pydatetime(), idx))

    return pl.DataFrame(
        rows,
        schema={
            "id": pl.String,
            "timestamp": pl.Datetime("us"),
            "segment_index": pl.Int64,
        },
        orient="row",
    )


def compare(tt: pl.DataFrame, mp: pl.DataFrame) -> bool:
    print("=" * 70)
    print("Per-trip segment counts")
    print("=" * 70)

    tt_counts = (
        tt.group_by("id")
        .agg(pl.col("segment_index").n_unique().alias("trucktrack"))
        .sort("id")
    )
    mp_counts = (
        mp.group_by("id")
        .agg(pl.col("segment_index").n_unique().alias("movingpandas"))
        .sort("id")
    )
    counts = tt_counts.join(mp_counts, on="id", how="full", coalesce=True).sort("id")
    print(counts)

    counts_match = counts.filter(
        pl.col("trucktrack") != pl.col("movingpandas")
    ).is_empty()

    # Per-row stop/move agreement: compare the set of timestamps each tool
    # classified as "moving" (i.e. present in the output) per id.
    print()
    print("=" * 70)
    print("Stop/move classification agreement (per id)")
    print("=" * 70)

    tt_keys = tt.select("id", "timestamp").with_columns(pl.lit(True).alias("in_tt"))
    mp_keys = mp.select("id", "timestamp").with_columns(pl.lit(True).alias("in_mp"))
    joined = tt_keys.join(mp_keys, on=["id", "timestamp"], how="full", coalesce=True)

    only_tt = joined.filter(pl.col("in_mp").is_null())
    only_mp = joined.filter(pl.col("in_tt").is_null())

    print(f"  rows kept by trucktrack only: {len(only_tt)}")
    print(f"  rows kept by movingpandas only: {len(only_mp)}")
    if len(only_tt) > 0:
        print("  sample (trucktrack only):")
        print(only_tt.head(5))
    if len(only_mp) > 0:
        print("  sample (movingpandas only):")
        print(only_mp.head(5))

    rows_match = len(only_tt) == 0 and len(only_mp) == 0

    # Segment-membership agreement: for ids where counts agree, check that the
    # row-to-segment mapping is identical.
    print()
    print("=" * 70)
    print("Segment membership (where counts agree)")
    print("=" * 70)
    membership_ok = True
    for row in counts.iter_rows(named=True):
        if row["trucktrack"] != row["movingpandas"]:
            continue
        tid = row["id"]
        a = tt.filter(pl.col("id") == tid).sort("timestamp")
        b = mp.filter(pl.col("id") == tid).sort("timestamp")
        merged = a.join(
            b, on=["id", "timestamp"], how="inner", suffix="_mp"
        )
        # Build a permutation map from tt's segment_index to mp's, then check
        # consistency (segment numbering may differ but partition should match).
        pairs = merged.select("segment_index", "segment_index_mp").unique()
        if pairs.group_by("segment_index").len().get_column("len").max() != 1:
            print(f"  {tid}: MISMATCH (a tt segment maps to multiple mp segments)")
            membership_ok = False
        elif pairs.group_by("segment_index_mp").len().get_column("len").max() != 1:
            print(f"  {tid}: MISMATCH (a mp segment maps to multiple tt segments)")
            membership_ok = False
        else:
            print(f"  {tid}: OK ({row['trucktrack']} segments)")

    print()
    print("=" * 70)
    ok = counts_match and rows_match and membership_ok
    if ok:
        print("PASS: trucktrack and movingpandas agree on all trips")
    else:
        print("FAIL: implementations disagree (see details above)")
    print("=" * 70)
    return ok


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--max-diameter", type=float, default=50.0, help="meters")
    ap.add_argument(
        "--min-duration", type=float, default=120.0, help="seconds"
    )
    args = ap.parse_args()

    min_duration = timedelta(seconds=args.min_duration)
    print(
        f"Parameters: max_diameter={args.max_diameter} m, "
        f"min_duration={min_duration}"
    )

    df = pl.read_parquet(DATA)
    print(f"Loaded {len(df)} rows, {df['id'].n_unique()} trips from {DATA.name}")
    print()

    tt = trucktrack_segments(df, args.max_diameter, min_duration)
    mp = movingpandas_segments(df, args.max_diameter, min_duration)

    return 0 if compare(tt, mp) else 1


if __name__ == "__main__":
    sys.exit(main())
