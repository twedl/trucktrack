use std::sync::Arc;

use polars::prelude::*;
use polars::series::ops::NullBehavior;

/// Split a trajectory DataFrame at observation gaps exceeding `gap_us` microseconds.
///
/// Appends a `segment_id` (i32) column. Rows within the same (id, segment) group
/// are contiguous in time with no gap larger than the threshold.
pub fn split_by_observation_gap(
    df: DataFrame,
    id_col: &str,
    time_col: &str,
    gap_us: i64,
    min_length: usize,
) -> PolarsResult<DataFrame> {
    // Cast to Int64 (microseconds since epoch) before diffing so that
    // both tz-aware and naive datetime columns are handled uniformly.
    let result = df
        .lazy()
        .sort([id_col, time_col], SortMultipleOptions::default())
        .with_column(
            col(time_col)
                .cast(DataType::Int64)
                .diff(lit(1), NullBehavior::Ignore)
                .over([col(id_col)])
                .gt(lit(gap_us))
                .fill_null(lit(false))
                .cum_sum(false)
                .over([col(id_col)])
                .alias("segment_id"),
        )
        .collect()?;

    if min_length > 0 {
        let cached = result.lazy().cache();

        let counts = cached
            .clone()
            .group_by([col(id_col), col("segment_id")])
            .agg([col(time_col).count().alias("_cnt")]);

        let filtered = cached
            .join(
                counts,
                [col(id_col), col("segment_id")],
                [col(id_col), col("segment_id")],
                JoinArgs::new(JoinType::Inner),
            )
            .filter(col("_cnt").gt_eq(lit(min_length as u32)))
            .drop(Selector::ByName {
                names: Arc::new([PlSmallStr::from_static("_cnt")]),
                strict: false,
            })
            .collect()?;

        Ok(filtered)
    } else {
        Ok(result)
    }
}
