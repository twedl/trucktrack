use std::sync::Arc;

use polars::prelude::*;

use crate::geo::haversine_distance_meters;

/// Detect stops and split trajectory into labelled segments.
///
/// A stop is a contiguous window where all points fit within `max_diameter_m`
/// meters and the duration is at least `min_duration_us` microseconds.
/// All rows are kept. Each contiguous movement or stop region gets a
/// sequential `segment_id`. A boolean `is_stop` column distinguishes
/// stop rows from movement rows.
#[allow(clippy::too_many_arguments)]
pub fn split_by_stops(
    df: DataFrame,
    id_col: &str,
    time_col: &str,
    lat_col: &str,
    lon_col: &str,
    max_diameter_m: f64,
    min_duration_us: i64,
    min_length: usize,
) -> PolarsResult<DataFrame> {
    let sorted = df
        .lazy()
        .sort([id_col, time_col], SortMultipleOptions::default())
        .collect()?;

    let groups = sorted.partition_by([id_col], true)?;
    let mut result_frames: Vec<DataFrame> = Vec::new();

    for group in groups {
        let lats = group.column(lat_col)?.f64()?.to_vec();
        let lons = group.column(lon_col)?.f64()?.to_vec();
        let times_us = datetime_to_microseconds(group.column(time_col)?)?;

        let n = group.height();
        let stops = detect_stops(&lats, &lons, &times_us, max_diameter_m, min_duration_us);

        // Mark stop rows
        let mut is_stop_vec = vec![false; n];
        for (start, end) in &stops {
            for val in &mut is_stop_vec[*start..=*end] {
                *val = true;
            }
        }

        // Assign sequential segment_id to all rows, incrementing on transitions
        let mut segment_ids: Vec<u32> = vec![0; n];
        if n > 0 {
            for i in 1..n {
                segment_ids[i] = if is_stop_vec[i] != is_stop_vec[i - 1] {
                    segment_ids[i - 1].saturating_add(1)
                } else {
                    segment_ids[i - 1]
                };
            }
        }

        // Attach segment_id and is_stop columns (keep all rows)
        let seg_col = UInt32Chunked::from_vec(PlSmallStr::from_static("segment_id"), segment_ids)
            .into_series()
            .into_column();

        let stop_col = BooleanChunked::from_slice(PlSmallStr::from_static("is_stop"), &is_stop_vec)
            .into_series()
            .into_column();

        let mut group = group;
        let _ = group.with_column(seg_col)?;
        let _ = group.with_column(stop_col)?;

        result_frames.push(group);
    }

    let mut combined = if result_frames.is_empty() {
        let mut empty = sorted.head(Some(0));
        let _ = empty.with_column(Column::new_empty(
            PlSmallStr::from_static("segment_id"),
            &DataType::UInt32,
        ));
        let _ = empty.with_column(Column::new_empty(
            PlSmallStr::from_static("is_stop"),
            &DataType::Boolean,
        ));
        empty
    } else {
        let lfs: Vec<LazyFrame> = result_frames.into_iter().map(|f| f.lazy()).collect();
        concat(lfs, UnionArgs::default())?.collect()?
    };

    if min_length > 0 {
        // Only filter short *movement* segments; stop segments always survive.
        // Count rows per movement segment, then mark each row with its count
        // (stop rows get null _cnt via left join and are always kept).
        let cached = combined.lazy().cache();

        let movement_counts = cached
            .clone()
            .filter(col("is_stop").eq(lit(false)))
            .group_by([col(id_col), col("segment_id")])
            .agg([col(time_col).count().alias("_cnt")]);

        combined = cached
            .join(
                movement_counts,
                [col(id_col), col("segment_id")],
                [col(id_col), col("segment_id")],
                JoinArgs::new(JoinType::Left),
            )
            .filter(
                col("is_stop")
                    .eq(lit(true))
                    .or(col("_cnt").gt_eq(lit(min_length as u32))),
            )
            .drop(Selector::ByName {
                names: Arc::new([PlSmallStr::from_static("_cnt")]),
                strict: false,
            })
            .collect()?;
    }

    Ok(combined)
}

/// Extract microsecond timestamps from a datetime column.
///
/// Normalises the column's TimeUnit to microseconds before casting to Int64
/// so the result is microseconds-since-epoch regardless of input TimeUnit
/// (ms / us / ns).  Without this normalisation, a `Datetime[ms]` column
/// would yield millisecond integers and silently miscompare against the
/// microsecond `min_duration_us` threshold.
///
/// Returns an error if the column is not a datetime type.
/// Null timestamps are mapped to `i64::MIN` so they sort before all valid times
/// and never satisfy duration checks in `detect_stops`.
fn datetime_to_microseconds(col: &Column) -> PolarsResult<Vec<i64>> {
    let tz = match col.dtype() {
        DataType::Datetime(_, tz) => tz.clone(),
        other => {
            return Err(PolarsError::ComputeError(
                format!("'{}' is not a datetime column (got {other})", col.name()).into(),
            ));
        }
    };
    let series = col
        .cast(&DataType::Datetime(TimeUnit::Microseconds, tz))?
        .cast(&DataType::Int64)?
        .take_materialized_series();
    let i64_ca = series.i64()?;
    Ok(i64_ca
        .into_iter()
        .map(|opt: Option<i64>| opt.unwrap_or(i64::MIN))
        .collect())
}

/// Detect stop intervals using a sliding window.
///
/// Returns a list of (start_idx, end_idx) inclusive ranges where the object
/// is stopped (within diameter, for at least min_duration).
///
/// Rows with null lat/lon are skipped (never start or extend a stop window).
fn detect_stops(
    lats: &[Option<f64>],
    lons: &[Option<f64>],
    times_us: &[i64],
    max_diameter_m: f64,
    min_duration_us: i64,
) -> Vec<(usize, usize)> {
    let n = lats.len();
    if n == 0 {
        return vec![];
    }

    // Rough degree threshold for bounding-box pre-filter
    let degree_threshold = max_diameter_m / 111_000.0;

    let mut stops: Vec<(usize, usize)> = Vec::new();
    let mut start = 0;

    while start < n {
        let (start_lat, start_lon) = match (lats[start], lons[start]) {
            (Some(la), Some(lo)) => (la, lo),
            _ => {
                start += 1;
                continue;
            }
        };

        let mut end = start;
        let mut best_end: Option<usize> = None;

        let mut min_lat = start_lat;
        let mut max_lat = start_lat;
        let mut min_lon = start_lon;
        let mut max_lon = start_lon;

        while end < n {
            let (lat, lon) = match (lats[end], lons[end]) {
                (Some(la), Some(lo)) => (la, lo),
                _ => break,
            };

            min_lat = min_lat.min(lat);
            max_lat = max_lat.max(lat);
            min_lon = min_lon.min(lon);
            max_lon = max_lon.max(lon);

            // Fast bbox reject
            if (max_lat - min_lat) > degree_threshold || (max_lon - min_lon) > degree_threshold {
                break;
            }

            // Precise check: diameter is max distance across the bbox corners
            let diameter = haversine_distance_meters(min_lat, min_lon, max_lat, max_lon);
            if diameter > max_diameter_m {
                break;
            }

            // Check duration
            let duration = times_us[end] - times_us[start];
            if duration >= min_duration_us {
                best_end = Some(end);
            }

            end += 1;
        }

        if let Some(stop_end) = best_end {
            stops.push((start, stop_end));
            start = stop_end + 1;
        } else {
            start += 1;
        }
    }

    stops
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detect_stops_empty_input() {
        let stops = detect_stops(&[], &[], &[], 100.0, 60_000_000);
        assert!(stops.is_empty());
    }

    #[test]
    fn detect_stops_single_point() {
        let lats = vec![Some(43.65)];
        let lons = vec![Some(-79.38)];
        let times = vec![0];
        let stops = detect_stops(&lats, &lons, &times, 100.0, 0);
        // Single point with min_duration=0 qualifies as a stop
        assert_eq!(stops, vec![(0, 0)]);
    }

    #[test]
    fn detect_stops_stationary_long_enough() {
        // 5 points at the same location, 1 second apart
        let lats = vec![Some(43.65); 5];
        let lons = vec![Some(-79.38); 5];
        let times: Vec<i64> = (0..5).map(|i| i * 1_000_000).collect();
        // Requires 3 seconds duration
        let stops = detect_stops(&lats, &lons, &times, 100.0, 3_000_000);
        assert_eq!(stops, vec![(0, 4)]);
    }

    #[test]
    fn detect_stops_too_short_duration() {
        let lats = vec![Some(43.65); 3];
        let lons = vec![Some(-79.38); 3];
        let times: Vec<i64> = (0..3).map(|i| i * 1_000_000).collect();
        // Requires 10 seconds — only 2 seconds of data
        let stops = detect_stops(&lats, &lons, &times, 100.0, 10_000_000);
        assert!(stops.is_empty());
    }

    #[test]
    fn detect_stops_null_coords_skipped() {
        let lats = vec![None, Some(43.65), Some(43.65), Some(43.65), None];
        let lons = vec![None, Some(-79.38), Some(-79.38), Some(-79.38), None];
        let times: Vec<i64> = (0..5).map(|i| i * 1_000_000).collect();
        let stops = detect_stops(&lats, &lons, &times, 100.0, 1_000_000);
        // Should detect stop at indices 1..3, skipping nulls at 0 and 4
        assert_eq!(stops, vec![(1, 3)]);
    }

    #[test]
    fn detect_stops_null_breaks_window() {
        // Null in the middle should break the stop window
        let lats = vec![Some(43.65), Some(43.65), None, Some(43.65), Some(43.65)];
        let lons = vec![Some(-79.38), Some(-79.38), None, Some(-79.38), Some(-79.38)];
        let times: Vec<i64> = (0..5).map(|i| i * 1_000_000).collect();
        // Requires 2 seconds: first window (0,1) is only 1s, second window (3,4) is only 1s
        let stops = detect_stops(&lats, &lons, &times, 100.0, 2_000_000);
        assert!(stops.is_empty());
    }

    #[test]
    fn detect_stops_movement_breaks_stop() {
        // Stationary then moves far away
        let lats = vec![
            Some(43.65),
            Some(43.65),
            Some(43.65),
            Some(50.0),
            Some(50.0),
        ];
        let lons = vec![
            Some(-79.38),
            Some(-79.38),
            Some(-79.38),
            Some(-79.38),
            Some(-79.38),
        ];
        let times: Vec<i64> = (0..5).map(|i| i * 1_000_000).collect();
        let stops = detect_stops(&lats, &lons, &times, 100.0, 1_000_000);
        assert_eq!(stops, vec![(0, 2)]);
    }
}
