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
        .sort([id_col, time_col], Default::default())
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
                if is_stop_vec[i] != is_stop_vec[i - 1] {
                    segment_ids[i] = segment_ids[i - 1] + 1;
                } else {
                    segment_ids[i] = segment_ids[i - 1];
                }
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
        concat(lfs, Default::default())?.collect()?
    };

    if min_length > 0 {
        // Only filter short *movement* segments; stop segments always survive.
        // Count rows per movement segment, then mark each row with its count
        // (stop rows get null _cnt via left join and are always kept).
        let movement_counts = combined
            .clone()
            .lazy()
            .filter(col("is_stop").eq(lit(false)))
            .group_by([col(id_col), col("segment_id")])
            .agg([col(time_col).count().alias("_cnt")])
            .collect()?;

        combined = combined
            .lazy()
            .join(
                movement_counts.lazy(),
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
fn datetime_to_microseconds(col: &Column) -> PolarsResult<Vec<i64>> {
    let ca = col.datetime().map_err(|_| {
        PolarsError::ComputeError(format!("Expected datetime column, got {:?}", col.dtype()).into())
    })?;
    // Access the underlying physical Int64Chunked via the Series
    let phys = ca.phys.clone().into_series();
    let i64_ca = phys.i64()?;
    Ok(i64_ca
        .to_vec()
        .into_iter()
        .map(|opt: Option<i64>| opt.unwrap_or(0))
        .collect())
}

/// Detect stop intervals using a sliding window.
///
/// Returns a list of (start_idx, end_idx) inclusive ranges where the object
/// is stopped (within diameter, for at least min_duration).
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
        let mut end = start;
        let mut best_end: Option<usize> = None;

        // Track bounding box
        let start_lat = lats[start].unwrap_or(0.0);
        let start_lon = lons[start].unwrap_or(0.0);
        let mut min_lat = start_lat;
        let mut max_lat = start_lat;
        let mut min_lon = start_lon;
        let mut max_lon = start_lon;

        while end < n {
            let lat = lats[end].unwrap_or(0.0);
            let lon = lons[end].unwrap_or(0.0);

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
