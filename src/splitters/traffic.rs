use polars::prelude::*;

use crate::geo::{bearing_degrees, haversine_distance_meters};

/// Filter out stops that are likely traffic (jams, red lights, congestion).
///
/// Takes a DataFrame with `segment_id` and `is_stop` columns (output of
/// `split_by_stops`).  For each stop segment, computes the approach bearing
/// from upstream movement points and the departure bearing from downstream
/// movement points.  If the angular change is below `max_angle_change`
/// degrees the stop is reclassified as movement — the truck never left the
/// road corridor.
///
/// Bearings are computed from successive GPS positions (not device-reported
/// heading).  Point pairs closer than `min_distance_m` are skipped to avoid
/// GPS-jitter noise.
///
/// After reclassification `segment_id` values are reassigned sequentially
/// and `is_stop` is updated in place.
pub fn filter_traffic_stops(
    df: DataFrame,
    id_col: &str,
    lat_col: &str,
    lon_col: &str,
    max_angle_change: f64,
    min_distance_m: f64,
) -> PolarsResult<DataFrame> {
    let groups = df.partition_by([id_col], true)?;
    let mut result_frames: Vec<DataFrame> = Vec::new();

    for group in groups {
        let lats = group.column(lat_col)?.f64()?.to_vec();
        let lons = group.column(lon_col)?.f64()?.to_vec();
        let is_stop_series = group.column("is_stop")?.bool()?;

        let n = group.height();
        let mut is_stop: Vec<bool> = (0..n)
            .map(|i| is_stop_series.get(i).unwrap_or(false))
            .collect();

        let stop_ranges = find_stop_ranges(&is_stop);

        for (s_start, s_end) in &stop_ranges {
            let entry = find_entry_bearing(&lats, &lons, *s_start, min_distance_m);
            let exit = find_exit_bearing(&lats, &lons, *s_end, n, min_distance_m);

            if let (Some(entry_b), Some(exit_b)) = (entry, exit) {
                if angular_difference(entry_b, exit_b) < max_angle_change {
                    for val in &mut is_stop[*s_start..=*s_end] {
                        *val = false;
                    }
                }
            }
        }

        // Reassign segment_ids after reclassification
        let mut segment_ids: Vec<u32> = vec![0; n];
        if n > 0 {
            for i in 1..n {
                segment_ids[i] = if is_stop[i] != is_stop[i - 1] {
                    segment_ids[i - 1].saturating_add(1)
                } else {
                    segment_ids[i - 1]
                };
            }
        }

        let mut group = group.drop("is_stop")?.drop("segment_id")?;

        let seg_col =
            UInt32Chunked::from_vec(PlSmallStr::from_static("segment_id"), segment_ids)
                .into_series()
                .into_column();

        let stop_col =
            BooleanChunked::from_slice(PlSmallStr::from_static("is_stop"), &is_stop)
                .into_series()
                .into_column();

        let _ = group.with_column(seg_col)?;
        let _ = group.with_column(stop_col)?;

        result_frames.push(group);
    }

    if result_frames.is_empty() {
        Ok(df.head(Some(0)))
    } else {
        let lfs: Vec<LazyFrame> = result_frames.into_iter().map(|f| f.lazy()).collect();
        concat(lfs, UnionArgs::default())?.collect()
    }
}

/// Find contiguous runs of `true` in `is_stop`, returned as inclusive ranges.
fn find_stop_ranges(is_stop: &[bool]) -> Vec<(usize, usize)> {
    let mut ranges = Vec::new();
    let mut i = 0;
    while i < is_stop.len() {
        if is_stop[i] {
            let start = i;
            while i < is_stop.len() && is_stop[i] {
                i += 1;
            }
            ranges.push((start, i - 1));
        } else {
            i += 1;
        }
    }
    ranges
}

/// Walk backward from the stop to compute the approach bearing.
///
/// Finds two valid points before the stop with at least `min_distance_m`
/// separation and returns the bearing from the farther to the nearer.
fn find_entry_bearing(
    lats: &[Option<f64>],
    lons: &[Option<f64>],
    stop_start: usize,
    min_distance_m: f64,
) -> Option<f64> {
    if stop_start == 0 {
        return None;
    }

    // Nearest valid point before the stop
    let mut near = None;
    let mut idx = stop_start - 1;
    loop {
        if let (Some(lat), Some(lon)) = (lats[idx], lons[idx]) {
            near = Some((idx, lat, lon));
            break;
        }
        if idx == 0 {
            break;
        }
        idx -= 1;
    }

    let (near_idx, near_lat, near_lon) = near?;

    if near_idx == 0 {
        return None;
    }

    // Walk further back for a point with sufficient distance
    let mut far_idx = near_idx - 1;
    loop {
        if let (Some(lat), Some(lon)) = (lats[far_idx], lons[far_idx]) {
            let dist = haversine_distance_meters(lat, lon, near_lat, near_lon);
            if dist >= min_distance_m {
                return Some(bearing_degrees(lat, lon, near_lat, near_lon));
            }
        }
        if far_idx == 0 {
            break;
        }
        far_idx -= 1;
    }

    None
}

/// Walk forward from the stop to compute the departure bearing.
///
/// Finds two valid points after the stop with at least `min_distance_m`
/// separation and returns the bearing from the nearer to the farther.
fn find_exit_bearing(
    lats: &[Option<f64>],
    lons: &[Option<f64>],
    stop_end: usize,
    n: usize,
    min_distance_m: f64,
) -> Option<f64> {
    if stop_end + 1 >= n {
        return None;
    }

    // Nearest valid point after the stop
    let mut near = None;
    for idx in (stop_end + 1)..n {
        if let (Some(lat), Some(lon)) = (lats[idx], lons[idx]) {
            near = Some((idx, lat, lon));
            break;
        }
    }

    let (near_idx, near_lat, near_lon) = near?;

    // Walk further forward for a point with sufficient distance
    for idx in (near_idx + 1)..n {
        if let (Some(lat), Some(lon)) = (lats[idx], lons[idx]) {
            let dist = haversine_distance_meters(near_lat, near_lon, lat, lon);
            if dist >= min_distance_m {
                return Some(bearing_degrees(near_lat, near_lon, lat, lon));
            }
        }
    }

    None
}

/// Smallest angle between two bearings (0–180°).
fn angular_difference(a: f64, b: f64) -> f64 {
    let diff = (a - b).rem_euclid(360.0);
    if diff > 180.0 {
        360.0 - diff
    } else {
        diff
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn angular_difference_same() {
        assert!((angular_difference(90.0, 90.0)).abs() < 1e-9);
    }

    #[test]
    fn angular_difference_opposite() {
        assert!((angular_difference(0.0, 180.0) - 180.0).abs() < 1e-9);
    }

    #[test]
    fn angular_difference_wraparound() {
        assert!((angular_difference(10.0, 350.0) - 20.0).abs() < 1e-9);
    }

    #[test]
    fn find_stop_ranges_basic() {
        let stops = vec![false, true, true, false, true, false];
        let ranges = find_stop_ranges(&stops);
        assert_eq!(ranges, vec![(1, 2), (4, 4)]);
    }

    #[test]
    fn find_stop_ranges_empty() {
        let ranges = find_stop_ranges(&[]);
        assert!(ranges.is_empty());
    }

    #[test]
    fn traffic_stop_reclassified() {
        // Truck travels north, "stops", continues north — should be filtered
        // Points along roughly the same latitude line going north
        let lats: Vec<Option<f64>> = vec![
            Some(43.0),
            Some(43.001),
            Some(43.002), // approach
            Some(43.0025),
            Some(43.0025), // stop (small cluster)
            Some(43.003),
            Some(43.004), // departure
        ];
        let lons: Vec<Option<f64>> = vec![
            Some(-79.0),
            Some(-79.0),
            Some(-79.0),
            Some(-79.0),
            Some(-79.0),
            Some(-79.0),
            Some(-79.0),
        ];

        let is_stop = vec![false, false, false, true, true, false, false];
        let segment_ids: Vec<u32> = vec![0, 0, 0, 1, 1, 2, 2];

        let df = DataFrame::new(7, vec![
            Column::new("id".into(), vec!["a"; 7]),
            Series::new("lat".into(), &lats)
                .cast(&DataType::Float64)
                .unwrap()
                .into_column(),
            Series::new("lon".into(), &lons)
                .cast(&DataType::Float64)
                .unwrap()
                .into_column(),
            BooleanChunked::from_slice("is_stop".into(), &is_stop)
                .into_series()
                .into_column(),
            UInt32Chunked::from_vec("segment_id".into(), segment_ids)
                .into_series()
                .into_column(),
        ])
        .unwrap();

        let result = filter_traffic_stops(df, "id", "lat", "lon", 30.0, 5.0).unwrap();

        // Stop should be reclassified — all is_stop should be false
        let stop_col = result.column("is_stop").unwrap().bool().unwrap();
        for i in 0..result.height() {
            assert!(!stop_col.get(i).unwrap(), "row {i} should not be a stop");
        }
    }

    #[test]
    fn real_stop_preserved() {
        // Truck travels north, stops, departs east — bearing change ~90°
        let lats: Vec<Option<f64>> = vec![
            Some(43.0),
            Some(43.001),
            Some(43.002), // approach heading north
            Some(43.0025),
            Some(43.0025), // stop
            Some(43.0025),
            Some(43.0025), // departure heading east
        ];
        let lons: Vec<Option<f64>> = vec![
            Some(-79.0),
            Some(-79.0),
            Some(-79.0),
            Some(-79.0),
            Some(-79.0),
            Some(-78.999),
            Some(-78.998),
        ];

        let is_stop = vec![false, false, false, true, true, false, false];
        let segment_ids: Vec<u32> = vec![0, 0, 0, 1, 1, 2, 2];

        let df = DataFrame::new(7, vec![
            Column::new("id".into(), vec!["a"; 7]),
            Series::new("lat".into(), &lats)
                .cast(&DataType::Float64)
                .unwrap()
                .into_column(),
            Series::new("lon".into(), &lons)
                .cast(&DataType::Float64)
                .unwrap()
                .into_column(),
            BooleanChunked::from_slice("is_stop".into(), &is_stop)
                .into_series()
                .into_column(),
            UInt32Chunked::from_vec("segment_id".into(), segment_ids)
                .into_series()
                .into_column(),
        ])
        .unwrap();

        let result = filter_traffic_stops(df, "id", "lat", "lon", 30.0, 5.0).unwrap();

        // Stop should be preserved — rows 3,4 should still be is_stop=true
        let stop_col = result.column("is_stop").unwrap().bool().unwrap();
        assert!(stop_col.get(3).unwrap(), "row 3 should still be a stop");
        assert!(stop_col.get(4).unwrap(), "row 4 should still be a stop");
    }
}
