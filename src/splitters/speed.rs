use polars::prelude::*;

use crate::geo::haversine_distance_meters;

/// Drop GPS points whose implied speed from the last kept point is
/// physically impossible for a truck (e.g. > 200 km/h).
///
/// Per truck, sorted by ``(id_col, time_col)``, walks each point and
/// computes Haversine-distance ÷ time-delta against the last *kept* point.
/// If the implied speed exceeds ``max_speed_mps`` the current row is
/// dropped and the last-kept anchor is held — so a burst of adjacent
/// spikes all collapse against the last clean fix rather than
/// re-anchoring on a glitch.
///
/// First valid point per truck is always kept.  Rows with a null in any
/// of ``time_col`` / ``lat_col`` / ``lon_col`` can't be evaluated; they
/// pass through and do not advance the anchor, so the next valid point is
/// still compared against the previous clean fix.  If two consecutive
/// valid points share a timestamp the current row is kept without
/// advancing the anchor — the stale-ping filter is expected to have
/// removed exact duplicates upstream; guarding here avoids divide-by-zero
/// on any that slip through.
///
/// Output is sorted by ``(id_col, time_col)`` regardless of input order.
/// Tz-aware and tz-naive time columns are both handled — the column is
/// cast to ``Int64`` (microseconds since epoch) for the diff.
pub fn filter_impossible_speeds(
    df: DataFrame,
    id_col: &str,
    time_col: &str,
    lat_col: &str,
    lon_col: &str,
    max_speed_mps: f64,
) -> PolarsResult<DataFrame> {
    if df.height() == 0 {
        return Ok(df);
    }

    let sorted = df
        .lazy()
        .sort([id_col, time_col], SortMultipleOptions::default())
        .collect()?;

    let groups = sorted.partition_by([id_col], true)?;
    let mut result_frames: Vec<DataFrame> = Vec::with_capacity(groups.len());

    for group in groups {
        let n = group.height();
        let lats = group.column(lat_col)?.f64()?;
        let lons = group.column(lon_col)?.f64()?;
        // Normalise the time column to microsecond TimeUnit before casting to
        // Int64 so the resulting integers are microseconds-since-epoch
        // regardless of input TimeUnit (ms / us / ns).  Without this, a
        // `Datetime[ms]` column would silently miscompute speed.
        let time_col_data = group.column(time_col)?;
        let tz = match time_col_data.dtype() {
            DataType::Datetime(_, tz) => tz.clone(),
            other => {
                return Err(PolarsError::ComputeError(
                    format!("'{time_col}' is not a datetime column (got {other})").into(),
                ));
            }
        };
        let times_col = time_col_data
            .cast(&DataType::Datetime(TimeUnit::Microseconds, tz))?
            .cast(&DataType::Int64)?;
        let times = times_col.i64()?;

        let mut keep = vec![true; n];
        // Index of the last kept point with non-null (time, lat, lon).
        let mut anchor: Option<usize> = None;

        for (i, slot) in keep.iter_mut().enumerate().take(n) {
            let (t, lat, lon) = match (times.get(i), lats.get(i), lons.get(i)) {
                (Some(t), Some(lat), Some(lon)) => (t, lat, lon),
                _ => continue, // null — keep row, don't advance anchor
            };

            let Some(anchor_idx) = anchor else {
                // First valid point in this group — always keep; set anchor.
                anchor = Some(i);
                continue;
            };

            // Safe unwraps: anchor points to a row that was accepted above.
            let t_prev = times.get(anchor_idx).unwrap();
            let lat_prev = lats.get(anchor_idx).unwrap();
            let lon_prev = lons.get(anchor_idx).unwrap();

            let dt_us = t - t_prev;
            if dt_us <= 0 {
                // Equal timestamps (or non-monotonic input) — can't compute
                // a speed.  Keep the row but hold the anchor.
                continue;
            }

            let dist_m = haversine_distance_meters(lat_prev, lon_prev, lat, lon);
            let dt_s = dt_us as f64 * 1e-6;
            let implied_mps = dist_m / dt_s;

            if implied_mps > max_speed_mps {
                *slot = false;
                // Anchor unchanged: next point still compared to the last clean fix.
            } else {
                anchor = Some(i);
            }
        }

        let mask = BooleanChunked::from_slice(PlSmallStr::from_static("_keep"), &keep);
        result_frames.push(group.filter(&mask)?);
    }

    if result_frames.is_empty() {
        Ok(sorted.head(Some(0)))
    } else {
        let lfs: Vec<LazyFrame> = result_frames.into_iter().map(|f| f.lazy()).collect();
        concat(lfs, UnionArgs::default())?.collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 200 km/h as m/s.
    const MAX_MPS: f64 = 200.0 / 3.6;

    fn make_df(ids: Vec<&str>, times: Vec<i64>, lats: Vec<f64>, lons: Vec<f64>) -> DataFrame {
        let n = ids.len();
        DataFrame::new(
            n,
            vec![
                Column::new("id".into(), ids),
                Series::new("time".into(), times)
                    .cast(&DataType::Datetime(TimeUnit::Microseconds, None))
                    .unwrap()
                    .into_column(),
                Column::new("lat".into(), lats),
                Column::new("lon".into(), lons),
            ],
        )
        .unwrap()
    }

    #[test]
    fn keeps_plausible_trajectory() {
        // ~111 m between rows over 60 s → ~1.85 m/s, well under threshold.
        let df = make_df(
            vec!["a", "a", "a"],
            vec![0, 60_000_000, 120_000_000],
            vec![43.0, 43.001, 43.002],
            vec![-79.0, -79.0, -79.0],
        );
        let out = filter_impossible_speeds(df, "id", "time", "lat", "lon", MAX_MPS).unwrap();
        assert_eq!(out.height(), 3);
    }

    #[test]
    fn drops_single_spike() {
        // Row 1 jumps ~111 km in 1 s → ~400,000 km/h, clearly impossible.
        let df = make_df(
            vec!["a", "a", "a"],
            vec![0, 1_000_000, 2_000_000],
            vec![43.0, 44.0, 43.00001],
            vec![-79.0, -79.0, -79.0],
        );
        let out = filter_impossible_speeds(df, "id", "time", "lat", "lon", MAX_MPS).unwrap();
        assert_eq!(out.height(), 2);
        let lats = out.column("lat").unwrap().f64().unwrap();
        assert_eq!(lats.get(0), Some(43.0));
        assert_eq!(lats.get(1), Some(43.00001));
    }

    #[test]
    fn consecutive_spikes_all_drop() {
        // Two impossible jumps in a row — both should collapse against row 0,
        // so only rows 0 and 3 survive.
        let df = make_df(
            vec!["a", "a", "a", "a"],
            vec![0, 1_000_000, 2_000_000, 3_000_000],
            vec![43.0, 44.0, 45.0, 43.00002],
            vec![-79.0, -79.0, -79.0, -79.0],
        );
        let out = filter_impossible_speeds(df, "id", "time", "lat", "lon", MAX_MPS).unwrap();
        assert_eq!(out.height(), 2);
        let lats = out.column("lat").unwrap().f64().unwrap();
        assert_eq!(lats.get(0), Some(43.0));
        assert_eq!(lats.get(1), Some(43.00002));
    }

    #[test]
    fn null_coords_pass_through() {
        // Null lat on row 1: kept, doesn't advance anchor.  Row 2 is still
        // compared to row 0 — ~222 m / 120 s ≈ 1.85 m/s, kept.
        let df = DataFrame::new(
            3,
            vec![
                Column::new("id".into(), vec!["a", "a", "a"]),
                Series::new("time".into(), vec![0i64, 60_000_000, 120_000_000])
                    .cast(&DataType::Datetime(TimeUnit::Microseconds, None))
                    .unwrap()
                    .into_column(),
                Series::new("lat".into(), &[Some(43.0), None, Some(43.002)])
                    .cast(&DataType::Float64)
                    .unwrap()
                    .into_column(),
                Series::new("lon".into(), &[Some(-79.0), Some(-79.0), Some(-79.0)])
                    .cast(&DataType::Float64)
                    .unwrap()
                    .into_column(),
            ],
        )
        .unwrap();
        let out = filter_impossible_speeds(df, "id", "time", "lat", "lon", MAX_MPS).unwrap();
        assert_eq!(out.height(), 3);
    }

    #[test]
    fn trucks_are_independent() {
        // Truck a has a spike; truck b is clean.  Sorting puts all of a then
        // all of b; filter runs per-group.
        let df = make_df(
            vec!["a", "a", "b", "b"],
            vec![0, 1_000_000, 0, 60_000_000],
            vec![43.0, 44.0, 50.0, 50.001],
            vec![-79.0, -79.0, -100.0, -100.0],
        );
        let out = filter_impossible_speeds(df, "id", "time", "lat", "lon", MAX_MPS).unwrap();
        // a: row 0 kept, row 1 (spike) dropped.  b: both kept.
        assert_eq!(out.height(), 3);
        let ids = out.column("id").unwrap().str().unwrap();
        let mut counts = std::collections::HashMap::<&str, usize>::new();
        for i in 0..out.height() {
            *counts.entry(ids.get(i).unwrap()).or_insert(0) += 1;
        }
        assert_eq!(counts["a"], 1);
        assert_eq!(counts["b"], 2);
    }

    #[test]
    fn first_point_kept_even_alone() {
        let df = make_df(vec!["a"], vec![0], vec![43.0], vec![-79.0]);
        let out = filter_impossible_speeds(df, "id", "time", "lat", "lon", MAX_MPS).unwrap();
        assert_eq!(out.height(), 1);
    }

    #[test]
    fn duplicate_timestamp_kept() {
        // dt == 0: can't compute a speed; keep the row, hold the anchor.
        let df = make_df(
            vec!["a", "a"],
            vec![0, 0],
            vec![43.0, 43.001],
            vec![-79.0, -79.0],
        );
        let out = filter_impossible_speeds(df, "id", "time", "lat", "lon", MAX_MPS).unwrap();
        assert_eq!(out.height(), 2);
    }

    #[test]
    fn empty_input_returns_empty() {
        let df = make_df(vec![], vec![], vec![], vec![]);
        let out = filter_impossible_speeds(df, "id", "time", "lat", "lon", MAX_MPS).unwrap();
        assert_eq!(out.height(), 0);
    }
}
