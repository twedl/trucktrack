use std::collections::VecDeque;

use polars::prelude::*;

/// Drop stale GPS pings: verbatim re-emissions of an earlier record.
///
/// Some devices buffer readings and occasionally re-emit an older record
/// later in the stream with only the timestamp advanced.  The result is a
/// sequence like ``T1 → T2 → T3`` where ``T3`` has the same
/// ``(lat, lon, speed, heading)`` as ``T1`` but a later time — i.e. the
/// truck appears to teleport back to an earlier location.
///
/// This filter walks each truck's timeline and, for every row, compares its
/// ``(lat, lon, speed, heading)`` tuple bit-for-bit against a ring buffer of
/// the last ``window`` distinct rows.  A match means the row is a stale
/// re-emission and is dropped.  Float comparison is bit-exact
/// (``f64::to_bits``): the error mode is verbatim re-emission, not rounding
/// drift, and exact equality avoids false positives when a truck
/// legitimately returns to a previously visited location (it only gets
/// dropped if the window still contains it).
///
/// Input is sorted by ``(id_col, time_col)`` before filtering, so output is
/// ``(id, time)``-sorted regardless of input order.  Tz-aware and tz-naive
/// time columns are both handled — only the sort key cares about type.
///
/// Rows with a null in any of the four fields never match (treated as
/// distinct) and pass through.  Rows with ``speed == 0`` are also exempt:
/// a stopped truck legitimately emits many repeated identical pings, so
/// zero-speed rows are neither checked against the buffer nor added to it.
#[allow(clippy::too_many_arguments)]
pub fn filter_stale_pings(
    df: DataFrame,
    id_col: &str,
    time_col: &str,
    lat_col: &str,
    lon_col: &str,
    speed_col: &str,
    heading_col: &str,
    window: usize,
) -> PolarsResult<DataFrame> {
    if window == 0 || df.height() == 0 {
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
        let speeds = group.column(speed_col)?.f64()?;
        let headings = group.column(heading_col)?.f64()?;

        let mut keep = vec![true; n];
        let mut buf: VecDeque<(u64, u64, u64, u64)> = VecDeque::with_capacity(window);

        for (i, slot) in keep.iter_mut().enumerate().take(n) {
            let (lat, lon, spd, hdg) =
                match (lats.get(i), lons.get(i), speeds.get(i), headings.get(i)) {
                    (Some(a), Some(b), Some(c), Some(d)) => (a, b, c, d),
                    _ => continue, // null in any field → can't match, skip buffer update
                };

            // Stopped trucks legitimately emit repeated identical pings.
            // Don't flag them as stale, and don't let them poison the buffer
            // (otherwise a later moving ping at the same coords could be
            // compared against a stationary sample and spuriously dropped).
            if spd == 0.0 {
                continue;
            }

            let key = (lat.to_bits(), lon.to_bits(), spd.to_bits(), hdg.to_bits());

            if buf.iter().any(|k| *k == key) {
                *slot = false;
                continue;
            }

            if buf.len() == window {
                buf.pop_front();
            }
            buf.push_back(key);
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

    fn make_df(
        ids: Vec<&str>,
        times: Vec<i64>,
        lats: Vec<f64>,
        lons: Vec<f64>,
        speeds: Vec<f64>,
        headings: Vec<f64>,
    ) -> DataFrame {
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
                Column::new("speed".into(), speeds),
                Column::new("heading".into(), headings),
            ],
        )
        .unwrap()
    }

    #[test]
    fn drops_verbatim_reemission() {
        // T1 → T2 → T3 where T3 has identical (lat,lon,speed,heading) to T1.
        let df = make_df(
            vec!["a", "a", "a"],
            vec![0, 60_000_000, 63_000_000],
            vec![43.0, 43.001, 43.0],
            vec![-79.0, -79.0, -79.0],
            vec![55.0, 56.0, 55.0],
            vec![90.0, 91.0, 90.0],
        );

        let out =
            filter_stale_pings(df, "id", "time", "lat", "lon", "speed", "heading", 5).unwrap();
        assert_eq!(out.height(), 2);
        let times = out.column("time").unwrap().cast(&DataType::Int64).unwrap();
        let t = times.i64().unwrap();
        assert_eq!(t.get(0), Some(0));
        assert_eq!(t.get(1), Some(60_000_000));
    }

    #[test]
    fn keeps_clean_monotonic() {
        let df = make_df(
            vec!["a", "a", "a"],
            vec![0, 60_000_000, 120_000_000],
            vec![43.0, 43.001, 43.002],
            vec![-79.0, -79.0, -79.0],
            vec![55.0, 56.0, 57.0],
            vec![90.0, 91.0, 92.0],
        );

        let out =
            filter_stale_pings(df, "id", "time", "lat", "lon", "speed", "heading", 5).unwrap();
        assert_eq!(out.height(), 3);
    }

    #[test]
    fn legitimate_return_outside_window_kept() {
        // Truck parks, drives away, returns to same coords — but the first
        // occurrence has fallen out of the lookback window.
        let df = make_df(
            vec!["a", "a", "a", "a", "a", "a"],
            vec![0, 1, 2, 3, 4, 5],
            vec![43.0, 43.1, 43.2, 43.3, 43.4, 43.0],
            vec![-79.0, -79.1, -79.2, -79.3, -79.4, -79.0],
            vec![0.0, 30.0, 40.0, 50.0, 30.0, 0.0],
            vec![0.0, 90.0, 90.0, 90.0, 270.0, 0.0],
        );

        // window=3: only last 3 rows are compared; row 5 matches row 0 but 0 is evicted.
        let out =
            filter_stale_pings(df, "id", "time", "lat", "lon", "speed", "heading", 3).unwrap();
        assert_eq!(out.height(), 6);
    }

    #[test]
    fn trucks_are_independent() {
        // Truck b's record has the same coords as truck a's T1 — must not be dropped.
        let df = make_df(
            vec!["a", "b", "a"],
            vec![0, 0, 60_000_000],
            vec![43.0, 43.0, 43.0],
            vec![-79.0, -79.0, -79.0],
            vec![55.0, 55.0, 55.0],
            vec![90.0, 90.0, 90.0],
        );

        // Within truck a: T1 and T3 match, T3 dropped. Truck b kept.
        let out =
            filter_stale_pings(df, "id", "time", "lat", "lon", "speed", "heading", 5).unwrap();
        assert_eq!(out.height(), 2);
        let ids = out.column("id").unwrap().str().unwrap();
        let id_set: std::collections::HashSet<&str> =
            (0..out.height()).map(|i| ids.get(i).unwrap()).collect();
        assert!(id_set.contains("a"));
        assert!(id_set.contains("b"));
    }

    #[test]
    fn stopped_truck_repeats_kept() {
        // speed=0 across multiple consecutive pings must not be flagged stale.
        let df = make_df(
            vec!["a", "a", "a", "a"],
            vec![0, 60_000_000, 120_000_000, 180_000_000],
            vec![43.0, 43.0, 43.0, 43.0],
            vec![-79.0, -79.0, -79.0, -79.0],
            vec![0.0, 0.0, 0.0, 0.0],
            vec![0.0, 0.0, 0.0, 0.0],
        );

        let out =
            filter_stale_pings(df, "id", "time", "lat", "lon", "speed", "heading", 5).unwrap();
        assert_eq!(out.height(), 4);
    }
}
