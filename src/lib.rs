pub mod bucket;
pub mod geo;
pub mod splitters;
pub mod transform;

use pyo3::prelude::*;
use pyo3_polars::PyDataFrame;

use transform::{transform_df, transform_file};

// Every wrapper runs its Rust work without the GIL (see `transform_df` /
// `transform_file`), so Python threads process frames in parallel.

// ── Splitter PyO3 wrappers ──────────────────────────────────────────────

#[pyfunction]
fn split_by_gap_df(
    py: Python<'_>,
    df: PyDataFrame,
    id_col: &str,
    time_col: &str,
    gap_us: i64,
    min_length: usize,
) -> PyResult<PyDataFrame> {
    transform_df(py, df, |df| {
        splitters::gap::split_by_observation_gap(df, id_col, time_col, gap_us, min_length)
    })
}

#[pyfunction]
fn split_by_gap_file(
    py: Python<'_>,
    input_path: &str,
    output_path: &str,
    id_col: &str,
    time_col: &str,
    gap_us: i64,
    min_length: usize,
) -> PyResult<usize> {
    transform_file(py, input_path, output_path, |df| {
        splitters::gap::split_by_observation_gap(df, id_col, time_col, gap_us, min_length)
    })
}

#[pyfunction]
#[pyo3(signature = (df, id_col, time_col, lat_col, lon_col, max_diameter_m, min_duration_us, min_length))]
#[allow(clippy::too_many_arguments)]
fn split_by_stops_df(
    py: Python<'_>,
    df: PyDataFrame,
    id_col: &str,
    time_col: &str,
    lat_col: &str,
    lon_col: &str,
    max_diameter_m: f64,
    min_duration_us: i64,
    min_length: usize,
) -> PyResult<PyDataFrame> {
    transform_df(py, df, |df| {
        splitters::stop::split_by_stops(
            df,
            id_col,
            time_col,
            lat_col,
            lon_col,
            max_diameter_m,
            min_duration_us,
            min_length,
        )
    })
}

#[pyfunction]
#[pyo3(signature = (input_path, output_path, id_col, time_col, lat_col, lon_col, max_diameter_m, min_duration_us, min_length))]
#[allow(clippy::too_many_arguments)]
fn split_by_stops_file(
    py: Python<'_>,
    input_path: &str,
    output_path: &str,
    id_col: &str,
    time_col: &str,
    lat_col: &str,
    lon_col: &str,
    max_diameter_m: f64,
    min_duration_us: i64,
    min_length: usize,
) -> PyResult<usize> {
    transform_file(py, input_path, output_path, |df| {
        splitters::stop::split_by_stops(
            df,
            id_col,
            time_col,
            lat_col,
            lon_col,
            max_diameter_m,
            min_duration_us,
            min_length,
        )
    })
}

// ── Traffic filter PyO3 wrappers ────────────────────────────────────────

#[pyfunction]
#[pyo3(signature = (df, id_col, lat_col, lon_col, max_angle_change, min_distance_m))]
fn filter_traffic_stops_df(
    py: Python<'_>,
    df: PyDataFrame,
    id_col: &str,
    lat_col: &str,
    lon_col: &str,
    max_angle_change: f64,
    min_distance_m: f64,
) -> PyResult<PyDataFrame> {
    transform_df(py, df, |df| {
        splitters::traffic::filter_traffic_stops(
            df,
            id_col,
            lat_col,
            lon_col,
            max_angle_change,
            min_distance_m,
        )
    })
}

#[pyfunction]
#[pyo3(signature = (input_path, output_path, id_col, lat_col, lon_col, max_angle_change, min_distance_m))]
#[allow(clippy::too_many_arguments)]
fn filter_traffic_stops_file(
    py: Python<'_>,
    input_path: &str,
    output_path: &str,
    id_col: &str,
    lat_col: &str,
    lon_col: &str,
    max_angle_change: f64,
    min_distance_m: f64,
) -> PyResult<usize> {
    transform_file(py, input_path, output_path, |df| {
        splitters::traffic::filter_traffic_stops(
            df,
            id_col,
            lat_col,
            lon_col,
            max_angle_change,
            min_distance_m,
        )
    })
}

// ── Stale-ping filter PyO3 wrappers ─────────────────────────────────────

#[pyfunction]
#[pyo3(signature = (df, id_col, time_col, lat_col, lon_col, speed_col, heading_col, window))]
#[allow(clippy::too_many_arguments)]
fn filter_stale_pings_df(
    py: Python<'_>,
    df: PyDataFrame,
    id_col: &str,
    time_col: &str,
    lat_col: &str,
    lon_col: &str,
    speed_col: &str,
    heading_col: &str,
    window: usize,
) -> PyResult<PyDataFrame> {
    transform_df(py, df, |df| {
        splitters::stale::filter_stale_pings(
            df,
            id_col,
            time_col,
            lat_col,
            lon_col,
            speed_col,
            heading_col,
            window,
        )
    })
}

#[pyfunction]
#[pyo3(signature = (input_path, output_path, id_col, time_col, lat_col, lon_col, speed_col, heading_col, window))]
#[allow(clippy::too_many_arguments)]
fn filter_stale_pings_file(
    py: Python<'_>,
    input_path: &str,
    output_path: &str,
    id_col: &str,
    time_col: &str,
    lat_col: &str,
    lon_col: &str,
    speed_col: &str,
    heading_col: &str,
    window: usize,
) -> PyResult<usize> {
    transform_file(py, input_path, output_path, |df| {
        splitters::stale::filter_stale_pings(
            df,
            id_col,
            time_col,
            lat_col,
            lon_col,
            speed_col,
            heading_col,
            window,
        )
    })
}

// ── Impossible-speed filter PyO3 wrappers ──────────────────────────────

#[pyfunction]
#[pyo3(signature = (df, id_col, time_col, lat_col, lon_col, max_speed_mps))]
fn filter_impossible_speeds_df(
    py: Python<'_>,
    df: PyDataFrame,
    id_col: &str,
    time_col: &str,
    lat_col: &str,
    lon_col: &str,
    max_speed_mps: f64,
) -> PyResult<PyDataFrame> {
    transform_df(py, df, |df| {
        splitters::speed::filter_impossible_speeds(
            df,
            id_col,
            time_col,
            lat_col,
            lon_col,
            max_speed_mps,
        )
    })
}

#[pyfunction]
#[pyo3(signature = (input_path, output_path, id_col, time_col, lat_col, lon_col, max_speed_mps))]
#[allow(clippy::too_many_arguments)]
fn filter_impossible_speeds_file(
    py: Python<'_>,
    input_path: &str,
    output_path: &str,
    id_col: &str,
    time_col: &str,
    lat_col: &str,
    lon_col: &str,
    max_speed_mps: f64,
) -> PyResult<usize> {
    transform_file(py, input_path, output_path, |df| {
        splitters::speed::filter_impossible_speeds(
            df,
            id_col,
            time_col,
            lat_col,
            lon_col,
            max_speed_mps,
        )
    })
}

// ── Module registration ─────────────────────────────────────────────────

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(transform::process_tracks_file, m)?)?;
    m.add_function(wrap_pyfunction!(transform::tracks_from_df, m)?)?;
    m.add_function(wrap_pyfunction!(split_by_gap_df, m)?)?;
    m.add_function(wrap_pyfunction!(split_by_gap_file, m)?)?;
    m.add_function(wrap_pyfunction!(split_by_stops_df, m)?)?;
    m.add_function(wrap_pyfunction!(split_by_stops_file, m)?)?;
    m.add_function(wrap_pyfunction!(filter_traffic_stops_df, m)?)?;
    m.add_function(wrap_pyfunction!(filter_traffic_stops_file, m)?)?;
    m.add_function(wrap_pyfunction!(filter_stale_pings_df, m)?)?;
    m.add_function(wrap_pyfunction!(filter_stale_pings_file, m)?)?;
    m.add_function(wrap_pyfunction!(filter_impossible_speeds_df, m)?)?;
    m.add_function(wrap_pyfunction!(filter_impossible_speeds_file, m)?)?;
    m.add_function(wrap_pyfunction!(geo::haversine_km, m)?)?;
    m.add_function(wrap_pyfunction!(bucket::py_bucket_of, m)?)?;
    m.add_function(wrap_pyfunction!(bucket::bucket_ids, m)?)?;
    m.add("BUCKET_HASH_VERSION", bucket::HASH_VERSION)?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
