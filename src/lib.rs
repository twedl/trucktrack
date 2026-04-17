pub mod geo;
pub mod partition;
pub mod splitters;
pub mod transform;

use polars::prelude::*;
use pyo3::prelude::*;
use pyo3_polars::PyDataFrame;

use transform::{io_err, polars_err};

// ── Splitter PyO3 wrappers ──────────────────────────────────────────────

#[pyfunction]
fn split_by_gap_df(
    df: PyDataFrame,
    id_col: &str,
    time_col: &str,
    gap_us: i64,
    min_length: usize,
) -> PyResult<PyDataFrame> {
    let result =
        splitters::gap::split_by_observation_gap(df.into(), id_col, time_col, gap_us, min_length)
            .map_err(polars_err)?;
    Ok(PyDataFrame(result))
}

#[pyfunction]
fn split_by_gap_file(
    input_path: &str,
    output_path: &str,
    id_col: &str,
    time_col: &str,
    gap_us: i64,
    min_length: usize,
) -> PyResult<usize> {
    let in_file = std::fs::File::open(input_path).map_err(io_err)?;
    let df = ParquetReader::new(in_file).finish().map_err(polars_err)?;

    let mut result =
        splitters::gap::split_by_observation_gap(df, id_col, time_col, gap_us, min_length)
            .map_err(polars_err)?;

    let n = result.height();
    let out_file = std::fs::File::create(output_path).map_err(io_err)?;
    ParquetWriter::new(out_file)
        .finish(&mut result)
        .map_err(polars_err)?;
    Ok(n)
}

#[pyfunction]
#[pyo3(signature = (df, id_col, time_col, lat_col, lon_col, max_diameter_m, min_duration_us, min_length))]
#[allow(clippy::too_many_arguments)]
fn split_by_stops_df(
    df: PyDataFrame,
    id_col: &str,
    time_col: &str,
    lat_col: &str,
    lon_col: &str,
    max_diameter_m: f64,
    min_duration_us: i64,
    min_length: usize,
) -> PyResult<PyDataFrame> {
    let result = splitters::stop::split_by_stops(
        df.into(),
        id_col,
        time_col,
        lat_col,
        lon_col,
        max_diameter_m,
        min_duration_us,
        min_length,
    )
    .map_err(polars_err)?;
    Ok(PyDataFrame(result))
}

#[pyfunction]
#[pyo3(signature = (input_path, output_path, id_col, time_col, lat_col, lon_col, max_diameter_m, min_duration_us, min_length))]
#[allow(clippy::too_many_arguments)]
fn split_by_stops_file(
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
    let in_file = std::fs::File::open(input_path).map_err(io_err)?;
    let df = ParquetReader::new(in_file).finish().map_err(polars_err)?;

    let mut result = splitters::stop::split_by_stops(
        df,
        id_col,
        time_col,
        lat_col,
        lon_col,
        max_diameter_m,
        min_duration_us,
        min_length,
    )
    .map_err(polars_err)?;

    let n = result.height();
    let out_file = std::fs::File::create(output_path).map_err(io_err)?;
    ParquetWriter::new(out_file)
        .finish(&mut result)
        .map_err(polars_err)?;
    Ok(n)
}

// ── Traffic filter PyO3 wrappers ────────────────────────────────────────

#[pyfunction]
#[pyo3(signature = (df, id_col, lat_col, lon_col, max_angle_change, min_distance_m))]
fn filter_traffic_stops_df(
    df: PyDataFrame,
    id_col: &str,
    lat_col: &str,
    lon_col: &str,
    max_angle_change: f64,
    min_distance_m: f64,
) -> PyResult<PyDataFrame> {
    let result = splitters::traffic::filter_traffic_stops(
        df.into(),
        id_col,
        lat_col,
        lon_col,
        max_angle_change,
        min_distance_m,
    )
    .map_err(polars_err)?;
    Ok(PyDataFrame(result))
}

#[pyfunction]
#[pyo3(signature = (input_path, output_path, id_col, lat_col, lon_col, max_angle_change, min_distance_m))]
fn filter_traffic_stops_file(
    input_path: &str,
    output_path: &str,
    id_col: &str,
    lat_col: &str,
    lon_col: &str,
    max_angle_change: f64,
    min_distance_m: f64,
) -> PyResult<usize> {
    let in_file = std::fs::File::open(input_path).map_err(io_err)?;
    let df = ParquetReader::new(in_file).finish().map_err(polars_err)?;

    let mut result = splitters::traffic::filter_traffic_stops(
        df,
        id_col,
        lat_col,
        lon_col,
        max_angle_change,
        min_distance_m,
    )
    .map_err(polars_err)?;

    let n = result.height();
    let out_file = std::fs::File::create(output_path).map_err(io_err)?;
    ParquetWriter::new(out_file)
        .finish(&mut result)
        .map_err(polars_err)?;
    Ok(n)
}

// ── Stale-ping filter PyO3 wrappers ─────────────────────────────────────

#[pyfunction]
#[pyo3(signature = (df, id_col, time_col, lat_col, lon_col, speed_col, heading_col, window))]
#[allow(clippy::too_many_arguments)]
fn filter_stale_pings_df(
    df: PyDataFrame,
    id_col: &str,
    time_col: &str,
    lat_col: &str,
    lon_col: &str,
    speed_col: &str,
    heading_col: &str,
    window: usize,
) -> PyResult<PyDataFrame> {
    let result = splitters::stale::filter_stale_pings(
        df.into(),
        id_col,
        time_col,
        lat_col,
        lon_col,
        speed_col,
        heading_col,
        window,
    )
    .map_err(polars_err)?;
    Ok(PyDataFrame(result))
}

#[pyfunction]
#[pyo3(signature = (input_path, output_path, id_col, time_col, lat_col, lon_col, speed_col, heading_col, window))]
#[allow(clippy::too_many_arguments)]
fn filter_stale_pings_file(
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
    let in_file = std::fs::File::open(input_path).map_err(io_err)?;
    let df = ParquetReader::new(in_file).finish().map_err(polars_err)?;

    let mut result = splitters::stale::filter_stale_pings(
        df,
        id_col,
        time_col,
        lat_col,
        lon_col,
        speed_col,
        heading_col,
        window,
    )
    .map_err(polars_err)?;

    let n = result.height();
    let out_file = std::fs::File::create(output_path).map_err(io_err)?;
    ParquetWriter::new(out_file)
        .finish(&mut result)
        .map_err(polars_err)?;
    Ok(n)
}

// ── Impossible-speed filter PyO3 wrappers ──────────────────────────────

#[pyfunction]
#[pyo3(signature = (df, id_col, time_col, lat_col, lon_col, max_speed_mps))]
fn filter_impossible_speeds_df(
    df: PyDataFrame,
    id_col: &str,
    time_col: &str,
    lat_col: &str,
    lon_col: &str,
    max_speed_mps: f64,
) -> PyResult<PyDataFrame> {
    let result = splitters::speed::filter_impossible_speeds(
        df.into(),
        id_col,
        time_col,
        lat_col,
        lon_col,
        max_speed_mps,
    )
    .map_err(polars_err)?;
    Ok(PyDataFrame(result))
}

#[pyfunction]
#[pyo3(signature = (input_path, output_path, id_col, time_col, lat_col, lon_col, max_speed_mps))]
fn filter_impossible_speeds_file(
    input_path: &str,
    output_path: &str,
    id_col: &str,
    time_col: &str,
    lat_col: &str,
    lon_col: &str,
    max_speed_mps: f64,
) -> PyResult<usize> {
    let in_file = std::fs::File::open(input_path).map_err(io_err)?;
    let df = ParquetReader::new(in_file).finish().map_err(polars_err)?;

    let mut result = splitters::speed::filter_impossible_speeds(
        df,
        id_col,
        time_col,
        lat_col,
        lon_col,
        max_speed_mps,
    )
    .map_err(polars_err)?;

    let n = result.height();
    let out_file = std::fs::File::create(output_path).map_err(io_err)?;
    ParquetWriter::new(out_file)
        .finish(&mut result)
        .map_err(polars_err)?;
    Ok(n)
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
    m.add_function(wrap_pyfunction!(partition::haversine_km, m)?)?;
    m.add_function(wrap_pyfunction!(partition::valhalla_tile_id, m)?)?;
    m.add_function(wrap_pyfunction!(partition::classify_and_partition_key, m)?)?;
    m.add_function(wrap_pyfunction!(partition::hilbert_indices, m)?)?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
