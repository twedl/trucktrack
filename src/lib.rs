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

// ── Module registration ─────────────────────────────────────────────────

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(transform::process_tracks_file, m)?)?;
    m.add_function(wrap_pyfunction!(transform::tracks_from_df, m)?)?;
    m.add_function(wrap_pyfunction!(split_by_gap_df, m)?)?;
    m.add_function(wrap_pyfunction!(split_by_gap_file, m)?)?;
    m.add_function(wrap_pyfunction!(split_by_stops_df, m)?)?;
    m.add_function(wrap_pyfunction!(split_by_stops_file, m)?)?;
    m.add_function(wrap_pyfunction!(partition::haversine_km, m)?)?;
    m.add_function(wrap_pyfunction!(partition::valhalla_tile_id, m)?)?;
    m.add_function(wrap_pyfunction!(partition::classify_and_partition_key, m)?)?;
    m.add_function(wrap_pyfunction!(partition::hilbert_indices, m)?)?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
