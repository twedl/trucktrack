use polars::prelude::*;
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

pub fn polars_err(e: PolarsError) -> PyErr {
    PyRuntimeError::new_err(e.to_string())
}

pub fn io_err(e: std::io::Error) -> PyErr {
    PyRuntimeError::new_err(e.to_string())
}

pub fn df_from_ipc(ipc_bytes: &[u8]) -> PolarsResult<DataFrame> {
    let cursor = std::io::Cursor::new(ipc_bytes);
    IpcReader::new(cursor).finish()
}

pub fn df_to_ipc(df: &mut DataFrame) -> PolarsResult<Vec<u8>> {
    let mut out: Vec<u8> = Vec::new();
    IpcWriter::new(&mut out).finish(df)?;
    Ok(out)
}

/// Add a `speed_mps` column (km/h -> m/s) to a DataFrame.
pub fn add_speed_mps(df: DataFrame) -> PolarsResult<DataFrame> {
    df.lazy()
        .with_column((col("speed") * lit(1000.0_f64 / 3600.0_f64)).alias("speed_mps"))
        .collect()
}

/// Read a parquet file, compute derived columns, write results to a new parquet file.
#[pyfunction]
pub fn process_tracks_file(input_path: &str, output_path: &str) -> PyResult<usize> {
    let in_file = std::fs::File::open(input_path).map_err(io_err)?;
    let df = ParquetReader::new(in_file).finish().map_err(polars_err)?;

    let mut result = add_speed_mps(df).map_err(polars_err)?;

    let n = result.height();
    let out_file = std::fs::File::create(output_path).map_err(io_err)?;
    ParquetWriter::new(out_file)
        .finish(&mut result)
        .map_err(polars_err)?;

    Ok(n)
}

/// Accept an Arrow IPC buffer from Python, process in Rust, return an Arrow IPC buffer.
#[pyfunction]
pub fn tracks_from_ipc(ipc_bytes: &[u8]) -> PyResult<Vec<u8>> {
    let df = df_from_ipc(ipc_bytes).map_err(polars_err)?;
    let mut result = add_speed_mps(df).map_err(polars_err)?;
    df_to_ipc(&mut result).map_err(polars_err)
}
