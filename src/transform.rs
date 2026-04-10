use polars::prelude::*;
use pyo3::exceptions::{PyOSError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3_polars::PyDataFrame;

pub fn polars_err(e: PolarsError) -> PyErr {
    match &e {
        PolarsError::SchemaMismatch(_)
        | PolarsError::SchemaFieldNotFound(_)
        | PolarsError::ColumnNotFound(_)
        | PolarsError::ShapeMismatch(_) => PyValueError::new_err(e.to_string()),
        _ => PyRuntimeError::new_err(e.to_string()),
    }
}

pub fn io_err(e: std::io::Error) -> PyErr {
    PyOSError::new_err(e.to_string())
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

/// Accept a Polars DataFrame from Python via the Arrow C Data Interface,
/// process in Rust, and return a Polars DataFrame the same way (zero-copy).
#[pyfunction]
pub fn tracks_from_df(df: PyDataFrame) -> PyResult<PyDataFrame> {
    let result = add_speed_mps(df.into()).map_err(polars_err)?;
    Ok(PyDataFrame(result))
}
