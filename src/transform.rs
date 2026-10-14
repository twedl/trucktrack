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

/// Apply `f` to a DataFrame shared from Python, without the GIL.
///
/// Every `*_df` wrapper goes through here so Python threads (the split
/// pipeline's `ThreadPoolExecutor`) run Rust work in parallel.
pub fn transform_df(
    py: Python<'_>,
    df: PyDataFrame,
    f: impl FnOnce(DataFrame) -> PolarsResult<DataFrame> + Send,
) -> PyResult<PyDataFrame> {
    let df: DataFrame = df.into();
    let result = py.detach(|| f(df)).map_err(polars_err)?;
    Ok(PyDataFrame(result))
}

/// Read `input_path`, apply `f`, write `output_path`, all without the GIL.
/// Returns the number of rows written.
pub fn transform_file(
    py: Python<'_>,
    input_path: &str,
    output_path: &str,
    f: impl FnOnce(DataFrame) -> PolarsResult<DataFrame> + Send,
) -> PyResult<usize> {
    py.detach(|| {
        let in_file = std::fs::File::open(input_path).map_err(io_err)?;
        let df = ParquetReader::new(in_file).finish().map_err(polars_err)?;
        let mut result = f(df).map_err(polars_err)?;
        let n = result.height();
        let out_file = std::fs::File::create(output_path).map_err(io_err)?;
        ParquetWriter::new(out_file)
            .finish(&mut result)
            .map_err(polars_err)?;
        Ok(n)
    })
}

/// Add a `speed_mps` column (km/h -> m/s) to a DataFrame.
pub fn add_speed_mps(df: DataFrame) -> PolarsResult<DataFrame> {
    df.lazy()
        .with_column((col("speed") * lit(1000.0_f64 / 3600.0_f64)).alias("speed_mps"))
        .collect()
}

/// Read a parquet file, compute derived columns, write results to a new parquet file.
#[pyfunction]
pub fn process_tracks_file(py: Python<'_>, input_path: &str, output_path: &str) -> PyResult<usize> {
    transform_file(py, input_path, output_path, add_speed_mps)
}

/// Accept a Polars DataFrame from Python via the Arrow C Data Interface,
/// process in Rust, and return a Polars DataFrame the same way (zero-copy).
#[pyfunction]
pub fn tracks_from_df(py: Python<'_>, df: PyDataFrame) -> PyResult<PyDataFrame> {
    transform_df(py, df, add_speed_mps)
}
