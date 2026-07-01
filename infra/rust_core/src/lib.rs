use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

mod math_ops;

#[pyfunction]
fn calculate_ofi<'py>(
    py: Python<'py>,
    high: PyReadonlyArray1<'py, f64>,
    low: PyReadonlyArray1<'py, f64>,
    close: PyReadonlyArray1<'py, f64>,
    volume: PyReadonlyArray1<'py, f64>,
) -> &'py PyArray1<f64> {
    let result = math_ops::orderflow::calculate_ofi(
        high.as_slice().expect("high must be contiguous"),
        low.as_slice().expect("low must be contiguous"),
        close.as_slice().expect("close must be contiguous"),
        volume.as_slice().expect("volume must be contiguous"),
    );
    result.into_pyarray(py)
}

#[pyfunction]
fn rolling_volatility<'py>(
    py: Python<'py>,
    values: PyReadonlyArray1<'py, f64>,
    window: usize,
) -> &'py PyArray1<f64> {
    let result = math_ops::rolling_vol::rolling_volatility(
        values.as_slice().expect("values must be contiguous"),
        window,
    );
    result.into_pyarray(py)
}

#[pymodule]
fn rust_core(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(calculate_ofi, module)?)?;
    module.add_function(wrap_pyfunction!(rolling_volatility, module)?)?;
    Ok(())
}

