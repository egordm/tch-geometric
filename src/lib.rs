use pyo3::prelude::*;
use tch::Tensor;

#[pyfunction]
fn sum_as_string(a: Tensor, b: Tensor) -> PyResult<Tensor> {
    Ok(a + &b)
}

#[pymodule]
fn tch_geometric(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(sum_as_string, m)?)?;
    Ok(())
}