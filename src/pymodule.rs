use pyo3::prelude::{
    pymodule,
    Python,
    wrap_pyfunction,
    PyModule,
    PyResult,
};

use crate::primitives::{
    apply_q1_gate,
    apply_q1_gate_transposed,
    apply_q2_gate,
    apply_q2_gate_transposed,
    get_q1_density_matrix,
    get_q2_density_matrix,
    apply_q1_gate_conj_transposed,
    apply_q2_gate_conj_transposed,
    q2_gradient,
    q1_gradient,
    fast_copy,
};
use crate::differentiable_density_matrices::{
    grad_wrt_q1_density_matrix,
    _q1_density_matrix,
    grad_wrt_q2_density_matrix,
    _q2_density_matrix
};
use crate::circuit::Circuit;

#[pymodule]
fn differentiable_circuit(py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(apply_q1_gate, m)?)?;
    m.add_function(wrap_pyfunction!(apply_q2_gate, m)?)?;
    m.add_function(wrap_pyfunction!(apply_q1_gate_transposed, m)?)?;
    m.add_function(wrap_pyfunction!(apply_q2_gate_transposed, m)?)?;
    m.add_function(wrap_pyfunction!(apply_q1_gate_conj_transposed, m)?)?;
    m.add_function(wrap_pyfunction!(apply_q2_gate_conj_transposed, m)?)?;
    m.add_function(wrap_pyfunction!(get_q1_density_matrix, m)?)?;
    m.add_function(wrap_pyfunction!(get_q2_density_matrix, m)?)?;
    m.add_function(wrap_pyfunction!(q2_gradient, m)?)?;
    m.add_function(wrap_pyfunction!(q1_gradient, m)?)?;
    m.add_function(wrap_pyfunction!(_q2_density_matrix, m)?)?;
    m.add_function(wrap_pyfunction!(_q1_density_matrix, m)?)?;
    m.add_function(wrap_pyfunction!(grad_wrt_q2_density_matrix, m)?)?;
    m.add_function(wrap_pyfunction!(grad_wrt_q1_density_matrix, m)?)?;
    m.add_function(wrap_pyfunction!(fast_copy, m)?)?;
    m.add_class::<Circuit>()?;
    Ok(())
}