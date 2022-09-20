use crate::primitives::QuantumState;
use numpy::{
  PyArray1,
  PyArray2,
  PyReadonlyArray2,
};
use pyo3::{
  Python,
  PyResult,
  pyfunction
};
use num::Complex;
use rayon::prelude::*;

fn conj_and_double(state: &PyArray1<Complex<f64>>) {
  unsafe { state.as_slice_mut().unwrap() }
    .into_par_iter()
    .for_each(|x| { *x = 2. * x.conj() })
}

#[pyfunction]
pub fn _q2_density_matrix<'py>(
  state: &PyArray1<Complex<f64>>,
  pos2: usize,
  pos1: usize,
  py: Python<'py>,
) -> PyResult<&'py PyArray2<Complex<f64>>> {
  state.get_q2_density_matrix(pos2, pos1, py)
}

#[pyfunction]
pub fn grad_wrt_q2_density_matrix(
  state_to_grad: &PyArray1<Complex<f64>>,
  grad: PyReadonlyArray2<Complex<f64>>,
  pos2: usize,
  pos1: usize,
) -> PyResult<()> {
  conj_and_double(state_to_grad);
  state_to_grad.apply_q2_gate_transposed(grad, pos2, pos1)?;
  Ok(())
}

#[pyfunction]
pub fn _q1_density_matrix<'py>(
  state: &PyArray1<Complex<f64>>,
  pos: usize,
  py: Python<'py>,
) -> PyResult<&'py PyArray2<Complex<f64>>> {
  state.get_q1_density_matrix(pos, py)
}

#[pyfunction]
pub fn grad_wrt_q1_density_matrix(
  state_to_grad: &PyArray1<Complex<f64>>,
  grad: PyReadonlyArray2<Complex<f64>>,
  pos: usize,
) -> PyResult<()> {
  conj_and_double(state_to_grad);
  state_to_grad.apply_q1_gate_transposed(grad, pos)?;
  Ok(())
}