use std::collections::VecDeque;

use pyo3::{
  pyclass,
  pymethods,
  PyResult, PyErr,
  Python,
};
use pyo3::exceptions::PyTypeError;
use numpy::{
  PyReadonlyArray2,
  PyArray2,
  PyArray1,
};
use num::Complex;
use crate::primitives::{
  QuantumState,
  q2_gradient,
  q1_gradient,
};

enum Instruction {
  ConstQ2Gate((usize, usize)),
  VarQ2Gate((usize, usize)),
  ConstQ1Gate(usize),
  VarQ1Gate(usize),
  Q2Density((usize, usize)),
  Q1Density(usize),
}

#[pyclass]
pub struct Circuit {
  instructions: Vec<Instruction>,
}

#[pymethods]
impl Circuit {
  #[new]
  fn new() -> Self {
    Self { instructions: Vec::new() }
  }
  
  fn add_q2_const_gate(&mut self, pos2: usize, pos1: usize) {
    self.instructions.push(Instruction::ConstQ2Gate((pos2, pos1)))
  }

  fn add_q2_var_gate(&mut self, pos2: usize, pos1: usize) {
    self.instructions.push(Instruction::VarQ2Gate((pos2, pos1)))
  }

  fn add_q1_const_gate(&mut self, pos: usize) {
    self.instructions.push(Instruction::ConstQ1Gate(pos))
  }

  fn add_q1_var_gate(&mut self, pos: usize) {
    self.instructions.push(Instruction::VarQ1Gate(pos))
  }

  fn get_q2_dens_op(&mut self, pos2: usize, pos1: usize) {
    self.instructions.push(Instruction::Q2Density((pos2, pos1)))
  }

  fn get_q1_dens_op(&mut self, pos: usize) {
    self.instructions.push(Instruction::Q1Density(pos))
  }

  fn forward<'py>(
    &self,
    state: &PyArray1<Complex<f64>>,
    const_gates: Vec<PyReadonlyArray2<Complex<f64>>>,
    var_gates: Vec<PyReadonlyArray2<Complex<f64>>>,
    py: Python<'py>,
  ) -> PyResult<Vec<&'py PyArray2<Complex<f64>>>> {
    let mut density_matrices = Vec::new();
    let mut const_gates = VecDeque::from(const_gates);
    let mut var_gates = VecDeque::from(var_gates);
    for inst in &self.instructions {
      match inst {
        Instruction::ConstQ1Gate(pos) => {
          let gate = const_gates.pop_front().ok_or(PyErr::new::<PyTypeError, _>("The number of constant gates is less than required."))?;
          state.apply_q1_gate(gate, *pos)?
        },
        Instruction::ConstQ2Gate((pos2, pos1)) => {
          let gate = const_gates.pop_front().ok_or(PyErr::new::<PyTypeError, _>("The number of constant gates is less than required."))?;
          state.apply_q2_gate(gate, *pos2, *pos1)?
        },
        Instruction::VarQ1Gate(pos) => {
          let gate = var_gates.pop_front().ok_or(PyErr::new::<PyTypeError, _>("The number of variable gates is less than required."))?;
          state.apply_q1_gate(gate, *pos)?
        },
        Instruction::VarQ2Gate((pos2, pos1)) => {
          let gate = var_gates.pop_front().ok_or(PyErr::new::<PyTypeError, _>("The number of variable gates is less than required."))?;
          state.apply_q2_gate(gate, *pos2, *pos1)?
        },
        Instruction::Q1Density(pos) => {
          density_matrices.push(state.get_q1_density_matrix(*pos, py)?)
        },
        Instruction::Q2Density((pos2, pos1)) => {
          density_matrices.push(state.get_q2_density_matrix(*pos2, *pos1, py)?)
        },
      }
    }
    if !const_gates.is_empty() { return Err(PyErr::new::<PyTypeError, _>("Number of constant gates is more than required.")) }
    if !var_gates.is_empty() { return Err(PyErr::new::<PyTypeError, _>("Number of variable gates is more than required.")) }
    Ok(density_matrices)
  }

  fn forward_wo_density_matrices<'py>(
    &self,
    state: &PyArray1<Complex<f64>>,
    const_gates: Vec<PyReadonlyArray2<Complex<f64>>>,
    var_gates: Vec<PyReadonlyArray2<Complex<f64>>>,
  ) -> PyResult<()> {
    let mut const_gates = VecDeque::from(const_gates);
    let mut var_gates = VecDeque::from(var_gates);
    for inst in &self.instructions {
      match inst {
        Instruction::ConstQ1Gate(pos) => {
          let gate = const_gates.pop_front().ok_or(PyErr::new::<PyTypeError, _>("The number of constant gates is less than required."))?;
          state.apply_q1_gate(gate, *pos)?
        },
        Instruction::ConstQ2Gate((pos2, pos1)) => {
          let gate = const_gates.pop_front().ok_or(PyErr::new::<PyTypeError, _>("The number of constant gates is less than required."))?;
          state.apply_q2_gate(gate, *pos2, *pos1)?
        },
        Instruction::VarQ1Gate(pos) => {
          let gate = var_gates.pop_front().ok_or(PyErr::new::<PyTypeError, _>("The number of variable gates is less than required."))?;
          state.apply_q1_gate(gate, *pos)?
        },
        Instruction::VarQ2Gate((pos2, pos1)) => {
          let gate = var_gates.pop_front().ok_or(PyErr::new::<PyTypeError, _>("The number of variable gates is less than required."))?;
          state.apply_q2_gate(gate, *pos2, *pos1)?
        },
        Instruction::Q1Density(_) => {},
        Instruction::Q2Density((_, _)) => {},
      }
    }
    if !const_gates.is_empty() { return Err(PyErr::new::<PyTypeError, _>("Number of constant gates is more than required.")) }
    if !var_gates.is_empty() { return Err(PyErr::new::<PyTypeError, _>("Number of variable gates is more than required.")) }
    Ok(())
  }

  fn backward<'py>(
    &self,
    state: &PyArray1<Complex<f64>>,
    grad: &PyArray1<Complex<f64>>,
    mut const_gates: Vec<PyReadonlyArray2<Complex<f64>>>,
    mut var_gates: Vec<PyReadonlyArray2<Complex<f64>>>,
    py: Python<'py>,
  ) -> PyResult<Vec<&'py PyArray2<Complex<f64>>>>
  {
    let mut grads = VecDeque::new();
    for inst in self.instructions.iter().rev() {
      match inst {
        Instruction::ConstQ1Gate(pos) => {
          let gate = const_gates.pop().ok_or(PyErr::new::<PyTypeError, _>("The number of gates is less than required."))?;
          state.apply_q1_gate_conj_transposed(gate.clone(), *pos)?;
          grad.apply_q1_gate_transposed(gate, *pos)?;
        },
        Instruction::ConstQ2Gate((pos2, pos1)) => {
          let gate = const_gates.pop().ok_or(PyErr::new::<PyTypeError, _>("The number of gates is less than required."))?;
          state.apply_q2_gate_conj_transposed(gate.clone(), *pos2, *pos1)?;
          grad.apply_q2_gate_transposed(gate, *pos2, *pos1)?;
        },
        Instruction::VarQ1Gate(pos) => {
          let gate = var_gates.pop().ok_or(PyErr::new::<PyTypeError, _>("The number of gates is less than required."))?;
          state.apply_q1_gate_conj_transposed(gate.clone(), *pos)?;
          grads.push_front(q1_gradient(grad, state, *pos, py)?);
          grad.apply_q1_gate_transposed(gate, *pos)?;
        },
        Instruction::VarQ2Gate((pos2, pos1)) => {
          let gate = var_gates.pop().ok_or(PyErr::new::<PyTypeError, _>("The number of gates is less than required."))?;
          state.apply_q2_gate_conj_transposed(gate.clone(), *pos2, *pos1)?;
          grads.push_front(q2_gradient(grad, state, *pos2, *pos1, py)?);
          grad.apply_q2_gate_transposed(gate, *pos2, *pos1)?;
        },
        Instruction::Q1Density(_) => {},
        Instruction::Q2Density((_, _)) => {},
      }
    }
    if !const_gates.is_empty() { return Err(PyErr::new::<PyTypeError, _>("Number of constant gates is more than required.")) }
    if !var_gates.is_empty() { return Err(PyErr::new::<PyTypeError, _>("Number of constant gates is more than required.")) }
    Ok(Vec::from(grads))
  }
}