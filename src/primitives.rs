use std::ops::Deref;
use numpy::{
  PyArray1,
  PyArray2,
  PyReadonlyArray2,
  PyReadonlyArray1,
};
use pyo3::{
  Python,
  PyErr,
  pyfunction,
  PyResult,
};
use pyo3::exceptions::PyTypeError;
use num::Complex;
use rayon::{prelude::*, ThreadPoolBuilder};
use num_cpus::get_physical;
use std::sync::Mutex;
use std::iter::zip;

/////////////////////////////////////////////////////////////

//check if the given value is a power of 2
fn is_pow_2(val: usize) -> bool {
  (val - 1) & val == 0
}

// this gets number of qubits if size is a power of two
fn get_number_of_qubits(mut size: usize) -> usize {
  let mut pow = 0;
  while size != 1 {
    size = size >> 1;
    pow += 1;
  }
  pow
}

// insert a single zero in the binary representation of a number
fn insert_single_zero(idx: usize, pos: usize) -> usize {
  let mask = usize::MAX << pos;
  ((mask & idx) << 1) | ((!mask) & idx)
}

// insert two zeros in the binary representation of a number
fn insert_two_zeros(
  mut idx: usize,
  pos1: usize,
  pos2: usize,
) -> usize {
  let min_pos = std::cmp::min(pos1, pos2);
  let max_pos = std::cmp::max(pos1, pos2);
  let min_mask = usize::MAX << min_pos;
  let max_mask = usize::MAX << max_pos;
  idx = ((min_mask & idx) << 1) | ((!min_mask) & idx);
  ((max_mask & idx) << 1) | ((!max_mask) & idx)
}

///////////////////////////////////////////////////////////////////

// this is a mut raw ptr wrapper to make it sync
#[derive(Clone, Copy)]
struct MutPtrWrapper<T>(*mut Complex<T>);

impl<T> Deref for MutPtrWrapper<T> {
  type Target = *mut Complex<T>;
  fn deref(&self) -> &Self::Target {
      &self.0
  }
}

unsafe impl<T> Send for MutPtrWrapper<T>{}
unsafe impl<T> Sync for MutPtrWrapper<T>{}

// this is a raw ptr wrapper to make it sync
#[derive(Clone, Copy)]
struct PtrWrapper<T>(*const Complex<T>);

impl<T> Deref for PtrWrapper<T> {
  type Target = *const Complex<T>;
  fn deref(&self) -> &Self::Target {
      &self.0
  }
}

unsafe impl<T> Send for PtrWrapper<T>{}
unsafe impl<T> Sync for PtrWrapper<T>{}

///////////////////////////////////////////////////////////////

pub trait QuantumState {
  fn apply_q2_gate(
    &self,
    gate: PyReadonlyArray2<Complex<f64>>,
    pos2: usize,
    pos1: usize,
  ) -> PyResult<()>;
  fn apply_q2_gate_transposed(
    &self,
    gate: PyReadonlyArray2<Complex<f64>>,
    pos2: usize,
    pos1: usize,
  ) -> PyResult<()>;
  fn apply_q2_gate_conj_transposed(
    &self,
    gate: PyReadonlyArray2<Complex<f64>>,
    pos2: usize,
    pos1: usize,
  ) -> PyResult<()>;
  fn apply_q1_gate(
    &self,
    gate: PyReadonlyArray2<Complex<f64>>,
    pos: usize,
  ) -> PyResult<()>;
  fn apply_q1_gate_transposed(
    &self,
    gate: PyReadonlyArray2<Complex<f64>>,
    pos: usize,
  ) -> PyResult<()>;
  fn apply_q1_gate_conj_transposed(
    &self,
    gate: PyReadonlyArray2<Complex<f64>>,
    pos: usize,
  ) -> PyResult<()>;
  fn get_q2_density_matrix<'py>(
    &self,
    pos2: usize,
    pos1: usize,
    py: Python<'py>,
  ) -> PyResult<&'py PyArray2<Complex<f64>>>;
  fn get_q1_density_matrix<'py>(
    &self,
    pos: usize,
    py: Python<'py>,
  ) -> PyResult<&'py PyArray2<Complex<f64>>>;
}

//TODO: too much code duplication, refactor
impl QuantumState for PyArray1<Complex<f64>> {
  fn apply_q2_gate(
    &self,
    gate: PyReadonlyArray2<Complex<f64>>,
    pos2: usize,
    pos1: usize,
  ) -> PyResult<()>
  {
    if self.len() < 4 { return Err(PyErr::new::<PyTypeError, _>("Number of qubits in a state must be >= 2.")) }
    if !is_pow_2(self.len()) { return Err(PyErr::new::<PyTypeError, _>("The size of the state is not a power of 2.")) };
    if gate.len() != 16 { return Err(PyErr::new::<PyTypeError, _>("Incorrect number of elements in a gate."))};
    if pos2 == pos1 { return Err(PyErr::new::<PyTypeError, _>("pos1 and pos2 must be different."))};
    let qubits_number = get_number_of_qubits(self.len());
    if pos2 >= qubits_number { return Err(PyErr::new::<PyTypeError, _>("pos2 out of range"))};
    if pos1 >= qubits_number { return Err(PyErr::new::<PyTypeError, _>("pos1 out of range"))};
    let buff_ptr = unsafe {
      MutPtrWrapper(
        self.as_slice_mut()
        .expect("Array is not contiguous in memory.")
        .as_mut_ptr()
      )
    };
    let gate_ptr = PtrWrapper(
      gate.as_slice()
        .expect("Array is not contiguous in memory.")
        .as_ptr()
    );
    let stride1 = 2usize.pow(pos1 as u32);
    let stride2 = 2usize.pow(pos2 as u32);
    let batch_size = 2usize.pow(qubits_number as u32 - 2);
    (0..batch_size).into_par_iter()
      .for_each(|bi| {
        let bi = insert_two_zeros(bi, pos1, pos2);
        let mut tmp00 = Complex::new(0., 0.);
        let mut tmp01 = Complex::new(0., 0.);
        let mut tmp10 = Complex::new(0., 0.);
        let mut tmp11 = Complex::new(0., 0.);
        for q in 0..2 {
          for p in 0..2 {
            unsafe {
              tmp00 = tmp00 + *gate_ptr.add(2 * q + p)
                            * *buff_ptr.add(q * stride2 + p * stride1 + bi);
              tmp01 = tmp01 + *gate_ptr.add(4 + 2 * q + p)
                            * *buff_ptr.add(q * stride2 + p * stride1 + bi);
              tmp10 = tmp10 + *gate_ptr.add(8 + 2 * q + p)
                            * *buff_ptr.add(q * stride2 + p * stride1 + bi);
              tmp11 = tmp11 + *gate_ptr.add(12 + 2 * q + p)
                            * *buff_ptr.add(q * stride2 + p * stride1 + bi);
            }
          }
        }
        unsafe {
          *buff_ptr.add(bi) = tmp00;
          *buff_ptr.add(stride1 + bi) = tmp01;
          *buff_ptr.add(stride2 + bi) = tmp10;
          *buff_ptr.add(stride1 + stride2 + bi) = tmp11;
        }
      });
      Ok(())
  }

  fn apply_q2_gate_transposed(
    &self,
    gate: PyReadonlyArray2<Complex<f64>>,
    pos2: usize,
    pos1: usize,
  ) -> PyResult<()>
  {
    if self.len() < 4 { return Err(PyErr::new::<PyTypeError, _>("Number of qubits in a state must be >= 2.")) }
    if !is_pow_2(self.len()) { return Err(PyErr::new::<PyTypeError, _>("The size of the state is not a power of 2.")) };
    if gate.len() != 16 { return Err(PyErr::new::<PyTypeError, _>("Incorrect number of elements in a gate."))};
    if pos2 == pos1 { return Err(PyErr::new::<PyTypeError, _>("pos1 and pos2 must be different."))};
    let qubits_number = get_number_of_qubits(self.len());
    if pos2 >= qubits_number { return Err(PyErr::new::<PyTypeError, _>("pos2 out of range"))};
    if pos1 >= qubits_number { return Err(PyErr::new::<PyTypeError, _>("pos1 out of range"))};
    let buff_ptr = unsafe {
      MutPtrWrapper(
        self.as_slice_mut()
        .expect("Array is not contiguous in memory.")
        .as_mut_ptr()
      )
    };
    let gate_ptr = PtrWrapper(
      gate.as_slice()
        .expect("Array is not contiguous in memory.")
        .as_ptr()
    );
    let stride1 = 2usize.pow(pos1 as u32);
    let stride2 = 2usize.pow(pos2 as u32);
    let batch_size = 2usize.pow(qubits_number as u32 - 2);
    (0..batch_size).into_par_iter()
      .for_each(|bi| {
        let bi = insert_two_zeros(bi, pos1, pos2);
        let mut tmp00 = Complex::new(0., 0.);
        let mut tmp01 = Complex::new(0., 0.);
        let mut tmp10 = Complex::new(0., 0.);
        let mut tmp11 = Complex::new(0., 0.);
        for q in 0..2 {
          for p in 0..2 {
            unsafe {
              tmp00 = tmp00 + *gate_ptr.add(8 * q + 4 * p)
                            * *buff_ptr.add(q * stride2 + p * stride1 + bi);
              tmp01 = tmp01 + *gate_ptr.add(1 + 8 * q + 4 * p)
                            * *buff_ptr.add(q * stride2 + p * stride1 + bi);
              tmp10 = tmp10 + *gate_ptr.add(2 + 8 * q + 4 * p)
                            * *buff_ptr.add(q * stride2 + p * stride1 + bi);
              tmp11 = tmp11 + *gate_ptr.add(3 + 8 * q + 4 * p)
                            * *buff_ptr.add(q * stride2 + p * stride1 + bi);
            }
          }
        }
        unsafe {
          *buff_ptr.add(bi) = tmp00;
          *buff_ptr.add(stride1 + bi) = tmp01;
          *buff_ptr.add(stride2 + bi) = tmp10;
          *buff_ptr.add(stride1 + stride2 + bi) = tmp11;
        }
      });
      Ok(())
  }

  fn apply_q2_gate_conj_transposed(
    &self,
    gate: PyReadonlyArray2<Complex<f64>>,
    pos2: usize,
    pos1: usize,
  ) -> PyResult<()>
  {
    if self.len() < 4 { return Err(PyErr::new::<PyTypeError, _>("Number of qubits in a state must be >= 2.")) }
    if !is_pow_2(self.len()) { return Err(PyErr::new::<PyTypeError, _>("The size of the state is not a power of 2.")) };
    if gate.len() != 16 { return Err(PyErr::new::<PyTypeError, _>("Incorrect number of elements in a gate."))};
    if pos2 == pos1 { return Err(PyErr::new::<PyTypeError, _>("pos1 and pos2 must be different."))};
    let qubits_number = get_number_of_qubits(self.len());
    if pos2 >= qubits_number { return Err(PyErr::new::<PyTypeError, _>("pos2 out of range"))};
    if pos1 >= qubits_number { return Err(PyErr::new::<PyTypeError, _>("pos1 out of range"))};
    let buff_ptr = unsafe {
      MutPtrWrapper(
        self.as_slice_mut()
        .expect("Array is not contiguous in memory.")
        .as_mut_ptr()
      )
    };
    let gate_ptr = PtrWrapper(
      gate.as_slice()
        .expect("Array is not contiguous in memory.")
        .as_ptr()
    );
    let stride1 = 2usize.pow(pos1 as u32);
    let stride2 = 2usize.pow(pos2 as u32);
    let batch_size = 2usize.pow(qubits_number as u32 - 2);
    (0..batch_size).into_par_iter()
      .for_each(|bi| {
        let bi = insert_two_zeros(bi, pos1, pos2);
        let mut tmp00 = Complex::new(0., 0.);
        let mut tmp01 = Complex::new(0., 0.);
        let mut tmp10 = Complex::new(0., 0.);
        let mut tmp11 = Complex::new(0., 0.);
        for q in 0..2 {
          for p in 0..2 {
            unsafe {
              tmp00 = tmp00 + (*gate_ptr.add(8 * q + 4 * p)).conj()
                            * *buff_ptr.add(q * stride2 + p * stride1 + bi);
              tmp01 = tmp01 + (*gate_ptr.add(1 + 8 * q + 4 * p)).conj()
                            * *buff_ptr.add(q * stride2 + p * stride1 + bi);
              tmp10 = tmp10 + (*gate_ptr.add(2 + 8 * q + 4 * p)).conj()
                            * *buff_ptr.add(q * stride2 + p * stride1 + bi);
              tmp11 = tmp11 + (*gate_ptr.add(3 + 8 * q + 4 * p)).conj()
                            * *buff_ptr.add(q * stride2 + p * stride1 + bi);
            }
          }
        }
        unsafe {
          *buff_ptr.add(bi) = tmp00;
          *buff_ptr.add(stride1 + bi) = tmp01;
          *buff_ptr.add(stride2 + bi) = tmp10;
          *buff_ptr.add(stride1 + stride2 + bi) = tmp11;
        }
      });
      Ok(())
  }

  fn apply_q1_gate(
    &self,
    gate: PyReadonlyArray2<Complex<f64>>,
    pos: usize,
  ) -> PyResult<()>
  {
    if self.len() < 2 { return Err(PyErr::new::<PyTypeError, _>("Number of qubits in a state must be >= 1.")) }
    if !is_pow_2(self.len()) { return Err(PyErr::new::<PyTypeError, _>("The size of the state is not a power of 2.")) };
    if gate.len() != 4 { return Err(PyErr::new::<PyTypeError, _>("Incorrect number of elements in a gate."))};
    let qubits_number = get_number_of_qubits(self.len());
    if pos >= qubits_number { return Err(PyErr::new::<PyTypeError, _>("pos out of range"))};
    let buff_ptr = unsafe {
      MutPtrWrapper(
        self.as_slice_mut()
        .expect("Array is not contiguous in memory.")
        .as_mut_ptr()
      )
    };
    let gate_ptr = PtrWrapper(
      gate.as_slice()
        .expect("Array is not contiguous in memory.")
        .as_ptr()
    );
    let stride = 2usize.pow(pos as u32);
    let batch_size = 2usize.pow(qubits_number as u32 - 1);
    (0..batch_size).into_par_iter()
      .for_each(|bi| {
        let bi = insert_single_zero(bi, pos);
        let mut tmp0 = Complex::new(0., 0.);
        let mut tmp1 = Complex::new(0., 0.);
        for q in 0..2 {
          unsafe {
            tmp0 = tmp0 + *gate_ptr.add(q)
                          * *buff_ptr.add(q * stride + bi);
            tmp1 = tmp1 + *gate_ptr.add(2 + q)
                          * *buff_ptr.add(q * stride + bi);
          }
        }
        unsafe {
          *buff_ptr.add(bi) = tmp0;
          *buff_ptr.add(stride + bi) = tmp1;
        }
      });
      Ok(())
  }

  fn apply_q1_gate_transposed(
    &self,
    gate: PyReadonlyArray2<Complex<f64>>,
    pos: usize,
  ) -> PyResult<()>
  {
    if self.len() < 2 { return Err(PyErr::new::<PyTypeError, _>("Number of qubits in a state must be >= 1.")) }
    if !is_pow_2(self.len()) { return Err(PyErr::new::<PyTypeError, _>("The size of the state is not a power of 2.")) };
    if gate.len() != 4 { return Err(PyErr::new::<PyTypeError, _>("Incorrect number of elements in a gate."))};
    let qubits_number = get_number_of_qubits(self.len());
    if pos >= qubits_number { return Err(PyErr::new::<PyTypeError, _>("pos out of range"))};
    let buff_ptr = unsafe {
      MutPtrWrapper(
        self.as_slice_mut()
        .expect("Array is not contiguous in memory.")
        .as_mut_ptr()
      )
    };
    let gate_ptr = PtrWrapper(
      gate.as_slice()
        .expect("Array is not contiguous in memory.")
        .as_ptr()
    );
    let stride = 2usize.pow(pos as u32);
    let batch_size = 2usize.pow(qubits_number as u32 - 1);
    (0..batch_size).into_par_iter()
      .for_each(|bi| {
        let bi = insert_single_zero(bi, pos);
        let mut tmp0 = Complex::new(0., 0.);
        let mut tmp1 = Complex::new(0., 0.);
        for q in 0..2 {
          unsafe {
            tmp0 = tmp0 + *gate_ptr.add(2 * q)
                          * *buff_ptr.add(q * stride + bi);
            tmp1 = tmp1 + *gate_ptr.add(1 + 2 * q)
                          * *buff_ptr.add(q * stride + bi);
          }
        }
        unsafe {
          *buff_ptr.add(bi) = tmp0;
          *buff_ptr.add(stride + bi) = tmp1;
        }
      });
      Ok(())
  }

  fn apply_q1_gate_conj_transposed(
    &self,
    gate: PyReadonlyArray2<Complex<f64>>,
    pos: usize,
  ) -> PyResult<()>
  {
    if self.len() < 2 { return Err(PyErr::new::<PyTypeError, _>("Number of qubits in a state must be >= 1.")) }
    if !is_pow_2(self.len()) { return Err(PyErr::new::<PyTypeError, _>("The size of the state is not a power of 2.")) };
    if gate.len() != 4 { return Err(PyErr::new::<PyTypeError, _>("Incorrect number of elements in a gate."))};
    let qubits_number = get_number_of_qubits(self.len());
    if pos >= qubits_number { return Err(PyErr::new::<PyTypeError, _>("pos out of range"))};
    let buff_ptr = unsafe {
      MutPtrWrapper(
        self.as_slice_mut()
        .expect("Array is not contiguous in memory.")
        .as_mut_ptr()
      )
    };
    let gate_ptr = PtrWrapper(
      gate.as_slice()
        .expect("Array is not contiguous in memory.")
        .as_ptr()
    );
    let stride = 2usize.pow(pos as u32);
    let batch_size = 2usize.pow(qubits_number as u32 - 1);
    (0..batch_size).into_par_iter()
      .for_each(|bi| {
        let bi = insert_single_zero(bi, pos);
        let mut tmp0 = Complex::new(0., 0.);
        let mut tmp1 = Complex::new(0., 0.);
        for q in 0..2 {
          unsafe {
            tmp0 = tmp0 + (*gate_ptr.add(2 * q)).conj()
                          * *buff_ptr.add(q * stride + bi);
            tmp1 = tmp1 + (*gate_ptr.add(1 + 2 * q)).conj()
                          * *buff_ptr.add(q * stride + bi);
          }
        }
        unsafe {
          *buff_ptr.add(bi) = tmp0;
          *buff_ptr.add(stride + bi) = tmp1;
        }
      });
      Ok(())
  }

  fn get_q2_density_matrix<'py>(
      &self,
      pos2: usize,
      pos1: usize,
      py: Python<'py>,
    ) -> PyResult<&'py PyArray2<Complex<f64>>>
    {
      if self.len() < 4 { return Err(PyErr::new::<PyTypeError, _>("Number of qubits in a state must be >= 2.")) }
      if !is_pow_2(self.len()) { return Err(PyErr::new::<PyTypeError, _>("The size of the state is not a power of 2.")) };
      if pos2 == pos1 { return Err(PyErr::new::<PyTypeError, _>("pos1 and pos2 must be different."))};
      let qubits_number = get_number_of_qubits(self.len());
      if pos2 >= qubits_number { return Err(PyErr::new::<PyTypeError, _>("pos2 out of range"))};
      if pos1 >= qubits_number { return Err(PyErr::new::<PyTypeError, _>("pos1 out of range"))};
      let pyoutput = unsafe { PyArray2::new(py, [4, 4], false) };
      unsafe { pyoutput.as_slice_mut().unwrap().iter_mut().for_each(|x| { *x = Complex::new(0., 0.) }) };
      let output = unsafe {
        pyoutput.as_slice_mut()
          .expect("Array is not contiguous in memory.")
      };
      let buff_ptr = unsafe {
        PtrWrapper(
          self.as_slice()
            .expect("Array is not contiguous in memory.")
            .as_ptr()
        )
      };
      let stride1 = 2usize.pow(pos1 as u32);
      let stride2 = 2usize.pow(pos2 as u32);
      let cores_number = get_physical();
      let size = 2usize.pow(qubits_number as u32 - 2);
      let (chank_size, num_threads) = if size / (cores_number + 1) > 128 { // maybe tune this threshold later
        (size / (cores_number + 1), cores_number + 1)
      } else {
        (size, 1)
      };
      let pool = ThreadPoolBuilder::new().num_threads(num_threads).build().unwrap();
      let output_dens_matrix = Mutex::new(output);
      pool.scope(|s| {
        for i in 0..(num_threads + 1) {
          let ref_output_dens_matrix = &output_dens_matrix;
          s.spawn(move |_| {
            let mut local_density_matrix = [Complex::new(0., 0.); 16];
            let local_density_matrix_ptr = MutPtrWrapper(local_density_matrix.as_mut_ptr());
            for bi in (i * chank_size)..std::cmp::min((i + 1) * chank_size, size) {
              let bi = insert_two_zeros(bi, pos1, pos2);
              for q in 0..2 {
                for p in 0..2 {
                  for m in 0..2 {
                    for n in 0..2 {
                      unsafe {
                        *local_density_matrix_ptr.add(8 * q + 4 * p + 2 * m + n) =
                        *local_density_matrix_ptr.add(8 * q + 4 * p + 2 * m + n) +
                        *buff_ptr.add(stride2 * q + stride1 * p + bi) *
                        (*buff_ptr.add(stride2 * m + stride1 * n + bi)).conj()
                      }
                    }
                  }
                }
              }
            }
            zip(
              ref_output_dens_matrix.lock().unwrap().iter_mut(),
              local_density_matrix.iter()
            ).for_each(|(to, from)| {
              *to = *to + *from;
            });
          })
        }
      });
      Ok(pyoutput)
  }

  fn get_q1_density_matrix<'py>(
    &self,
    pos: usize,
    py: Python<'py>,
  ) -> PyResult<&'py PyArray2<Complex<f64>>>
  {
    if self.len() < 2 { return Err(PyErr::new::<PyTypeError, _>("Number of qubits in a state must be >= 1.")) }
    if !is_pow_2(self.len()) { return Err(PyErr::new::<PyTypeError, _>("The size of the state is not a power of 2.")) };
    let qubits_number = get_number_of_qubits(self.len());
    if pos >= qubits_number { return Err(PyErr::new::<PyTypeError, _>("pos out of range"))};
    let pyoutput = unsafe { PyArray2::new(py, [2, 2], false) };
    unsafe { pyoutput.as_slice_mut().unwrap().iter_mut().for_each(|x| { *x = Complex::new(0., 0.) }) };
    let output = unsafe {
      pyoutput.as_slice_mut()
        .expect("Array is not contiguous in memory.")
    };
    let buff_ptr = unsafe {
      PtrWrapper(
        self.as_slice()
          .expect("Array is not contiguous in memory.")
          .as_ptr()
      )
    };
    let stride = 2usize.pow(pos as u32);
    let cores_number = get_physical();
    let size = 2usize.pow(qubits_number as u32 - 1);
    let (chank_size, num_threads) = if size / (cores_number + 1) > 128 { // maybe tune this threshold later
      (size / (cores_number + 1), cores_number + 1)
    } else {
      (size, 1)
    };
    let pool = ThreadPoolBuilder::new().num_threads(num_threads).build().unwrap();
    let output_dens_matrix = Mutex::new(output);
    pool.scope(|s| {
      for i in 0..(num_threads + 1) {
        let ref_output_dens_matrix = &output_dens_matrix;
        s.spawn(move |_| {
          let mut local_density_matrix = [Complex::new(0., 0.); 4];
          let local_density_matrix_ptr = MutPtrWrapper(local_density_matrix.as_mut_ptr());
          for bi in (i * chank_size)..std::cmp::min((i + 1) * chank_size, size) {
            let bi = insert_single_zero(bi, pos);
            for q in 0..2 {
              for m in 0..2 {
                unsafe {
                  *local_density_matrix_ptr.add(2 * q + m) =
                  *local_density_matrix_ptr.add(2 * q + m) +
                  *buff_ptr.add(stride * q + bi) *
                  (*buff_ptr.add(stride * m + bi)).conj()
                }
              }
            }
          }
          zip(
            ref_output_dens_matrix.lock().unwrap().iter_mut(),
            local_density_matrix.iter()
          ).for_each(|(to, from)| {
            *to = *to + *from;
          });
        })
      }
    });
    Ok(pyoutput)
  }
}
////////////////////////////////////////////////////

#[pyfunction]
pub fn q2_gradient<'py>(
  bwd_state: &PyArray1<Complex<f64>>,
  fwd_state: &PyArray1<Complex<f64>>,
  pos2: usize,
  pos1: usize,
  py: Python<'py>,
) -> PyResult<&'py PyArray2<Complex<f64>>>
{
  if bwd_state.len() != fwd_state.len() { return Err(PyErr::new::<PyTypeError, _>("bwd_state and fwd state have different sizes.")) }
  if bwd_state.len() < 4 { return Err(PyErr::new::<PyTypeError, _>("Number of qubits in a system must be >= 2.")) }
  if !is_pow_2(bwd_state.len()) { return Err(PyErr::new::<PyTypeError, _>("The size of states is not a pow of 2")) }
  if pos2 == pos1 { return Err(PyErr::new::<PyTypeError, _>("pos1 and pos2 must be different."))};
  let size = bwd_state.len();
  let qubits_number = get_number_of_qubits(size);
  if pos2 >= qubits_number { return Err(PyErr::new::<PyTypeError, _>("pos2 out of range"))};
  if pos1 >= qubits_number { return Err(PyErr::new::<PyTypeError, _>("pos1 out of range"))};
  let pyoutput = unsafe { PyArray2::new(py, [4, 4], false) };
  unsafe { pyoutput.as_slice_mut().unwrap().iter_mut().for_each(|x| { *x = Complex::new(0., 0.) }) };
  let stride1 = 2usize.pow(pos1 as u32);
  let stride2 = 2usize.pow(pos2 as u32);
  let bwd_buff_ptr = unsafe { PtrWrapper(bwd_state.as_slice().unwrap().as_ptr()) };
  let fwd_buff_ptr = unsafe { PtrWrapper(fwd_state.as_slice().unwrap().as_ptr()) };
  let cores_number = get_physical();
  let size = 2usize.pow(qubits_number as u32 - 2);
  let (chank_size, num_threads) = if size / (cores_number + 1) > 128 { // maybe tune this threshold later
    (size / (cores_number + 1), cores_number + 1)
  } else {
    (size, 1)
  };
  let pool = ThreadPoolBuilder::new().num_threads(num_threads).build().unwrap();
  let output = Mutex::new(unsafe { pyoutput.as_slice_mut().unwrap() });
  pool.scope(|s| {
    for i in 0..(num_threads + 1) {
      let ref_output = &output;
      s.spawn(move |_| {
        let mut local_output = [Complex::new(0., 0.); 16];
        let local_output_ptr = MutPtrWrapper(local_output.as_mut_ptr());
        for bi in (i * chank_size)..std::cmp::min((i + 1) * chank_size, size) {
          let bi = insert_two_zeros(bi, pos1, pos2);
          for q in 0..2 {
            for p in 0..2 {
              for m in 0..2 {
                for n in 0..2 {
                  unsafe {
                    *local_output_ptr.add(8 * q + 4 * p + 2 * m + n) =
                    *local_output_ptr.add(8 * q + 4 * p + 2 * m + n) +
                    *bwd_buff_ptr.add(stride2 * q + stride1 * p + bi) *
                    *fwd_buff_ptr.add(stride2 * m + stride1 * n + bi)
                  }
                }
              }
            }
          }
        }
        zip(
          ref_output.lock().unwrap().iter_mut(),
          local_output.iter()
        ).for_each(|(to, from)| {
          *to = *to + *from;
        });
      })
    }
  });
  Ok(pyoutput)
}

#[pyfunction]
pub fn q1_gradient<'py>(
  bwd_state: &PyArray1<Complex<f64>>,
  fwd_state: &PyArray1<Complex<f64>>,
  pos: usize,
  py: Python<'py>,
) -> PyResult<&'py PyArray2<Complex<f64>>>
{
  if bwd_state.len() != fwd_state.len() { return Err(PyErr::new::<PyTypeError, _>("bwd_state and fwd state have different sizes.")) }
  if bwd_state.len() < 2 { return Err(PyErr::new::<PyTypeError, _>("Number of qubits in a system must be >= 1.")) }
  if !is_pow_2(bwd_state.len()) { return Err(PyErr::new::<PyTypeError, _>("The size of states is not a pow of 2")) }
  let size = bwd_state.len();
  let qubits_number = get_number_of_qubits(size);
  if pos >= qubits_number { return Err(PyErr::new::<PyTypeError, _>("pos out of range"))};
  let pyoutput = unsafe { PyArray2::new(py, [2, 2], false) };
  unsafe { pyoutput.as_slice_mut().unwrap().iter_mut().for_each(|x| { *x = Complex::new(0., 0.) }) };
  let stride = 2usize.pow(pos as u32);
  let bwd_buff_ptr = unsafe { PtrWrapper(bwd_state.as_slice().unwrap().as_ptr()) };
  let fwd_buff_ptr = unsafe { PtrWrapper(fwd_state.as_slice().unwrap().as_ptr()) };
  let cores_number = get_physical();
  let size = 2usize.pow(qubits_number as u32 - 1);
  let (chank_size, num_threads) = if size / (cores_number + 1) > 128 { // maybe tune this threshold later
    (size / (cores_number + 1), cores_number + 1)
  } else {
    (size, 1)
  };
  let pool = ThreadPoolBuilder::new().num_threads(num_threads).build().unwrap();
  let output = Mutex::new(unsafe { pyoutput.as_slice_mut().unwrap() });
  pool.scope(|s| {
    for i in 0..(num_threads + 1) {
      let ref_output = &output;
      s.spawn(move |_| {
        let mut local_output = [Complex::new(0., 0.); 4];
        let local_output_ptr = MutPtrWrapper(local_output.as_mut_ptr());
        for bi in (i * chank_size)..std::cmp::min((i + 1) * chank_size, size) {
          let bi = insert_single_zero(bi, pos);
          for q in 0..2 {
              for m in 0..2 {
                  unsafe {
                    *local_output_ptr.add(2 * q + m) =
                    *local_output_ptr.add(2 * q + m) +
                    *bwd_buff_ptr.add(stride * q + bi) *
                    *fwd_buff_ptr.add(stride * m + bi)
                  }
              }
          }
        }
        zip(
          ref_output.lock().unwrap().iter_mut(),
          local_output.iter()
        ).for_each(|(to, from)| {
          *to = *to + *from;
        });
      })
    }
  });
  Ok(pyoutput)
}

////////////////////////////////////////////////////

#[pyfunction]
pub fn fast_copy<'py>(
  arr: PyReadonlyArray1<Complex<f64>>,
  py: Python<'py>
) -> &'py PyArray1<Complex<f64>> {
  let size = arr.len();
  let new_arr: &PyArray1<Complex<f64>> = unsafe { PyArray1::new(py, [size], false) };
  let from_ptr = PtrWrapper(arr.as_slice().unwrap().as_ptr());
  let to_ptr = unsafe { MutPtrWrapper(new_arr.as_slice_mut().unwrap().as_mut_ptr()) };
  (0..size).into_par_iter().for_each(|i| {
    unsafe { *to_ptr.add(i) = *from_ptr.add(i)};
  });
  new_arr
}

///////////////////////////////////////////////////

#[pyfunction]
pub fn apply_q2_gate(
  state: &PyArray1<Complex<f64>>,
  gate: PyReadonlyArray2<Complex<f64>>,
  pos2: usize,
  pos1: usize,
) -> PyResult<()>
{
  state.apply_q2_gate(gate, pos2, pos1)
}

#[pyfunction]
pub fn apply_q2_gate_transposed(
  state: &PyArray1<Complex<f64>>,
  gate: PyReadonlyArray2<Complex<f64>>,
  pos2: usize,
  pos1: usize,
) -> PyResult<()>
{
  state.apply_q2_gate_transposed(gate, pos2, pos1)
}

#[pyfunction]
pub fn apply_q2_gate_conj_transposed(
  state: &PyArray1<Complex<f64>>,
  gate: PyReadonlyArray2<Complex<f64>>,
  pos2: usize,
  pos1: usize,
) -> PyResult<()>
{
  state.apply_q2_gate_conj_transposed(gate, pos2, pos1)
}

#[pyfunction]
pub fn apply_q1_gate(
  state: &PyArray1<Complex<f64>>,
  gate: PyReadonlyArray2<Complex<f64>>,
  pos: usize,
) -> PyResult<()>
{
  state.apply_q1_gate(gate, pos)
}

#[pyfunction]
pub fn apply_q1_gate_transposed(
  state: &PyArray1<Complex<f64>>,
  gate: PyReadonlyArray2<Complex<f64>>,
  pos: usize,
) -> PyResult<()>
{
  state.apply_q1_gate_transposed(gate, pos)
}

#[pyfunction]
pub fn apply_q1_gate_conj_transposed(
  state: &PyArray1<Complex<f64>>,
  gate: PyReadonlyArray2<Complex<f64>>,
  pos: usize,
) -> PyResult<()>
{
  state.apply_q1_gate_conj_transposed(gate, pos)
}

#[pyfunction]
pub fn get_q2_density_matrix<'py>(
  state: &PyArray1<Complex<f64>>,
  pos2: usize,
  pos1: usize,
  py: Python<'py>,
) -> PyResult<&'py PyArray2<Complex<f64>>>
{
  state.get_q2_density_matrix(pos2, pos1, py)
}

#[pyfunction]
pub fn get_q1_density_matrix<'py>(
  state: &PyArray1<Complex<f64>>,
  pos: usize,
  py: Python<'py>,
) -> PyResult<&'py PyArray2<Complex<f64>>>
{
  state.get_q1_density_matrix(pos, py)
}