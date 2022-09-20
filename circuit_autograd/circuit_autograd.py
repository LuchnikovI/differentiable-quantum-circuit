import numpy as np
from jax import custom_vjp
from typing import Callable, List, Tuple

from differentiable_circuit import (
  Circuit,
  _q2_density_matrix,
  _q1_density_matrix,
  grad_wrt_q2_density_matrix,
  grad_wrt_q1_density_matrix,
  fast_copy,
)

class AutoGradCircuit:

  def __init__(self):
    """Quantum circuit that supports automatic differentiation."""
    self.circuit = Circuit()

  def add_q2_const_gate(self, pos2: int, pos1: int):
    """Adds a constant two-qubit gate to a circuit.
    Args:
      pos2: int, a position of a qubit that is considered as 'control' qubit,
      pos1: int, a position of a qubit that is conisdered as 'target' qubit.
    """
    self.circuit.add_q2_const_gate(pos2, pos1)

  def add_q2_var_gate(self, pos2: int, pos1: int):
    """Adds a variable two-qubit gate to a circuit.
    Args:
      pos2: int, a position of a qubit that is considered as 'control' qubit,
      pos1: int, a position of a qubit that is conisdered as 'target' qubit.
    """
    self.circuit.add_q2_var_gate(pos2, pos1)

  def add_q1_const_gate(self, pos: int):
    """Adds a constant one-qubit gate to a circuit.
    Args:
      pos: int, a position of a qubit that the gate is being applied to.
    """
    self.circuit.add_q1_const_gate(pos)

  def add_q1_var_gate(self, pos: int):
    """Adds a variable one-qubit gate to a circuit.
    Args:
      pos: int, a position of a qubit that the gate is being applied to.
    """
    self.circuit.add_q1_var_gate(pos)

  def get_q2_dens_op(self, pos2: int, pos1: int):
    """Adds an operation that evaluates a two-qubit density matrix
    when the circuit is being run.
    Args:
      pos2 and pos1: positions of qubits whose density matrix is being evaluated."""
    self.circuit.get_q2_dens_op(pos2, pos1)

  def get_q1_dens_op(self, pos: int):
    """Adds an operation that evaluates a one-qubit density matrix
    when the circuit is being run.
    Args:
      pos: position of a qubit whose density matrix is being evaluated."""
    self.circuit.get_q1_dens_op(pos)

  def build(
    self
  ) -> Tuple[
      Callable[[np.array, List[np.array], List[np.array]], Tuple[np.array, List[np.array]]],
      Callable[[np.array, List[np.array], List[np.array]], Tuple[np.array]]
    ]:
    """Returns two functions. The first function runs the circuit given an
    initial state and a list of constant and variable gates and evaluates all required density matrices
    and the final state. The second function evaluates only the final state but
    supports backpropagation for optimization purposes."""
    def circuit_run(state, var_gates, const_gates):
      state_copy = fast_copy(np.asarray(state))
      density_matrices = self.circuit.forward(
        state_copy,
        list(map(lambda x: np.asarray(x), const_gates)), 
        list(map(lambda x: np.asarray(x), var_gates)),
      )
      return state_copy, density_matrices
    @custom_vjp
    def circuit_run_wo_density_matrices(state, var_gates, const_gates):
      state_copy = fast_copy(np.asarray(state))
      self.circuit.forward_wo_density_matrices(
        state_copy,
        list(map(lambda x: np.asarray(x), const_gates)), 
        list(map(lambda x: np.asarray(x), var_gates)),
      )
      return state_copy
    def circuit_run_wo_density_matrices_fwd(state, var_gates, const_gates):
      state_copy = fast_copy(np.asarray(state))
      self.circuit.forward_wo_density_matrices(
        state_copy,
        list(map(lambda x: np.asarray(x), const_gates)), 
        list(map(lambda x: np.asarray(x), var_gates)),
      )
      return state_copy, (state_copy, const_gates, var_gates)
    def circuit_run_wo_density_matrices_bwd(res, state_grad):
      state, const_gates, var_gates = res
      state_copy = fast_copy(np.asarray(state))
      state_grad_copy = fast_copy(np.asarray(state_grad))
      gates_grads = self.circuit.backward(
        state_copy,
        state_grad_copy,
        list(map(lambda x: np.asarray(x), const_gates)), 
        list(map(lambda x: np.asarray(x), var_gates)),
      )
      return state_grad_copy, gates_grads, None
    circuit_run_wo_density_matrices.defvjp(
      circuit_run_wo_density_matrices_fwd,
      circuit_run_wo_density_matrices_bwd
    )
    return circuit_run, circuit_run_wo_density_matrices

@custom_vjp
def q2_density_matrix(state, pos2, pos1):
  return _q2_density_matrix(np.asarray(state), pos2, pos1)
def q2_density_matrix_fwd(state, pos2, pos1):
  state = np.asarray(state)
  return _q2_density_matrix(state, pos2, pos1), (state, pos2, pos1)
def q2_density_matrix_bwd(res, dens_grad):
  state, pos2, pos1 = res
  state_copy = np.asarray(fast_copy(state))
  grad_wrt_q2_density_matrix(
    state_copy, 
    np.asarray(dens_grad).conj(),
    pos2,
    pos1,
  )
  return state_copy, None, None
q2_density_matrix.defvjp(q2_density_matrix_fwd, q2_density_matrix_bwd)

@custom_vjp
def q1_density_matrix(state, pos):
  return _q1_density_matrix(np.asarray(state), pos)
def q1_density_matrix_fwd(state, pos):
  state = np.asarray(state)
  return _q1_density_matrix(state, pos), (state, pos)
def q1_density_matrix_bwd(res, dens_grad):
  state, pos = res
  state_copy = np.asarray(fast_copy(state))
  grad_wrt_q1_density_matrix(
    state_copy,
    np.asarray(dens_grad).conj(),
    pos,
  )
  return state_copy, None
q1_density_matrix.defvjp(q1_density_matrix_fwd, q1_density_matrix_bwd)

'''def build(
  self
) -> Tuple[
    Callable[[np.array, List[np.array], List[np.array]], Tuple[np.array, List[np.array]]],
    Callable[[np.array, List[np.array], List[np.array]], Tuple[np.array]]
  ]:
  """Returns two functions. The first function runs the circuit given an
  initial state and a list of constant and variable gates and evaluates all required density matrices
  and the final state. The second function evaluates only the final state but
  supports backpropagation for optimization purposes."""
  def circuit_run(state, var_gates, const_gates):
    state_copy = fast_copy(state)
    density_matrices = self.circuit.forward(state_copy, const_gates, var_gates)
    return state_copy, density_matrices
  @primitive
  def circuit_forward(state, var_gates, const_gates):
    state_copy = fast_copy(state)
    self.circuit.forward_wo_density_matrices(state_copy, const_gates, var_gates)
    return state_copy
  def circuit_forward_vjp(state, _, var_gates, const_gates):
    def return_function(g):
      state_copy = fast_copy(state)
      state_grad_copy = fast_copy(g)
      gates_grads = self.circuit.backward(state_copy, state_grad_copy, const_gates, var_gates)
      return state_grad_copy, gates_grads
    return return_function
  defvjp(circuit_forward, circuit_forward_vjp, argnums=[0, 1])
  return circuit_run, circuit_forward

q2_density_matrix = primitive(q2_density_matrix)
q1_density_matrix = primitive(q1_density_matrix)

def q2_density_matrix_vjp(_, state, pos2, pos1):
def return_function(g):
  state_copy = fast_copy(state)
  grad_wrt_q2_density_matrix(state_copy, g, pos2, pos1)
  return state_copy
return return_function

def q1_density_matrix_vjp(_, state, pos):
def return_function(g):
  state_copy = fast_copy(state)
  grad_wrt_q1_density_matrix(state_copy, g, pos)
  return state_copy
return return_function

defvjp(q2_density_matrix, q2_density_matrix_vjp, argnums=[0])
defvjp(q1_density_matrix, q1_density_matrix_vjp, argnums=[0])
'''