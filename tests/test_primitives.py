from jax.config import config
config.update('jax_platform_name', 'cpu')
config.update("jax_enable_x64", True)
from jax import random
import jax.numpy as jnp
import numpy as np
from differentiable_circuit import (
  apply_q1_gate,
  apply_q2_gate,
  apply_q1_gate_transposed,
  apply_q2_gate_transposed,
  apply_q1_gate_conj_transposed,
  apply_q2_gate_conj_transposed,
  get_q1_density_matrix,
  get_q2_density_matrix,
  q2_gradient,
  q1_gradient,
)
from circuit_autograd.testing_utils import (
  jnp_apply_q2_gate,
  jnp_apply_q2_gate_transposed,
  jnp_apply_q1_gate,
  jnp_apply_q1_gate_transposed,
  jnp_get_q2_density_matrix,
  jnp_get_q1_density_matrix,
  jnp_apply_q1_gate_conj_transposed,
  jnp_apply_q2_gate_conj_transposed,
  jnp_q2_gradient,
  jnp_q1_gradient,
)


def apply_q2_gate_test(key: jnp.ndarray, qubits_number: int, pos2: int, pos1: int):
  key, subkey = random.split(key)
  state = random.normal(subkey, shape = (2 ** qubits_number, 2))
  state = state[..., 0] + 1j * state[..., 1]
  state_copy = state.copy().reshape(qubits_number * (2,))
  key, subkey = random.split(key)
  gate = random.normal(subkey, shape = (4, 4, 2))
  gate = gate[..., 0] + 1j * gate[..., 1]
  apply_q2_gate(np.asarray(state), np.asarray(gate), pos2, pos1)
  assert jnp.isclose(state, jnp_apply_q2_gate(state_copy, gate, pos2, pos1)).all()

def apply_q1_gate_test(key: jnp.ndarray, qubits_number: int, pos):
  key, subkey = random.split(key)
  state = random.normal(subkey, shape = (2 ** qubits_number, 2))
  state = state[..., 0] + 1j * state[..., 1]
  state_copy = state.copy().reshape(qubits_number * (2,))
  key, subkey = random.split(key)
  gate = random.normal(subkey, shape = (2, 2, 2))
  gate = gate[..., 0] + 1j * gate[..., 1]
  apply_q1_gate(np.asarray(state), np.asarray(gate), pos)
  assert jnp.isclose(state, jnp_apply_q1_gate(state_copy, gate, pos)).all()

def apply_q2_gate_transposed_test(key: jnp.ndarray, qubits_number: int, pos2: int, pos1: int):
  key, subkey = random.split(key)
  state = random.normal(subkey, shape = (2 ** qubits_number, 2))
  state = state[..., 0] + 1j * state[..., 1]
  state_copy = state.copy().reshape(qubits_number * (2,))
  key, subkey = random.split(key)
  gate = random.normal(subkey, shape = (4, 4, 2))
  gate = gate[..., 0] + 1j * gate[..., 1]
  apply_q2_gate_transposed(np.asarray(state), np.asarray(gate), pos2, pos1)
  assert jnp.isclose(state, jnp_apply_q2_gate_transposed(state_copy, gate, pos2, pos1)).all()

def apply_q2_gate_conj_transposed_test(key: jnp.ndarray, qubits_number: int, pos2: int, pos1: int):
  key, subkey = random.split(key)
  state = random.normal(subkey, shape = (2 ** qubits_number, 2))
  state = state[..., 0] + 1j * state[..., 1]
  state_copy = state.copy().reshape(qubits_number * (2,))
  key, subkey = random.split(key)
  gate = random.normal(subkey, shape = (4, 4, 2))
  gate = gate[..., 0] + 1j * gate[..., 1]
  apply_q2_gate_conj_transposed(np.asarray(state), np.asarray(gate), pos2, pos1)
  assert jnp.isclose(state, jnp_apply_q2_gate_conj_transposed(state_copy, gate, pos2, pos1)).all()

def apply_q1_gate_transposed_test(key: jnp.ndarray, qubits_number: int, pos):
  key, subkey = random.split(key)
  state = random.normal(subkey, shape = (2 ** qubits_number, 2))
  state = state[..., 0] + 1j * state[..., 1]
  state_copy = state.copy().reshape(qubits_number * (2,))
  key, subkey = random.split(key)
  gate = random.normal(subkey, shape = (2, 2, 2))
  gate = gate[..., 0] + 1j * gate[..., 1]
  apply_q1_gate_transposed(np.asarray(state), np.asarray(gate), pos)
  assert jnp.isclose(state, jnp_apply_q1_gate_transposed(state_copy, gate, pos)).all()

def apply_q1_gate_conj_transposed_test(key: jnp.ndarray, qubits_number: int, pos):
  key, subkey = random.split(key)
  state = random.normal(subkey, shape = (2 ** qubits_number, 2))
  state = state[..., 0] + 1j * state[..., 1]
  state_copy = state.copy().reshape(qubits_number * (2,))
  key, subkey = random.split(key)
  gate = random.normal(subkey, shape = (2, 2, 2))
  gate = gate[..., 0] + 1j * gate[..., 1]
  apply_q1_gate_conj_transposed(np.asarray(state), np.asarray(gate), pos)
  assert jnp.isclose(state, jnp_apply_q1_gate_conj_transposed(state_copy, gate, pos)).all()

def get_q2_density_matrix_test(key: jnp.ndarray, qubits_number: int, pos2: int, pos1: int):
  key, subkey = random.split(key)
  state = random.normal(subkey, shape = (2,) * (qubits_number + 1))
  state = state[..., 0] + 1j * state[..., 1]
  jnp_dens = jnp_get_q2_density_matrix(state, pos2, pos1)
  state = state.reshape((-1,))
  dens = get_q2_density_matrix(np.asarray(state), pos2, pos1)
  assert jnp.isclose(dens, jnp_dens).all()

def get_q1_density_matrix_test(key: jnp.ndarray, qubits_number: int, pos: int):
  key, subkey = random.split(key)
  state = state = random.normal(subkey, shape = (2,) * (qubits_number + 1))
  state = state[..., 0] + 1j * state[..., 1]
  jnp_dens = jnp_get_q1_density_matrix(state, pos)
  state = state.reshape((-1,))
  dens = get_q1_density_matrix(np.asarray(state), pos)
  assert jnp.isclose(dens, jnp_dens).all()

def q2_gradient_test(key: jnp.ndarray, qubits_number: int, pos2: int, pos1: int):
  key, subkey = random.split(key)
  bwd_state = random.normal(subkey, shape = (2,) * (qubits_number + 1))
  key, subkey = random.split(key)
  fwd_state = random.normal(subkey, shape = (2,) * (qubits_number + 1))
  bwd_state = bwd_state[..., 0] + 1j * bwd_state[..., 1]
  fwd_state = fwd_state[..., 0] + 1j * fwd_state[..., 1]
  jnp_grad = jnp_q2_gradient(bwd_state, fwd_state, pos2, pos1)
  fwd_state = fwd_state.reshape((-1,))
  bwd_state = bwd_state.reshape((-1,))
  grad = q2_gradient(np.asarray(bwd_state), np.asarray(fwd_state), pos2, pos1)
  assert jnp.isclose(grad, jnp_grad).all()

def q1_gradient_test(key: jnp.ndarray, qubits_number: int, pos: int):
  key, subkey = random.split(key)
  bwd_state = random.normal(subkey, shape = (2,) * (qubits_number + 1))
  key, subkey = random.split(key)
  fwd_state = random.normal(subkey, shape = (2,) * (qubits_number + 1))
  bwd_state = bwd_state[..., 0] + 1j * bwd_state[..., 1]
  fwd_state = fwd_state[..., 0] + 1j * fwd_state[..., 1]
  jnp_grad = jnp_q1_gradient(bwd_state, fwd_state, pos)
  fwd_state = fwd_state.reshape((-1,))
  bwd_state = bwd_state.reshape((-1,))
  grad = q1_gradient(np.asarray(bwd_state), np.asarray(fwd_state), pos)
  assert jnp.isclose(grad, jnp_grad).all()
  

def test_all_primitives():
  key = random.split(random.PRNGKey(42), 55)
  apply_q2_gate_test(key[0], 15, 0, 1)
  apply_q2_gate_test(key[1], 15, 1, 0)
  apply_q2_gate_test(key[2], 15, 0, 14)
  apply_q2_gate_test(key[3], 15, 14, 0)
  apply_q2_gate_test(key[4], 15, 14, 13)
  apply_q2_gate_test(key[5], 15, 13, 14)
  apply_q2_gate_test(key[6], 15, 4, 8)
  apply_q2_gate_test(key[7], 15, 9, 5)

  apply_q2_gate_transposed_test(key[8], 15, 0, 1)
  apply_q2_gate_transposed_test(key[9], 15, 1, 0)
  apply_q2_gate_transposed_test(key[10], 15, 0, 14)
  apply_q2_gate_transposed_test(key[11], 15, 14, 0)
  apply_q2_gate_transposed_test(key[12], 15, 14, 13)
  apply_q2_gate_transposed_test(key[13], 15, 13, 14)
  apply_q2_gate_transposed_test(key[14], 15, 4, 8)
  apply_q2_gate_transposed_test(key[15], 15, 9, 5)

  apply_q2_gate_conj_transposed_test(key[16], 15, 0, 1)
  apply_q2_gate_conj_transposed_test(key[17], 15, 1, 0)
  apply_q2_gate_conj_transposed_test(key[18], 15, 0, 14)
  apply_q2_gate_conj_transposed_test(key[19], 15, 14, 0)
  apply_q2_gate_conj_transposed_test(key[20], 15, 14, 13)
  apply_q2_gate_conj_transposed_test(key[21], 15, 13, 14)
  apply_q2_gate_conj_transposed_test(key[22], 15, 4, 8)
  apply_q2_gate_conj_transposed_test(key[23], 15, 9, 5)

  apply_q1_gate_test(key[24], 15, 0)
  apply_q1_gate_test(key[25], 15, 5)
  apply_q1_gate_test(key[26], 15, 14)

  apply_q1_gate_transposed_test(key[27], 15, 0)
  apply_q1_gate_transposed_test(key[28], 15, 5)
  apply_q1_gate_transposed_test(key[29], 15, 14)

  apply_q1_gate_conj_transposed_test(key[30], 15, 0)
  apply_q1_gate_conj_transposed_test(key[31], 15, 5)
  apply_q1_gate_conj_transposed_test(key[32], 15, 14)

  get_q2_density_matrix_test(key[33], 15, 0, 1)
  get_q2_density_matrix_test(key[34], 15, 1, 0)
  get_q2_density_matrix_test(key[35], 15, 0, 14)
  get_q2_density_matrix_test(key[36], 15, 14, 0)
  get_q2_density_matrix_test(key[37], 15, 14, 13)
  get_q2_density_matrix_test(key[38], 15, 13, 14)
  get_q2_density_matrix_test(key[39], 15, 4, 8)
  get_q2_density_matrix_test(key[40], 15, 9, 5)

  get_q1_density_matrix_test(key[41], 15, 0)
  get_q1_density_matrix_test(key[42], 15, 5)
  get_q1_density_matrix_test(key[43], 15, 14)

  q2_gradient_test(key[44], 15, 0, 1)
  q2_gradient_test(key[45], 15, 1, 0)
  q2_gradient_test(key[46], 15, 0, 14)
  q2_gradient_test(key[47], 15, 14, 0)
  q2_gradient_test(key[48], 15, 14, 13)
  q2_gradient_test(key[49], 15, 13, 14)
  q2_gradient_test(key[50], 15, 4, 8)
  q2_gradient_test(key[51], 15, 9, 5)

  q1_gradient_test(key[52], 15, 0)
  q1_gradient_test(key[53], 15, 5)
  q1_gradient_test(key[54], 15, 14)