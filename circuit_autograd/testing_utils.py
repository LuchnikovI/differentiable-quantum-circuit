import jax.numpy as jnp
from jax import jit
from functools import partial

@partial(jit, static_argnums=[2, 3])
def jnp_apply_q2_gate(
  state: jnp.ndarray,
  gate: jnp.ndarray,
  pos2: int,
  pos1: int
) -> jnp.array:
  qubits_number = len(state.shape)
  nppos1 = qubits_number - 1 - pos1
  nppos2 = qubits_number - 1 - pos2
  state = jnp.tensordot(gate.reshape((2, 2, 2, 2)), state, axes=[[2, 3], [nppos2, nppos1]])
  min_pos = min(nppos1, nppos2)
  max_pos = max(nppos1, nppos2)
  state = state.reshape((2, 2, 2 ** min_pos, 2 ** (max_pos - min_pos - 1), 2 ** (qubits_number - max_pos - 1)))
  if nppos2 > nppos1:
    state = state.transpose((2, 1, 3, 0, 4))
  else:
    state = state.transpose((2, 0, 3, 1, 4))
  return state.reshape((-1,))

@partial(jit, static_argnums=[2, 3])
def jnp_apply_q2_gate_transposed(
  state: jnp.ndarray,
  gate: jnp.ndarray,
  pos2: int,
  pos1: int
) -> jnp.array:
  qubits_number = len(state.shape)
  nppos1 = qubits_number - 1 - pos1
  nppos2 = qubits_number - 1 - pos2
  state = jnp.tensordot(gate.reshape((2, 2, 2, 2)), state, axes=[[0, 1], [nppos2, nppos1]])
  min_pos = min(nppos1, nppos2)
  max_pos = max(nppos1, nppos2)
  state = state.reshape((2, 2, 2 ** min_pos, 2 ** (max_pos - min_pos - 1), 2 ** (qubits_number - max_pos - 1)))
  if nppos2 > nppos1:
    state = state.transpose((2, 1, 3, 0, 4))
  else:
    state = state.transpose((2, 0, 3, 1, 4))
  return state.reshape((-1,))

@partial(jit, static_argnums=[2, 3])
def jnp_apply_q2_gate_conj_transposed(
  state: jnp.ndarray,
  gate: jnp.ndarray,
  pos2: int,
  pos1: int
) -> jnp.ndarray:
  qubits_number = len(state.shape)
  nppos1 = qubits_number - 1 - pos1
  nppos2 = qubits_number - 1 - pos2
  gate = gate.conj()
  state = jnp.tensordot(gate.reshape((2, 2, 2, 2)), state, axes=[[0, 1], [nppos2, nppos1]])
  min_pos = min(nppos1, nppos2)
  max_pos = max(nppos1, nppos2)
  state = state.reshape((2, 2, 2 ** min_pos, 2 ** (max_pos - min_pos - 1), 2 ** (qubits_number - max_pos - 1)))
  if nppos2 > nppos1:
    state = state.transpose((2, 1, 3, 0, 4))
  else:
    state = state.transpose((2, 0, 3, 1, 4))
  return state.reshape((-1,))

@partial(jit, static_argnums=2)
def jnp_apply_q1_gate(
  state: jnp.ndarray,
  gate: jnp.ndarray,
  pos: int,
) -> jnp.ndarray:
  qubits_number = len(state.shape)
  nppos = qubits_number - 1 - pos
  state = jnp.tensordot(gate, state, axes=[[1], [nppos]])
  state = state.reshape((2, 2 ** nppos, 2 ** (qubits_number - nppos - 1)))
  state = state.transpose((1, 0, 2))
  return state.reshape((-1,))

@partial(jit, static_argnums=2)
def jnp_apply_q1_gate_transposed(
  state: jnp.ndarray,
  gate: jnp.ndarray,
  pos: int,
) -> jnp.array:
  qubits_number = len(state.shape)
  nppos = qubits_number - 1 - pos
  state = jnp.tensordot(gate, state, axes=[[0], [nppos]])
  state = state.reshape((2, 2 ** nppos, 2 ** (qubits_number - nppos - 1)))
  state = state.transpose((1, 0, 2))
  return state.reshape((-1,))

@partial(jit, static_argnums=2)
def jnp_apply_q1_gate_conj_transposed(
  state: jnp.ndarray,
  gate: jnp.ndarray,
  pos: int,
) -> jnp.ndarray:
  qubits_number = len(state.shape)
  nppos = qubits_number - 1 - pos
  gate = gate.conj()
  state = jnp.tensordot(gate, state, axes=[[0], [nppos]])
  state = state.reshape((2, 2 ** nppos, 2 ** (qubits_number - nppos - 1)))
  state = state.transpose((1, 0, 2))
  return state.reshape((-1,))

@partial(jit, static_argnums=[1, 2])
def jnp_get_q2_density_matrix(
  state: jnp.ndarray,
  pos2: int,
  pos1: int,
) -> jnp.ndarray:
  qubits_number = len(state.shape)
  nppos1 = qubits_number - 1 - pos1
  nppos2 = qubits_number - 1 - pos2
  min_pos = min(nppos1, nppos2)
  max_pos = max(nppos1, nppos2)
  state = state.reshape((2 ** min_pos, 2, 2 ** (max_pos - min_pos - 1), 2, 2 ** (qubits_number - max_pos - 1)))
  conj_state = jnp.conj(state)
  dens = jnp.tensordot(state, conj_state, axes=[[0, 2, 4], [0, 2, 4]])
  if nppos2 > nppos1:
    dens = dens.transpose((1, 0, 3, 2))
  return dens.reshape((4, 4))

@partial(jit, static_argnums=1)
def jnp_get_q1_density_matrix(
  state: jnp.ndarray,
  pos: int,
) -> jnp.ndarray:
  qubits_number = len(state.shape)
  nppos = qubits_number - 1 - pos
  state = state.reshape((2 ** nppos, 2, 2 ** (qubits_number - nppos - 1)))
  conj_state = jnp.conj(state)
  dens = jnp.tensordot(state, conj_state, axes=[[0, 2], [0, 2]])
  return dens.reshape((2, 2))

@partial(jit, static_argnums=[2, 3])
def jnp_q2_gradient(bwd_state: jnp.ndarray, fwd_state: jnp.ndarray, pos2: int, pos1: int):
  qubits_number = len(bwd_state.shape)
  nppos1 = qubits_number - 1 - pos1
  nppos2 = qubits_number - 1 - pos2
  min_pos = min(nppos1, nppos2)
  max_pos = max(nppos1, nppos2)
  bwd_state = bwd_state.reshape((2 ** min_pos, 2, 2 ** (max_pos - min_pos - 1), 2, 2 ** (qubits_number - max_pos - 1)))
  fwd_state = fwd_state.reshape((2 ** min_pos, 2, 2 ** (max_pos - min_pos - 1), 2, 2 ** (qubits_number - max_pos - 1)))
  grad = jnp.tensordot(bwd_state, fwd_state, axes=[[0, 2, 4], [0, 2, 4]])
  if nppos2 > nppos1:
    grad = grad.transpose((1, 0, 3, 2))
  return grad.reshape((4, 4))

@partial(jit, static_argnums=2)
def jnp_q1_gradient(bwd_state: jnp.ndarray, fwd_state: jnp.ndarray, pos: int):
  qubits_number = len(bwd_state.shape)
  nppos = qubits_number - 1 - pos
  bwd_state = bwd_state.reshape((2 ** nppos, 2, 2 ** (qubits_number - nppos - 1)))
  fwd_state = fwd_state.reshape((2 ** nppos, 2, 2 ** (qubits_number - nppos - 1)))
  grad = jnp.tensordot(bwd_state, fwd_state, axes=[[0, 2], [0, 2]])
  return grad
