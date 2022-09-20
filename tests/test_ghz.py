from jax.config import config
config.update("jax_enable_x64", True)
config.update('jax_platform_name', 'cpu')
import jax.numpy as jnp
from circuit_autograd import AutoGradCircuit

def test_ghz():
  qubits_number = 15
  c = AutoGradCircuit()
  c.add_q1_const_gate(0)
  for i in range(qubits_number - 1):
    c.add_q2_const_gate(i, i+1)
  for i in range(qubits_number):
    c.get_q1_dens_op(i)
  for i in range(1, qubits_number):
    c.get_q2_dens_op(0, i)
  circ_run, _ = c.build()

  state = jnp.zeros(2 ** qubits_number, dtype=jnp.complex128)
  state = state.at[0].set(1.)
  h = (1. / jnp.sqrt(2.)) * jnp.array([[1., 1.], [1., -1.]], dtype=jnp.complex128)
  cnot = jnp.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=jnp.complex128)
  gates = [h] + (qubits_number - 1) * [cnot]
  state, density_matrices = circ_run(state, [], gates)

  ghz_state = jnp.zeros(2 ** qubits_number)
  ghz_state = ghz_state.at[0].set(1. / jnp.sqrt(2.))
  ghz_state = ghz_state.at[-1].set(1. / jnp.sqrt(2.))

  q1_dens = jnp.zeros((2, 2))
  q1_dens = q1_dens.at[0, 0].set(0.5)
  q1_dens = q1_dens.at[-1, -1].set(0.5)

  q2_dens = jnp.zeros((4, 4))
  q2_dens = q2_dens.at[0, 0].set(0.5)
  q2_dens = q2_dens.at[-1, -1].set(0.5)

  assert jnp.isclose(ghz_state, state).all()

  for dens in density_matrices[:qubits_number]:
    assert jnp.isclose(q1_dens, dens).all()

  for dens in density_matrices[qubits_number:]:
    assert jnp.isclose(q2_dens, dens).all()