from jax.config import config
config.update('jax_platform_name', 'cpu')
config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import random
from jax import grad
from circuit_autograd import (
  q2_density_matrix,
  q1_density_matrix,
)
from circuit_autograd.testing_utils import (
  jnp_get_q1_density_matrix,
  jnp_get_q2_density_matrix,
)

def test_density_matrix_autodiff():
  qubits_number = 10
  key = random.PRNGKey(42)
  key, subkey = random.split(key)
  state = random.normal(subkey, shape = (2,) * (qubits_number + 1))
  state = state[..., 0] + 1j * state[..., 1]
  state /= jnp.linalg.norm(state)

  def reney_entropy_2(dens_matrix: jnp.ndarray):
    return -jnp.log(jnp.einsum("ij,ji->", dens_matrix, jnp.conj(dens_matrix)))

  def reney_sum(state: jnp.ndarray):
    s = 0.
    for i in range(qubits_number):
      s = s + reney_entropy_2(q1_density_matrix(state, i))
    for i in range(1, qubits_number):
      s = s + reney_entropy_2(q2_density_matrix(state, i, 0))
    return jnp.abs(s)

  def jnp_reney_sum(state: jnp.ndarray):
    s = 0.
    for i in range(qubits_number):
      s = s + reney_entropy_2(jnp_get_q1_density_matrix(state, i))
    for i in range(1, qubits_number):
      s = s + reney_entropy_2(jnp_get_q2_density_matrix(state, i, 0))
    return jnp.abs(s)
  jnp_dens_matrix_grad = grad(jnp_reney_sum)(state)
  state = state.reshape((-1,))
  dens_matrix_grad = grad(reney_sum)(state)
  assert jnp.isclose(dens_matrix_grad.reshape((2,) * qubits_number), jnp_dens_matrix_grad).all()
