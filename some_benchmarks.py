from jax.config import config
config.update('jax_platform_name', 'cpu')
config.update("jax_enable_x64", True)
import time
import jax.numpy as jnp
from jax import random, jit, grad
import numpy as np
# --------this is not a public API------- #
from differentiable_circuit import (
  apply_q1_gate,
  apply_q2_gate,
  get_q1_density_matrix,
  get_q2_density_matrix,
  q2_gradient,
  q1_gradient,
)
# ---------------------------------------- #
from circuit_autograd import q1_density_matrix, q2_density_matrix
from circuit_autograd.testing_utils import (
  jnp_apply_q2_gate,
  jnp_apply_q1_gate,
  jnp_get_q2_density_matrix,
  jnp_get_q1_density_matrix,
  jnp_q2_gradient,
  jnp_q1_gradient,
)

key = random.PRNGKey(42)

# Note, that here we mutate jnp.ndarray that is normaly forbidden
# but in the public API it does not happen

# --------------------------------------------------- #
# Here we benchmark primitive operations over a state
# --------------------------------------------------- #

# positions for a two-qubit operations
pos2 = 18
pos1 = 10

# posiyion for a one-qubit operation
pos = 14

# random state
key, subkey = random.split(key)
state = random.normal(subkey, shape = 25 * (2,) + (2,))
state = state[..., 0] + 1j * state[..., 1]

# random two-qubit gate
key, subkey = random.split(key)
q2gate = random.normal(subkey, shape = (4, 4, 2))
q2gate = q2gate[..., 0] + 1j * q2gate[..., 1]

# random one-qubit gate
key, subkey = random.split(key)
q1gate = random.normal(subkey, shape = (2, 2, 2))
q1gate = q1gate[..., 0] + 1j * q1gate[..., 1]

jnp_apply_q2_gate(state, q2gate, pos2, pos1).block_until_ready()  # precompile before run
jnp_start = time.time()
for _ in range(10):
  jnp_apply_q2_gate(state, q2gate, pos2, pos1).block_until_ready()
jnp_end = time.time()
jnp_time = (jnp_end - jnp_start) / 10

state = state.reshape((2 ** 25,))
rust_start = time.time()
for _ in range(10):
  apply_q2_gate(np.asarray(state), np.asarray(q2gate), pos2, pos1)
rust_end = time.time()
rust_time = (rust_end - rust_start) / 10

print("For 25 qubits and a two-qubit operation one has jax time: {} secs, rust time: {} secs".format(jnp_time, rust_time))

state = state.reshape((2,) * 25)
jnp_apply_q1_gate(state, q1gate, pos).block_until_ready()  # precompile before run
jnp_start = time.time()
for _ in range(10):
  jnp_apply_q1_gate(state, q1gate, pos).block_until_ready()
jnp_end = time.time()
jnp_time = (jnp_end - jnp_start) / 10

state = state.reshape((2 ** 25,))
rust_start = time.time()
for _ in range(10):
  apply_q1_gate(np.asarray(state), np.asarray(q1gate), pos)
rust_end = time.time()
rust_time = (rust_end - rust_start) / 10

print("For 25 qubits and a one-qubit operation one has jax time: {} secs, rust time: {} secs".format(jnp_time, rust_time))

state = state.reshape((2,) * 25)
jnp_get_q2_density_matrix(state, pos2, pos1).block_until_ready()  # precompile before run
jnp_start = time.time()
for _ in range(10):
  jnp_get_q2_density_matrix(state, pos2, pos1).block_until_ready()
jnp_end = time.time()
jnp_time = (jnp_end - jnp_start) / 10

state = state.reshape((2 ** 25,))
rust_start = time.time()
for _ in range(10):
  get_q2_density_matrix(np.asarray(state), pos2, pos1)
rust_end = time.time()
rust_time = (rust_end - rust_start) / 10

print("For 25 qubits one evaluates a two-qubit density matrix in {} secs for jax, in: {} secs for rust".format(jnp_time, rust_time))

state = state.reshape((2,) * 25)
jnp_get_q1_density_matrix(state, pos).block_until_ready()  # precompile before run
jnp_start = time.time()
for _ in range(10):
  jnp_get_q1_density_matrix(state, pos).block_until_ready()
jnp_end = time.time()
jnp_time = (jnp_end - jnp_start) / 10

state = state.reshape((2 ** 25,))
rust_start = time.time()
for _ in range(10):
  get_q1_density_matrix(np.asarray(state), pos)
rust_end = time.time()
rust_time = (rust_end - rust_start) / 10

print("For 25 qubits one evaluates a one-qubit density matrix in {} secs for jax, in: {} secs for rust".format(jnp_time, rust_time))

bwd_state = state.reshape((2,) * 25)
key, subkey = random.split(key)
fwd_state = random.normal(subkey, shape = 25 * (2,) + (2,))
fwd_state = fwd_state[..., 0] + 1j * fwd_state[..., 1]

jnp_q2_gradient(bwd_state, fwd_state, pos2, pos1).block_until_ready()  # precompile before run
jnp_start = time.time()
for _ in range(10):
  jnp_q2_gradient(bwd_state, fwd_state, pos2, pos1).block_until_ready()
jnp_end = time.time()
jnp_time = (jnp_end - jnp_start) / 10

fwd_state = fwd_state.reshape((2 ** 25,))
bwd_state = bwd_state.reshape((2 ** 25,))
rust_start = time.time()
for _ in range(10):
  q2_gradient(np.asarray(bwd_state), np.asarray(fwd_state), pos2, pos1)
rust_end = time.time()
rust_time = (rust_end - rust_start) / 10

print("For 25 qubits one evaluates a two-qubit gate gradient in {} secs for jax, in: {} secs for rust".format(jnp_time, rust_time))

bwd_state = bwd_state.reshape((2,) * 25)
fwd_state = fwd_state.reshape((2,) * 25)
jnp_q1_gradient(bwd_state, fwd_state, pos).block_until_ready()  # precompile before run
jnp_start = time.time()
for _ in range(10):
  jnp_q1_gradient(bwd_state, fwd_state, pos).block_until_ready()
jnp_end = time.time()
jnp_time = (jnp_end - jnp_start) / 10

fwd_state = fwd_state.reshape((2 ** 25,))
bwd_state = bwd_state.reshape((2 ** 25,))
rust_start = time.time()
for _ in range(10):
  q1_gradient(np.asarray(bwd_state), np.asarray(fwd_state), pos)
rust_end = time.time()
rust_time = (rust_end - rust_start) / 10

print("For 25 qubits one evaluates a one-qubit gate gradient in {} secs for jax, in: {} secs for rust".format(jnp_time, rust_time))

# ------------------------------------------------------------ #
# Here we benchmark differentiable density matrices evaluation
# ------------------------------------------------------------ #

def reney_entropy_2(dens_matrix: jnp.array):
  return -jnp.log(jnp.einsum("ij,ji->", dens_matrix, jnp.conj(dens_matrix)))

@grad
def two_qubit_entropy_grad(state: jnp.array):
  return jnp.abs(reney_entropy_2(q2_density_matrix(state, 10, 17)))

@jit
@grad
def jnp_two_qubit_entropy_grad(state: jnp.array):
  return jnp.abs(reney_entropy_2(jnp_get_q2_density_matrix(state, 10, 17)))

@grad
def one_qubit_entropy_grad(state: jnp.array):
  return jnp.abs(reney_entropy_2(q1_density_matrix(state, 16)))

@jit
@grad
def jnp_one_qubit_entropy_grad(state: jnp.array):
  return jnp.abs(reney_entropy_2(jnp_get_q1_density_matrix(state, 16)))

state = state.reshape((-1,))
two_qubit_entropy_grad(state)  # to make sure that the function is compiled
rust_start = time.time()
for _ in range(10):
  two_qubit_entropy_grad(state)
rust_end = time.time()
rust_time = (rust_end - rust_start) / 10

state = state.reshape((2,) * 25)
jnp_two_qubit_entropy_grad(state).block_until_ready()  # to make sure that the function is compiled
jnp_start = time.time()
for _ in range(10):
  jnp_two_qubit_entropy_grad(state).block_until_ready()  # to make sure that the function is compiled
jnp_end = time.time()
jnp_time = (jnp_end - jnp_start) / 10
print("For 25 qubits one evaluates a gradient of the Reney entropy of a two-qubit density matrix in {} secs for jax, in: {} secs for rust".format(jnp_time, rust_time))

state = state.reshape((-1,))
one_qubit_entropy_grad(state)  # to make sure that the function is compiled
rust_start = time.time()
for _ in range(10):
  one_qubit_entropy_grad(state)
rust_end = time.time()
rust_time = (rust_end - rust_start) / 10

state = state.reshape((2,) * 25)
jnp_one_qubit_entropy_grad(state).block_until_ready()  # to make sure that the function is compiled
jnp_start = time.time()
for _ in range(10):
  jnp_one_qubit_entropy_grad(state).block_until_ready()  # to make sure that the function is compiled
jnp_end = time.time()
jnp_time = (jnp_end - jnp_start) / 10
print("For 25 qubits one evaluates a gradient of the Reney entropy of a one-qubit density matrix in {} secs for jax, in: {} secs for rust".format(jnp_time, rust_time))
