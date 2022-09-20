from jax.config import config
config.update('jax_platform_name', 'cpu')
config.update("jax_enable_x64", True)
import time
from typing import List
import jax.numpy as jnp
from jax import grad, random, jit
import matplotlib.pyplot as plt
from jax.scipy.linalg import expm
from circuit_autograd.circuit_autograd import AutoGradCircuit, q1_density_matrix

#------------------------------- Params -----------------------------------#
dt =0.05
num_layers_before_control = 90
num_layers_after_control = 90
qubits_number = 21
#--------------------------------------------------------------------------#

key = random.PRNGKey(42)

# two-qubit evolution operator
x = jnp.array([[0, 1], [1, 0]], dtype=jnp.complex128)
y = jnp.array([[0, -1j], [1j, 0]], dtype=jnp.complex128)
z = jnp.array([[1, 0], [0, -1]], dtype=jnp.complex128)

xx = jnp.tensordot(x, x, axes=0).transpose((0, 2, 1, 3)).reshape((4, 4))
yy = jnp.tensordot(y, y, axes=0).transpose((0, 2, 1, 3)).reshape((4, 4))
zz = jnp.tensordot(z, z, axes=0).transpose((0, 2, 1, 3)).reshape((4, 4))

u = expm(-1j * dt * (1.2 * xx + yy + 0.8 * zz))

# Hadamard gate
h = (1. / jnp.sqrt(2.)) * jnp.array([[1., 1.], [1., -1.]], dtype=jnp.complex128)

# initial state
state = jnp.zeros((2 ** qubits_number,), dtype=jnp.complex128)
state = state.at[0].set(1.)

# this function returns one-qubit gate given the parameters
@jit
def q1_gate(param):
  return jnp.array([
    [jnp.cos(param[0] / 2), -jnp.exp(1j * param[1]) * jnp.sin(param[0] / 2)],
    [jnp.exp(1j * param[2]) * jnp.sin(param[0] / 2), jnp.exp(1j * (param[2] + param[1])) * jnp.cos(param[0] / 2)]])

# here we define a circuit
c = AutoGradCircuit()
c.add_q1_const_gate(6)
c.add_q1_const_gate(15)
for layer in range(num_layers_before_control):
  for i in range(0, qubits_number-1, 2):
    c.add_q2_const_gate(i, i+1)
  for i in range(1, qubits_number-1, 2):
    c.add_q2_const_gate(i, i+1)
  if layer % 2 == 0:
    for i in range(qubits_number):
      c.get_q1_dens_op(i)
for layer in range(num_layers_after_control):
  for i in range(0, qubits_number-1, 2):
    c.add_q2_const_gate(i, i+1)
  for i in range(1, qubits_number-1, 2):
    c.add_q2_const_gate(i, i+1)
  for i in range(qubits_number):
    c.add_q1_var_gate(i)
  if layer % 2 == 0:
    for i in range(qubits_number):
      c.get_q1_dens_op(i)
run_circ, fwd_circ = c.build()

# here we define constant gates
const_gates = 2 * [h] + num_layers_before_control * (qubits_number - 1) * [u] +\
    num_layers_after_control * ((qubits_number - 1) * [u])

key, subkey = random.split(key)
# here we define parameters of the circuit
params = random.normal(subkey, shape=(num_layers_after_control * qubits_number, 3))

# Reney entropy necessary for a loss function
@jit
def reney_entropy_2(dens_matrix: jnp.array):
    return -jnp.log(jnp.einsum("ij,ji->", dens_matrix, jnp.conj(dens_matrix)))

# here we define a penalty function
def penalty(state: jnp.array, var_gates: List[jnp.array], const_gates: List[jnp.array]):
  final_state = fwd_circ(state, var_gates, const_gates)
  entropy = 0.
  for i in range(qubits_number):
    entropy = entropy + reney_entropy_2(q1_density_matrix(final_state, i))
  return jnp.abs(entropy) + jnp.abs(jnp.array(var_gates)).sum()

# here we define a loss function
def loss(state, params, const_gates):
  var_gates = [q1_gate(param) for param in params]
  return penalty(state, var_gates, const_gates)

loss_grad = grad(loss, argnums=[0, 1])

start = time.time()
loss_grad(state, params, const_gates)
end = time.time()
print("First gradient computation time (compilation included): {} secs".format(end - start))

start = time.time()
_, params_grad = loss_grad(state, params, const_gates)
end = time.time()
print("Second gradient computation time: {} secs".format(end - start))

# here we check correctness of the gradient
eta = 1e-6
key, subkey = random.split(key)
params_perturbation = random.normal(subkey, shape=params.shape)
params + eta * params_perturbation
l1 = loss(state, params + eta * params_perturbation, const_gates)
l0 = loss(state, params - eta * params_perturbation, const_gates)
dl = jnp.tensordot(params_perturbation, params_grad, axes=[[0, 1], [0, 1]])

print("Exact derivative: {}".format(dl))
print("Approximate derivative: {}".format((l1 - l0) / (2 * eta)))

# let us also plot population dynamics with random control

var_gates = [q1_gate(param) for param in params]
_, density_matrices = run_circ(state, var_gates, const_gates)
population = jnp.abs(jnp.array(density_matrices).reshape((-1, qubits_number, 4))[..., 0])
plt.imshow(population)
plt.show()


