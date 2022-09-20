from jax.config import config
config.update('jax_platform_name', 'cpu')
config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import random
from jax import grad
from circuit_autograd import AutoGradCircuit


def test_autodiff():
  layers = 10  # number of layers in a circuit
  qubits_number = 21  # number of qubits in a circuit
  eta = 1e-10  # perturbation step

  key = random.PRNGKey(42)
  
  # target state
  key, subkey = random.split(key)
  target_state_conj = random.normal(subkey, shape = (2 ** qubits_number, 2))
  target_state_conj = target_state_conj[..., 0] + 1j * target_state_conj[..., 1]
  target_state_conj /= jnp.linalg.norm(target_state_conj)

  # initial state
  initial_state = jnp.zeros(2 ** qubits_number, dtype=jnp.complex128)
  initial_state = initial_state.at[0].set(1.)

  # cnot gate
  cnot = jnp.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=jnp.complex128)

  # random unitary matrix generator
  def random_unitary(key: random.PRNGKey, size: int):
    u = random.normal(key, shape = (size, size, 2))
    u = u[..., 0] + 1j * u[..., 1]
    q, _ = jnp.linalg.qr(u)
    return q


  # random complex matrix generator
  def random_complex(key: random.PRNGKey, size: int):
    a = random.normal(key, shape = (size, size, 2))
    return a[..., 0] + 1j * a[..., 1]

  # here we define a circuit structure
  c = AutoGradCircuit()
  for _ in range(layers):
    for i in range(qubits_number):
      c.add_q1_var_gate(i)
    for i in range(0, qubits_number-1, 2):
      c.add_q2_var_gate(i+1, i)
    for i in range(qubits_number):
      c.add_q1_const_gate(i)
    for i in range(1, qubits_number-1, 2):
      c.add_q2_const_gate(i+1, i)
    for i in range(i):
      c.get_q1_dens_op(i)

  # this finction run a circuit and supports backprop.
  _, fwd_circ = c.build()

  # fidelity function
  def fidelity(state, var_gates, const_gates):
    return jnp.abs(jnp.dot(target_state_conj, fwd_circ(state, var_gates, const_gates)))

  # here we define circuit gates
  const_gates = []
  for _ in range(layers):
    key, subkey = random.split(key)
    const_gates += [random_unitary(k, 2) for k in random.split(subkey, qubits_number)]
    const_gates += int((qubits_number - 1) / 2) * [cnot]

  var_gates = []
  for _ in range(layers):
    key, subkey = random.split(key)
    var_gates += [random_unitary(k, 2) for k in random.split(subkey, qubits_number)]
    var_gates += int((qubits_number - 1) / 2) * [cnot]

  # here we define perturbated gates
  gates_perturbation = []
  for _ in range(layers):
    key, subkey = random.split(key)
    gates_perturbation += [random_complex(k, 2) for k in random.split(subkey, qubits_number)]
    key, subkey = random.split(key)
    gates_perturbation += [random_complex(k, 4) for k in random.split(subkey, int((qubits_number - 1) / 2))]
  perturbated_var_gates = [lhs + eta * rhs for (lhs, rhs) in zip(var_gates, gates_perturbation)]

  # here we define a perturbated state
  key, subkey = random.split(key)
  state_perturbation = random.normal(subkey, shape = (2 ** qubits_number, 2))
  state_perturbation = state_perturbation[..., 0] + 1j * state_perturbation[..., 1]
  perturbated_initial_state = initial_state + eta * state_perturbation

  # a fidelity value and a perturbated fidelity value
  f = fidelity(initial_state, var_gates, const_gates)
  perturb_f = fidelity(perturbated_initial_state, perturbated_var_gates, const_gates)

  # gradients wrt initials state and gates
  state_grad, gates_grad = grad(fidelity, argnums=[0, 1])(initial_state, var_gates, const_gates)

  # here we calculate df via gradients
  df = 0.
  for (p, g) in zip(gates_perturbation, gates_grad):
    df += eta * jnp.tensordot(g, p, axes = [[0, 1], [0, 1]]).real

  df += eta * (jnp.tensordot(state_grad, state_perturbation, axes=1)).real
  assert jnp.abs(perturb_f - f - df) / jnp.abs(df) < 1e-4