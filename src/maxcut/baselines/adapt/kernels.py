from typing import List
import cudaq


@cudaq.kernel
def _adapt_initial_state(qubits_num: int):
    qubits = cudaq.qvector(qubits_num)
    h(qubits)


@cudaq.kernel
def _adapt_echo_kernel(state: cudaq.State):
    q = cudaq.qvector(state)


@cudaq.kernel
def _adapt_grad_kernel(
    state: cudaq.State,
    ham_word: List[cudaq.pauli_word],
    ham_coef: List[complex],
    init_gamma: float
):
    q = cudaq.qvector(state)
    for i in range(len(ham_coef)):
        exp_pauli(init_gamma * ham_coef[i].real, q, ham_word[i])


@cudaq.kernel
def adapt_qaoa_kernel(
    qubits_num: int,
    ham_word: List[cudaq.pauli_word],
    ham_coef: List[complex],
    mixer_pool: List[List[cudaq.pauli_word]],
    gamma: List[float],
    beta: List[float],
    num_layer: int
):
    qubits = cudaq.qvector(qubits_num)
    h(qubits)
    for p in range(num_layer):
        for i in range(len(ham_coef)):
            exp_pauli(gamma[p] * ham_coef[i].real, qubits, ham_word[i])
        for word in mixer_pool[p]:
            exp_pauli(beta[p], qubits, word)
